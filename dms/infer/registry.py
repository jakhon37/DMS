"""ModelRegistry: engine lifetime, per-capability backends, honest degradation.

One object owns every engine the DMS perception stack uses.  It reads
``engines/MANIFEST.json`` to learn which file implements which role, loads the
ones the requested backends need, hands out ready-to-use wrappers by capability
name, and keeps a :class:`dms.types.ModelHealth` record per capability.

THE CONTRACT THE MAIN LOOP RELIES ON: a missing, corrupt, or failing engine
NEVER raises into the frame loop.  It degrades.  Each capability resolves to one
of three states and says which:

    ``model``      the engine loaded and is answering        source=MODEL
    ``geometry``   no engine, a classical implementation ran  source=GEOMETRY
    ``unavailable``nothing can answer                         source=UNAVAILABLE

:meth:`ModelRegistry.degraded_capabilities` lists everything that asked for
``model`` and did not get it, which is what an ``AlertType.MODEL_DEGRADED``
signal and the ``/healthz`` engine map should be built from.  A health surface
that claims a safety model it is not running is worse than one that says "off",
so :meth:`health_map` reports the state actually achieved, never the state
requested.

Backends are chosen per capability by the caller::

    reg = ModelRegistry(backends={"headpose": "model", "gaze": "off"})
    reg.load()
    pose = reg.headpose.infer(bgr, face.box_for_crop(), landmarks)

Accepted backend values are ``"model"``, ``"geometry"`` and ``"off"``.
``"geometry"`` skips loading the engine entirely even if it exists, which is how
a memory-constrained profile keeps head pose without the 4.9 MB engine.
``"off"`` disables the capability; the wrapper still exists and still returns a
well-formed dataclass, with ``source=UNAVAILABLE``.  A capability the caller does
not name takes its own default, which is ``"model"`` for everything except gaze -
gaze defaults to ``"off"`` because its weights are research-licence-only, and
even when it is explicitly requested the engine is not deserialised unless
``research_models_allowed`` is set.

Units and ownership: the registry owns the :class:`dms.runtime.trt_engine.TrtEngine`
objects and closes them in :meth:`close`.  Wrappers borrow them and must not
close them independently.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

from dms.infer.eyestate import EyeStateEstimator
from dms.infer.gaze import GazeEstimator
from dms.infer.headpose import HeadPoseEstimator
from dms.infer.identity import IdentityEmbedder
from dms.infer.mouthstate import MouthStateEstimator
from dms.infer.objects import ObjectDetector
from dms.infer.occlusion import OcclusionDetector
from dms.types import ModelHealth

log = logging.getLogger("dms.infer.registry")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_ENGINE_DIR = os.path.join(ROOT, "engines")

BACKEND_MODEL = "model"
BACKEND_GEOMETRY = "geometry"
BACKEND_OFF = "off"
VALID_BACKENDS = (BACKEND_MODEL, BACKEND_GEOMETRY, BACKEND_OFF)


class Capability(object):
    """Static description of one capability's engine needs.

    ``roles`` are MANIFEST roles, never filenames: the manifest is the source of
    truth for which file implements a role, so an engine rebuilt at a different
    resolution or swapped for a faster variant needs no code change here.

    ``alternatives`` says whether the roles are interchangeable.  When they are,
    loading stops at the first one that succeeds - which matters on a board with
    ~2.5 GB free, because loading the second head-pose variant "just in case"
    costs 4.9 MB of weights plus 1.7 MB of context scratch that nothing will ever
    execute.  When they are not (the occlusion gate needs the sunglasses head AND
    the mask head; they answer different questions) every role is loaded.

    ``has_geometry`` says whether a classical implementation can stand in.

    ``default_backend`` is what the capability does when the caller does not
    name it.  Everything defaults to ``"model"`` except gaze, which defaults to
    ``"off"`` because its weights are licence-restricted (see below).

    ``licence_gated`` marks a capability whose engine must not even be
    deserialised unless ``research_models_allowed`` is set.  Loading
    licence-restricted weights into a production process and then declining to
    call them is a weaker gate than never loading them, and it costs 5.9 MB of
    weights plus 16.9 MB of context scratch for nothing.
    """

    __slots__ = ("roles", "alternatives", "has_geometry", "default_backend", "licence_gated")

    def __init__(self, roles, alternatives, has_geometry, default_backend=None, licence_gated=False):
        self.roles = tuple(roles)
        self.alternatives = bool(alternatives)
        self.has_geometry = bool(has_geometry)
        self.default_backend = default_backend or BACKEND_MODEL
        self.licence_gated = bool(licence_gated)


#: ``head_pose_fallback`` is listed second because mobilenetv3-small is SLOWER
#: than mobilenetv2 (2.45 vs 1.49 ms median, with a 9.4 ms p99 tail) while
#: needing half the context scratch - it is the memory-pressure option, not the
#: accuracy option, so it is only reached when the primary is absent.
CAPABILITIES: Dict[str, Capability] = {
    "headpose": Capability(("head_pose", "head_pose_fallback"), True, True),
    "eyestate": Capability(("eye_state_pair", "eye_state"), True, True),
    "mouthstate": Capability(("mouth_state",), True, True),
    "occlusion": Capability(("eye_occlusion", "mouth_occlusion"), False, False),
    "gaze": Capability(("gaze",), True, False, default_backend=BACKEND_OFF, licence_gated=True),
    "objects": Capability(("cabin_objects",), True, False),
    "identity": Capability(("driver_identity",), True, False),
}


class ModelRegistry:
    """Loads and owns the DMS model layer."""

    def __init__(
        self,
        *,
        engine_dir: Optional[str] = None,
        manifest: Optional[dict] = None,
        manifest_path: Optional[str] = None,
        backends: Optional[Dict[str, str]] = None,
        capability_cfg: Optional[Dict[str, dict]] = None,
        research_models_allowed: bool = False,
        engine_factory=None,
    ) -> None:
        self.engine_dir = engine_dir or DEFAULT_ENGINE_DIR
        self.manifest_path = manifest_path
        self._manifest = manifest
        self.research_models_allowed = bool(research_models_allowed)
        self.capability_cfg = dict(capability_cfg or {})
        self._engine_factory = engine_factory
        self.backends: Dict[str, str] = {}
        for cap, spec in CAPABILITIES.items():
            want = (backends or {}).get(cap, spec.default_backend)
            if want not in VALID_BACKENDS:
                raise ValueError(
                    "backend for %r must be one of %s, got %r" % (cap, VALID_BACKENDS, want)
                )
            self.backends[cap] = want

        self._engines: Dict[str, object] = {}  # role -> engine
        self._role_files: Dict[str, str] = {}  # role -> absolute path
        self._state: Dict[str, str] = {cap: BACKEND_OFF for cap in CAPABILITIES}
        self._load_errors: Dict[str, str] = {}

        self.headpose: Optional[HeadPoseEstimator] = None
        self.eyestate: Optional[EyeStateEstimator] = None
        self.mouthstate: Optional[MouthStateEstimator] = None
        self.occlusion: Optional[OcclusionDetector] = None
        self.gaze: Optional[GazeEstimator] = None
        self.objects: Optional[ObjectDetector] = None
        self.identity: Optional[IdentityEmbedder] = None
        self._loaded = False

    # ------------------------------------------------------------ manifest --
    def manifest(self) -> dict:
        """The parsed MANIFEST, loaded lazily. Returns ``{}`` if unreadable."""
        if self._manifest is None:
            try:
                from dms.runtime.manifest import load_manifest

                self._manifest = load_manifest(self.manifest_path)
            except Exception as exc:  # noqa: BLE001 - a missing manifest degrades
                log.warning("MANIFEST unreadable (%s); no engines will be loaded", exc)
                self._manifest = {}
        return self._manifest

    def role_path(self, role: str) -> Optional[str]:
        """Absolute path of the engine implementing ``role``, if it is on disk."""
        engines = (self.manifest().get("engines") or {})
        for rec in engines.values():
            if not isinstance(rec, dict) or rec.get("role") != role:
                continue
            fname = rec.get("file")
            if not fname:
                continue
            path = os.path.join(self.engine_dir, str(fname))
            if os.path.isfile(path):
                return path
            log.warning("MANIFEST role %s names %s, which is not on disk", role, fname)
        return None

    # -------------------------------------------------------------- loading --
    def _make_engine(self, path: str):
        if self._engine_factory is not None:
            return self._engine_factory(path)
        from dms.runtime.trt_engine import TrtEngine

        return TrtEngine(path)

    def _engine_for_role(self, role: str):
        """Load (or reuse) the engine for a role. Returns None on any failure."""
        if role in self._engines:
            return self._engines[role]
        path = self.role_path(role)
        if path is None:
            return None
        try:
            engine = self._make_engine(path)
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the loop
            log.error("failed to load %s for role %s: %s", os.path.basename(path), role, exc)
            self._load_errors[role] = str(exc)
            return None
        self._engines[role] = engine
        self._role_files[role] = path
        log.info("loaded role=%s engine=%s", role, os.path.basename(path))
        return engine

    def _engines_for(self, cap: str) -> Dict[str, object]:
        """The engines to give a capability's wrapper, keyed by role.

        Stops after the first success when the roles are alternatives, so a
        preferred engine that loads means the fallback engine is never
        deserialised and never occupies device memory.
        """
        spec = CAPABILITIES[cap]
        if self.backends[cap] != BACKEND_MODEL:
            return {}
        if spec.licence_gated and not self.research_models_allowed:
            log.warning(
                "capability %r is licence-gated and research_models_allowed is "
                "false: its engine will not be loaded at all",
                cap,
            )
            return {}
        found: Dict[str, object] = {}
        for role in spec.roles:
            eng = self._engine_for_role(role)
            if eng is not None:
                found[role] = eng
                if spec.alternatives:
                    break
        return found

    def load(self) -> "ModelRegistry":
        """Construct every wrapper. Idempotent; safe to call once at startup."""
        if self._loaded:
            return self

        cfg = self.capability_cfg

        hp = self._engines_for("headpose")
        hp_engine = hp.get("head_pose") or hp.get("head_pose_fallback")
        self.headpose = HeadPoseEstimator(
            engine=hp_engine,
            cfg=cfg.get("headpose"),
            geometry_fallback=self.backends["headpose"] != BACKEND_OFF,
        )

        eye = self._engines_for("eyestate")
        self.eyestate = EyeStateEstimator(
            engine=eye.get("eye_state_pair"),
            single_engine=eye.get("eye_state"),
            cfg=cfg.get("eyestate"),
            geometry_fallback=self.backends["eyestate"] != BACKEND_OFF,
        )

        mouth = self._engines_for("mouthstate")
        self.mouthstate = MouthStateEstimator(
            engine=mouth.get("mouth_state"),
            cfg=cfg.get("mouthstate"),
            geometry_fallback=self.backends["mouthstate"] != BACKEND_OFF,
        )

        occ = self._engines_for("occlusion")
        self.occlusion = OcclusionDetector(
            sunglasses_engine=occ.get("eye_occlusion"),
            mask_engine=occ.get("mouth_occlusion"),
            cfg=cfg.get("occlusion"),
        )

        gz = self._engines_for("gaze")
        self.gaze = GazeEstimator(
            engine=gz.get("gaze"),
            cfg=cfg.get("gaze"),
            allow_research_models=self.research_models_allowed,
        )

        obj = self._engines_for("objects")
        self.objects = ObjectDetector(engine=obj.get("cabin_objects"), cfg=cfg.get("objects"))

        idt = self._engines_for("identity")
        self.identity = IdentityEmbedder(
            engine=idt.get("driver_identity"), cfg=cfg.get("identity")
        )

        for cap in CAPABILITIES:
            self._state[cap] = self._resolve_state(cap)
        self._loaded = True
        log.info("model registry: %s", self.describe())
        return self

    def _resolve_state(self, cap: str) -> str:
        want = self.backends[cap]
        if want == BACKEND_OFF:
            return BACKEND_OFF
        wrapper = self.get(cap)
        if wrapper is not None and wrapper.available:
            return BACKEND_MODEL
        if CAPABILITIES[cap].has_geometry and want != BACKEND_OFF:
            return BACKEND_GEOMETRY
        return BACKEND_OFF

    # -------------------------------------------------------------- queries --
    def get(self, capability: str):
        """The wrapper for a capability, or None if the name is unknown."""
        return getattr(self, capability, None) if capability in CAPABILITIES else None

    def state(self, capability: str) -> str:
        """``"model"`` / ``"geometry"`` / ``"off"`` - what actually happened."""
        return self._state.get(capability, BACKEND_OFF)

    def describe(self) -> Dict[str, str]:
        """capability -> achieved state, for logs and /healthz."""
        return dict(self._state)

    def health_map(self) -> Dict[str, ModelHealth]:
        """capability -> live :class:`dms.types.ModelHealth`."""
        out: Dict[str, ModelHealth] = {}
        for cap in CAPABILITIES:
            wrapper = self.get(cap)
            if wrapper is not None:
                out[cap] = wrapper.health
        return out

    def degraded_capabilities(self) -> List[str]:
        """Capabilities that asked for a model and are not running one.

        This is the list an ``AlertType.MODEL_DEGRADED`` event should carry.  A
        capability explicitly configured ``"geometry"`` or ``"off"`` is not
        degraded - it is doing what it was told.
        """
        out: List[str] = []
        for cap, want in self.backends.items():
            if want == BACKEND_MODEL and self._state.get(cap) != BACKEND_MODEL:
                out.append(cap)
        return sorted(out)

    def engine_files(self) -> Dict[str, str]:
        """role -> the engine file actually loaded for it."""
        return {role: os.path.basename(p) for role, p in self._role_files.items()}

    def load_errors(self) -> Dict[str, str]:
        """role -> the error text from a failed load, for the health surface."""
        return dict(self._load_errors)

    def forget_track(self, track_id: int) -> None:
        """Drop per-track caches when a track dies. Call from the tracker."""
        if self.occlusion is not None:
            self.occlusion.forget(track_id)
        if self.identity is not None:
            self.identity.forget(track_id)

    def close(self) -> None:
        """Release every engine. Safe to call twice."""
        for wrapper in (
            self.headpose,
            self.eyestate,
            self.mouthstate,
            self.occlusion,
            self.gaze,
            self.objects,
            self.identity,
        ):
            if wrapper is not None:
                wrapper.engine = None
                wrapper.health.loaded = False
                if hasattr(wrapper, "mask_engine"):
                    wrapper.mask_engine = None
                if hasattr(wrapper, "sunglasses_engine"):
                    wrapper.sunglasses_engine = None
        for role, engine in list(self._engines.items()):
            if hasattr(engine, "close"):
                try:
                    engine.close()
                except Exception as exc:  # noqa: BLE001
                    log.warning("closing engine for role %s failed: %s", role, exc)
        self._engines.clear()
        self._loaded = False
