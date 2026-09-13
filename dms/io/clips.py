"""Incident clip ring: a rolling H.264 buffer that is only ever written on an alert.

This is PR-10 of ``docs/production-dms-design.md`` (§ "clip ring"), which every
shipped config already advertises — ``clips.quota_mb: 2048``, ``clips.retain_h: 24``
— and which nothing implemented. An operator reading ``/etc/dms/default.yaml`` was
being told a 2 GB / 24 h quota was enforced on ``/var/lib/dms`` when no quota was
enforced on anything.

Privacy gate (the important part)
---------------------------------
Nothing here encodes a single frame unless **both** ``clips.enabled`` and
``privacy.record_faces`` are true. Both default to false, so a default install writes
zero mp4s and never even creates the clip directory. ``privacy.record_faces`` was
previously dead config — a switch presented to an auditor that did nothing — and
in-cabin face video is biometric-adjacent personal data, so the gate is checked in
:meth:`ClipRing.start`, re-checked in :meth:`ClipRing.push`, and reported verbatim on
``/healthz`` as ``clips.reason`` so the posture of a running unit is readable.

How the ring works
------------------
``appsrc`` (BGR frames pushed from the main loop) → ``nvvidconv`` → ``nvv4l2h264enc``
→ ``h264parse`` → ``splitmuxsink`` writing ``segment_s``-second mp4 segments into
``<dir>/ring`` with ``max-files`` set so the ring self-truncates. On an alert edge the
caller calls :meth:`trigger`; ``post_s`` seconds later :meth:`tick` concatenates the
segments covering ``[t - pre_s, t + post_s]`` with ``ffmpeg -c copy`` (no re-encode,
no GPU) into ``<dir>/<timestamp>_<reason>.mp4``.

Only *finalised* segments can be concatenated — the segment splitmuxsink is currently
writing has no moov atom yet. Worst case the incident clip therefore starts up to
``segment_s`` late relative to ``pre_s``; that is why ``pre_s`` (10 s) is longer than
``segment_s``.

Quota
-----
:meth:`sweep` enforces two independent bounds on the whole clip directory, oldest
first: ``quota_mb`` total bytes and ``retain_h`` age. It runs after every write and on
every ``tick``. Live ring segments are counted against the quota but never deleted by
the sweeper — ``splitmuxsink``'s ``max-files`` bounds them.

Units: ``pre_s``/``post_s``/``segment_s`` seconds, ``quota_mb`` MiB, ``retain_h``
hours, ``bitrate`` bits per second, ``min_free_mb`` MiB.

Failure behaviour: nothing raises. ENOSPC/EDQUOT anywhere (push, mux, concat) stops
recording, publishes ``DISK_FULL`` through ``on_disk_full`` and leaves the process
running — clips are evidence, not a safety function, and a full disk must never take
driver monitoring offline. Every refusal is counted in
``dms_clips_dropped_total{reason}``.

Testability: the encoder is injected. ``writer_factory`` returns any object with
``start()``, ``push(frame, pts_ns)`` and ``stop()``, and ``concat`` is a plain
callable, so the whole ring — quota sweeps, ENOSPC, the privacy gate, trigger timing —
is unit-tested with no GPU and no GStreamer.
"""
from __future__ import annotations

import errno
import logging
import os
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from dms.io import metrics
from dms.state.alerts import AlertEvent, AlertType, Severity

log = logging.getLogger("dms.clips")

#: Subdirectory holding the live rolling segments.
RING_SUBDIR = "ring"
#: Extension of both ring segments and incident clips.
CLIP_EXT = ".mp4"
#: Errnos meaning "out of room".
_FULL_ERRNOS = frozenset((errno.ENOSPC, errno.EDQUOT))
#: ffmpeg says this on stderr when it runs out of space.
_FULL_TEXT = ("no space left", "disk quota exceeded")

_DIR_MODE = 0o750
_FILE_MODE = 0o640


def dir_size_bytes(path):
    # type: (str) -> int
    """Total size of the regular files under *path*, recursively. 0 if absent."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def list_clips(path, ext=CLIP_EXT):
    # type: (str, str) -> List[Tuple[float, int, str]]
    """``(mtime, size, path)`` for every clip under *path*, oldest first."""
    out = []  # type: List[Tuple[float, int, str]]
    for root, _dirs, files in os.walk(path):
        for name in files:
            if not name.endswith(ext):
                continue
            full = os.path.join(root, name)
            try:
                st = os.stat(full)
            except OSError:
                continue
            out.append((st.st_mtime, st.st_size, full))
    out.sort()
    return out


def ffmpeg_concat(ffmpeg, segments, out_path, timeout_s=60.0):
    # type: (str, List[str], str, float) -> None
    """Stream-copy *segments* into *out_path*. Raises on failure.

    ``-c copy`` means no decode and no encode: the GPU is not touched and the call
    costs milliseconds even for a 15 s clip. The concat demuxer needs a list file, so
    one is written next to the output and removed afterwards.
    """
    listing = out_path + ".concat.txt"
    with open(listing, "w") as f:
        for seg in segments:
            # The concat demuxer's quoting rule: wrap in single quotes, escape any.
            f.write("file '%s'\n" % seg.replace("'", "'\\''"))
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", listing,
        "-c", "copy", "-movflags", "+faststart", out_path,
    ]
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s
        )
    finally:
        try:
            os.unlink(listing)
        except OSError:
            pass
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError("ffmpeg concat failed (rc=%d): %s" % (proc.returncode, err[-400:]))


class GstSegmentWriter(object):
    """Hardware-encoding ring writer: appsrc → nvv4l2h264enc → splitmuxsink.

    Constructed lazily by :class:`ClipRing` so importing this module never pulls in
    GStreamer. Frames are pushed as BGR ``numpy`` arrays with an explicit monotonic
    PTS in nanoseconds; ``do-timestamp`` is off because the capture clock, not the
    pipeline clock, is what the incident timeline is built from.

    Failure behaviour: :meth:`start` raises ``RuntimeError`` if the pipeline will not
    reach PLAYING (:class:`ClipRing` catches it and disables clips); :meth:`push`
    raises ``RuntimeError`` on a flow error so the ring can stop cleanly.
    """

    def __init__(self, directory, width, height, fps, segment_s, max_files, bitrate):
        # type: (str, int, int, int, float, int, int) -> None
        self.directory = directory
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps) or 30
        self.segment_s = float(segment_s)
        self.max_files = int(max_files)
        self.bitrate = int(bitrate)
        self._pipeline = None  # type: Any
        self._appsrc = None  # type: Any
        self._gst = None  # type: Any

    def description(self):
        # type: () -> str
        """The gst-launch description, so it can be asserted in a test and logged."""
        keyframe = max(1, int(self.fps * self.segment_s))
        location = os.path.join(self.directory, "seg_%05d" + CLIP_EXT)
        return (
            "appsrc name=src is-live=true do-timestamp=false format=time block=true "
            "max-bytes=8388608 "
            "caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 "
            "! videoconvert ! video/x-raw,format=I420 "
            "! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 "
            "! nvv4l2h264enc bitrate={bitrate} insert-sps-pps=true "
            "iframeinterval={kf} idrinterval={kf} maxperf-enable=true "
            "! h264parse "
            "! splitmuxsink name=mux muxer=mp4mux location={loc} "
            "max-size-time={maxns} max-files={files}"
        ).format(
            w=self.width, h=self.height, fps=self.fps, bitrate=self.bitrate,
            kf=keyframe, loc=location, maxns=int(self.segment_s * 1e9), files=self.max_files,
        )

    def start(self):
        # type: () -> None
        import gi  # noqa: E402  (lazy: the board has python3-gi, tests do not need it)

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # type: ignore

        if not Gst.is_initialized():
            Gst.init(None)
        self._gst = Gst
        desc = self.description()
        log.info("clip ring pipeline: %s", desc)
        self._pipeline = Gst.parse_launch(desc)
        self._appsrc = self._pipeline.get_by_name("src")
        rc = self._pipeline.set_state(Gst.State.PLAYING)
        if rc == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
            raise RuntimeError("clip ring pipeline refused to start")

    def push(self, frame, pts_ns):
        # type: (Any, int) -> None
        if self._appsrc is None or self._gst is None:
            raise RuntimeError("clip writer is not started")
        Gst = self._gst
        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.pts = int(pts_ns)
        buf.duration = int(1e9 / max(1, self.fps))
        ret = self._appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            raise RuntimeError("clip ring push-buffer returned %s" % ret)

    def stop(self):
        # type: () -> None
        """EOS the pipeline so splitmuxsink finalises the segment it is writing."""
        if self._pipeline is None:
            return
        Gst = self._gst
        try:
            if self._appsrc is not None:
                self._appsrc.emit("end-of-stream")
            bus = self._pipeline.get_bus()
            bus.timed_pop_filtered(3 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
        except Exception as exc:  # a bus that is already torn down
            log.debug("clip ring EOS handling: %s", exc)
        finally:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
            self._appsrc = None


class _Pending(object):
    """One triggered incident waiting for its post-roll to elapse."""

    __slots__ = ("reason", "severity", "t_mono_s", "deadline")

    def __init__(self, reason, severity, t_mono_s, deadline):
        # type: (str, str, float, float) -> None
        self.reason = reason
        self.severity = severity
        self.t_mono_s = t_mono_s
        self.deadline = deadline


class ClipRing(object):
    """The clip ring, its privacy gate, its quota and its ENOSPC behaviour."""

    def __init__(
        self,
        enabled=False,
        record_faces=False,
        directory="/var/lib/dms/clips",
        pre_s=10.0,
        post_s=5.0,
        segment_s=10.0,
        quota_mb=2048,
        retain_h=24,
        ffmpeg="/usr/bin/ffmpeg",
        min_free_mb=256.0,
        width=1280,
        height=720,
        fps=30,
        bitrate=4000000,
        writer_factory=None,
        concat=None,
        on_disk_full=None,
        clock=time.monotonic,
    ):
        # type: (bool, bool, str, float, float, float, int, int, str, float, int, int, int, int, Optional[Callable[..., Any]], Optional[Callable[..., None]], Optional[Callable[[AlertEvent], None]], Callable[[], float]) -> None
        self.enabled = bool(enabled)
        self.record_faces = bool(record_faces)
        self.directory = directory
        self.ring_dir = os.path.join(directory, RING_SUBDIR)
        self.pre_s = float(pre_s)
        self.post_s = float(post_s)
        self.segment_s = float(segment_s)
        self.quota_bytes = int(quota_mb) * 1024 * 1024
        self.retain_s = float(retain_h) * 3600.0
        self.ffmpeg = ffmpeg
        self.min_free_mb = float(min_free_mb)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps) or 30
        self.bitrate = int(bitrate)
        self._writer_factory = writer_factory
        self._concat = concat or ffmpeg_concat
        self._on_disk_full = on_disk_full
        self._clock = clock

        self.active = False
        self.reason = "not started"
        self.disk_full = False
        self.written = 0
        self.dropped = 0
        self.swept = 0
        self._writer = None  # type: Any
        self._pending = []  # type: List[_Pending]
        self._frames = 0
        self._last_sweep = 0.0

        metrics.CLIPS_ENABLED.set_bool(False)
        metrics.CLIPS_QUOTA_BYTES.set(self.quota_bytes)

    # ------------------------------------------------------------------ config

    @classmethod
    def from_config(cls, cfg, **kwargs):
        # type: (Any, **Any) -> "ClipRing"
        """Build from an :class:`dms.config.schema.AppConfig`.

        Reads ``clips.*``, ``privacy.record_faces`` and the capture geometry. Optional
        keys (``clips.segment_s``, ``clips.bitrate``, ``clips.min_free_mb``) are read
        with defaults so this works against the current schema unchanged.
        """
        clips = getattr(cfg, "clips", None)
        privacy = getattr(cfg, "privacy", None)
        source = getattr(cfg, "source", None)
        full = getattr(getattr(cfg, "capture", None), "full_size", (1280, 720))
        return cls(
            enabled=bool(getattr(clips, "enabled", False)),
            record_faces=bool(getattr(privacy, "record_faces", False)),
            directory=str(getattr(clips, "dir", "/var/lib/dms/clips")),
            pre_s=float(getattr(clips, "pre_s", 10.0)),
            post_s=float(getattr(clips, "post_s", 5.0)),
            segment_s=float(getattr(clips, "segment_s", 10.0)),
            quota_mb=int(getattr(clips, "quota_mb", 2048)),
            retain_h=int(getattr(clips, "retain_h", 24)),
            ffmpeg=str(getattr(clips, "ffmpeg", "/usr/bin/ffmpeg")),
            min_free_mb=float(getattr(clips, "min_free_mb", 256.0)),
            width=int(full[0]),
            height=int(full[1]),
            fps=int(getattr(source, "fps", 30)),
            bitrate=int(getattr(clips, "bitrate", 4000000)),
            **kwargs
        )

    # ----------------------------------------------------------------- gating

    def gate_reason(self):
        # type: () -> str
        """Empty string when recording is permitted, else why it is not.

        Both switches are named in the message because "clips off" is an answer an
        auditor has to be able to act on.
        """
        if not self.enabled and not self.record_faces:
            return "clips.enabled=false and privacy.record_faces=false"
        if not self.enabled:
            return "clips.enabled=false"
        if not self.record_faces:
            return "privacy.record_faces=false (refusing to encode in-cabin video)"
        return ""

    @property
    def max_ring_files(self):
        # type: () -> int
        """Segments to keep: enough for the pre-roll, plus the open one, plus slack."""
        need = self.pre_s + self.post_s
        return max(3, int(need / max(0.5, self.segment_s)) + 3)

    # ---------------------------------------------------------------- lifecycle

    def start(self):
        # type: () -> bool
        """Create the directory and the encoder, if and only if both switches allow it.

        Returns True when the ring is recording. A refusal is not an error: it logs at
        INFO, sets :attr:`reason`, and creates nothing on disk.
        """
        gate = self.gate_reason()
        if gate:
            self.active = False
            self.reason = gate
            metrics.CLIPS_ENABLED.set_bool(False)
            log.info("clip ring disabled: %s", gate)
            return False
        if not os.path.isfile(self.ffmpeg) and self._concat is ffmpeg_concat:
            self.active = False
            self.reason = "ffmpeg not found at %s" % self.ffmpeg
            metrics.CLIPS_ENABLED.set_bool(False)
            log.error("clip ring disabled: %s", self.reason)
            return False
        try:
            os.makedirs(self.ring_dir, mode=_DIR_MODE, exist_ok=True)
            os.chmod(self.directory, _DIR_MODE)
        except OSError as exc:
            self.active = False
            self.reason = "cannot create %s: %s" % (self.ring_dir, exc)
            metrics.CLIPS_ENABLED.set_bool(False)
            log.error("clip ring disabled: %s", self.reason)
            return False
        free = self._free_mb()
        if free is not None and free < self.min_free_mb:
            self.active = False
            self.reason = "only %.0f MiB free, need %.0f" % (free, self.min_free_mb)
            metrics.CLIPS_ENABLED.set_bool(False)
            metrics.CLIPS_DROPPED_TOTAL.inc(reason="low_disk")
            log.error("clip ring disabled: %s", self.reason)
            return False
        factory = self._writer_factory or self._default_writer
        try:
            self._writer = factory(
                directory=self.ring_dir,
                width=self.width,
                height=self.height,
                fps=self.fps,
                segment_s=self.segment_s,
                max_files=self.max_ring_files,
                bitrate=self.bitrate,
            )
            self._writer.start()
        except Exception as exc:
            self._writer = None
            self.active = False
            self.reason = "encoder failed: %s" % exc
            metrics.CLIPS_ENABLED.set_bool(False)
            log.error("clip ring disabled: %s", self.reason, exc_info=True)
            return False
        self.active = True
        self.reason = ""
        metrics.CLIPS_ENABLED.set_bool(True)
        self.sweep(self._clock())
        log.warning(
            "CLIP_RING_ACTIVE dir=%s segment=%.0fs pre=%.0fs post=%.0fs quota=%d MiB — "
            "in-cabin video is being recorded because clips.enabled and "
            "privacy.record_faces are both true",
            self.directory, self.segment_s, self.pre_s, self.post_s,
            self.quota_bytes // (1024 * 1024),
        )
        return True

    @staticmethod
    def _default_writer(**kwargs):
        # type: (**Any) -> GstSegmentWriter
        return GstSegmentWriter(**kwargs)

    def stop(self):
        # type: () -> None
        """Finalise the open segment and release the encoder. Idempotent."""
        writer, self._writer = self._writer, None
        self.active = False
        metrics.CLIPS_ENABLED.set_bool(False)
        if writer is None:
            return
        try:
            writer.stop()
        except Exception:
            log.exception("clip ring writer failed to stop cleanly")

    def __enter__(self):
        # type: () -> "ClipRing"
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        # type: (Any, Any, Any) -> bool
        self.stop()
        return False

    # -------------------------------------------------------------------- feed

    def push(self, frame_bgr, t_mono_s=None):
        # type: (Any, Optional[float]) -> bool
        """Feed one BGR frame into the ring. Returns False if it was not recorded.

        The privacy gate is re-checked here, not just at start, so that flipping
        ``record_faces`` off at runtime (a future reload) stops the encoder rather
        than relying on ``start`` never having run.
        """
        if not self.active or self._writer is None:
            return False
        if self.gate_reason():
            log.warning("clip ring stopping: privacy gate closed at runtime")
            self.stop()
            return False
        now = self._clock() if t_mono_s is None else float(t_mono_s)
        try:
            self._writer.push(frame_bgr, int(now * 1e9))
        except OSError as exc:
            self._on_io_error(exc, "push")
            return False
        except Exception as exc:
            self.dropped += 1
            metrics.CLIPS_DROPPED_TOTAL.inc(reason="encoder_error")
            log.error("clip ring push failed, stopping the ring: %s", exc)
            self.stop()
            return False
        self._frames += 1
        return True

    # ----------------------------------------------------------------- trigger

    def trigger(self, reason, severity="warn", t_mono_s=None):
        # type: (str, Any, Optional[float]) -> bool
        """Mark an incident. The clip is written ``post_s`` later by :meth:`tick`.

        ``reason`` becomes part of the filename, so it is sanitised to
        ``[a-z0-9_-]``. Repeated triggers inside one post-roll window are coalesced
        into the pending incident rather than producing overlapping clips.
        """
        if not self.active:
            metrics.CLIPS_DROPPED_TOTAL.inc(reason="ring_inactive")
            return False
        sev = str(getattr(severity, "value", severity))
        if sev not in ("warn", "critical"):
            return False
        now = self._clock() if t_mono_s is None else float(t_mono_s)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(reason).lower())[:48]
        for pend in self._pending:
            if now - pend.t_mono_s <= self.post_s:
                # Same incident window: keep the earliest start, take the worst
                # severity, extend the post-roll.
                if sev == "critical":
                    pend.severity = sev
                pend.deadline = max(pend.deadline, now + self.post_s)
                return True
        self._pending.append(_Pending(safe, sev, now, now + self.post_s))
        log.info("clip trigger %s/%s at t=%.1f, writing in %.0f s", safe, sev, now, self.post_s)
        return True

    def tick(self, t_mono_s=None):
        # type: (Optional[float]) -> List[str]
        """Write out any incident whose post-roll has elapsed. Call once per frame.

        Cheap when there is nothing pending (one list check). Returns the paths of
        clips written by this call.
        """
        now = self._clock() if t_mono_s is None else float(t_mono_s)
        written = []  # type: List[str]
        if self._pending:
            still = []  # type: List[_Pending]
            for pend in self._pending:
                if now < pend.deadline:
                    still.append(pend)
                    continue
                path = self._write_clip(pend, now)
                if path:
                    written.append(path)
            self._pending = still
        if now - self._last_sweep >= 60.0:
            self.sweep(now)
        return written

    def _write_clip(self, pend, now):
        # type: (_Pending, float) -> str
        if self.disk_full:
            self.dropped += 1
            metrics.CLIPS_DROPPED_TOTAL.inc(reason="disk_full")
            return ""
        free = self._free_mb()
        if free is not None and free < self.min_free_mb:
            self._enter_disk_full("free=%.0f MiB < min_free_mb=%.0f" % (free, self.min_free_mb), now)
            self.dropped += 1
            metrics.CLIPS_DROPPED_TOTAL.inc(reason="low_disk")
            return ""
        segments = self._segments_for(pend, now)
        if not segments:
            self.dropped += 1
            metrics.CLIPS_DROPPED_TOTAL.inc(reason="no_segments")
            log.warning("clip %s: no finalised ring segments cover the incident", pend.reason)
            return ""
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        out = os.path.join(self.directory, "%s_%s_%s%s" % (stamp, pend.severity, pend.reason, CLIP_EXT))
        try:
            self._concat(self.ffmpeg, segments, out)
        except OSError as exc:
            self._on_io_error(exc, "concat")
            return ""
        except Exception as exc:
            text = str(exc).lower()
            if any(t in text for t in _FULL_TEXT):
                self._enter_disk_full(str(exc)[:200], now)
                metrics.CLIPS_DROPPED_TOTAL.inc(reason="disk_full")
            else:
                metrics.CLIPS_DROPPED_TOTAL.inc(reason="mux_error")
                log.error("clip %s could not be muxed: %s", out, exc)
            self.dropped += 1
            return ""
        try:
            os.chmod(out, _FILE_MODE)
        except OSError:
            pass
        self.written += 1
        metrics.CLIPS_WRITTEN_TOTAL.inc(severity=pend.severity)
        log.info("clip written %s from %d segments", out, len(segments))
        self.sweep(now)
        return out

    def _segments_for(self, pend, now):
        # type: (_Pending, float) -> List[str]
        """Finalised ring segments covering ``[t-pre_s, now]``, in time order.

        Selection is by mtime, which for splitmuxsink is when the segment was closed.
        A segment closed at ``T`` holds roughly ``[T-segment_s, T]``, so the window is
        widened by one segment on the early side.
        """
        oldest_allowed = time.time() - (now - pend.t_mono_s) - self.pre_s - self.segment_s
        segs = [p for mtime, _size, p in list_clips(self.ring_dir) if mtime >= oldest_allowed]
        return segs

    # --------------------------------------------------------------- quota

    def sweep(self, now=None):
        # type: (Optional[float]) -> Tuple[int, int]
        """Delete incident clips, oldest first, until under quota and retention.

        Returns ``(files_deleted, bytes_freed)``. Ring segments are never deleted
        here: ``splitmuxsink max-files`` owns them, and deleting the file it is
        writing would corrupt the very clip an incident needs.
        """
        self._last_sweep = self._clock() if now is None else float(now)
        if not os.path.isdir(self.directory):
            return 0, 0
        ring_bytes = dir_size_bytes(self.ring_dir)
        incidents = [
            (m, s, p) for m, s, p in list_clips(self.directory)
            if os.path.dirname(os.path.abspath(p)) != os.path.abspath(self.ring_dir)
        ]
        total = ring_bytes + sum(s for _m, s, _p in incidents)
        deleted = 0
        freed = 0
        wall = time.time()
        for mtime, size, path in incidents:
            too_old = self.retain_s > 0 and (wall - mtime) > self.retain_s
            over_quota = self.quota_bytes > 0 and total > self.quota_bytes
            if not (too_old or over_quota):
                break
            try:
                os.unlink(path)
            except OSError as exc:
                log.warning("could not delete %s: %s", path, exc)
                continue
            total -= size
            freed += size
            deleted += 1
            self.swept += 1
            metrics.CLIPS_SWEPT_TOTAL.inc(reason="retention" if too_old else "quota")
        metrics.CLIPS_BYTES.set(total)
        if deleted:
            log.info("clip sweep removed %d files (%.1f MiB)", deleted, freed / 1048576.0)
        return deleted, freed

    # ------------------------------------------------------------- disk full

    def _free_mb(self):
        # type: () -> Optional[float]
        target = self.directory
        while target and not os.path.exists(target):
            parent = os.path.dirname(target)
            if parent == target:
                return None
            target = parent
        try:
            return shutil.disk_usage(target).free / 1048576.0
        except OSError:
            return None

    def _on_io_error(self, exc, where):
        # type: (OSError, str) -> None
        if exc.errno in _FULL_ERRNOS:
            self._enter_disk_full("%s during %s" % (errno.errorcode.get(exc.errno, exc.errno), where), self._clock())
        else:
            log.error("clip ring I/O error during %s: %s", where, exc)
        self.dropped += 1
        metrics.CLIPS_DROPPED_TOTAL.inc(reason="disk_full" if exc.errno in _FULL_ERRNOS else "io_error")

    def _enter_disk_full(self, detail, now):
        # type: (str, float) -> None
        if self.disk_full:
            return
        self.disk_full = True
        metrics.DISK_FULL.set(1.0)
        log.error("DISK_FULL clips %s: %s — stopping the ring, the process stays up",
                  self.directory, detail)
        self.stop()
        self.reason = "disk full: %s" % detail
        if self._on_disk_full is not None:
            event = AlertEvent(
                t_mono_s=now,
                type=AlertType.DISK_FULL,
                severity=Severity.WARN,
                edge="enter",
                track_id=None,
                extra={"clips": 1.0},
            )
            try:
                self._on_disk_full(event)
            except Exception:
                log.exception("clips on_disk_full callback raised")

    # ------------------------------------------------------------------ status

    def snapshot(self):
        # type: () -> Dict[str, Any]
        """State for ``/healthz`` under the ``clips`` key."""
        return {
            "enabled": self.active,
            "reason": self.reason or "recording",
            "configured": self.enabled,
            "record_faces": self.record_faces,
            "dir": self.directory,
            "written": self.written,
            "dropped": self.dropped,
            "swept": self.swept,
            "pending": len(self._pending),
            "frames": self._frames,
            "disk_full": self.disk_full,
            "quota_mb": self.quota_bytes // (1024 * 1024),
            "bytes": dir_size_bytes(self.directory) if os.path.isdir(self.directory) else 0,
        }


__all__ = [
    "CLIP_EXT",
    "ClipRing",
    "GstSegmentWriter",
    "RING_SUBDIR",
    "dir_size_bytes",
    "ffmpeg_concat",
    "list_clips",
]
