"""Overlay lifecycle: one switch, one reused buffer, and a real no-op when off.

The overlay is a debugging aid.  In a shipped vehicle it renders to nothing and
must therefore cost nothing.  :class:`OverlayRenderer` makes that structural
rather than a matter of remembering an ``if``:

* :attr:`OverlayRenderer.enabled` is decided once, at construction, from the
  three switches that can want an overlay (``--save-video``, ``--save-preview``,
  ``display.enabled``).  :meth:`render` returns ``None`` immediately when it is
  false - before any copy, resize or draw.
* When enabled it owns the destination buffer.  A steady-state frame performs
  one ``cv2.resize`` (or one ``np.copyto`` at ``scale=1.0``) into that buffer and
  allocates nothing else.
* :meth:`thermal_gate` lets the caller shed the overlay under thermal pressure
  without losing the user's setting - the previous code did the opposite, and
  because of an ``or`` where ``and`` was meant it drew the overlay *only* when
  not degraded, i.e. always in normal operation and never when it would have
  been the right thing to drop.

Typical wiring in the frame loop::

    overlay = OverlayRenderer.from_flags(
        save_video=bool(args.save_video),
        save_preview=bool(args.save_preview),
        display=cfg.display.enabled,
    )
    ...
    if overlay.enabled:                      # skip building overlay-only inputs
        with budget.stage("overlay"):
            vis = overlay.render(bgr, faces, driver=driver_face, ...)

``vis`` is ``None`` when the overlay is off; the video writer and the preview
save must both be guarded by ``overlay.enabled`` for the same reason.

Units: ``scale`` is a dimensionless render factor in (0, 1]; ``max_width`` is
canvas pixels.  Failure behaviour: :meth:`render` propagates a dtype/shape error
from :func:`~dms.viz.overlay.draw_faces` (a programming error) but never raises
on out-of-frame geometry.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from dms.viz.overlay import draw_faces

#: Render scale used when a caller asks for "cheap".  0.5 of 1280x720 is
#: 640x360: still readable in a saved preview, roughly a third of the cost.
PREVIEW_SCALE = 0.5


class OverlayRenderer(object):
    """Owns the overlay enable flag, the render scale and the reused canvas."""

    __slots__ = ("_enabled", "_scale", "_max_width", "_buf", "_buf_key", "_thermal_off", "frames_drawn")

    def __init__(self, enabled=False, scale=1.0, max_width=0):
        # type: (bool, float, int) -> None
        s = float(scale)
        if not (0.0 < s <= 1.0):
            raise ValueError("scale must be in (0, 1], got %r" % (scale,))
        self._enabled = bool(enabled)
        self._scale = s
        self._max_width = int(max_width)
        self._buf = None  # type: Optional[np.ndarray]
        self._buf_key = None  # type: Optional[Tuple[int, int]]
        self._thermal_off = False
        self.frames_drawn = 0

    # ------------------------------------------------------------ factories
    @classmethod
    def from_flags(cls, save_video=False, save_preview=False, display=False, scale=None, max_width=0):
        # type: (bool, bool, bool, Optional[float], int) -> OverlayRenderer
        """Enable if and only if something will actually consume the pixels.

        ``scale`` defaults to 1.0 when a video or preview file is being written
        (an artifact a human will inspect frame by frame deserves full
        resolution) and to :data:`PREVIEW_SCALE` for a live display, where the
        frame is glanced at and the latency matters more.
        """
        enabled = bool(save_video or save_preview or display)
        if scale is None:
            scale = 1.0 if (save_video or save_preview) else PREVIEW_SCALE
        return cls(enabled=enabled, scale=scale, max_width=max_width)

    @classmethod
    def disabled(cls):
        # type: () -> OverlayRenderer
        """A renderer that will never draw. :meth:`render` returns ``None``."""
        return cls(enabled=False)

    # ------------------------------------------------------------ properties
    @property
    def enabled(self):
        # type: () -> bool
        """True when :meth:`render` will produce a frame right now."""
        return self._enabled and not self._thermal_off

    @property
    def configured(self):
        # type: () -> bool
        """True when the operator asked for an overlay, ignoring the thermal gate."""
        return self._enabled

    @property
    def scale(self):
        # type: () -> float
        return self._scale

    def thermal_gate(self, degraded):
        # type: (bool) -> None
        """Shed the overlay while the board is thermally degraded.

        This is the one documented degrade action the overlay participates in.
        It is separate from :attr:`configured` so the operator's setting is
        restored the moment the board cools, with no reconfiguration.
        """
        self._thermal_off = bool(degraded)

    # ---------------------------------------------------------------- buffer
    def _buffer(self, src_h, src_w):
        # type: (int, int) -> np.ndarray
        s = self._scale
        dst_w = max(1, int(round(src_w * s)))
        dst_h = max(1, int(round(src_h * s)))
        if self._max_width and dst_w > self._max_width:
            k = float(self._max_width) / dst_w
            dst_w = self._max_width
            dst_h = max(1, int(round(dst_h * k)))
        key = (dst_h, dst_w)
        if self._buf_key != key or self._buf is None:
            self._buf = np.empty((dst_h, dst_w, 3), dtype=np.uint8)
            self._buf_key = key
        return self._buf

    def effective_scale(self, src_h, src_w):
        # type: (int, int) -> float
        """The scale actually used for this frame size, after ``max_width``."""
        buf = self._buffer(src_h, src_w)
        return buf.shape[1] / float(src_w)

    # ---------------------------------------------------------------- render
    def render(self, bgr, faces, **kw):
        # type: (np.ndarray, object, object) -> Optional[np.ndarray]
        """Draw the overlay, or return ``None`` when the overlay is off.

        The returned array is owned by this renderer and is overwritten by the
        next call.  A caller that needs to keep it (an encoder queue) must copy.
        """
        if not self._enabled or self._thermal_off:
            return None
        src_h, src_w = bgr.shape[:2]
        buf = self._buffer(src_h, src_w)
        eff = buf.shape[1] / float(src_w)
        kw.pop("out", None)
        kw.pop("scale", None)
        out = draw_faces(bgr, faces, out=buf, scale=eff, **kw)
        self.frames_drawn += 1
        return out


#: Shared singleton for "there is definitely no overlay".  Stateless: it holds no
#: buffer and never draws, so sharing it between callers is safe.
NULL_RENDERER = OverlayRenderer.disabled()


__all__ = ["NULL_RENDERER", "PREVIEW_SCALE", "OverlayRenderer"]
