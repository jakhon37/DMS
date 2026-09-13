"""Debug visualisation. Opt-in, cheap when on, a true no-op when off.

Nothing in this package is on the production path: an overlay is only rendered
when the operator asked for a preview file, a video file, or a live display.
:class:`~dms.viz.renderer.OverlayRenderer` is the switch, and it is the only
thing the frame loop should hold - :func:`~dms.viz.overlay.draw_faces` is the
drawing primitive underneath it and is exported for tests and one-off tools.

All coordinates passed in are full-frame source pixels; the renderer applies its
own scale.  Colours are BGR.  See ``docs/performance.md`` for the measured cost
of each rendering mode on this board.
"""
from __future__ import annotations

from dms.viz.overlay import draw_faces
from dms.viz.renderer import NULL_RENDERER, PREVIEW_SCALE, OverlayRenderer

__all__ = ["NULL_RENDERER", "PREVIEW_SCALE", "OverlayRenderer", "draw_faces"]
