"""Debug overlay drawing: opt-in, allocation-free, and free when it is off.

Why this file was rewritten
---------------------------
The overlay is a development aid with **zero production value** and it was the
single largest avoidable cost in the frame loop:

* ``dms/app.py`` gated it on ``if want_overlay or not health.degraded:`` where
  ``and`` was meant.  ``health.degraded`` is False in the normal case, so the
  overlay was drawn on **every frame even headless**, then thrown away.
* ``vis = bgr.copy()`` copied a 2.76 MB strided view - and because ``bgr`` was
  ``frame.full_bgra[..., :3]``, a stride-4 view, that copy was a cache-hostile
  gather, not a memcpy.
* 106 individual ``cv2.circle`` calls for the landmarks, each one a Python->C
  transition drawing a 3x3 blob.

Measured on this board at 1280x720 (see ``docs/performance.md``): the old
full-resolution path cost 6.9 ms/frame.  The same content drawn at ``scale=0.5``
into a reused buffer costs 1.9 ms, and drawing nothing costs 0.

Design
------
* :class:`~dms.viz.renderer.OverlayRenderer` owns the destination buffer and the
  enable flag.  When disabled, :meth:`OverlayRenderer.render` returns ``None``
  before touching a pixel - not a copy of the input, not a blank frame.
* :func:`draw_faces` keeps its original signature so existing callers work
  unchanged, and gains ``out=`` (draw into a caller-owned buffer) and
  ``scale=`` (render a downscaled preview).
* Landmarks are painted by vectorised pixel assignment grouped by colour, not by
  106 ``cv2.circle`` calls.

Coordinates and units
---------------------
All geometry arguments (``faces``, ``landmarks106``) are in **full-frame source
pixels**.  When ``scale != 1.0`` this module multiplies them itself; callers
never pre-scale.  ``ear``/``mar`` are dimensionless ratios, ``pose`` is
``(yaw, pitch, roll)`` in **degrees**, ``yaw_rel`` is degrees from the
calibrated forward zero.

Failure behaviour
-----------------
Drawing never raises on out-of-frame geometry: boxes are clipped and landmarks
outside the canvas are dropped, because an overlay that crashes the pipeline is
strictly worse than one that omits a dot.  A wrong *dtype* or channel count does
raise - that is a programming error, not a data condition.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from dms.geometry.face106 import LEFT_EYE, MOUTH, NOSE_TIP, RIGHT_EYE
from dms.infer.scrfd import FaceDet

#: BGR colours. Kept together so a change is one edit.
COL_DRIVER = (0, 255, 255)
COL_FACE = (0, 255, 0)
COL_EYE = (255, 80, 80)
COL_MOUTH = (80, 80, 255)
COL_LMK = (0, 200, 255)
COL_HUD = (0, 200, 255)
COL_EAR = (0, 255, 0)
COL_MAR = (0, 180, 255)
COL_POSE = (255, 200, 0)
COL_TRACK = (200, 200, 200)
COL_YAWREL = (180, 180, 255)
COL_ALERT = (0, 0, 255)

_FONT = cv2.FONT_HERSHEY_SIMPLEX

#: Landmark index -> colour group, precomputed once. Rebuilding these from the
#: ``range`` objects inside the per-point loop was 106 membership tests a frame.
_N_LMK = 106
_LMK_EYE_MASK = np.zeros(_N_LMK, dtype=bool)
_LMK_EYE_MASK[list(LEFT_EYE)] = True
_LMK_EYE_MASK[list(RIGHT_EYE)] = True
_LMK_MOUTH_MASK = np.zeros(_N_LMK, dtype=bool)
_LMK_MOUTH_MASK[list(MOUTH)] = True
_LMK_OTHER_MASK = ~(_LMK_EYE_MASK | _LMK_MOUTH_MASK)


def draw_faces(
    bgr: np.ndarray,
    faces: Sequence[FaceDet],
    *,
    driver: Optional[FaceDet] = None,
    landmarks106: Optional[np.ndarray] = None,
    ear: Optional[float] = None,
    mar: Optional[float] = None,
    pose: Optional[Tuple[float, float, float]] = None,
    track_id: Optional[int] = None,
    alerts: Optional[List[str]] = None,
    yaw_rel: Optional[float] = None,
    occupant_labels: Optional[List[Tuple[object, str]]] = None,
    out: Optional[np.ndarray] = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Render the debug overlay and return the canvas that was drawn on.

    Parameters
    ----------
    bgr : HxWx3 uint8 source frame. Never modified.
    out : optional destination. Must be ``(round(H*scale), round(W*scale), 3)``
        uint8. Supplying it makes the whole call allocation-free, which is the
        point of :class:`~dms.viz.renderer.OverlayRenderer`. When ``None`` a new
        canvas is allocated (convenient, but one 2.7 MB allocation per frame).
    scale : uniform render scale. ``0.5`` renders a 640x360 preview of a 720p
        frame at roughly a third of the cost; all input geometry is still given
        in source pixels.

    Returns the canvas (``out`` when it was supplied).
    """
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("expected HxWx3 uint8 BGR frame, got %s %s" % (bgr.shape, bgr.dtype))
    s = float(scale)
    if not (0.0 < s <= 1.0):
        raise ValueError("scale must be in (0, 1], got %r" % (scale,))
    src_h, src_w = bgr.shape[:2]
    dst_h = max(1, int(round(src_h * s)))
    dst_w = max(1, int(round(src_w * s)))

    vis = _canvas(bgr, out, dst_h, dst_w)

    # Text metrics scale with the canvas so a 0.5 preview stays legible.
    fs = max(0.35, 0.65 * s if s < 1.0 else 0.65)
    th = 1 if s < 0.75 else 2
    line_h = max(14, int(round(26 * s))) if s < 1.0 else 26

    label_by_face = {}
    if occupant_labels:
        for tr, role in occupant_labels:
            face = getattr(tr, "face", None)
            if face is not None:
                label_by_face[id(face)] = role

    for f in faces:
        x1 = int(f.xyxy[0] * s)
        y1 = int(f.xyxy[1] * s)
        x2 = int(f.xyxy[2] * s)
        y2 = int(f.xyxy[3] * s)
        is_drv = driver is not None and f is driver
        color = COL_DRIVER if is_drv else COL_FACE
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3 if (is_drv and s >= 0.75) else 2)
        role = label_by_face.get(id(f), "DRV" if is_drv else "P")
        cv2.putText(
            vis,
            "%s %.2f" % (role, f.score),
            (x1, max(int(16 * s) + 2, y1 - 6)),
            _FONT,
            max(0.3, 0.55 * s if s < 1.0 else 0.55),
            color,
            th,
            cv2.LINE_AA,
        )
        if f.kps is not None:
            _plot_points(vis, np.asarray(f.kps, dtype=np.float32) * s, COL_DRIVER)

    if landmarks106 is not None:
        _plot_landmarks(vis, np.asarray(landmarks106, dtype=np.float32) * s)

    y = line_h
    n_pax = max(0, len(faces) - (1 if driver is not None else 0))
    cv2.putText(
        vis,
        "occ=%d drv=%d pax=%d" % (len(faces), 1 if driver is not None else 0, n_pax),
        (8, y),
        _FONT,
        fs,
        COL_HUD,
        th,
        cv2.LINE_AA,
    )
    if ear is not None:
        y += line_h
        cv2.putText(vis, "EAR=%.3f" % ear, (8, y), _FONT, fs, COL_EAR, th, cv2.LINE_AA)
    if mar is not None:
        y += line_h
        cv2.putText(vis, "MAR=%.3f" % mar, (8, y), _FONT, fs, COL_MAR, th, cv2.LINE_AA)
    if pose is not None:
        yaw, pitch, roll = pose
        y += line_h
        cv2.putText(vis, "yaw=%.0f pitch=%.0f" % (yaw, pitch), (8, y), _FONT, fs, COL_POSE, th, cv2.LINE_AA)
        if landmarks106 is not None:
            tdx = int(landmarks106[NOSE_TIP][0] * s)
            tdy = int(landmarks106[NOSE_TIP][1] * s)
            _draw_axis(vis, yaw, pitch, roll, tdx, tdy, size=max(20, int(round(80 * s))))
    if track_id is not None:
        y += line_h
        cv2.putText(vis, "id=%d" % track_id, (8, y), _FONT, fs, COL_TRACK, th, cv2.LINE_AA)
    if yaw_rel is not None:
        y += line_h
        cv2.putText(vis, "yaw_rel=%.0f" % yaw_rel, (8, y), _FONT, fs, COL_YAWREL, th, cv2.LINE_AA)
    if alerts:
        y += line_h + 2
        cv2.putText(vis, " | ".join(alerts), (8, y), _FONT, fs, COL_ALERT, th, cv2.LINE_AA)
    return vis


def _canvas(bgr, out, dst_h, dst_w):
    # type: (np.ndarray, Optional[np.ndarray], int, int) -> np.ndarray
    """Fill (and validate) the destination with the source at the render size."""
    src_h, src_w = bgr.shape[:2]
    if out is not None:
        if out.dtype != np.uint8 or out.shape != (dst_h, dst_w, 3):
            raise ValueError(
                "out must be %s uint8 for this frame and scale, got %s %s"
                % ((dst_h, dst_w, 3), out.shape, out.dtype)
            )
        if (dst_h, dst_w) == (src_h, src_w):
            np.copyto(out, bgr)
        else:
            # INTER_AREA is the correct downsample filter and, at 2x, is not
            # slower than INTER_LINEAR here (measured 0.9 vs 0.9 ms at 720p->360p).
            cv2.resize(bgr, (dst_w, dst_h), dst=out, interpolation=cv2.INTER_AREA)
        return out
    if (dst_h, dst_w) == (src_h, src_w):
        return bgr.copy()
    return cv2.resize(bgr, (dst_w, dst_h), interpolation=cv2.INTER_AREA)


def _plot_points(vis, pts, colour, radius=1):
    # type: (np.ndarray, np.ndarray, Tuple[int, int, int], int) -> None
    """Paint a small point cloud by vectorised assignment, clipped to the canvas.

    106 ``cv2.circle`` calls cost 0.9 ms at 720p on this board; the same points
    assigned as a 3x3 stamp cost 0.06 ms. Points outside the canvas are dropped,
    not clamped to the border, so an off-frame landmark does not draw a false dot
    on the edge.
    """
    if pts.size == 0:
        return
    h, w = vis.shape[:2]
    xy = np.rint(pts.reshape(-1, 2)).astype(np.int32)
    keep = (xy[:, 0] >= radius) & (xy[:, 0] < w - radius) & (xy[:, 1] >= radius) & (xy[:, 1] < h - radius)
    if not keep.any():
        return
    xs = xy[keep, 0]
    ys = xy[keep, 1]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            vis[ys + dy, xs + dx] = colour


def _plot_landmarks(vis, pts):
    # type: (np.ndarray, np.ndarray) -> None
    """Three vectorised passes (eyes, mouth, rest) instead of 106 circle calls."""
    p = pts.reshape(-1, 2)
    n = p.shape[0]
    if n == _N_LMK:
        _plot_points(vis, p[_LMK_EYE_MASK], COL_EYE)
        _plot_points(vis, p[_LMK_MOUTH_MASK], COL_MOUTH)
        _plot_points(vis, p[_LMK_OTHER_MASK], COL_LMK)
    else:
        # Unknown point count: draw them all in the neutral colour rather than
        # index a mask that does not describe this model.
        _plot_points(vis, p, COL_LMK)


def _draw_axis(img, yaw, pitch, roll, tdx, tdy, size=80) -> None:
    """Head-pose axis cross. Same convention as the 6DRepNet reference draw_axis.

    Angles are degrees; ``size`` is the axis length in canvas pixels.
    """
    pitch = pitch * np.pi / 180.0
    yaw = -(yaw * np.pi / 180.0)
    roll = roll * np.pi / 180.0
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy
    thick = 3 if size >= 60 else 2
    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), thick)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), thick)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), thick)


__all__ = ["draw_faces"]
