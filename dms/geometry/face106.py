"""InsightFace 2d106det landmark topology.

VERIFIED ON-BOARD, NOT COPIED FROM A COOKBOOK.  The previous revision of this
module used the group table published by the ``uniface`` docs (contour 0-32,
brows 33-50, nose 51-62, eyes 63-86, mouth 87-105).  That table is wrong for the
engine this project actually ships (``2d106det_192_fp16_gpu.engine``).

How the table below was established: SCRFD-500m emits five keypoints whose order
is fixed by InsightFace (image-left eye, image-right eye, nose tip, image-left
mouth corner, image-right mouth corner).  Running the detector and the landmark
engine on the same faces from the replay clips and asking "which of the 106
points lands on each SCRFD keypoint" gives an unambiguous answer, and the
resulting groups are geometrically self-consistent (two 10-point eye clusters
above two 9-point brow clusters, a 15-point nose column on the face midline, a
20-point mouth ring, and a 33-point jaw contour that closes at index 0).

    index range   group                      count
    ------------  -------------------------  -----
    0 - 32        jaw / face contour            33
    33 - 42       image-LEFT eye                10
    43 - 51       image-LEFT eyebrow             9
    52 - 71       mouth (12 outer + 8 inner)    20
    72 - 86       nose                          15
    87 - 96       image-RIGHT eye               10
    97 - 105      image-RIGHT eyebrow            9
                                             = 106

"LEFT" here always means *image* left (smaller x for a roughly frontal face),
which is the same convention SCRFD uses for ``kps[0]``.  For a driver facing the
camera that is the driver's own right eye; nothing downstream depends on the
anatomical naming, only on the consistency of it.

Both eye groups have the SAME internal ordering, so a single offset table serves
both.  With ``base`` = 33 (left) or 87 (right):

    base + 0  lower lid, middle
    base + 1  eye centre  (duplicated at base + 5 by the network)
    base + 2  corner at smaller x
    base + 3  lower lid, smaller x
    base + 4  lower lid, larger x
    base + 5  eye centre  (duplicate of base + 1)
    base + 6  corner at larger x
    base + 7  upper lid, middle
    base + 8  upper lid, smaller x
    base + 9  upper lid, larger x

Only this module may define 106-point indices.  EAR / MAR / PnP and every model
wrapper must import names from here.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

N_LANDMARKS = 106

# ---------------------------------------------------------------- groups ----
CONTOUR = range(0, 33)
LEFT_EYE = range(33, 43)
BROW_L = range(43, 52)
MOUTH = range(52, 72)
NOSE = range(72, 87)
RIGHT_EYE = range(87, 97)
BROW_R = range(97, 106)

_EYE_BASE_L = 33
_EYE_BASE_R = 87

# Offsets inside a 10-point eye group (see module docstring).
EYE_OFF_LOWER_MID = 0
EYE_OFF_CENTRE = 1
EYE_OFF_CORNER_LOX = 2
EYE_OFF_LOWER_LOX = 3
EYE_OFF_LOWER_HIX = 4
EYE_OFF_CENTRE_DUP = 5
EYE_OFF_CORNER_HIX = 6
EYE_OFF_UPPER_MID = 7
EYE_OFF_UPPER_LOX = 8
EYE_OFF_UPPER_HIX = 9

#: The 8 distinct lid/corner points of an eye (centre and its duplicate removed).
EYE_CONTOUR_OFFSETS: Tuple[int, ...] = (0, 2, 3, 4, 6, 7, 8, 9)

#: (upper, lower) offset pairs used by the aspect ratio, outer -> inner.
EYE_LID_PAIR_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (EYE_OFF_UPPER_LOX, EYE_OFF_LOWER_LOX),
    (EYE_OFF_UPPER_MID, EYE_OFF_LOWER_MID),
    (EYE_OFF_UPPER_HIX, EYE_OFF_LOWER_HIX),
)

# ------------------------------------------------------------- singletons ---
CHIN = 0  # contour closes at the chin, it is not the middle of the range
NOSE_TIP = 86  # bottom of the 72/73/74/86 midline bridge column
NOSE_BRIDGE_TOP = 72

LEFT_EYE_CENTRE = _EYE_BASE_L + EYE_OFF_CENTRE  # 34
RIGHT_EYE_CENTRE = _EYE_BASE_R + EYE_OFF_CENTRE  # 88
LEFT_EYE_OUTER = _EYE_BASE_L + EYE_OFF_CORNER_LOX  # 35, temple side
LEFT_EYE_INNER = _EYE_BASE_L + EYE_OFF_CORNER_HIX  # 39, nose side
RIGHT_EYE_INNER = _EYE_BASE_R + EYE_OFF_CORNER_LOX  # 89, nose side
RIGHT_EYE_OUTER = _EYE_BASE_R + EYE_OFF_CORNER_HIX  # 93, temple side

# Kept under the historical names so nothing that imports them breaks; both now
# point at the *outer* corner, which is what PnP wants.
LEFT_EYE_CORNER = LEFT_EYE_OUTER
RIGHT_EYE_CORNER = RIGHT_EYE_OUTER

#: Outer lip ring, traced as a closed polygon: left corner, along the lower lip
#: to the right corner, back along the upper lip.
OUTER_LIP: Tuple[int, ...] = (52, 55, 56, 53, 59, 58, 61, 68, 67, 71, 63, 64)
#: Inner lip ring, same winding.
INNER_LIP: Tuple[int, ...] = (65, 54, 60, 57, 69, 70, 62, 66)

MOUTH_LEFT = 52  # outer corner, smaller x
MOUTH_RIGHT = 61  # outer corner, larger x
MOUTH_INNER_LEFT = 65
MOUTH_INNER_RIGHT = 69

#: (upper inner, lower inner) pairs, outer -> inner.  Used by MAR.
INNER_LIP_PAIRS: Tuple[Tuple[int, int], ...] = ((66, 54), (62, 60), (70, 57))
MOUTH_UP = 62  # upper inner lip, middle
MOUTH_LOW = 60  # lower inner lip, middle

# ------------------------------------------------------------------- PnP ----
PNP_INDICES: Tuple[int, int, int, int, int, int] = (
    NOSE_TIP,
    CHIN,
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
    MOUTH_LEFT,
    MOUTH_RIGHT,
)

#: Millimetres, in the OPENCV CAMERA FRAME: +X image right, +Y image DOWN,
#: +Z AWAY from the camera into the scene.  Origin at the nose tip.
#:
#: Because +Z points away from the camera and the nose tip is the part of a face
#: closest to it, the nose tip is the origin and EVERY other point has POSITIVE
#: z.  Getting that sign wrong builds a mirrored head whose nose points backwards
#: and inverts the recovered yaw; the sign is pinned by
#: ``test_pnp_yaw_sign_matches_the_documented_convention``, which checks the
#: solve against a landmark cue that does not use solvePnP at all.
#:
#: The previous table placed the chin at z = -28 against eye corners at z = -30,
#: i.e. the chin sat as deep in the head as the eye sockets, and had the depth
#: axis inverted on top of that.  Real heads are not shaped like that, and the
#: inversion biases solvePnP's rotation about the X axis - the documented
#: +/-80 deg pitch noise.  These values are the canonical OpenCV 6-point head
#: divided by 5 (millimetres at adult scale: 90 mm between the outer eye
#: corners, 66 mm nose tip to chin), rotated into the camera frame, with the
#: chin at roughly half the depth of the eye corners as anatomy requires.
PNP_OBJECT_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],  # nose tip - the closest point to the camera
        [0.0, 66.0, 13.0],  # chin
        [-45.0, -34.0, 27.0],  # eye outer corner, image left
        [45.0, -34.0, 27.0],  # eye outer corner, image right
        [-30.0, 30.0, 25.0],  # mouth corner, image left
        [30.0, 30.0, 25.0],  # mouth corner, image right
    ],
    dtype=np.float64,
)


# ------------------------------------------------------------- accessors ----
def points(landmarks: np.ndarray, idxs: Sequence[int]) -> np.ndarray:
    """Gather rows ``idxs`` from a (106, 2) array."""
    return np.asarray(landmarks)[list(idxs)]


def eye_base(left: bool) -> int:
    """First index of the requested 10-point eye group."""
    return _EYE_BASE_L if left else _EYE_BASE_R


def eye_contour(landmarks106: np.ndarray, left: bool) -> np.ndarray:
    """The 8 distinct lid/corner points of one eye, as (8, 2) pixels."""
    base = eye_base(left)
    return np.asarray(landmarks106)[[base + o for o in EYE_CONTOUR_OFFSETS]]


def eye_corners(landmarks106: np.ndarray, left: bool) -> Tuple[np.ndarray, np.ndarray]:
    """(corner at smaller x, corner at larger x) for one eye, in pixels."""
    base = eye_base(left)
    lm = np.asarray(landmarks106)
    return lm[base + EYE_OFF_CORNER_LOX], lm[base + EYE_OFF_CORNER_HIX]


def eye_lid_pairs(landmarks106: np.ndarray, left: bool) -> np.ndarray:
    """(3, 2, 2) array of (upper, lower) lid point pairs for one eye."""
    base = eye_base(left)
    lm = np.asarray(landmarks106)
    return np.stack([np.stack([lm[base + u], lm[base + d]]) for u, d in EYE_LID_PAIR_OFFSETS])


def eye_centre(landmarks106: np.ndarray, left: bool) -> np.ndarray:
    """The network's own eye-centre point, (2,) pixels."""
    return np.asarray(landmarks106)[eye_base(left) + EYE_OFF_CENTRE]


def inter_ocular(landmarks106: np.ndarray) -> float:
    """Distance between the two eye centres, in pixels.

    This is the natural scale for every landmark-derived threshold.  Returns 0.0
    if the points coincide, which callers must treat as "unusable".
    """
    lm = np.asarray(landmarks106)
    return float(np.linalg.norm(lm[LEFT_EYE_CENTRE] - lm[RIGHT_EYE_CENTRE]))


def roll_from_eyes(landmarks106: np.ndarray) -> float:
    """In-plane roll in degrees from the eye-centre axis.

    Positive = the image-right eye sits lower than the image-left eye.  Uses the
    landmark eye centres, so it is available whenever landmarks are.
    """
    lm = np.asarray(landmarks106)
    d = lm[RIGHT_EYE_CENTRE] - lm[LEFT_EYE_CENTRE]
    return float(np.degrees(np.arctan2(float(d[1]), float(d[0]))))


def roll_from_kps(kps5: np.ndarray) -> float:
    """In-plane roll in degrees from the SCRFD 5-keypoint eye pair.

    Same sign convention as :func:`roll_from_eyes`.  Available one stage earlier
    than the landmarks, so it can gate whether the landmark crop is worth doing.
    """
    k = np.asarray(kps5, dtype=np.float64)
    d = k[1] - k[0]
    return float(np.degrees(np.arctan2(float(d[1]), float(d[0]))))


def _bbox(pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float32)
    return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()], dtype=np.float32)


def eye_box(
    landmarks106: np.ndarray,
    left: bool,
    margin: float = 0.0,
) -> np.ndarray:
    """Tight xyxy box around one eye's 8 lid/corner points, in full-frame pixels.

    ``margin`` is a fraction of the eye width added on every side.  OCEC was
    trained on zero-margin crops, so the default is 0.0; the parameter exists so
    the operating point can be swept on replay without touching call sites.
    """
    box = _bbox(eye_contour(landmarks106, left))
    if margin > 0.0:
        w = float(box[2] - box[0])
        h = float(box[3] - box[1])
        box = box + np.array([-margin * w, -margin * h, margin * w, margin * h], dtype=np.float32)
    return box


def mouth_box(landmarks106: np.ndarray) -> np.ndarray:
    """Tight xyxy box around the outer lip ring, in full-frame pixels."""
    return _bbox(points(landmarks106, OUTER_LIP))


def face_box_from_landmarks(landmarks106: np.ndarray) -> np.ndarray:
    """xyxy hull of all 106 points. Used to sanity-check the detector box."""
    return _bbox(np.asarray(landmarks106))


def landmarks_agree_with_kps(
    landmarks106: np.ndarray,
    kps5: Optional[np.ndarray],
    tol_frac: float = 0.20,
) -> Tuple[bool, float]:
    """Cross-check the 106 points against SCRFD's 5 keypoints.

    Compares the landmark eye centres, nose tip and mouth corners against the
    detector's independent estimate of the same five features.  Returns
    ``(agree, worst_error_fraction)`` where the error is normalised by the
    inter-ocular distance.  ``agree`` is False when the worst feature is further
    than ``tol_frac`` of the inter-ocular distance away, which is the signal that
    the landmark crop went to the wrong place (occlusion, profile, a box that
    fell off the frame) and that EAR/MAR/PnP must be marked unreliable rather
    than trusted.

    With no keypoints available the check cannot run and returns ``(True, 0.0)``
    - absence of evidence is not evidence of failure, and the caller has other
    gates.
    """
    if kps5 is None:
        return True, 0.0
    k = np.asarray(kps5, dtype=np.float64).reshape(-1, 2)
    if k.shape[0] != 5:
        return True, 0.0
    lm = np.asarray(landmarks106, dtype=np.float64)
    iod = inter_ocular(lm)
    if iod < 1e-3:
        return False, float("inf")
    mine = np.stack(
        [
            lm[LEFT_EYE_CENTRE],
            lm[RIGHT_EYE_CENTRE],
            lm[NOSE_TIP],
            lm[MOUTH_LEFT],
            lm[MOUTH_RIGHT],
        ]
    )
    err = float(np.max(np.linalg.norm(mine - k, axis=1)) / iod)
    return err <= tol_frac, err
