"""Classical geometry over the 106 landmarks.

Every function here is the documented FALLBACK for a neural capability in
:mod:`dms.infer`.  They are kept working, kept tested and kept honest: each one
either produces a value or says it could not, and none of them returns a
plausible constant.
"""
from dms.geometry.ear import (
    contour_ear,
    eye_aspect_ratio,
    eye_crop_boxes,
    eye_ratio,
    face_ear,
    face_ear_detail,
)
from dms.geometry.face106 import (
    LEFT_EYE,
    MOUTH,
    NOSE_TIP,
    PNP_INDICES,
    PNP_OBJECT_POINTS,
    RIGHT_EYE,
    eye_box,
    inter_ocular,
    landmarks_agree_with_kps,
    mouth_box,
    roll_from_eyes,
    roll_from_kps,
)
from dms.geometry.mar import mouth_aspect_ratio, mouth_aspect_ratio_detail
from dms.geometry.pnp import (
    camera_matrix,
    camera_matrix_from_fov,
    rotation_to_ypr,
    solve_head_pose,
    solve_head_pose_full,
)

__all__ = [
    "LEFT_EYE",
    "MOUTH",
    "NOSE_TIP",
    "PNP_INDICES",
    "PNP_OBJECT_POINTS",
    "RIGHT_EYE",
    "camera_matrix",
    "camera_matrix_from_fov",
    "contour_ear",
    "eye_aspect_ratio",
    "eye_box",
    "eye_crop_boxes",
    "eye_ratio",
    "face_ear",
    "face_ear_detail",
    "inter_ocular",
    "landmarks_agree_with_kps",
    "mouth_aspect_ratio",
    "mouth_aspect_ratio_detail",
    "mouth_box",
    "roll_from_eyes",
    "roll_from_kps",
    "rotation_to_ypr",
    "solve_head_pose",
    "solve_head_pose_full",
]
