"""

Camera models and calibration helpers.

The Cozmo camera is a fixed-focus QVGA camera.  This module provides a small
pinhole camera model that is sufficient for projecting detections onto the
ground plane and for building an occupancy map from a single camera.  Full
lens calibration is optionally available through OpenCV.

"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .utils import as_rgb, cv2


__all__ = [
    "CameraModel",
    "CalibrationResult",
    "calibrate_camera",
    "find_chessboard_corners",
    "estimate_homography",
    "apply_homography",
]


def _rotation_matrix(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    """ Rotation from camera coordinates (x right, y down, z forward).

    The returned matrix maps camera coordinates to the robot world frame
    (x forward, y left, z up).  Positive pitch looks downward.
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    right = np.array([sy, -cy, 0.0])
    down = np.array([-cy * sp, -sy * sp, -cp])
    forward = np.array([cy * cp, sy * cp, -sp])

    if abs(roll) > 1e-12:
        # Rotate the right and down axes around the forward axis.
        cr, sr = math.cos(roll), math.sin(roll)
        right, down = right * cr + down * sr, -right * sr + down * cr

    return np.column_stack([right, down, forward])


@dataclass
class CameraModel:
    """ Intrinsic and extrinsic parameters of a pinhole camera.

    Distances are in millimetres.  The camera is located at
    ``(position_x, position_y, position_z)`` in the robot world frame.  The
    default pose points along the robot's forward axis with no pitch.
    """

    width: int = 320
    height: int = 240
    fx: float = 277.13
    fy: float = 277.13
    cx: float = 160.0
    cy: float = 120.0
    distortion: Optional[np.ndarray] = field(default=None, repr=False)
    position: Tuple[float, float, float] = (0.0, 0.0, 40.0)
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    name: str = "camera"

    def __post_init__(self) -> None:
        self.width = int(self.width)
        self.height = int(self.height)
        self.fx = float(self.fx)
        self.fy = float(self.fy)
        self.cx = float(self.cx)
        self.cy = float(self.cy)
        self.position = tuple(float(v) for v in self.position)  # type: ignore
        if len(self.position) != 3:
            raise ValueError("Camera position must have three components.")
        if self.distortion is not None:
            self.distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)

    @classmethod
    def from_fov(cls, width: int = 320, height: int = 240,
                 hfov_deg: float = 60.0, position: Tuple[float, float, float] = (0.0, 0.0, 40.0),
                 pitch_deg: float = 0.0, name: str = "camera") -> "CameraModel":
        """ Create a camera model from a horizontal field of view. """
        width = int(width)
        height = int(height)
        hfov = math.radians(float(hfov_deg))
        fx = (width / 2.0) / math.tan(hfov / 2.0)
        return cls(width=width, height=height, fx=fx, fy=fx,
                   cx=width / 2.0, cy=height / 2.0, position=position,
                   pitch=math.radians(pitch_deg), name=name)

    @classmethod
    def cozmo(cls, width: int = 320, height: int = 240, **kwargs) -> "CameraModel":
        """ A camera model matching Cozmo's QVGA camera. """
        kwargs.setdefault("position", (0.0, 0.0, 40.0))
        kwargs.setdefault("name", "cozmo")
        return cls.from_fov(width=width, height=height, hfov_deg=60.0, **kwargs)

    @classmethod
    def vector(cls, width: int = 640, height: int = 480, **kwargs) -> "CameraModel":
        """ A camera model matching Vector's wide-angle camera. """
        kwargs.setdefault("position", (0.0, 0.0, 100.0))
        kwargs.setdefault("name", "vector")
        return cls.from_fov(width=width, height=height, hfov_deg=120.0, **kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "CameraModel":
        distortion = data.get("distortion")
        pose = data.get("pose", {})
        return cls(
            width=data.get("width", 320),
            height=data.get("height", 240),
            fx=data.get("fx", 277.13),
            fy=data.get("fy", 277.13),
            cx=data.get("cx", data.get("width", 320) / 2.0),
            cy=data.get("cy", data.get("height", 240) / 2.0),
            distortion=np.asarray(distortion) if distortion is not None else None,
            position=tuple(pose.get("position", (0.0, 0.0, 40.0))),
            yaw=float(pose.get("yaw", 0.0)),
            pitch=float(pose.get("pitch", 0.0)),
            roll=float(pose.get("roll", 0.0)),
            name=data.get("name", "camera"),
        )

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "CameraModel":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "distortion": self.distortion.tolist() if self.distortion is not None else None,
            "pose": {
                "position": list(self.position),
                "yaw": self.yaw,
                "pitch": self.pitch,
                "roll": self.roll,
            },
        }

    def save(self, path: Union[str, os.PathLike]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """ Rotation from camera to world coordinates. """
        return _rotation_matrix(self.yaw, self.pitch, self.roll)

    def set_pose(self, position: Optional[Sequence[float]] = None,
                 yaw_deg: Optional[float] = None, pitch_deg: Optional[float] = None,
                 roll_deg: Optional[float] = None) -> "CameraModel":
        """ Update the camera extrinsics and return ``self`` for chaining. """
        if position is not None:
            if len(position) != 3:
                raise ValueError("Camera position must have three components.")
            self.position = tuple(float(v) for v in position)  # type: ignore
        if yaw_deg is not None:
            self.yaw = math.radians(float(yaw_deg))
        if pitch_deg is not None:
            self.pitch = math.radians(float(pitch_deg))
        if roll_deg is not None:
            self.roll = math.radians(float(roll_deg))
        return self

    def camera_to_world(self, points: np.ndarray) -> np.ndarray:
        """ Transform ``N x 3`` camera points to world coordinates. """
        points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        rotation = self.rotation_matrix
        return points @ rotation.T + np.asarray(self.position, dtype=np.float64)

    def world_to_camera(self, points: np.ndarray) -> np.ndarray:
        """ Transform ``N x 3`` world points to camera coordinates. """
        points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        rotation = self.rotation_matrix
        return (points - np.asarray(self.position, dtype=np.float64)) @ rotation

    def project_camera(self, points: np.ndarray) -> np.ndarray:
        """ Project camera coordinates to pixel coordinates.

        Points behind the camera are returned as ``nan``.
        """
        points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        result = np.full((len(points), 2), np.nan, dtype=np.float64)
        valid = points[:, 2] > 1e-6
        result[valid, 0] = self.fx * points[valid, 0] / points[valid, 2] + self.cx
        result[valid, 1] = self.fy * points[valid, 1] / points[valid, 2] + self.cy
        return result

    def project(self, points: np.ndarray) -> np.ndarray:
        """ Project world points to pixel coordinates. """
        return self.project_camera(self.world_to_camera(points))

    def unproject(self, u: float, v: float, depth: float) -> np.ndarray:
        """ Convert a pixel and depth to a camera-space point. """
        return np.array([(float(u) - self.cx) / self.fx * depth,
                         (float(v) - self.cy) / self.fy * depth,
                         float(depth)], dtype=np.float64)

    def ray(self, u: float, v: float) -> np.ndarray:
        """ Return a unit-length world-space ray through a pixel. """
        direction = np.array([(float(u) - self.cx) / self.fx,
                              (float(v) - self.cy) / self.fy,
                              1.0], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        world = self.rotation_matrix @ direction
        return world / np.linalg.norm(world)

    def ground_intersection(self, u: float, v: float) -> Optional[Tuple[float, float, float]]:
        """ Intersect a pixel ray with the ground plane ``z = 0``. """
        ray = self.ray(u, v)
        if ray[2] >= -1e-9:
            return None
        t = -self.position[2] / ray[2]
        if t <= 0.0:
            return None
        point = np.asarray(self.position, dtype=np.float64) + ray * t
        return float(point[0]), float(point[1]), 0.0

    def ground_distance(self, u: float, v: float) -> Optional[float]:
        point = self.ground_intersection(u, v)
        if point is None:
            return None
        return float(math.hypot(point[0] - self.position[0], point[1] - self.position[1]))

    def ground_point_to_pixel(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """ Project a ground plane point into the image. """
        point = self.project(np.array([[x, y, 0.0]], dtype=np.float64))[0]
        if not np.all(np.isfinite(point)):
            return None
        if point[0] < 0 or point[0] >= self.width or point[1] < 0 or point[1] >= self.height:
            return None
        return float(point[0]), float(point[1])

    def horizon_row(self) -> float:
        """ Row of the horizon in the image, in pixels. """
        return float(self.cy - self.fy * math.tan(self.pitch))

    def ground_distance_for_row(self, row: float) -> float:
        """ Estimate the ground distance of a row below the horizon. """
        depression = math.atan2(float(row) - self.cy, self.fy) + self.pitch
        if depression <= 1e-6:
            return float("inf")
        return float(self.position[2] / math.tan(depression))

    def row_for_ground_distance(self, distance: float) -> float:
        """ Row at which the ground is ``distance`` millimetres away. """
        if distance <= 0:
            return float("inf")
        depression = math.atan2(self.position[2], float(distance))
        return float(self.cy + self.fy * math.tan(depression - self.pitch))

    @property
    def hfov(self) -> float:
        """ Horizontal field of view in radians. """
        return 2.0 * math.atan(self.width / (2.0 * self.fx))

    @property
    def vfov(self) -> float:
        """ Vertical field of view in radians. """
        return 2.0 * math.atan(self.height / (2.0 * self.fy))

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """ Undistort an image.  Requires OpenCV and calibration distortion. """
        if cv2 is None:
            raise RuntimeError("Undistorting images requires OpenCV.")
        if self.distortion is None:
            return as_rgb(image)
        return cv2.undistort(as_rgb(image), self.intrinsic_matrix, self.distortion)

    def scaled(self, width: int, height: int) -> "CameraModel":
        """ Return a copy of this model scaled to another image resolution. """
        sx = float(width) / self.width
        sy = float(height) / self.height
        return CameraModel(width=width, height=height, fx=self.fx * sx, fy=self.fy * sy,
                           cx=self.cx * sx, cy=self.cy * sy,
                           distortion=None if self.distortion is None else self.distortion.copy(),
                           position=self.position, yaw=self.yaw, pitch=self.pitch,
                           roll=self.roll, name=self.name)

    def __repr__(self) -> str:
        return "CameraModel({}x{}, fx={:.1f}, fy={:.1f}, pos={})".format(
            self.width, self.height, self.fx, self.fy, self.position)


@dataclass
class CalibrationResult:
    """ Result of a chessboard calibration. """

    camera: CameraModel
    rms: float
    image_count: int
    per_image_errors: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = self.camera.to_dict()
        data.update({"rms": self.rms, "image_count": self.image_count,
                     "per_image_errors": list(self.per_image_errors)})
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationResult":
        return cls(camera=CameraModel.from_dict(data), rms=float(data.get("rms", 0.0)),
                   image_count=int(data.get("image_count", 0)),
                   per_image_errors=list(data.get("per_image_errors", [])))


def find_chessboard_corners(image: Union[np.ndarray, object],
                            pattern_size: Tuple[int, int] = (9, 6)) -> Optional[np.ndarray]:
    """ Find inner chessboard corners.  Requires OpenCV. """
    if cv2 is None:
        raise RuntimeError("Chessboard detection requires OpenCV.")
    gray = cv2.cvtColor(as_rgb(image), cv2.COLOR_RGB2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH
    if hasattr(cv2, "CALIB_CB_NORMALIZE_IMAGE"):
        flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, tuple(pattern_size), flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2)


def calibrate_camera(images: Iterable[Union[np.ndarray, object]],
                     pattern_size: Tuple[int, int] = (9, 6),
                     square_size: float = 25.0,
                     camera: Optional[CameraModel] = None) -> CalibrationResult:
    """ Calibrate camera intrinsics from chessboard images.

    ``pattern_size`` is the number of inner corners per row and column, and
    ``square_size`` is expressed in millimetres.  OpenCV is required.
    """
    if cv2 is None:
        raise RuntimeError("Camera calibration requires OpenCV. Install pycozmo[vision].")
    object_points = []
    image_points = []
    image_size = None
    used = 0
    for image in images:
        rgb = as_rgb(image)
        if image_size is None:
            image_size = (rgb.shape[1], rgb.shape[0])
        corners = find_chessboard_corners(rgb, pattern_size)
        if corners is None:
            continue
        grid = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
        grid[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
        object_points.append(grid)
        image_points.append(corners.astype(np.float32))
        used += 1

    if used < 3:
        raise ValueError("At least three valid chessboard images are required (found {}).".format(used))
    if image_size is None:
        raise ValueError("No images supplied.")

    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None)
    result_camera = CameraModel(
        width=image_size[0], height=image_size[1],
        fx=matrix[0, 0], fy=matrix[1, 1], cx=matrix[0, 2], cy=matrix[1, 2],
        distortion=distortion.reshape(-1),
        position=camera.position if camera else (0.0, 0.0, 40.0),
        yaw=camera.yaw if camera else 0.0,
        pitch=camera.pitch if camera else 0.0,
        name=camera.name if camera else "calibrated",
    )
    return CalibrationResult(camera=result_camera, rms=float(rms), image_count=used)


def estimate_homography(source: Sequence[Sequence[float]], destination: Sequence[Sequence[float]]) -> np.ndarray:
    """ Estimate a 3x3 homography from point correspondences.

    At least four correspondences are required.  OpenCV is used when
    available; otherwise a normalized DLT implementation is used.
    """
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(destination, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 4:
        raise ValueError("Expected at least four 2D point correspondences of equal shape.")
    if cv2 is not None:
        homography, _ = cv2.findHomography(src, dst, 0)
        return np.asarray(homography, dtype=np.float64)

    # Normalize the points for numerical stability.
    def normalizer(points: np.ndarray) -> np.ndarray:
        center = points.mean(axis=0)
        scale = np.sqrt(2.0) / max(np.mean(np.linalg.norm(points - center, axis=1)), 1e-12)
        return np.array([[scale, 0.0, -scale * center[0]],
                         [0.0, scale, -scale * center[1]],
                         [0.0, 0.0, 1.0]])

    src_norm = normalizer(src)
    dst_norm = normalizer(dst)
    src_h = np.column_stack([src, np.ones(len(src))]) @ src_norm.T
    dst_h = np.column_stack([dst, np.ones(len(dst))]) @ dst_norm.T
    rows = []
    for (x, y, _), (u, v, _) in zip(src_h, dst_h):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, vh = np.linalg.svd(np.asarray(rows))
    homography = vh[-1].reshape(3, 3)
    homography = np.linalg.inv(dst_norm) @ homography @ src_norm
    return homography / homography[2, 2]


def apply_homography(homography: np.ndarray, points: Sequence[Sequence[float]]) -> np.ndarray:
    """ Apply a homography to 2D points. """
    homography = np.asarray(homography, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ homography.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]
