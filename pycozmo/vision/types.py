"""

Common data types used by the off-board vision and navigation modules.

"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np


__all__ = [
    "BoundingBox",
    "Detection",
    "Pose2D",
]


@dataclass(frozen=True)
class BoundingBox:
    """ Axis-aligned bounding box.

    The canonical representation is ``(x1, y1, x2, y2)`` with ``x2 >= x1``
    and ``y2 >= y1``.  For convenience :meth:`from_xywh` accepts the common
    ``(x, y, width, height)`` form used by many detectors.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        x1, x2 = sorted((float(self.x1), float(self.x2)))
        y1, y2 = sorted((float(self.y1), float(self.y2)))
        object.__setattr__(self, "x1", x1)
        object.__setattr__(self, "y1", y1)
        object.__setattr__(self, "x2", x2)
        object.__setattr__(self, "y2", y2)

    @classmethod
    def from_xywh(cls, x: float, y: float, width: float, height: float) -> "BoundingBox":
        """ Create a box from its top-left corner and size. """
        return cls(x, y, x + width, y + height)

    @classmethod
    def from_center(cls, cx: float, cy: float, width: float, height: float) -> "BoundingBox":
        """ Create a box from its center and size. """
        return cls(cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> "BoundingBox":
        """ Create a box from ``(x1, y1, x2, y2)`` or ``(x, y, width, height)``.

        Four values are interpreted as ``x1, y1, x2, y2`` for consistency
        with :class:`BoundingBox`.  Use :meth:`from_xywh` when the last two
        values are a width and a height.
        """
        values = tuple(float(v) for v in values)
        if len(values) != 4:
            raise ValueError("Expected four values for a bounding box.")
        return cls(*values)

    @property
    def x(self) -> float:
        return self.x1

    @property
    def y(self) -> float:
        return self.y1

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def xywh(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.width, self.height

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return self.xyxy

    def to_xywh(self) -> Tuple[float, float, float, float]:
        return self.xywh

    def to_int(self) -> Tuple[int, int, int, int]:
        """ Return ``(x1, y1, x2, y2)`` rounded to integers. """
        return (int(round(self.x1)), int(round(self.y1)),
                int(round(self.x2)), int(round(self.y2)))

    def intersection(self, other: "BoundingBox") -> "BoundingBox":
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        if x2 < x1 or y2 < y1:
            return BoundingBox(0.0, 0.0, 0.0, 0.0)
        return BoundingBox(x1, y1, x2, y2)

    def iou(self, other: "BoundingBox") -> float:
        """ Intersection over union in the range ``[0, 1]``. """
        intersection = self.intersection(other).area
        if intersection <= 0.0:
            return 0.0
        union = self.area + other.area - intersection
        return float(intersection / union) if union > 0.0 else 0.0

    def expand(self, amount: float, clip_size: Optional[Tuple[int, int]] = None) -> "BoundingBox":
        """ Return a box expanded by ``amount`` pixels on every side. """
        box = BoundingBox(self.x1 - amount, self.y1 - amount, self.x2 + amount, self.y2 + amount)
        if clip_size is not None:
            box = box.clip(clip_size[0], clip_size[1])
        return box

    def clip(self, width: int, height: int) -> "BoundingBox":
        """ Clip the box to an image of ``width`` by ``height`` pixels. """
        return BoundingBox(max(0.0, self.x1), max(0.0, self.y1),
                           min(float(width), self.x2), min(float(height), self.y2))

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.to_int()

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    def __iter__(self) -> Iterator[float]:
        # Iteration follows the canonical order and is handy when drawing.
        return iter(self.xyxy)

    def __repr__(self) -> str:
        return "BoundingBox(x={:.1f}, y={:.1f}, w={:.1f}, h={:.1f})".format(
            self.x, self.y, self.width, self.height)


@dataclass
class Detection:
    """ A single object detected in an image.

    ``kind`` is a short stable identifier such as ``"face"``, ``"cube"``,
    ``"marker"``, ``"pet"`` or ``"robot"``.  Extra detector-specific values
    are stored in :attr:`attributes` and can also be read as attributes
    (``detection.fur_color`` for example).
    """

    kind: str
    bbox: BoundingBox
    confidence: float = 1.0
    label: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    track_id: Optional[int] = None
    timestamp: Optional[float] = None
    mask: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.bbox, BoundingBox):
            self.bbox = BoundingBox.from_iterable(self.bbox)
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        if self.label is None:
            self.label = ""

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails.  Keeping detector
        # specific values accessible makes client code much more readable.
        attributes = self.__dict__.get("attributes", {})
        if name in attributes:
            return attributes[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    @property
    def kind_name(self) -> str:
        """ The part of :attr:`kind` before a ``:`` qualifier. """
        return self.kind.split(":", 1)[0]

    @property
    def center(self) -> Tuple[float, float]:
        return self.bbox.center

    def to_dict(self, include_mask: bool = False) -> Dict[str, Any]:
        data = {
            "kind": self.kind,
            "label": self.label,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict(),
            "attributes": dict(self.attributes),
            "track_id": self.track_id,
            "timestamp": self.timestamp,
        }
        if include_mask and self.mask is not None:
            data["mask"] = self.mask.tolist()
        return data

    def __repr__(self) -> str:
        return "Detection(kind={!r}, label={!r}, confidence={:.2f}, bbox={!r})".format(
            self.kind, self.label, self.confidence, self.bbox)


@dataclass
class Pose2D:
    """ A planar pose (position in millimetres, heading in radians).

    Heading is measured counter-clockwise from the positive X axis, matching
    the Cozmo world frame where X points forward and Y points left.
    """

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def __post_init__(self) -> None:
        self.x = float(self.x)
        self.y = float(self.y)
        self.yaw = float(self.yaw)

    @classmethod
    def from_pose(cls, pose: Any) -> "Pose2D":
        """ Create from a :class:`pycozmo.util.Pose`-like object. """
        position = getattr(pose, "position", pose)
        rotation = getattr(pose, "rotation", None)
        x = float(getattr(position, "x", 0.0))
        y = float(getattr(position, "y", 0.0))
        if rotation is not None:
            yaw = float(rotation.angle_z.radians)
        else:
            yaw = float(getattr(pose, "yaw", 0.0))
        return cls(x, y, yaw)

    @property
    def position(self) -> Tuple[float, float]:
        return self.x, self.y

    def to_tuple(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.yaw

    def distance_to(self, other: "Pose2D") -> float:
        return math.hypot(other.x - self.x, other.y - self.y)

    def angle_to(self, other: "Pose2D") -> float:
        return math.atan2(other.y - self.y, other.x - self.x)

    def relative_angle_to(self, other: "Pose2D") -> float:
        """ Bearing to ``other`` relative to this pose, in ``[-pi, pi]``. """
        return _wrap_angle(self.angle_to(other) - self.yaw)

    def transform(self, other: "Pose2D") -> "Pose2D":
        """ Compose with a pose expressed in this pose's frame. """
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        return Pose2D(
            self.x + c * other.x - s * other.y,
            self.y + s * other.x + c * other.y,
            _wrap_angle(self.yaw + other.yaw),
        )

    def inverse(self) -> "Pose2D":
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        return Pose2D(-c * self.x - s * self.y, s * self.x - c * self.y, -self.yaw)

    def moved(self, distance: float, angle: Optional[float] = None) -> "Pose2D":
        """ Return a pose translated by ``distance`` in the given direction. """
        heading = self.yaw if angle is None else angle
        return Pose2D(self.x + math.cos(heading) * distance,
                      self.y + math.sin(heading) * distance, self.yaw)

    def __iter__(self) -> Iterator[float]:
        return iter(self.to_tuple())

    def __repr__(self) -> str:
        return "Pose2D(x={:.1f}, y={:.1f}, yaw={:.3f})".format(self.x, self.y, self.yaw)


def _wrap_angle(angle: float) -> float:
    """ Wrap an angle to ``[-pi, pi)``. """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def boxes_to_array(boxes: Sequence[BoundingBox]) -> np.ndarray:
    """ Convert boxes to an ``(N, 4)`` float array. """
    return np.asarray([b.xyxy for b in boxes], dtype=np.float32).reshape((-1, 4))


def detections_to_array(detections: Sequence[Detection]) -> np.ndarray:
    """ Convert detection boxes to an ``(N, 4)`` float array. """
    return boxes_to_array([d.bbox for d in detections])


__all__.extend(["boxes_to_array", "detections_to_array"])
