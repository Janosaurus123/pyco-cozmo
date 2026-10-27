"""

Drivable-area estimation from a single camera frame.

The estimator combines a flat-ground camera model with a colour model of the
floor.  It is deliberately model based: no trained segmentation network is
required, and the same code works on Cozmo's 320x240 frames and on larger
Vector images.

"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .calibration import CameraModel
from .types import BoundingBox
from .utils import as_rgb, connected_components


__all__ = [
    "GroundContact",
    "DrivableArea",
    "DrivableAreaEstimator",
    "estimate_drivable_area",
]


@dataclass
class GroundContact:
    """ An obstacle standing on the floor. """

    bbox: BoundingBox
    distance_mm: float
    bearing: float
    area: int = 0
    world: Optional[Tuple[float, float]] = None
    label: str = "obstacle"

    def to_dict(self) -> dict:
        data = {
            "bbox": self.bbox.to_dict(),
            "distance_mm": self.distance_mm,
            "bearing": self.bearing,
            "area": self.area,
            "label": self.label,
        }
        if self.world is not None:
            data["world"] = list(self.world)
        return data


@dataclass
class DrivableArea:
    """ Result of one drivable-area estimation. """

    mask: np.ndarray = field(repr=False)
    free_ratio: float = 0.0
    best_heading: float = 0.0
    clearance_mm: float = 0.0
    contacts: List[GroundContact] = field(default_factory=list)
    horizon_row: float = 0.0
    floor_row: float = 0.0
    floor_color: Tuple[int, int, int] = (0, 0, 0)
    distance_map: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def free(self) -> np.ndarray:
        return self.mask

    @property
    def obstacle_mask(self) -> np.ndarray:
        if self.distance_map is None:
            return np.zeros_like(self.mask, dtype=np.bool_)
        valid = np.isfinite(self.distance_map) & (self.distance_map > 0)
        return valid & ~self.mask

    @property
    def heading_degrees(self) -> float:
        return math.degrees(self.best_heading)

    def to_dict(self) -> dict:
        return {
            "free_ratio": self.free_ratio,
            "best_heading": self.best_heading,
            "clearance_mm": self.clearance_mm,
            "horizon_row": self.horizon_row,
            "floor_row": self.floor_row,
            "floor_color": list(self.floor_color),
            "contacts": [contact.to_dict() for contact in self.contacts],
        }

    def annotate(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """ Return a Pillow image with a translucent drivable-area overlay. """
        base = Image.fromarray(as_rgb(image)).convert("RGBA")
        overlay = np.zeros((base.height, base.width, 4), dtype=np.uint8)
        overlay[self.mask] = (40, 220, 90, 90)
        for contact in self.contacts:
            x1, y1, x2, y2 = contact.bbox.to_int()
            overlay[max(0, y1):max(0, y2), max(0, x1):x1 + 2] = (255, 60, 40, 200)
            overlay[max(0, y1):max(0, y2), max(0, x2 - 2):max(0, x2)] = (255, 60, 40, 200)
            overlay[max(0, y1):y1 + 2, max(0, x1):max(0, x2)] = (255, 60, 40, 200)
            overlay[max(0, y2 - 2):max(0, y2), max(0, x1):max(0, x2)] = (255, 60, 40, 200)
        return Image.alpha_composite(base, Image.fromarray(overlay)).convert("RGB")


class DrivableAreaEstimator:
    """ Segment the floor and find the safest heading.

    Args:
        max_distance_mm: Ignore ground farther away than this.
        min_distance_mm: Ignore ground closer than this (robot body, bumper).
        color_tolerance: Maximum chromaticity distance from the floor colour.
        brightness_range: Accepted brightness ratio relative to the floor.
        min_contact_area: Minimum pixel area of an obstacle contact.
        heading_samples: Number of ray directions evaluated for the best
            heading.
        clearance_step_mm: Ray-marching step along each heading.
    """

    def __init__(self,
                 max_distance_mm: float = 2500.0,
                 min_distance_mm: float = 20.0,
                 color_tolerance: float = 0.11,
                 brightness_range: Tuple[float, float] = (0.45, 1.75),
                 min_contact_area: int = 120,
                 heading_samples: int = 41,
                 clearance_step_mm: float = 25.0) -> None:
        self.max_distance_mm = float(max_distance_mm)
        self.min_distance_mm = float(min_distance_mm)
        self.color_tolerance = float(color_tolerance)
        self.brightness_range = (float(brightness_range[0]), float(brightness_range[1]))
        self.min_contact_area = int(min_contact_area)
        self.heading_samples = max(3, int(heading_samples))
        self.clearance_step_mm = max(5.0, float(clearance_step_mm))

    @staticmethod
    def horizon_row(camera: CameraModel) -> float:
        """ Row of the geometric horizon for ``camera``. """
        return float(camera.horizon_row())

    @staticmethod
    def _dominant_color(rgb: np.ndarray) -> Tuple[int, int, int]:
        pixels = rgb.reshape(-1, 3)
        if len(pixels) == 0:
            return (128, 128, 128)
        quantized = (pixels.astype(np.int32) // 16)
        codes = (quantized[:, 0] << 8) | (quantized[:, 1] << 4) | quantized[:, 2]
        counts = np.bincount(codes, minlength=4096)
        code = int(np.argmax(counts))
        selected = pixels[codes == code]
        mean = selected.mean(axis=0)
        return tuple(int(round(value)) for value in mean)  # type: ignore

    def _distance_map(self, camera: CameraModel) -> np.ndarray:
        distances = np.full(camera.height, np.inf, dtype=np.float32)
        for row in range(camera.height):
            distances[row] = camera.ground_distance_for_row(row)
        return distances

    def _floor_start(self, color_match: np.ndarray, horizon: float) -> int:
        height, width = color_match.shape
        x1 = int(width * 0.10)
        x2 = max(x1 + 1, int(width * 0.90))
        start = max(0, int(math.ceil(horizon)))
        if start >= height:
            return height
        fractions = color_match[start:, x1:x2].mean(axis=1)
        threshold = 0.12
        minimum_run = max(3, int(height * 0.015))
        run = 0
        for offset, fraction in enumerate(fractions):
            if fraction >= threshold:
                run += 1
                if run >= minimum_run:
                    return start + offset - run + 1
            else:
                run = 0
        # The floor may be visible only in a small part of the frame.  Use
        # the first row with any reasonable support instead of giving up.
        candidates = np.nonzero(fractions >= threshold)[0]
        if len(candidates):
            return start + int(candidates[0])
        return start

    def estimate(self, image: Union[np.ndarray, Image.Image],
                 camera: Optional[CameraModel] = None) -> DrivableArea:
        rgb = as_rgb(image)
        height, width = rgb.shape[:2]
        if camera is None:
            camera = CameraModel.cozmo(width=width, height=height)
        elif camera.width != width or camera.height != height:
            camera = camera.scaled(width, height)

        horizon = float(np.clip(camera.horizon_row(), 0, height))
        distance_map = self._distance_map(camera)

        # Estimate the floor colour from the lower central part of the frame.
        y1 = int(height * 0.82)
        x1 = int(width * 0.20)
        x2 = max(x1 + 1, int(width * 0.80))
        sample = rgb[y1:height, x1:x2]
        if sample.size == 0:
            sample = rgb
        floor_color = self._dominant_color(sample)
        floor = np.asarray(floor_color, dtype=np.float32)
        denominator = max(1.0, float(floor.sum()))
        floor_chroma = floor / denominator
        floor_brightness = max(1.0, float(floor.mean()))

        values = rgb.astype(np.float32)
        sums = np.maximum(1.0, values.sum(axis=2, keepdims=True))
        chroma = values / sums
        chroma_distance = np.linalg.norm(chroma - floor_chroma.reshape(1, 1, 3), axis=2)
        brightness = values.mean(axis=2) / floor_brightness
        color_match = ((chroma_distance <= self.color_tolerance) &
                       (brightness >= self.brightness_range[0]) &
                       (brightness <= self.brightness_range[1]))

        floor_row = self._floor_start(color_match, horizon)
        rows = np.arange(height, dtype=np.float32).reshape(-1, 1)
        valid = ((rows >= floor_row) &
                 (distance_map.reshape(-1, 1) >= self.min_distance_mm) &
                 (distance_map.reshape(-1, 1) <= self.max_distance_mm))
        valid = np.broadcast_to(valid, (height, width)).copy()
        free = valid & color_match

        # Remove isolated one-pixel free/obstacle specks.  This is done with
        # NumPy only so the estimator also works without OpenCV.
        free = _majority_filter(free, iterations=1)
        free &= valid

        obstacle_mask = valid & ~free
        contacts: List[GroundContact] = []
        for component in connected_components(obstacle_mask, min_area=self.min_contact_area):
            bottom_x = component.x + component.width / 2.0
            bottom_y = component.y + component.height - 1
            row = int(np.clip(round(bottom_y), 0, height - 1))
            distance = float(distance_map[row])
            if not math.isfinite(distance):
                point = camera.ground_intersection(bottom_x, bottom_y)
                if point is None:
                    continue
                distance = float(math.hypot(point[0] - camera.position[0],
                                            point[1] - camera.position[1]))
            bearing = -math.atan2(bottom_x - camera.cx, max(1e-6, camera.fx))
            world = camera.ground_intersection(bottom_x, bottom_y)
            contacts.append(GroundContact(
                bbox=BoundingBox(component.x, component.y,
                                 component.x + component.width,
                                 component.y + component.height),
                distance_mm=distance,
                bearing=bearing,
                area=component.area,
                world=None if world is None else (world[0], world[1]),
            ))
        contacts.sort(key=lambda contact: contact.distance_mm)

        total_valid = int(valid.sum())
        free_ratio = float(free.sum()) / total_valid if total_valid else 0.0

        best_heading, clearance = self._best_heading(camera, free)
        return DrivableArea(
            mask=free,
            free_ratio=free_ratio,
            best_heading=best_heading,
            clearance_mm=clearance,
            contacts=contacts,
            horizon_row=horizon,
            floor_row=float(floor_row),
            floor_color=floor_color,
            distance_map=distance_map.reshape(-1, 1) * np.ones((1, width), dtype=np.float32),
        )

    def _best_heading(self, camera: CameraModel, free: np.ndarray) -> Tuple[float, float]:
        height, width = free.shape
        half_fov = camera.hfov / 2.0
        margin = min(0.04, half_fov * 0.25)
        offsets = np.linspace(-half_fov + margin, half_fov - margin, self.heading_samples)
        distances = np.arange(self.min_distance_mm, self.max_distance_mm + 1e-6,
                              self.clearance_step_mm)
        if not len(distances):
            return 0.0, 0.0

        best_heading = 0.0
        best_clearance = -1.0
        best_score = -float("inf")
        for offset in offsets:
            world_angle = camera.yaw + float(offset)
            points = np.column_stack([
                camera.position[0] + np.cos(world_angle) * distances,
                camera.position[1] + np.sin(world_angle) * distances,
                np.zeros_like(distances),
            ])
            pixels = camera.project(points)
            free_flags = np.zeros(len(distances), dtype=np.bool_)
            for index, (u, v) in enumerate(pixels):
                if not (np.isfinite(u) and np.isfinite(v)):
                    continue
                column = int(round(float(u)))
                row = int(round(float(v)))
                if 0 <= row < height and 0 <= column < width:
                    # A tiny 3x3 window makes the estimate tolerant of
                    # quantization holes between adjacent ground pixels.
                    window = free[max(0, row - 1):row + 2, max(0, column - 1):column + 2]
                    free_flags[index] = bool(window.any())

            clearance = 0.0
            blocked_run = 0
            started = False
            for index, is_free in enumerate(free_flags):
                if is_free:
                    started = True
                    blocked_run = 0
                    clearance = float(distances[index])
                elif started:
                    blocked_run += 1
                    if blocked_run >= 2:
                        break
            if not started or clearance <= 0.0:
                continue
            # Prefer a long clear path, and only mildly prefer driving
            # straight ahead when several directions are equally open.
            score = clearance - 20.0 * abs(float(offset))
            if score > best_score:
                best_score = score
                best_heading = float(offset)
                best_clearance = clearance
        if best_clearance < 0.0:
            return 0.0, 0.0
        return best_heading, best_clearance


def _majority_filter(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """ Majority (3x3) filter for a boolean mask. """
    result = np.asarray(mask, dtype=np.bool_)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(result, 1, mode="edge").astype(np.uint8)
        total = np.zeros_like(result, dtype=np.uint8)
        for dy in range(3):
            for dx in range(3):
                total += padded[dy:dy + result.shape[0], dx:dx + result.shape[1]]
        result = total >= 5
    return result


def estimate_drivable_area(image: Union[np.ndarray, Image.Image],
                           camera: Optional[CameraModel] = None,
                           **kwargs) -> DrivableArea:
    """ Convenience wrapper around :class:`DrivableAreaEstimator`. """
    return DrivableAreaEstimator(**kwargs).estimate(image, camera=camera)
