"""

Pet detection.

Cozmo's camera is a low-resolution colour camera, so pet detection is based
on warm fur colours (brown, orange and tan) combined with blob shape and the
ground-plane geometry.  The detector is intentionally conservative: false
positives are more annoying than a missed pet because they can make the robot
avoid an empty piece of floor.

"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .calibration import CameraModel
from .types import BoundingBox, Detection
from .utils import (as_hsv, as_rgb, color_name, connected_components,
                    dominant_color, nms)


__all__ = [
    "Pet",
    "PetDetector",
    "detect_pets",
]


@dataclass
class Pet(Detection):
    """ A detected pet. """

    fur_color: str = "unknown"
    species: str = "pet"
    distance_mm: Optional[float] = None
    bearing: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind == "pet" and not self.label:
            self.label = self.species

    def to_dict(self, include_mask: bool = False) -> dict:
        data = super().to_dict(include_mask=include_mask)
        data.update({
            "fur_color": self.fur_color,
            "species": self.species,
            "distance_mm": self.distance_mm,
            "bearing": self.bearing,
        })
        return data


class PetDetector:
    """ Detect warm-coloured pets on the floor.

    Args:
        min_area: Minimum blob area in pixels.
        max_area_ratio: Maximum blob area as a fraction of the image.
        min_confidence: Detections below this confidence are discarded.
        detect_dark: Also accept black/dark fur blobs.  Disabled by default
            because robot bodies are also dark.
    """

    def __init__(self,
                 min_area: int = 260,
                 max_area_ratio: float = 0.35,
                 min_confidence: float = 0.35,
                 detect_dark: bool = False) -> None:
        self.min_area = int(min_area)
        self.max_area_ratio = float(max_area_ratio)
        self.min_confidence = float(min_confidence)
        self.detect_dark = bool(detect_dark)

    def _mask(self, rgb: np.ndarray) -> np.ndarray:
        hsv = as_hsv(rgb)
        hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # Brown, orange and tan fur.  The saturation floor keeps beige
        # floors and grey walls out of the mask.
        warm = (((hue >= 4) & (hue <= 38)) &
                (saturation >= 65) & (saturation <= 250) &
                (value >= 35) & (value <= 225))
        if not self.detect_dark:
            return warm
        dark = (value >= 15) & (value <= 85)
        return warm | dark

    def detect(self, image: Union[np.ndarray, Image.Image],
               camera: Optional[CameraModel] = None) -> List[Pet]:
        rgb = as_rgb(image)
        height, width = rgb.shape[:2]
        if camera is None:
            camera = CameraModel.cozmo(width=width, height=height)
        elif camera.width != width or camera.height != height:
            camera = camera.scaled(width, height)

        mask = self._mask(rgb)
        horizon = max(0.0, camera.horizon_row() - 8.0)
        mask[:int(horizon), :] = False
        maximum_area = int(self.max_area_ratio * width * height)
        detections: List[Pet] = []

        for component in connected_components(mask, min_area=self.min_area):
            if component.area > maximum_area:
                continue
            aspect = component.width / float(component.height) if component.height else 0.0
            if not 0.30 <= aspect <= 3.4:
                continue
            if component.extent < 0.35:
                continue
            bbox = BoundingBox(component.x, component.y,
                               component.x + component.width,
                               component.y + component.height)
            local_mask = np.zeros((height, width), dtype=np.bool_)
            local_mask[component.y:component.y + component.height,
                       component.x:component.x + component.width] = component.mask
            fur = dominant_color(rgb, local_mask)
            # Grey, blue and red objects are not warm-fur candidates.
            if not self.detect_dark:
                maximum = max(fur)
                minimum = min(fur)
                if maximum <= 0 or (maximum - minimum) / float(maximum) < 0.12:
                    continue

            area_score = min(1.0, component.area / 2200.0)
            extent_score = min(1.0, component.extent / 0.80)
            distance_mm, bearing = _ground_geometry(camera, bbox)
            distance_score = 0.0
            if distance_mm is not None:
                distance_score = float(np.clip(1.0 - distance_mm / 3500.0, 0.0, 1.0))
            confidence = float(np.clip(
                0.30 + 0.30 * area_score + 0.20 * extent_score + 0.15 * distance_score,
                0.0, 1.0))
            if confidence < self.min_confidence:
                continue
            detections.append(Pet(
                kind="pet",
                label="pet",
                confidence=confidence,
                bbox=bbox,
                fur_color=color_name(fur),
                distance_mm=distance_mm,
                bearing=bearing,
                attributes={
                    "area": component.area,
                    "extent": component.extent,
                    "color_rgb": fur,
                },
            ))
        return nms(detections, iou_threshold=0.35)


def _ground_geometry(camera: CameraModel, bbox: BoundingBox
                     ) -> Tuple[Optional[float], float]:
    u = bbox.center_x
    v = bbox.y2 - 1
    point = camera.ground_intersection(u, v)
    bearing = -math.atan2(u - camera.cx, max(1e-6, camera.fx))
    if point is None:
        distance = camera.ground_distance_for_row(v)
        if not math.isfinite(distance):
            return None, bearing
        return float(distance), bearing
    distance = math.hypot(point[0] - camera.position[0],
                          point[1] - camera.position[1])
    return float(distance), bearing


def detect_pets(image: Union[np.ndarray, Image.Image],
                camera: Optional[CameraModel] = None,
                **kwargs) -> List[Pet]:
    """ Convenience wrapper around :class:`PetDetector`. """
    return PetDetector(**kwargs).detect(image, camera=camera)
