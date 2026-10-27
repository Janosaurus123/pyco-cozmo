"""

Detection of other Cozmo and Vector robots.

Both robots are recognised by their illuminated face screen and the dark body
below it.  Looking for a bright green/cyan screen first keeps the detector
cheap and avoids the classic "dark blob on the floor" false positive.

"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .calibration import CameraModel
from .types import BoundingBox, Detection
from .utils import (as_gray, as_hsv, as_rgb, color_name, connected_components,
                    dominant_color, nms)


__all__ = [
    "Robot",
    "RobotDetector",
    "detect_robots",
]


@dataclass
class Robot(Detection):
    """ A detected robot with a face screen. """

    screen_color: str = "unknown"
    screen_bbox: Optional[BoundingBox] = None
    distance_mm: Optional[float] = None
    bearing: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.label:
            self.label = self.kind

    def to_dict(self, include_mask: bool = False) -> dict:
        data = super().to_dict(include_mask=include_mask)
        data.update({
            "screen_color": self.screen_color,
            "screen_bbox": self.screen_bbox.to_dict() if self.screen_bbox else None,
            "distance_mm": self.distance_mm,
            "bearing": self.bearing,
        })
        return data


class RobotDetector:
    """ Detect Cozmo and Vector robots.

    Args:
        min_screen_area: Minimum face-screen area in pixels.
        min_confidence: Detections below this confidence are discarded.
        vector_screen_ratio: A screen wider than this fraction of the image
            is labelled as Vector rather than Cozmo.
    """

    def __init__(self,
                 min_screen_area: int = 70,
                 min_confidence: float = 0.40,
                 vector_screen_ratio: float = 0.18) -> None:
        self.min_screen_area = int(min_screen_area)
        self.min_confidence = float(min_confidence)
        self.vector_screen_ratio = float(vector_screen_ratio)

    def _screen_mask(self, rgb: np.ndarray) -> np.ndarray:
        hsv = as_hsv(rgb)
        hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        coloured = (((hue >= 35) & (hue <= 100)) &
                    (saturation >= 90) & (value >= 80))
        # A purely white screen has no hue.  Accept very bright, low
        # saturation blobs only when they are large and compact.
        white = (value >= 215) & (saturation <= 35)
        return coloured | white

    def detect(self, image: Union[np.ndarray, Image.Image],
               camera: Optional[CameraModel] = None) -> List[Robot]:
        rgb = as_rgb(image)
        height, width = rgb.shape[:2]
        if camera is None:
            camera = CameraModel.cozmo(width=width, height=height)
        elif camera.width != width or camera.height != height:
            camera = camera.scaled(width, height)

        screen_mask = self._screen_mask(rgb)
        screen_mask[:int(max(0.0, camera.horizon_row() - 20.0)), :] = False
        gray = as_gray(rgb)
        hsv = as_hsv(rgb)
        dark_mask = (gray <= 85) & (hsv[:, :, 1] <= 150)
        detections: List[Robot] = []

        for screen in connected_components(screen_mask, min_area=self.min_screen_area):
            if screen.area < self.min_screen_area:
                continue
            aspect = screen.width / float(screen.height) if screen.height else 0.0
            if not 0.45 <= aspect <= 4.5 or screen.extent < 0.35:
                continue
            screen_bbox = BoundingBox(screen.x, screen.y,
                                      screen.x + screen.width,
                                      screen.y + screen.height)

            # Search just below the screen for a dark, roughly body-sized
            # component.  The body may extend a little to either side.
            margin_x = int(max(3, screen.width * 0.20))
            search_x1 = max(0, screen.x - margin_x)
            search_x2 = min(width, screen.x + screen.width + margin_x)
            search_y1 = max(0, screen.y + screen.height - 2)
            search_y2 = min(height, screen.y + screen.height + int(max(12, screen.height * 3.2)))
            body_region = dark_mask[search_y1:search_y2, search_x1:search_x2]
            body_bbox = None
            body_area = 0
            best_overlap = 0
            if body_region.size:
                for body in connected_components(body_region, min_area=120):
                    if body.width < screen.width * 0.55:
                        continue
                    if body.height < screen.height * 0.35:
                        continue
                    body_x1 = search_x1 + body.x
                    body_y1 = search_y1 + body.y
                    body_x2 = body_x1 + body.width
                    body_y2 = body_y1 + body.height
                    horizontal_overlap = max(0, min(body_x2, screen.x + screen.width) -
                                             max(body_x1, screen.x))
                    if horizontal_overlap < 0.5 * screen.width:
                        continue
                    if horizontal_overlap > best_overlap:
                        best_overlap = horizontal_overlap
                        body_area = body.area
                        body_bbox = BoundingBox(body_x1, body_y1, body_x2, body_y2)

            if body_bbox is None:
                # A screen without a visible body is still useful, but it is
                # less trustworthy.  Keep it only when it is large.
                if screen.area < self.min_screen_area * 3:
                    continue
                body_bbox = BoundingBox(screen.x - margin_x,
                                        screen.y + screen.height,
                                        screen.x + screen.width + margin_x,
                                        screen.y + screen.height + int(screen.height * 1.4))
            bbox = BoundingBox(min(screen_bbox.x1, body_bbox.x1),
                               screen_bbox.y1,
                               max(screen_bbox.x2, body_bbox.x2),
                               max(screen_bbox.y2, body_bbox.y2)).clip(width, height)

            screen_local = np.zeros((height, width), dtype=np.bool_)
            screen_local[screen.y:screen.y + screen.height,
                         screen.x:screen.x + screen.width] = screen.mask
            coloured_only = screen_local & (hsv[:, :, 1] >= 60)
            if coloured_only.any():
                screen_rgb = dominant_color(rgb, coloured_only)
            else:
                screen_rgb = dominant_color(rgb, screen_local)

            area_score = min(1.0, screen.area / 1800.0)
            body_score = min(1.0, body_area / 3000.0) if body_area else 0.0
            confidence = float(np.clip(0.35 + 0.35 * area_score + 0.30 * body_score,
                                       0.0, 1.0))
            if confidence < self.min_confidence:
                continue

            distance_mm, bearing = _ground_geometry(camera, bbox)
            kind = "vector" if screen.width >= width * self.vector_screen_ratio else "cozmo"
            detections.append(Robot(
                kind=kind,
                label=kind,
                confidence=confidence,
                bbox=bbox,
                screen_color=color_name(screen_rgb),
                screen_bbox=screen_bbox,
                distance_mm=distance_mm,
                bearing=bearing,
                attributes={
                    "screen_area": screen.area,
                    "body_area": body_area,
                    "screen_rgb": screen_rgb,
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


def detect_robots(image: Union[np.ndarray, Image.Image],
                  camera: Optional[CameraModel] = None,
                  **kwargs) -> List[Robot]:
    """ Convenience wrapper around :class:`RobotDetector`. """
    return RobotDetector(**kwargs).detect(image, camera=camera)
