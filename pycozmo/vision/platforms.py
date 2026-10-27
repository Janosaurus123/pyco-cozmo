"""

Detection of Cozmo charging platforms and Vector charging bases.

The charger is one of the hardest objects to find from a low camera: it is
mostly flat, often nearly the same colour as the floor, and only a few LEDs
reveal its presence.  This detector therefore starts from the illuminated
charging contacts/LEDs, associates nearby dark plastic, and verifies that the
result looks like a low, wide platform.

"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from .calibration import CameraModel
from .types import BoundingBox, Detection
from .utils import (as_gray, as_hsv, as_rgb, color_name, connected_components,
                    dominant_color, nms)


__all__ = [
    "Platform",
    "PlatformDetector",
    "ChargerDetector",
    "detect_platforms",
]


@dataclass
class Platform(Detection):
    """ A detected charging platform. """

    platform_type: str = "charger"
    led_count: int = 0
    led_color: Tuple[int, int, int] = (0, 0, 0)
    led_boxes: Tuple[BoundingBox, ...] = ()
    distance_mm: Optional[float] = None
    bearing: float = 0.0
    on_charger: Optional[bool] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.label:
            self.label = self.platform_type
        self.led_color = tuple(int(value) for value in self.led_color)  # type: ignore

    def to_dict(self, include_mask: bool = False) -> dict:
        data = super().to_dict(include_mask=include_mask)
        data.update({
            "platform_type": self.platform_type,
            "led_count": self.led_count,
            "led_color": list(self.led_color),
            "led_boxes": [box.to_dict() for box in self.led_boxes],
            "distance_mm": self.distance_mm,
            "bearing": self.bearing,
            "on_charger": self.on_charger,
        })
        return data


class PlatformDetector:
    """ Find chargers by their blue/cyan charging LEDs.

    Args:
        min_led_area: Minimum area of an individual LED in pixels.
        min_led_width: Minimum width of an LED in pixels.
        max_led_aspect: Maximum width/height ratio of a charging LED.
        min_confidence: Detections below this confidence are discarded.
        max_image_width: Large frames are downscaled before detection.
        detect_mats: Also look for large, uniformly coloured play mats.
            This is disabled by default because a coloured floor can easily
            be mistaken for a mat.
    """

    def __init__(self,
                 min_led_area: int = 8,
                 min_led_width: int = 5,
                 max_led_aspect: float = 9.0,
                 min_confidence: float = 0.35,
                 max_image_width: int = 640,
                 detect_mats: bool = False,
                 mat_min_area_ratio: float = 0.08) -> None:
        self.min_led_area = max(1, int(min_led_area))
        self.min_led_width = max(1, int(min_led_width))
        self.max_led_aspect = float(max_led_aspect)
        self.min_confidence = float(min_confidence)
        self.max_image_width = int(max_image_width)
        self.detect_mats = bool(detect_mats)
        self.mat_min_area_ratio = float(mat_min_area_ratio)
        self._scale = 1.0

    @staticmethod
    def _led_mask(rgb: np.ndarray) -> np.ndarray:
        hsv = as_hsv(rgb)
        hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # Cozmo's charger uses blue LEDs; some revisions and Vector's base
        # use cyan LEDs.  Keep the range narrow enough to reject blue jeans
        # and robot screens (which are usually green).
        blue = (((hue >= 88) & (hue <= 138)) &
                (saturation >= 120) & (value >= 110))
        # Very bright white contacts are accepted only when they are small
        # and lie next to a blue LED; they are added by the association step.
        return blue

    def _prepare(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        rgb = as_rgb(image)
        self._scale = 1.0
        if rgb.shape[1] > self.max_image_width:
            from .utils import resize
            self._scale = self.max_image_width / float(rgb.shape[1])
            width = self.max_image_width
            height = max(1, int(round(rgb.shape[0] * self._scale)))
            rgb = resize(rgb, (width, height), interpolation="area")
        return rgb

    def _visible_leds(self, rgb: np.ndarray) -> List[Tuple[BoundingBox, int]]:
        mask = self._led_mask(rgb)
        leds: List[Tuple[BoundingBox, int]] = []
        for component in connected_components(mask, min_area=self.min_led_area):
            if component.width < self.min_led_width:
                continue
            aspect = component.width / float(max(1, component.height))
            if aspect > self.max_led_aspect:
                continue
            # Charging LEDs are bright and compact.  Long horizontal light
            # reflections are rejected below by the grouping logic.
            box = BoundingBox(component.x, component.y,
                              component.x + component.width,
                              component.y + component.height)
            if box.width < 2.0 or box.height < 1.0:
                continue
            leds.append((box, component.area))
        return leds

    @staticmethod
    def _cluster_leds(leds: Sequence[Tuple[BoundingBox, int]],
                      max_gap: float = 55.0) -> List[List[Tuple[BoundingBox, int]]]:
        """ Group LEDs which plausibly belong to one charger. """
        remaining = list(leds)
        groups: List[List[Tuple[BoundingBox, int]]] = []
        while remaining:
            group = [remaining.pop(0)]
            changed = True
            while changed:
                changed = False
                box = group[-1][0]
                for index in range(len(remaining) - 1, -1, -1):
                    candidate = remaining[index][0]
                    vertical_gap = abs(candidate.center_y - box.center_y)
                    horizontal_gap = max(0.0, max(candidate.x1, box.x1) - min(candidate.x2, box.x2))
                    if (horizontal_gap <= max_gap and
                            vertical_gap <= max(3.0, max(candidate.height, box.height) * 1.5)):
                        group.append(remaining.pop(index))
                        changed = True
            groups.append(group)
        return groups

    @staticmethod
    def _platform_box(rgb: np.ndarray, group: Sequence[Tuple[BoundingBox, int]]) -> Optional[BoundingBox]:
        height, width = rgb.shape[:2]
        leds_box = BoundingBox(
            min(item[0].x1 for item in group),
            min(item[0].y1 for item in group),
            max(item[0].x2 for item in group),
            max(item[0].y2 for item in group),
        )
        led_span = max(1.0, leds_box.width)
        margin_x = max(5.0, led_span * 0.35)
        margin_y = max(4.0, leds_box.height * 2.4)
        search = BoundingBox(leds_box.x1 - margin_x,
                             leds_box.y1 - margin_y,
                             leds_box.x2 + margin_x,
                             leds_box.y2 + margin_y).clip(width, height)
        if search.width <= 0 or search.height <= 0:
            return None
        x1, y1, x2, y2 = search.to_int()
        region = rgb[y1:y2, x1:x2]
        if not region.size:
            return None
        gray = as_gray(region)
        # The plastic charger is usually darker than the floor.  Include
        # dark pixels and the LEDs themselves, then take the largest blob.
        mask = gray <= 105
        led_local = np.zeros(mask.shape, dtype=np.bool_)
        for box, _area in group:
            lx1 = int(max(0, box.x1 - x1))
            ly1 = int(max(0, box.y1 - y1))
            lx2 = int(min(mask.shape[1], box.x2 - x1))
            ly2 = int(min(mask.shape[0], box.y2 - y1))
            if lx2 > lx1 and ly2 > ly1:
                led_local[ly1:ly2, lx1:lx2] = True
        candidates = connected_components(mask | led_local, min_area=max(35, int(led_span * 4)))
        if not candidates:
            return None
        best = None
        for component in candidates:
            box = BoundingBox(component.x + x1, component.y + y1,
                              component.x + component.width + x1,
                              component.y + component.height + y1)
            # A charger is wide but not extremely tall and must contain
            # (most of) the LED group.
            if not 0.65 <= box.width / float(max(1.0, box.height)) <= 8.0:
                continue
            overlap = box.intersection(leds_box).area
            if overlap < 0.45 * max(1.0, leds_box.area):
                continue
            score = box.area
            if best is None or score > best[0]:
                best = (score, box)
        if best is None:
            # Fall back to a conservative box around the LEDs.  This makes
            # the detector useful when the platform blends into the floor,
            # while requiring enough LEDs to avoid random blue specks.
            if len(group) < 2 and leds_box.width < 18:
                return None
            box = BoundingBox(leds_box.x1 - led_span * 0.15,
                              leds_box.y1 - leds_box.height * 1.8,
                              leds_box.x2 + led_span * 0.15,
                              leds_box.y2 + leds_box.height * 0.8).clip(width, height)
            return box
        return best[1].clip(width, height)

    def _detect_mats(self, rgb: np.ndarray, camera: CameraModel) -> List[Platform]:
        """ Detect large, uniformly coloured play mats.

        This is intentionally conservative: a large flat region must have a
        high fill ratio and a different hue from the lower-floor sample.
        """
        height, width = rgb.shape[:2]
        hsv = as_hsv(rgb)
        hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        coloured = (saturation >= 70) & (value >= 45)
        if not coloured.any():
            return []
        # Remove the usual floor sample from consideration.
        floor_region = np.zeros_like(coloured)
        floor_region[int(height * 0.80):, :] = True
        if floor_region.any():
            floor_hue = np.median(hue[floor_region & coloured]) if (floor_region & coloured).any() else -1
        else:
            floor_hue = -1
        if floor_hue >= 0:
            diff = np.minimum(np.abs(hue.astype(np.int16) - int(floor_hue)),
                              np.abs(hue.astype(np.int16) - int(floor_hue) - 180))
            diff = np.minimum(diff, np.abs(hue.astype(np.int16) - int(floor_hue) + 180))
            coloured &= diff > 12
        detections: List[Platform] = []
        for component in connected_components(coloured, min_area=int(height * width * self.mat_min_area_ratio)):
            if component.width < width * 0.25 or component.height < height * 0.08:
                continue
            aspect = component.width / float(max(1, component.height))
            if not 1.2 <= aspect <= 12.0:
                continue
            if component.extent < 0.42:
                continue
            bbox = BoundingBox(component.x, component.y,
                               component.x + component.width,
                               component.y + component.height).clip(width, height)
            mask = np.zeros((height, width), dtype=np.bool_)
            mask[component.y:component.y + component.height,
                 component.x:component.x + component.width] = component.mask
            color = dominant_color(rgb, mask)
            distance, bearing = self._geometry(camera, bbox)
            detections.append(Platform(
                kind="platform",
                label="platform",
                platform_type="play_mat",
                confidence=float(np.clip(0.35 + component.extent * 0.4, 0.0, 1.0)),
                bbox=bbox,
                led_count=0,
                led_color=color,
                distance_mm=distance,
                bearing=bearing,
                attributes={"extent": component.extent},
            ))
        return detections

    @staticmethod
    def _geometry(camera: CameraModel, bbox: BoundingBox) -> Tuple[Optional[float], float]:
        u = bbox.center_x
        v = bbox.y2 - 1
        point = camera.ground_intersection(u, v)
        bearing = -math.atan2(u - camera.cx, max(1e-6, camera.fx))
        if point is None:
            distance = camera.ground_distance_for_row(v)
            return (float(distance), bearing) if math.isfinite(distance) else (None, bearing)
        return float(math.hypot(point[0] - camera.position[0],
                                point[1] - camera.position[1])), bearing

    def detect(self, image: Union[np.ndarray, Image.Image],
               camera: Optional[CameraModel] = None,
               on_charger: Optional[bool] = None) -> List[Platform]:
        rgb = self._prepare(image)
        height, width = rgb.shape[:2]
        if camera is None:
            camera = CameraModel.cozmo(width=width, height=height)
        elif camera.width != width or camera.height != height:
            camera = camera.scaled(width, height)

        detections: List[Platform] = []
        leds = self._visible_leds(rgb)
        for group in self._cluster_leds(leds):
            if not group:
                continue
            leds_box = BoundingBox(min(item[0].x1 for item in group),
                                   min(item[0].y1 for item in group),
                                   max(item[0].x2 for item in group),
                                   max(item[0].y2 for item in group))
            # Reject tiny isolated reflections.
            if len(group) < 2 and leds_box.width < 12:
                continue
            platform_box = self._platform_box(rgb, group)
            if platform_box is None:
                continue
            mask = np.zeros((height, width), dtype=np.bool_)
            for box, _area in group:
                x1, y1, x2, y2 = box.clip(width, height).to_int()
                mask[y1:y2, x1:x2] = True
            led_color = dominant_color(rgb, mask) if mask.any() else (0, 0, 0)
            distance, bearing = self._geometry(camera, platform_box)
            led_score = min(1.0, len(group) / 2.0)
            size_score = min(1.0, platform_box.width / (width * 0.55))
            confidence = float(np.clip(0.40 + 0.35 * led_score + 0.25 * size_score, 0.0, 1.0))
            if confidence < self.min_confidence:
                continue
            platform_type = "charger" if color_name(led_color) in ("blue", "cyan") else "charging_base"
            detections.append(Platform(
                kind="platform",
                label=platform_type,
                platform_type=platform_type,
                confidence=confidence,
                bbox=platform_box,
                led_count=len(group),
                led_color=led_color,
                led_boxes=tuple(item[0] for item in group),
                distance_mm=distance,
                bearing=bearing,
                on_charger=on_charger,
                attributes={
                    "led_color_name": color_name(led_color),
                    "led_span": leds_box.width,
                    "scale": 1.0 / self._scale if self._scale else 1.0,
                },
            ))

        if self.detect_mats:
            detections.extend(self._detect_mats(rgb, camera))
        return nms(detections, iou_threshold=0.35)


#: Compatibility alias - the charger is the usual platform of interest.
ChargerDetector = PlatformDetector


def detect_platforms(image: Union[np.ndarray, Image.Image],
                     camera: Optional[CameraModel] = None,
                     on_charger: Optional[bool] = None,
                     **kwargs) -> List[Platform]:
    """ Convenience wrapper around :class:`PlatformDetector`. """
    return PlatformDetector(**kwargs).detect(image, camera=camera,
                                             on_charger=on_charger)
