"""

Frame-difference motion detection.

The detector maintains a slowly adapting background model.  A large
fraction of changed pixels is treated as a global illumination change (for
example, the room lights being switched on) instead of a moving object.

"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import List, Optional

import numpy as np

from .types import BoundingBox, Detection
from .utils import as_gray, connected_components, dilate, gaussian_blur


__all__ = [
    "MotionResult",
    "MotionDetector",
    "MotionTracker",
    "detect_motion",
]


@dataclass
class MotionResult:
    """ Result of one motion detection update. """

    detections: List[Detection] = field(default_factory=list)
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    score: float = 0.0
    changed: bool = False
    global_change: bool = False
    timestamp: Optional[float] = None

    @property
    def regions(self) -> List[BoundingBox]:
        return [detection.bbox for detection in self.detections]

    @property
    def motion_detected(self) -> bool:
        return bool(self.detections)

    def __bool__(self) -> bool:
        return self.motion_detected


class MotionDetector:
    """ Detect motion between successive camera frames.

    Args:
        threshold: Per-pixel absolute difference threshold (0..255).
        min_area: Minimum connected region area in pixels.
        learning_rate: Background adaptation rate in the range ``(0, 1]``.
        blur_size: Odd Gaussian/box blur kernel size.
        dilate_iterations: Number of dilation passes used to join fragments.
        max_motion_ratio: Above this fraction of changed pixels the frame is
            considered a global lighting change.
        use_opencv: Use OpenCV when available (falls back automatically).
    """

    def __init__(self,
                 threshold: int = 25,
                 min_area: int = 150,
                 learning_rate: float = 0.05,
                 blur_size: int = 5,
                 dilate_iterations: int = 1,
                 max_motion_ratio: float = 0.55,
                 use_opencv: bool = True) -> None:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1].")
        self.threshold = int(threshold)
        self.min_area = int(min_area)
        self.learning_rate = float(learning_rate)
        self.blur_size = int(blur_size)
        self.dilate_iterations = int(dilate_iterations)
        self.max_motion_ratio = float(max_motion_ratio)
        self.use_opencv = bool(use_opencv)
        self.background: Optional[np.ndarray] = None
        self.last_result: Optional[MotionResult] = None
        self.frame_count = 0

    def reset(self) -> None:
        """ Forget the background model. """
        self.background = None
        self.last_result = None
        self.frame_count = 0

    def update(self, image: np.ndarray, timestamp: Optional[float] = None) -> MotionResult:
        """ Process a frame and return detected motion regions. """
        gray = as_gray(image).astype(np.float32)
        if self.blur_size > 1:
            gray = gaussian_blur(gray.astype(np.uint8), self.blur_size).astype(np.float32)

        if self.background is None or self.background.shape != gray.shape:
            self.background = gray.copy()
            self.frame_count = 1
            result = MotionResult(timestamp=timestamp)
            self.last_result = result
            return result

        difference = np.abs(gray - self.background)
        changed_mask = difference >= self.threshold
        ratio = float(changed_mask.mean())

        global_change = ratio > self.max_motion_ratio
        if global_change:
            # Adapt quickly to the new lighting level and do not report every
            # pixel in the frame as an object.
            self.background = (1.0 - 0.5) * self.background + 0.5 * gray
            result = MotionResult(mask=changed_mask, score=ratio, global_change=True,
                                  timestamp=timestamp)
            self.last_result = result
            self.frame_count += 1
            return result

        # Morphological cleanup; one dilation joins small fragments.
        mask = changed_mask
        if self.dilate_iterations > 0:
            mask = dilate(mask, iterations=self.dilate_iterations)

        detections: List[Detection] = []
        for component in connected_components(mask, min_area=self.min_area):
            region = difference[component.y:component.y + component.height,
                                component.x:component.x + component.width]
            strength = float(np.clip(region.mean() / 80.0, 0.0, 1.0))
            area_score = float(np.clip(component.area / 4000.0, 0.0, 1.0))
            confidence = float(np.clip(0.35 + 0.45 * strength + 0.20 * area_score, 0.0, 1.0))
            bbox = BoundingBox(component.x, component.y,
                               component.x + component.width, component.y + component.height)
            detections.append(Detection(
                kind="motion",
                label="motion",
                confidence=confidence,
                bbox=bbox,
                attributes={"area": component.area, "strength": strength},
                timestamp=timestamp,
                mask=component.mask,
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        result = MotionResult(detections=detections, mask=mask, score=ratio,
                              changed=bool(detections), timestamp=timestamp)
        self.background = (1.0 - self.learning_rate) * self.background + self.learning_rate * gray
        self.last_result = result
        self.frame_count += 1
        return result


class MotionTracker:
    """ Small helper that requires motion to persist across several frames.

    This suppresses one-frame sensor noise while still reporting real motion
    quickly.  It uses a simple overlap/nearest-region match.
    """

    def __init__(self, detector: Optional[MotionDetector] = None,
                 min_frames: int = 2, max_missed: int = 3,
                 iou_threshold: float = 0.05) -> None:
        self.detector = detector or MotionDetector()
        self.min_frames = max(1, int(min_frames))
        self.max_missed = max(0, int(max_missed))
        self.iou_threshold = float(iou_threshold)
        self._tracks = []  # type: ignore

    def reset(self) -> None:
        self.detector.reset()
        self._tracks = []

    def update(self, image: np.ndarray, timestamp: Optional[float] = None) -> MotionResult:
        result = self.detector.update(image, timestamp=timestamp)
        active = []
        unmatched = list(result.detections)

        for track in self._tracks:
            best = None
            best_iou = 0.0
            for detection in unmatched:
                iou = track["bbox"].iou(detection.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best = detection
            if best is not None and best_iou >= self.iou_threshold:
                track["bbox"] = best.bbox
                track["hits"] += 1
                track["misses"] = 0
                track["detection"] = best
                unmatched.remove(best)
                if track["hits"] >= self.min_frames:
                    active.append(best)
            else:
                track["misses"] += 1
                if track["misses"] <= self.max_missed:
                    active.append(track["detection"])

        for detection in unmatched:
            self._tracks.append({"bbox": detection.bbox, "hits": 1, "misses": 0,
                                 "detection": detection})
            if self.min_frames <= 1:
                active.append(detection)

        self._tracks = [track for track in self._tracks
                        if track["misses"] <= self.max_missed]
        active.sort(key=lambda d: d.confidence, reverse=True)
        return MotionResult(detections=active, mask=result.mask, score=result.score,
                            changed=bool(active), global_change=result.global_change,
                            timestamp=result.timestamp)


def detect_motion(previous: np.ndarray, current: np.ndarray,
                  threshold: int = 25, min_area: int = 150,
                  blur_size: int = 5) -> MotionResult:
    """ One-shot frame differencing without a persistent background. """
    previous_gray = as_gray(previous).astype(np.float32)
    current_gray = as_gray(current).astype(np.float32)
    if previous_gray.shape != current_gray.shape:
        raise ValueError("Both frames must have the same dimensions.")
    if blur_size > 1:
        previous_gray = gaussian_blur(previous_gray.astype(np.uint8), blur_size).astype(np.float32)
        current_gray = gaussian_blur(current_gray.astype(np.uint8), blur_size).astype(np.float32)
    difference = np.abs(current_gray - previous_gray)
    mask = difference >= threshold
    ratio = float(mask.mean())
    detections: List[Detection] = []
    if ratio <= 0.55:
        mask = dilate(mask, iterations=1)
        for component in connected_components(mask, min_area=min_area):
            confidence = float(np.clip(0.35 + difference[
                component.y:component.y + component.height,
                component.x:component.x + component.width].mean() / 80.0, 0.0, 1.0))
            detections.append(Detection(
                kind="motion", label="motion", confidence=confidence,
                bbox=BoundingBox(component.x, component.y,
                                 component.x + component.width, component.y + component.height),
                attributes={"area": component.area}, timestamp=time.time()))
    return MotionResult(detections=detections, mask=mask, score=ratio,
                        changed=bool(detections), timestamp=time.time())
