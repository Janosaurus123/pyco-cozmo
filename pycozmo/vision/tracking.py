"""

Simple multi-object tracking.

The tracker associates detections between frames using bounding-box overlap
and centre distance.  It is intentionally dependency-free and works well for
the small number of objects visible in a Cozmo camera frame.

"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import List, Optional, Sequence

from .types import Detection


__all__ = [
    "Track",
    "ObjectTracker",
    "FaceTracker",
]


@dataclass
class Track:
    """ State of one tracked object. """

    track_id: int
    detection: Detection
    age: int = 1
    hits: int = 1
    misses: int = 0
    last_seen: float = field(default_factory=time.time)

    @property
    def bbox(self):
        return self.detection.bbox

    @property
    def label(self) -> str:
        return self.detection.label

    @property
    def confidence(self) -> float:
        return self.detection.confidence

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 2


class ObjectTracker:
    """ Greedy IoU-based tracker.

    Args:
        max_age: Number of consecutive misses before a track is removed.
        min_hits: Number of hits before a track is considered confirmed.
        iou_threshold: Minimum IoU used to associate a detection.
        max_distance: Optional maximum centre distance in pixels.
        kind: Optional detection kind filter (``"face"`` for example).
    """

    def __init__(self, max_age: int = 8, min_hits: int = 2,
                 iou_threshold: float = 0.15, max_distance: Optional[float] = None,
                 kind: Optional[str] = None) -> None:
        self.max_age = max(0, int(max_age))
        self.min_hits = max(1, int(min_hits))
        self.iou_threshold = float(iou_threshold)
        self.max_distance = None if max_distance is None else float(max_distance)
        self.kind = kind
        self.tracks: List[Track] = []
        self._next_id = 1
        self.frame_count = 0

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 1
        self.frame_count = 0

    def update(self, detections: Sequence[Detection],
               timestamp: Optional[float] = None) -> List[Detection]:
        """ Associate detections and return detections with ``track_id`` set. """
        if self.kind is not None:
            detections = [d for d in detections if d.kind_name == self.kind]
        self.frame_count += 1
        unmatched = list(detections)
        used_tracks = set()

        for track in sorted(self.tracks, key=lambda item: item.last_seen):
            best = None
            best_score = -1.0
            for detection in unmatched:
                iou = track.bbox.iou(detection.bbox)
                if self.max_distance is not None:
                    distance = math.hypot(track.bbox.center_x - detection.bbox.center_x,
                                          track.bbox.center_y - detection.bbox.center_y)
                    if distance > self.max_distance:
                        continue
                else:
                    distance = None
                if iou < self.iou_threshold:
                    continue
                score = iou if distance is None else iou + 0.001 * self.max_distance - distance * 0.001
                if score > best_score:
                    best_score = score
                    best = detection
            if best is not None:
                self._update_track(track, best, timestamp)
                unmatched.remove(best)
                used_tracks.add(track.track_id)

        for detection in unmatched:
            self._start_track(detection, timestamp)

        self.tracks = [track for track in self.tracks if track.misses <= self.max_age]
        for track in self.tracks:
            track.age += 1
        return [d for d in detections if d.track_id is not None]

    def _update_track(self, track: Track, detection: Detection, timestamp: Optional[float]) -> None:
        track.detection = detection
        track.hits += 1
        track.misses = 0
        track.last_seen = timestamp if timestamp is not None else time.time()
        detection.track_id = track.track_id

    def _start_track(self, detection: Detection, timestamp: Optional[float]) -> None:
        detection.track_id = self._next_id
        self.tracks.append(Track(self._next_id, detection, last_seen=timestamp or time.time()))
        self._next_id += 1


class FaceTracker(ObjectTracker):
    """ Convenience tracker that only associates face detections. """

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("kind", "face")
        kwargs.setdefault("max_age", 10)
        kwargs.setdefault("min_hits", 1)
        super().__init__(**kwargs)
