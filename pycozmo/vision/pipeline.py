"""

Composable vision pipeline.

The pipeline runs the individual detectors in a sensible order and shares the
camera model, trackers and optional face recogniser between them.  Heavy
detectors are only instantiated when they are enabled in :class:`VisionConfig`.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image, ImageDraw

from .calibration import CameraModel
from .cube import CubeFaceDetector
from .faces import ExpressionEstimator, Face, FaceDetector, FaceRecognizer
from .ground import DrivableArea, DrivableAreaEstimator
from .markers import ArucoDetector, CustomMarkerDetector
from .motion import MotionTracker
from .pets import Pet, PetDetector
from .platforms import Platform, PlatformDetector
from .robots import Robot, RobotDetector
from .tracking import FaceTracker, ObjectTracker
from .types import Detection
from .utils import as_rgb


__all__ = [
    "VisionConfig",
    "VisionResult",
    "VisionPipeline",
]


@dataclass
class VisionConfig:
    """ Select which detectors run in a :class:`VisionPipeline`. """

    motion: bool = False
    faces: bool = True
    expressions: bool = True
    recognition: bool = False
    markers: bool = False
    aruco: bool = False
    cubes: bool = False
    pets: bool = False
    robots: bool = False
    platforms: bool = False
    ground: bool = False
    track: bool = True
    motion_min_frames: int = 2


@dataclass
class VisionResult:
    """ Output of one :meth:`VisionPipeline.process` call. """

    detections: List[Detection] = field(default_factory=list)
    faces: List[Face] = field(default_factory=list)
    pets: List[Pet] = field(default_factory=list)
    robots: List[Robot] = field(default_factory=list)
    platforms: List[Platform] = field(default_factory=list)
    cubes: List[Detection] = field(default_factory=list)
    markers: List[Detection] = field(default_factory=list)
    aruco: List[Detection] = field(default_factory=list)
    motion: List[Detection] = field(default_factory=list)
    ground: Optional[DrivableArea] = None
    timestamp: Optional[float] = None

    def by_kind(self, kind: str) -> List[Detection]:
        """ Return all object detections with a given kind prefix. """
        return [detection for detection in self.detections
                if detection.kind_name == kind]

    def to_dict(self) -> dict:
        data = {
            "timestamp": self.timestamp,
            "detections": [detection.to_dict() for detection in self.detections],
            "faces": [face.to_dict() for face in self.faces],
            "pets": [pet.to_dict() for pet in self.pets],
            "robots": [robot.to_dict() for robot in self.robots],
            "platforms": [platform.to_dict() for platform in self.platforms],
            "cubes": [cube.to_dict() for cube in self.cubes],
            "markers": [marker.to_dict() for marker in self.markers],
            "aruco": [marker.to_dict() for marker in self.aruco],
            "motion": [motion.to_dict() for motion in self.motion],
        }
        if self.ground is not None:
            data["ground"] = self.ground.to_dict()
        return data

    def annotate(self, image: Union[np.ndarray, Image.Image],
                 show_ground: bool = True) -> Image.Image:
        """ Draw detections on a copy of ``image``. """
        if show_ground and self.ground is not None:
            base = self.ground.annotate(image).convert("RGBA")
        else:
            base = Image.fromarray(as_rgb(image)).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        palette = {
            "face": (255, 220, 60, 255),
            "pet": (255, 150, 60, 255),
            "cozmo": (80, 220, 255, 255),
            "vector": (80, 220, 255, 255),
            "marker": (230, 80, 230, 255),
            "aruco": (230, 80, 230, 255),
            "cube": (120, 255, 120, 255),
            "platform": (100, 150, 255, 255),
            "motion": (255, 255, 255, 220),
        }
        for detection in self.detections:
            color = palette.get(detection.kind_name, (255, 80, 80, 255))
            x1, y1, x2, y2 = detection.bbox.to_int()
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label = detection.label or detection.kind
            if isinstance(detection, Face):
                if detection.identity:
                    label = "{} ({})".format(detection.identity, detection.expression)
                elif detection.expression != "neutral":
                    label = detection.expression
            elif isinstance(detection, Pet):
                label = "{} {}".format(detection.species, detection.fur_color)
            elif isinstance(detection, Robot):
                label = "{} {}".format(detection.kind, detection.screen_color)
            label = "{} {:.2f}".format(label, detection.confidence)
            text_y = max(0, y1 - 11)
            draw.rectangle([x1, text_y, x1 + 8 * len(label), text_y + 11],
                           fill=(0, 0, 0, 160))
            draw.text((x1 + 2, text_y), label, fill=color)
        return Image.alpha_composite(base, overlay).convert("RGB")


class VisionPipeline:
    """ Run several detectors on each frame.

    Args:
        camera: Default camera model.  It can be overridden per frame.
        config: Detector selection.
        face_detector: Optional pre-configured :class:`FaceDetector`.
        expression_estimator: Optional expression estimator.
        recognizer: Optional fitted :class:`FaceRecognizer`, required when
            ``config.recognition`` is enabled.
        detectors: Optional mapping from detector name to an existing
            detector instance (``"pet"``, ``"robot"``, ``"cube"``,
            ``"marker"``, ``"aruco"``, ``"ground"``).
    """

    def __init__(self,
                 camera: Optional[CameraModel] = None,
                 config: Optional[VisionConfig] = None,
                 face_detector: Optional[FaceDetector] = None,
                 expression_estimator: Optional[ExpressionEstimator] = None,
                 recognizer: Optional[FaceRecognizer] = None,
                 detectors: Optional[Dict[str, object]] = None) -> None:
        self.camera = camera
        self.config = config or VisionConfig()
        detectors = dict(detectors or {})
        self.face_detector = face_detector or FaceDetector()
        self.expression_estimator = expression_estimator or ExpressionEstimator()
        self.recognizer = recognizer
        self.ground_estimator = detectors.get("ground") or DrivableAreaEstimator()
        self.pet_detector = detectors.get("pet") or PetDetector()
        self.robot_detector = detectors.get("robot") or RobotDetector()
        self.platform_detector = detectors.get("platform") or PlatformDetector()
        self.cube_detector = detectors.get("cube") or CubeFaceDetector()
        self.marker_detector = detectors.get("marker") or CustomMarkerDetector()
        self.aruco_detector = detectors.get("aruco") or ArucoDetector()
        self.motion_tracker = (detectors.get("motion") or
                               MotionTracker(min_frames=self.config.motion_min_frames))
        self.face_tracker = FaceTracker()
        self.object_tracker = ObjectTracker(max_age=6, min_hits=1)
        self.frame_count = 0

    def reset(self) -> None:
        """ Reset all temporal state (motion background and trackers). """
        self.motion_tracker.reset()
        self.face_tracker.reset()
        self.object_tracker.reset()
        self.frame_count = 0

    def process(self, image: Union[np.ndarray, Image.Image],
                camera: Optional[CameraModel] = None,
                timestamp: Optional[float] = None) -> VisionResult:
        rgb = as_rgb(image)
        height, width = rgb.shape[:2]
        frame_camera = camera or self.camera
        if frame_camera is None:
            frame_camera = CameraModel.cozmo(width=width, height=height)
        elif frame_camera.width != width or frame_camera.height != height:
            frame_camera = frame_camera.scaled(width, height)

        config = self.config
        result = VisionResult(timestamp=timestamp)

        if config.ground:
            result.ground = self.ground_estimator.estimate(rgb, camera=frame_camera)

        if config.motion:
            motion = self.motion_tracker.update(rgb, timestamp=timestamp)
            result.motion = list(motion.detections)

        if config.faces:
            faces = self.face_detector.detect(rgb)
            for face in faces:
                if config.expressions:
                    self.expression_estimator.estimate(rgb, face)
                if config.recognition and self.recognizer is not None:
                    identity, identity_confidence = self.recognizer.predict_with_confidence(rgb)
                    face.identity = identity
                    face.identity_confidence = identity_confidence
            if config.track and faces:
                self.face_tracker.update(faces, timestamp=timestamp)
            result.faces = faces

        if config.markers:
            result.markers = self.marker_detector.detect(rgb)
        if config.aruco:
            result.aruco = self.aruco_detector.detect(rgb)
        if config.cubes:
            result.cubes = self.cube_detector.detect(rgb)
        if config.pets:
            result.pets = self.pet_detector.detect(rgb, camera=frame_camera)
        if config.robots:
            result.robots = self.robot_detector.detect(rgb, camera=frame_camera)
        if config.platforms:
            result.platforms = self.platform_detector.detect(rgb, camera=frame_camera)

        dynamic: List[Detection] = []
        dynamic.extend(result.faces)
        dynamic.extend(result.markers)
        dynamic.extend(result.aruco)
        dynamic.extend(result.cubes)
        dynamic.extend(result.pets)
        dynamic.extend(result.robots)
        dynamic.extend(result.platforms)
        if config.track:
            other = [detection for detection in dynamic if not isinstance(detection, Face)]
            if other:
                self.object_tracker.update(other, timestamp=timestamp)
        result.detections = sorted(dynamic, key=lambda detection: detection.confidence,
                                   reverse=True)
        if result.motion:
            result.detections.extend(result.motion)
        result.detections.sort(key=lambda detection: detection.confidence, reverse=True)

        self.frame_count += 1
        return result

    def __call__(self, image: Union[np.ndarray, Image.Image],
                 camera: Optional[CameraModel] = None,
                 timestamp: Optional[float] = None) -> VisionResult:
        return self.process(image, camera=camera, timestamp=timestamp)
