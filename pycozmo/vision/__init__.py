"""

Off-board computer vision for Cozmo and Vector.

The package is designed to work with plain NumPy arrays and Pillow images.
OpenCV is optional and enables the more advanced detectors (custom markers,
Power Cube templates and ArUco markers).

"""

from .types import BoundingBox, Detection, Pose2D
from .utils import CV2_AVAILABLE
from .calibration import (CameraModel, CalibrationResult, apply_homography,
                          calibrate_camera, estimate_homography,
                          find_chessboard_corners)
from .motion import MotionDetector, MotionResult, MotionTracker, detect_motion
from .markers import (CUSTOM_MARKER_TYPES, ArucoDetector, CustomMarkerDetector,
                      CustomMarkerType, detect_custom_markers,
                      render_custom_marker)
from .cube import (CubeFaceDetector, CubeFaceTemplate, detect_cubes,
                   load_cube_face_templates)
from .tracking import FaceTracker, ObjectTracker, Track
from .faces import (ExpressionEstimator, Face, FaceDetector, FaceRecognizer,
                    align_face, detect_faces)
from .ground import (DrivableArea, DrivableAreaEstimator, GroundContact,
                     estimate_drivable_area)
from .pets import Pet, PetDetector, detect_pets
from .platforms import (ChargerDetector, Platform, PlatformDetector,
                        detect_platforms)
from .robots import Robot, RobotDetector, detect_robots
from .pipeline import VisionConfig, VisionPipeline, VisionResult
from .mapping import AStarPlanner, MapBuilder, OccupancyGrid, plan_path


__all__ = [
    "BoundingBox",
    "Detection",
    "Pose2D",
    "CV2_AVAILABLE",

    "CameraModel",
    "CalibrationResult",
    "calibrate_camera",
    "find_chessboard_corners",
    "estimate_homography",
    "apply_homography",

    "MotionResult",
    "MotionDetector",
    "MotionTracker",
    "detect_motion",

    "CustomMarkerType",
    "CUSTOM_MARKER_TYPES",
    "CustomMarkerDetector",
    "ArucoDetector",
    "detect_custom_markers",
    "render_custom_marker",

    "CubeFaceTemplate",
    "CubeFaceDetector",
    "detect_cubes",
    "load_cube_face_templates",

    "Track",
    "ObjectTracker",
    "FaceTracker",

    "Face",
    "FaceDetector",
    "ExpressionEstimator",
    "FaceRecognizer",
    "align_face",
    "detect_faces",

    "GroundContact",
    "DrivableArea",
    "DrivableAreaEstimator",
    "estimate_drivable_area",

    "Pet",
    "PetDetector",
    "detect_pets",

    "Robot",
    "RobotDetector",
    "detect_robots",

    "Platform",
    "PlatformDetector",
    "ChargerDetector",
    "detect_platforms",

    "VisionConfig",
    "VisionResult",
    "VisionPipeline",

    "OccupancyGrid",
    "MapBuilder",
    "AStarPlanner",
    "plan_path",
]
