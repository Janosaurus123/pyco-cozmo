"""

Cozmo Power Cube detection.

The three Power Cubes each have six distinct black-and-white face glyphs.
Templates are extracted from Anki's publicly available cube textures and
shipped as a single montage in ``pycozmo/assets/cube_faces.png``.  Detection
uses normalized cross-correlation over a small scale and rotation pyramid.

"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

from .types import BoundingBox, Detection
from .utils import as_gray, as_rgb, cv2, nms


__all__ = [
    "CubeFaceTemplate",
    "CubeFaceDetector",
    "detect_cubes",
    "load_cube_face_templates",
]


ASSET_FILENAME = "cube_faces.png"
TEMPLATE_SIZE = 101
TEMPLATE_COLUMNS = 6
TEMPLATE_ROWS = 3
#: Cube IDs corresponding to the three template rows.
CUBE_IDS = (1, 2, 3)
#: Face indices corresponding to the six template columns.
FACE_INDICES = (0, 1, 2, 3, 4, 5)


@dataclass
class CubeFaceTemplate:
    """ A single cube face template. """

    cube_id: int
    face: int
    image: np.ndarray

    @property
    def label(self) -> str:
        return "cube{}_face{}".format(self.cube_id, self.face)


def _asset_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", ASSET_FILENAME)


def load_cube_face_templates() -> List[CubeFaceTemplate]:
    """ Load the bundled cube face templates. """
    path = _asset_path()
    if not os.path.exists(path):
        return []
    montage = np.asarray(Image.open(path).convert("L"))
    templates: List[CubeFaceTemplate] = []
    for row, cube_id in enumerate(CUBE_IDS):
        for column, face in enumerate(FACE_INDICES):
            x = column * TEMPLATE_SIZE
            y = row * TEMPLATE_SIZE
            image = montage[y:y + TEMPLATE_SIZE, x:x + TEMPLATE_SIZE]
            if image.shape != (TEMPLATE_SIZE, TEMPLATE_SIZE):
                continue
            templates.append(CubeFaceTemplate(cube_id, face, image.copy()))
    return templates


class CubeFaceDetector:
    """ Detect and identify Power Cube faces.

    Args:
        min_confidence: Correlation threshold in the range ``0..1``.
        scales: Template scales relative to the bundled 101 pixel template.
        rotations: Template rotations in degrees.
        use_opencv: Disable to keep the detector dormant when OpenCV is not
            available.  Template correlation itself needs OpenCV.
    """

    def __init__(self,
                 min_confidence: float = 0.52,
                 scales: Tuple[float, ...] = (0.45, 0.60, 0.75, 0.90, 1.10),
                 rotations: Tuple[int, ...] = (0, 90, 180, 270),
                 use_opencv: bool = True,
                 max_image_width: int = 640) -> None:
        self.min_confidence = float(min_confidence)
        self.scales = tuple(float(value) for value in scales)
        self.rotations = tuple(int(value) for value in rotations)
        self.enabled = cv2 is not None and use_opencv
        self.max_image_width = int(max_image_width)
        self.templates = load_cube_face_templates() if self.enabled else []
        self._cache = {}  # type: ignore

    def _prepared(self, scale: float, rotation: int) -> List[Tuple[CubeFaceTemplate, np.ndarray]]:
        key = (scale, rotation)
        if key not in self._cache:
            prepared = []
            for template in self.templates:
                size = max(12, int(round(TEMPLATE_SIZE * scale)))
                image = cv2.resize(template.image, (size, size), interpolation=cv2.INTER_AREA)
                if rotation:
                    if rotation == 90:
                        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    elif rotation == 180:
                        image = cv2.rotate(image, cv2.ROTATE_180)
                    elif rotation == 270:
                        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                prepared.append((template, image))
            self._cache[key] = prepared
        return self._cache[key]

    def detect(self, image: np.ndarray) -> List[Detection]:
        if not self.enabled or not self.templates:
            return []
        rgb = as_rgb(image)
        if rgb.shape[1] > self.max_image_width:
            scale = self.max_image_width / float(rgb.shape[1])
            size = (self.max_image_width, max(1, int(round(rgb.shape[0] * scale))))
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        gray = as_gray(rgb)
        detections: List[Detection] = []
        image_height, image_width = gray.shape

        for scale in self.scales:
            template_size = int(round(TEMPLATE_SIZE * scale))
            if template_size < 16 or template_size > min(image_width, image_height):
                continue
            for rotation in self.rotations:
                for template, prepared in self._prepared(scale, rotation):
                    if prepared.shape[0] >= gray.shape[0] or prepared.shape[1] >= gray.shape[1]:
                        continue
                    result = cv2.matchTemplate(gray, prepared, cv2.TM_CCOEFF_NORMED)
                    min_value, max_value, min_location, max_location = cv2.minMaxLoc(result)
                    confidence = float(max_value)
                    if confidence < self.min_confidence:
                        continue
                    x, y = max_location
                    bbox = BoundingBox(x, y, x + prepared.shape[1], y + prepared.shape[0])
                    detections.append(Detection(
                        kind="cube",
                        label="cube{}".format(template.cube_id),
                        confidence=confidence,
                        bbox=bbox,
                        attributes={
                            "cube_id": template.cube_id,
                            "face": template.face,
                            "template": template.label,
                            "rotation": rotation,
                            "scale": scale,
                        },
                    ))

        # The same physical face is usually found at several neighbouring
        # scales.  Keep the strongest detection at each location.
        return nms(detections, iou_threshold=0.35)


def detect_cubes(image: np.ndarray, **kwargs) -> List[Detection]:
    """ Convenience wrapper around :class:`CubeFaceDetector`. """
    return CubeFaceDetector(**kwargs).detect(image)
