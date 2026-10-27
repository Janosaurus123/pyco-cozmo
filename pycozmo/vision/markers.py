"""

Detection of Anki custom markers and ArUco markers.

Anki custom markers consist of a rounded black square border containing two
to five identical shapes (circles, diamonds, hexagons, or triangles).  The
combination of shape and count uniquely identifies one of the 16 markers
supported by the Cozmo and Vector SDKs.

"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .types import BoundingBox, Detection
from .utils import as_gray, cv2, nms


__all__ = [
    "CustomMarkerType",
    "CUSTOM_MARKER_TYPES",
    "CustomMarkerDetector",
    "ArucoDetector",
    "detect_custom_markers",
    "render_custom_marker",
]


_SHAPES = ("circles", "diamonds", "hexagons", "triangles")
_COUNTS = (2, 3, 4, 5)


@dataclass(frozen=True)
class CustomMarkerType:
    """ Description of one Anki custom marker type. """

    shape: str
    count: int
    marker_id: int

    @property
    def name(self) -> str:
        return "{}{}".format(self.shape[:-1].capitalize(), self.count)

    def __str__(self) -> str:
        return self.name


CUSTOM_MARKER_TYPES = tuple(
    CustomMarkerType(shape, count, shape_index * len(_COUNTS) + _COUNTS.index(count))
    for shape_index, shape in enumerate(_SHAPES)
    for count in _COUNTS
)
CUSTOM_MARKER_TYPES_BY_NAME = {m.name.lower(): m for m in CUSTOM_MARKER_TYPES}


def _marker_type(shape: str, count: int) -> CustomMarkerType:
    for marker_type in CUSTOM_MARKER_TYPES:
        if marker_type.shape == shape and marker_type.count == count:
            return marker_type
    raise ValueError("Unknown custom marker shape/count: {}/{}".format(shape, count))


class CustomMarkerDetector:
    """ Detect Anki custom markers in an image.

    OpenCV is required for contour hierarchy analysis.  If OpenCV is not
    installed the detector is disabled and :meth:`detect` returns an empty
    list rather than breaking the rest of the vision pipeline.
    """

    def __init__(self, min_side: float = 24.0, max_side: float = 600.0,
                 min_confidence: float = 0.35, use_opencv: bool = True) -> None:
        self.min_side = float(min_side)
        self.max_side = float(max_side)
        self.min_confidence = float(min_confidence)
        self.enabled = cv2 is not None and use_opencv

    def detect(self, image: np.ndarray) -> List[Detection]:
        if not self.enabled:
            return []
        gray = as_gray(image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or not contours:
            return []
        hierarchy = hierarchy[0]
        detections: List[Detection] = []

        for index, contour in enumerate(contours):
            if hierarchy[index][3] != -1:
                continue
            area = cv2.contourArea(contour)
            if area < self.min_side * self.min_side:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if max(width, height) > self.max_side:
                continue
            aspect = width / float(height) if height else 0.0
            if not 0.55 <= aspect <= 1.8:
                continue

            child = hierarchy[index][2]
            if child == -1:
                continue
            child_area = cv2.contourArea(contours[child])
            if child_area < 0.20 * area:
                continue
            # The inner white region must be reasonably square; this rejects
            # arbitrary rings and letter-like shapes.
            ix, iy, iw, ih = cv2.boundingRect(contours[child])
            if iw < self.min_side * 0.55 or ih < self.min_side * 0.55:
                continue
            if not 0.55 <= iw / float(ih) <= 1.8:
                continue

            shapes: List[Tuple[str, float]] = []
            shape_contour = hierarchy[child][2]
            while shape_contour != -1:
                shape = contours[shape_contour]
                contour_area = cv2.contourArea(shape)
                if contour_area >= max(12.0, 0.0015 * child_area):
                    shape_name, shape_confidence = _classify_shape(shape)
                    if shape_name is not None:
                        shapes.append((shape_name, shape_confidence))
                shape_contour = hierarchy[shape_contour][0]

            if not 2 <= len(shapes) <= 5:
                continue
            votes: Dict[str, int] = {}
            for shape_name, _ in shapes:
                votes[shape_name] = votes.get(shape_name, 0) + 1
            shape_name, vote_count = max(votes.items(), key=lambda item: item[1])
            agreement = vote_count / float(len(shapes))
            average_shape_confidence = float(np.mean([confidence for name, confidence in shapes
                                                      if name == shape_name]))
            side_score = min(1.0, max(width, height) / 100.0)
            confidence = float(np.clip(0.45 * agreement + 0.35 * average_shape_confidence +
                                       0.20 * side_score, 0.0, 1.0))
            if confidence < self.min_confidence:
                continue

            marker_type = _marker_type(shape_name, len(shapes))
            angle = 0.0
            if len(contour) >= 4:
                rect = cv2.minAreaRect(contour)
                angle = float(rect[2])
            bbox = BoundingBox(x, y, x + width, y + height)
            detections.append(Detection(
                kind="marker",
                label=marker_type.name,
                confidence=confidence,
                bbox=bbox,
                attributes={
                    "marker_id": marker_type.marker_id,
                    "marker_type": marker_type,
                    "shape": marker_type.shape,
                    "count": marker_type.count,
                    "angle": angle,
                },
            ))

        return nms(detections, iou_threshold=0.30)


def _classify_shape(contour) -> Tuple[Optional[str], float]:
    """ Classify a contour as an Anki marker shape. """
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 0.0 or perimeter <= 0.0:
        return None, 0.0
    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    vertices = len(approximation)
    if circularity >= 0.82 and vertices >= 6:
        return "circles", min(1.0, 0.55 + circularity * 0.5)
    if vertices == 3:
        return "triangles", 0.9
    if vertices == 4:
        # A diamond has low-ish circularity; a rounded square is also possible
        # but markers only use diamonds.
        return "diamonds", 0.9 if circularity < 0.82 else 0.65
    if vertices in (5, 6):
        return "hexagons", 0.85 if circularity >= 0.65 else 0.6
    if circularity >= 0.72 and vertices >= 5:
        return "circles", 0.6
    return None, 0.0


def detect_custom_markers(image: np.ndarray, **kwargs) -> List[Detection]:
    """ Convenience wrapper around :class:`CustomMarkerDetector`. """
    return CustomMarkerDetector(**kwargs).detect(image)


class ArucoDetector:
    """ Optional ArUco marker detector.

    The detector understands the OpenCV 4 and 5 APIs.  Marker corners are
    exposed in pixel coordinates; use :meth:`pycozmo.vision.CameraModel.project`
    to estimate their 3D pose when the marker size is known.
    """

    def __init__(self, dictionary: str = "DICT_4X4_50", use_opencv: bool = True) -> None:
        self.dictionary_name = dictionary
        self.enabled = cv2 is not None and use_opencv and hasattr(cv2, "aruco")
        self._detector = None
        self._parameters = None
        if self.enabled:
            try:
                dict_id = getattr(cv2.aruco, dictionary)
                self._dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
                self._parameters = cv2.aruco.DetectorParameters()
                if hasattr(cv2.aruco, "ArucoDetector"):
                    self._detector = cv2.aruco.ArucoDetector(self._dictionary, self._parameters)
            except Exception:
                self.enabled = False

    def detect(self, image: np.ndarray) -> List[Detection]:
        if not self.enabled:
            return []
        gray = as_gray(image)
        try:
            if self._detector is not None:
                corners, ids, _ = self._detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self._dictionary,
                                                          parameters=self._parameters)
        except Exception:
            return []
        if ids is None:
            return []
        detections: List[Detection] = []
        for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
            points = np.asarray(marker_corners, dtype=np.float64).reshape(-1, 2)
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            detections.append(Detection(
                kind="aruco",
                label=str(int(marker_id)),
                confidence=1.0,
                bbox=BoundingBox(x1, y1, x2, y2),
                attributes={"marker_id": int(marker_id), "corners": points.tolist()},
            ))
        return detections


# Positions and relative sizes used by the synthetic marker renderer.  They
# mirror the layout of the official Anki custom markers closely enough for
# documentation, examples and tests.
_LAYOUTS = {
    2: [((0.30, 0.62), 0.40), ((0.68, 0.30), 0.24)],
    3: [((0.30, 0.62), 0.44), ((0.70, 0.30), 0.19), ((0.74, 0.68), 0.18)],
    4: [((0.29, 0.58), 0.42), ((0.71, 0.25), 0.19), ((0.74, 0.62), 0.17),
        ((0.29, 0.26), 0.15)],
    5: [((0.29, 0.60), 0.40), ((0.70, 0.23), 0.18), ((0.74, 0.58), 0.17),
        ((0.30, 0.25), 0.15), ((0.56, 0.84), 0.13)],
}


def render_custom_marker(marker_type: CustomMarkerType, size: int = 384,
                         border: Optional[int] = None) -> Image.Image:
    """ Render a custom marker to a Pillow image.

    This is useful for printing markers, for examples and for tests.  The
    generated image has the same essential structure as the official SDK
    images: a rounded black border, a white interior and black shapes.
    """
    size = int(size)
    if size < 64:
        raise ValueError("Marker size must be at least 64 pixels.")
    border = int(border if border is not None else max(10, size // 12))
    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=max(8, size // 12),
                           fill=(0, 0, 0))
    inner_margin = border
    draw.rounded_rectangle([inner_margin, inner_margin, size - inner_margin - 1,
                            size - inner_margin - 1],
                           radius=max(4, size // 20), fill=(255, 255, 255))

    inner_x1 = inner_margin
    inner_y1 = inner_margin
    inner_size = size - 2 * inner_margin
    for (cx, cy), relative_size in _LAYOUTS[int(marker_type.count)]:
        center_x = inner_x1 + inner_size * cx
        center_y = inner_y1 + inner_size * cy
        radius = inner_size * relative_size / 2.0
        _draw_shape(draw, marker_type.shape, center_x, center_y, radius)
    return image


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, cx: float, cy: float,
                radius: float) -> None:
    if shape == "circles":
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(0, 0, 0))
        return
    if shape == "diamonds":
        points = [(cx, cy - radius), (cx + radius, cy), (cx, cy + radius),
                  (cx - radius, cy)]
        draw.polygon(points, fill=(0, 0, 0))
        return
    if shape == "triangles":
        points = [(cx, cy - radius), (cx + radius * 0.92, cy + radius * 0.75),
                  (cx - radius * 0.92, cy + radius * 0.75)]
        draw.polygon(points, fill=(0, 0, 0))
        return
    # Hexagon.
    points = [(cx + radius * math.cos(math.pi / 3.0 * i),
               cy + radius * math.sin(math.pi / 3.0 * i)) for i in range(6)]
    draw.polygon(points, fill=(0, 0, 0))
