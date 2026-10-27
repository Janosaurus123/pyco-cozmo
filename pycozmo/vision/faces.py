"""

Face detection, expression estimation, recognition and tracking.

The face detector uses OpenCV's Haar cascade when available and falls back
to a skin-colour plus connected-component detector.  Expression estimation
and Eigenface recognition are implemented with NumPy so they also work on
systems without OpenCV.

"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from .tracking import FaceTracker
from .types import BoundingBox, Detection
from .utils import (CV2_AVAILABLE, as_gray, as_rgb, cv2, connected_components,
                    dominant_color, nms, resize)


__all__ = [
    "Face",
    "FaceDetector",
    "ExpressionEstimator",
    "FaceRecognizer",
    "FaceTracker",
    "align_face",
    "detect_faces",
]


@dataclass
class Face(Detection):
    """ A detected face with optional expression and identity information. """

    expression: str = "neutral"
    smile: float = 0.0
    blink: float = 0.0
    gaze: Tuple[float, float] = (0.0, 0.0)
    mouth_open: float = 0.0
    brow_raise: float = 0.0
    eyes: List[BoundingBox] = field(default_factory=list)
    mouth_bbox: Optional[BoundingBox] = None
    identity: Optional[str] = None
    identity_confidence: float = 0.0
    skin_color: Tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind == "face" and not self.label:
            self.label = "face"

    def to_dict(self, include_mask: bool = False) -> Dict:
        data = super().to_dict(include_mask=include_mask)
        data.update({
            "expression": self.expression,
            "smile": self.smile,
            "blink": self.blink,
            "gaze": list(self.gaze),
            "mouth_open": self.mouth_open,
            "brow_raise": self.brow_raise,
            "eyes": [box.to_dict() for box in self.eyes],
            "mouth_bbox": self.mouth_bbox.to_dict() if self.mouth_bbox else None,
            "identity": self.identity,
            "identity_confidence": self.identity_confidence,
            "skin_color": list(self.skin_color),
        })
        return data


def _load_cascade():
    if cv2 is None:
        return None
    try:
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if not os.path.exists(path):
            return None
        cascade = cv2.CascadeClassifier(path)
        return None if cascade.empty() else cascade
    except Exception:
        return None


class FaceDetector:
    """ Detect faces in an image.

    Args:
        use_opencv: Use the Haar cascade if OpenCV is installed.
        min_size: Minimum face size in pixels.
        scale_factor: Haar cascade scale factor.
        min_neighbors: Haar cascade neighbour count.
        min_area: Minimum skin-component area for the fallback detector.
        skin_detection: Enable the skin-colour fallback.
    """

    def __init__(self,
                 use_opencv: bool = True,
                 min_size: Tuple[int, int] = (24, 24),
                 scale_factor: float = 1.1,
                 min_neighbors: int = 5,
                 min_area: int = 900,
                 skin_detection: bool = True) -> None:
        self.use_opencv = bool(use_opencv) and CV2_AVAILABLE
        self.min_size = (int(min_size[0]), int(min_size[1]))
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_area = int(min_area)
        self.skin_detection = bool(skin_detection)
        self._cascade = _load_cascade() if self.use_opencv else None

    def detect(self, image: Union[np.ndarray, Image.Image]) -> List[Face]:
        rgb = as_rgb(image)
        height, width = rgb.shape[:2]
        candidates: List[Face] = []

        if self._cascade is not None:
            gray = as_gray(rgb)
            try:
                boxes = self._cascade.detectMultiScale(
                    gray, scaleFactor=self.scale_factor,
                    minNeighbors=self.min_neighbors, minSize=self.min_size)
            except Exception:
                boxes = []
            for x, y, w, h in boxes:
                bbox = BoundingBox(int(x), int(y), int(x + w), int(y + h))
                candidates.append(Face(kind="face", bbox=bbox, confidence=0.92,
                                       label="face", skin_color=dominant_color(rgb, self._mask_for_box(rgb, bbox))))

        if self.skin_detection:
            candidates.extend(self._skin_candidates(rgb))

        if not candidates:
            # A last-resort detector for grayscale or strongly colour-cast
            # images.  It looks for a mid-tone, oval-ish region.
            candidates.extend(self._brightness_candidates(rgb))

        candidates = nms(candidates, iou_threshold=0.35)
        for face in candidates:
            face.bbox = face.bbox.clip(width, height)
            if not face.skin_color or face.skin_color == (0, 0, 0):
                face.skin_color = dominant_color(rgb, self._mask_for_box(rgb, face.bbox))
        return [face for face in candidates if face.bbox.width >= self.min_size[0] and
                face.bbox.height >= self.min_size[1]]

    def _skin_candidates(self, rgb: np.ndarray) -> List[Face]:
        mask = _skin_mask(rgb)
        if mask.sum() < self.min_area:
            return []
        faces = []
        for component in connected_components(mask, min_area=self.min_area):
            width, height = component.width, component.height
            if width < self.min_size[0] or height < self.min_size[1]:
                continue
            aspect = width / float(height) if height else 0.0
            if not 0.50 <= aspect <= 1.55:
                continue
            if component.extent < 0.38:
                continue
            bbox = BoundingBox(component.x, component.y,
                               component.x + width, component.y + height)
            area_score = min(1.0, component.area / 8000.0)
            extent_score = min(1.0, component.extent / 0.75)
            confidence = float(np.clip(0.45 + 0.30 * extent_score + 0.20 * area_score, 0.0, 0.95))
            local_mask = np.zeros(rgb.shape[:2], dtype=np.bool_)
            local_mask[component.y:component.y + component.height,
                       component.x:component.x + component.width] = component.mask
            faces.append(Face(kind="face", bbox=bbox, confidence=confidence, label="face",
                              skin_color=dominant_color(rgb, local_mask)))
        return faces

    def _brightness_candidates(self, rgb: np.ndarray) -> List[Face]:
        gray = as_gray(rgb)
        if gray.mean() < 5:
            return []
        threshold_value = max(35, int(np.percentile(gray, 55)))
        mask = gray >= threshold_value
        faces = []
        for component in connected_components(mask, min_area=self.min_area):
            width, height = component.width, component.height
            if width < self.min_size[0] or height < self.min_size[1]:
                continue
            aspect = width / float(height) if height else 0.0
            if not 0.55 <= aspect <= 1.45 or component.extent < 0.55:
                continue
            bbox = BoundingBox(component.x, component.y,
                               component.x + width, component.y + height)
            faces.append(Face(kind="face", bbox=bbox, confidence=0.35, label="face",
                              skin_color=dominant_color(rgb, component.mask)))
        return faces

    @staticmethod
    def _mask_for_box(rgb: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        mask = np.zeros(rgb.shape[:2], dtype=np.bool_)
        x1, y1, x2, y2 = bbox.clip(rgb.shape[1], rgb.shape[0]).to_int()
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
        return mask


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    """ Return a boolean skin-colour mask.

    Both HSV and YCrCb rules are combined to handle a range of skin tones
    and lighting conditions.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV) if cv2 is not None else None
    if hsv is None:
        from .utils import as_hsv
        hsv = as_hsv(rgb)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hsv_mask_value = (((h <= 28) | (h >= 160)) & (s >= 20) & (s <= 190) &
                      (v >= 35) & (v <= 255))

    if cv2 is not None:
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    else:
        # YCrCb conversion using the standard ITU-R BT.601 coefficients.
        r, g, b = [rgb[:, :, i].astype(np.float32) for i in range(3)]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cr = (r - y) * 0.713 + 128.0
        cb = (b - y) * 0.564 + 128.0
        ycrcb = np.stack([y, cr, cb], axis=2)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    ycc_mask = (cr >= 128) & (cr <= 185) & (cb >= 68) & (cb <= 140)

    mask = hsv_mask_value & ycc_mask
    if cv2 is not None:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(np.bool_)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(np.bool_)
    return mask


class ExpressionEstimator:
    """ Estimate simple facial expression parameters from a face crop.

    The estimator works with geometric image cues (dark regions for eyes,
    brows and mouth) and is therefore independent of any trained model.
    """

    def __init__(self, dark_ratio: float = 0.55) -> None:
        self.dark_ratio = float(dark_ratio)

    def estimate(self, image: Union[np.ndarray, Image.Image], face: Face) -> Face:
        rgb = as_rgb(image)
        gray = as_gray(rgb)
        height, width = gray.shape
        bbox = face.bbox.clip(width, height)
        x1, y1, x2, y2 = bbox.to_int()
        if x2 - x1 < 12 or y2 - y1 < 12:
            return face

        face_crop = gray[y1:y2, x1:x2]
        median = float(np.median(face_crop))
        dark_threshold = max(18.0, min(125.0, median * self.dark_ratio))
        face_width = x2 - x1
        face_height = y2 - y1

        # Eyes.
        eyes: List[BoundingBox] = []
        eye_components = []
        for x0, x1n in ((0.12, 0.47), (0.53, 0.88)):
            region, origin_x, origin_y = _crop_region_with_origin(
                gray, x1, y1, x2, y2, x0, 0.26, x1n, 0.54)
            component, component_bbox = _largest_dark_component(
                region, dark_threshold, max(4, int(0.0006 * face_width * face_height)),
                origin=(origin_x, origin_y))
            if component is not None:
                eyes.append(component_bbox)
                eye_components.append((component, component_bbox, region.shape, (origin_x, origin_y)))
            else:
                eyes.append(BoundingBox(0, 0, 0, 0))

        eye_ratio = 0.0
        if eye_components:
            ratios = []
            for component, _, _, _ in eye_components:
                if component.width > 0:
                    ratios.append(component.height / float(component.width))
            eye_ratio = float(np.median(ratios)) if ratios else 0.0
        blink = float(np.clip((0.34 - eye_ratio) / 0.20, 0.0, 1.0))

        # Gaze is estimated from the darkest pixels inside each eye
        # component (the pupil) relative to the centre of that component.
        # Using the component mask rather than its bounding box is important:
        # the box often contains light skin in its corners.
        gaze_x_values = []
        gaze_y_values = []
        for component, component_bbox, region_shape, origin in eye_components:
            region_width = region_shape[1]
            ex1 = int(origin[0] + component.x)
            ey1 = int(origin[1] + component.y)
            crop = gray[ey1:ey1 + component.height, ex1:ex1 + component.width]
            if crop.size != component.mask.size:
                continue
            darkness = crop[component.mask].astype(np.float32)
            if darkness.size < 2:
                continue
            minimum = float(darkness.min())
            maximum = float(darkness.max())
            threshold_value = minimum + max(4.0, 0.15 * (maximum - minimum))
            local_ys, local_xs = np.nonzero(component.mask &
                                            (crop <= threshold_value))
            if len(local_xs) < 2 or component.width <= 0 or component.height <= 0:
                continue
            pupil_x = float(local_xs.mean()) + component_bbox.x
            pupil_y = float(local_ys.mean()) + component_bbox.y
            # A small component is probably just the pupil, not the whole
            # eye.  In that case the eye-region centre is a better reference
            # for the horizontal gaze direction.
            if component.width < 0.35 * region_width:
                reference_x = origin[0] + region_width * 0.5
            else:
                reference_x = component_bbox.center_x
            reference_y = component_bbox.center_y
            gaze_x_values.append(float(np.clip(
                (pupil_x - reference_x) / max(1.0, component_bbox.width * 0.30),
                -1.0, 1.0)))
            gaze_y_values.append(float(np.clip(
                (pupil_y - reference_y) / max(1.0, component_bbox.height * 0.30),
                -1.0, 1.0)))
        gaze_x = float(np.median(gaze_x_values)) if gaze_x_values else 0.0
        gaze_y = float(np.median(gaze_y_values)) if gaze_y_values else 0.0
        if blink > 0.65:
            # Closed eyes make the pupil position meaningless.
            gaze_x = 0.0
            gaze_y = 0.0

        # Mouth.
        mouth_region, mouth_origin_x, mouth_origin_y = _crop_region_with_origin(
            gray, x1, y1, x2, y2, 0.20, 0.57, 0.80, 0.96)
        mouth_component, mouth_bbox = _largest_dark_component(
            mouth_region, dark_threshold, int(0.003 * face_width * face_height),
            origin=(mouth_origin_x, mouth_origin_y))
        smile = 0.0
        mouth_open = 0.0
        if mouth_component is not None and mouth_component.width > 2 and mouth_component.height > 1:
            # Mean darkness thickness per column separates a thin, curved
            # mouth line from a genuinely open mouth.  The bounding-box
            # height alone would mistake a strong smile for an open mouth.
            mean_thickness = float(mouth_component.area) / max(1.0, float(mouth_component.width))
            mouth_open = float(np.clip((mean_thickness / float(mouth_component.width) - 0.08) / 0.35,
                                       0.0, 1.0))
            ys, xs = np.nonzero(mouth_component.mask)
            if len(xs) > 4:
                width_norm = xs.astype(np.float32) / max(1.0, mouth_component.width - 1)
                edge = (width_norm <= 0.28) | (width_norm >= 0.72)
                centre = (width_norm >= 0.38) & (width_norm <= 0.62)
                if edge.any() and centre.any():
                    curve = (float(np.median(ys[centre])) - float(np.median(ys[edge]))) / \
                        max(1.0, mouth_component.height)
                    smile = float(np.clip(curve * 3.0, -1.0, 1.0))
            if mouth_open > 0.60:
                # For a wide open mouth the curvature of the dark blob is
                # dominated by its outline rather than a smile.
                smile = 0.0

        # Brows.
        brow_region, brow_origin_x, brow_origin_y = _crop_region_with_origin(
            gray, x1, y1, x2, y2, 0.12, 0.10, 0.88, 0.31)
        brow_component, _ = _largest_dark_component(
            brow_region, dark_threshold, int(0.002 * face_width * face_height),
            origin=(brow_origin_x, brow_origin_y))
        brow_raise = 0.0
        if brow_component is not None and brow_component.height > 0:
            brow_y = (brow_origin_y + brow_component.centroid[1] - y1) / max(1.0, face_height)
            brow_raise = float(np.clip((0.225 - brow_y) / 0.11, -1.0, 1.0))

        if smile > 0.35:
            expression = "happy"
        elif smile < -0.35:
            expression = "sad"
        elif mouth_open > 0.45:
            expression = "surprised"
        elif brow_raise < -0.35:
            expression = "angry"
        elif blink > 0.65:
            expression = "blinking"
        else:
            expression = "neutral"

        face.expression = expression
        face.smile = smile
        face.blink = blink
        face.gaze = (gaze_x, gaze_y)
        face.mouth_open = mouth_open
        face.brow_raise = brow_raise
        face.eyes = eyes
        face.mouth_bbox = mouth_bbox
        detection_score = (0.45 + 0.25 * (1.0 if eye_components else 0.0) +
                           0.20 * (1.0 if mouth_component is not None else 0.0))
        face.confidence = float(np.clip(face.confidence * detection_score + 0.05, 0.0, 1.0))
        return face


def _crop_region_with_origin(gray: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                             rx1: float, ry1: float, rx2: float, ry2: float):
    width = x2 - x1
    height = y2 - y1
    cx1 = x1 + int(round(width * rx1))
    cx2 = x1 + int(round(width * rx2))
    cy1 = y1 + int(round(height * ry1))
    cy2 = y1 + int(round(height * ry2))
    return gray[cy1:cy2, cx1:cx2], cx1, cy1


def _crop_region(gray: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                 rx1: float, ry1: float, rx2: float, ry2: float) -> np.ndarray:
    return _crop_region_with_origin(gray, x1, y1, x2, y2, rx1, ry1, rx2, ry2)[0]


def _largest_dark_component(region: np.ndarray, threshold: float, min_area: int,
                            origin: Tuple[int, int] = (0, 0)
                            ) -> Tuple[Optional[object], Optional[BoundingBox]]:
    if region.size == 0:
        return None, None
    mask = region <= threshold
    components = connected_components(mask, min_area=max(2, min_area))
    if not components:
        return None, None
    # Reject components that fill almost the entire region (usually a dark
    # background selected by mistake) unless that is all we have.
    components = [c for c in components if c.extent < 0.92] or components
    component = max(components, key=lambda c: c.area)
    origin_x, origin_y = origin
    bbox = BoundingBox(origin_x + component.x, origin_y + component.y,
                       origin_x + component.x + component.width,
                       origin_y + component.y + component.height)
    return component, bbox


def align_face(image: Union[np.ndarray, Image.Image], face: Union[Face, BoundingBox],
               size: Tuple[int, int] = (64, 64), margin: float = 0.08) -> np.ndarray:
    """ Crop and resize a face to a fixed size. """
    bbox = face.bbox if isinstance(face, Detection) else face
    rgb = as_rgb(image)
    height, width = rgb.shape[:2]
    bbox = bbox.clip(width, height)
    amount_x = bbox.width * margin
    amount_y = bbox.height * margin
    box = BoundingBox(bbox.x1 - amount_x, bbox.y1 - amount_y,
                      bbox.x2 + amount_x, bbox.y2 + amount_y).clip(width, height)
    x1, y1, x2, y2 = box.to_int()
    if x2 <= x1 or y2 <= y1:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    return resize(rgb[y1:y2, x1:x2], size, interpolation="area")


class FaceRecognizer:
    """ A small Eigenfaces recognizer.

    Args:
        n_components: Number of principal components to retain.
        threshold: Maximum distance to the nearest class centre before a
            face is reported as unknown.  The distance is measured on
            unit-length descriptors and includes the reconstruction error,
            so values are typically below ``0.5``.
        descriptor_size: Size of the aligned face patch used as descriptor.
    """

    def __init__(self, n_components: int = 20, threshold: float = 0.35,
                 descriptor_size: Tuple[int, int] = (64, 64)) -> None:
        self.n_components = int(n_components)
        self.threshold = float(threshold)
        self.descriptor_size = (int(descriptor_size[0]), int(descriptor_size[1]))
        self.mean: Optional[np.ndarray] = None
        self.components: Optional[np.ndarray] = None
        self.class_names: List[str] = []
        self.class_means: Optional[np.ndarray] = None
        self.detector = FaceDetector(use_opencv=False)

    def _descriptor(self, image: Union[np.ndarray, Image.Image, Face],
                    source: Optional[Union[np.ndarray, Image.Image]] = None) -> np.ndarray:
        if isinstance(image, Face) and source is not None:
            patch = align_face(source, image, self.descriptor_size)
        elif isinstance(image, Face):
            raise ValueError("A source image is required when passing a Face.")
        else:
            faces = self.detector.detect(image)
            if faces:
                patch = align_face(image, faces[0], self.descriptor_size)
            else:
                rgb = as_rgb(image)
                height, width = rgb.shape[:2]
                side = min(width, height)
                x = (width - side) // 2
                y = (height - side) // 2
                patch = resize(rgb[y:y + side, x:x + side], self.descriptor_size,
                               interpolation="area")
        gray = as_gray(patch).astype(np.float32)
        if cv2 is not None:
            gray = cv2.equalizeHist(gray.astype(np.uint8)).astype(np.float32)
        else:
            gray = (gray - gray.mean()) / max(1.0, gray.std())
        descriptor = gray.reshape(-1)
        descriptor = descriptor - descriptor.mean()
        descriptor /= max(1e-6, np.linalg.norm(descriptor))
        return descriptor

    def fit(self, images: Sequence[Union[np.ndarray, Image.Image]],
            labels: Sequence[str]) -> "FaceRecognizer":
        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length.")
        if not images:
            raise ValueError("At least one training image is required.")
        descriptors = np.stack([self._descriptor(image) for image in images])
        self.mean = descriptors.mean(axis=0)
        centered = descriptors - self.mean
        maximum = max(1, min(self.n_components, len(descriptors) - 1, centered.shape[1]))
        if len(descriptors) == 1:
            self.components = np.zeros((1, centered.shape[1]), dtype=np.float64)
            projections = np.zeros((1, 1), dtype=np.float64)
        else:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            self.components = vh[:maximum].copy()
            projections = centered @ self.components.T
        self.class_names = []
        class_means = []
        for label in labels:
            label = str(label)
            if label not in self.class_names:
                self.class_names.append(label)
                class_means.append([])
        class_means = {name: [] for name in self.class_names}
        for projection, label in zip(projections, labels):
            class_means[str(label)].append(projection)
        self.class_means = np.stack([np.mean(class_means[name], axis=0)
                                     for name in self.class_names])
        return self

    def _project(self, descriptor: np.ndarray) -> np.ndarray:
        if self.mean is None or self.components is None:
            raise RuntimeError("FaceRecognizer is not fitted.")
        return (descriptor - self.mean) @ self.components.T

    def predict_with_confidence(self, image: Union[np.ndarray, Image.Image]) -> Tuple[Optional[str], float]:
        if self.class_means is None:
            raise RuntimeError("FaceRecognizer is not fitted.")
        descriptor = self._descriptor(image)
        projection = self._project(descriptor)
        distances = np.linalg.norm(self.class_means - projection, axis=1)
        index = int(np.argmin(distances))
        distance = float(distances[index])
        # Faces that do not lie close to the learned face subspace are
        # likely unknown identities.  Adding the reconstruction error makes
        # the threshold meaningful even when the training set is tiny.
        reconstruction = descriptor - self.mean - projection @ self.components
        distance = float(math.hypot(distance, float(np.linalg.norm(reconstruction))))
        if distance > self.threshold:
            return None, 0.0
        confidence = float(np.exp(-distance / max(1e-6, self.threshold)))
        return self.class_names[index], confidence

    def predict(self, image: Union[np.ndarray, Image.Image]) -> Optional[str]:
        return self.predict_with_confidence(image)[0]

    def save(self, path: Union[str, os.PathLike]) -> None:
        if self.mean is None or self.components is None or self.class_means is None:
            raise RuntimeError("FaceRecognizer is not fitted.")
        np.savez_compressed(
            path,
            mean=self.mean,
            components=self.components,
            class_means=self.class_means,
            class_names=np.asarray(self.class_names, dtype=str),
            threshold=np.asarray([self.threshold]),
            n_components=np.asarray([self.n_components]),
            descriptor_size=np.asarray(self.descriptor_size),
        )

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "FaceRecognizer":
        data = np.load(path, allow_pickle=False)
        descriptor_size = tuple(int(v) for v in data["descriptor_size"])
        recognizer = cls(n_components=int(data["n_components"][0]),
                         threshold=float(data["threshold"][0]),
                         descriptor_size=descriptor_size)  # type: ignore
        recognizer.mean = np.asarray(data["mean"])
        recognizer.components = np.asarray(data["components"])
        recognizer.class_means = np.asarray(data["class_means"])
        recognizer.class_names = [str(value) for value in data["class_names"]]
        return recognizer


def detect_faces(image: Union[np.ndarray, Image.Image], **kwargs) -> List[Face]:
    """ Convenience wrapper around :class:`FaceDetector`. """
    return FaceDetector(**kwargs).detect(image)
