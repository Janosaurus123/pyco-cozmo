"""

Low-level image helpers shared by the vision modules.

OpenCV is used when it is available, but every function in this module has a
NumPy/Pillow fallback.  This keeps the base :mod:`pycozmo` package usable on
systems where installing OpenCV is not desirable.

"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import io
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover
    from .types import BoundingBox

try:  # pragma: no cover - the import itself is trivial
    import cv2
except Exception:  # pragma: no cover - depends on the environment
    cv2 = None  # type: ignore


__all__ = [
    "CV2_AVAILABLE",
    "Component",
    "as_array",
    "as_rgb",
    "as_gray",
    "as_hsv",
    "to_pil",
    "resize",
    "box_blur",
    "gaussian_blur",
    "threshold",
    "otsu_threshold",
    "dilate",
    "erode",
    "connected_components",
    "component_mask",
    "nms",
    "color_name",
    "dominant_color",
    "clip_box",
    "safe_crop",
    "hsv_mask",
    "Cv2Required",
]


CV2_AVAILABLE = cv2 is not None


class Cv2Required(RuntimeError):
    """ Raised when a feature needs OpenCV and it is not installed. """


def _require_cv2(feature: str) -> None:
    if cv2 is None:
        raise Cv2Required("{} requires OpenCV. Install it with 'pip install pycozmo[vision]'.".format(feature))


def as_array(image: Union[np.ndarray, Image.Image, bytes, bytearray]) -> np.ndarray:
    """ Convert an image-like object to a NumPy array. """
    if isinstance(image, np.ndarray):
        array = image
    elif isinstance(image, Image.Image):
        array = np.asarray(image)
    elif isinstance(image, (bytes, bytearray)):
        array = np.asarray(Image.open(io.BytesIO(bytes(image))))
    else:
        raise TypeError("Unsupported image type: {}".format(type(image).__name__))

    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError("Expected a grayscale, RGB, or RGBA image, got shape {}.".format(array.shape))
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    return array


def as_rgb(image: Union[np.ndarray, Image.Image, bytes, bytearray]) -> np.ndarray:
    """ Return an RGB ``uint8`` array. """
    array = as_array(image)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return array


def as_gray(image: Union[np.ndarray, Image.Image, bytes, bytearray]) -> np.ndarray:
    """ Return a grayscale ``uint8`` array. """
    array = as_array(image)
    if array.shape[2] == 1:
        return array[:, :, 0]
    if cv2 is not None:
        # OpenCV's RGB conversion is used because our arrays are RGB.
        return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    rgb = array.astype(np.float32)
    return np.clip(0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2],
                   0, 255).astype(np.uint8)


def as_hsv(image: Union[np.ndarray, Image.Image, bytes, bytearray]) -> np.ndarray:
    """ Return an OpenCV-style HSV array (H in 0..179). """
    array = as_rgb(image)
    if cv2 is not None:
        return cv2.cvtColor(array, cv2.COLOR_RGB2HSV)

    rgb = array.astype(np.float32) / 255.0
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    delta = maxc - minc
    h = np.zeros_like(maxc)
    mask = delta > 1e-6
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    idx = mask & (maxc == r)
    h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6.0
    idx = mask & (maxc == g) & ~(maxc == r)
    h[idx] = (b[idx] - r[idx]) / delta[idx] + 2.0
    idx = mask & (maxc == b) & ~(maxc == r) & ~(maxc == g)
    h[idx] = (r[idx] - g[idx]) / delta[idx] + 4.0
    h = (h * 30.0) % 180.0
    s = np.where(maxc > 0.0, delta / np.maximum(maxc, 1e-6), 0.0) * 255.0
    v = maxc * 255.0
    return np.stack([h, s, v], axis=2).astype(np.uint8)


def to_pil(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """ Convert an image-like object to a Pillow image. """
    if isinstance(image, Image.Image):
        return image
    array = as_array(image)
    if array.shape[2] == 1:
        array = array[:, :, 0]
    return Image.fromarray(array)


def resize(image: Union[np.ndarray, Image.Image], size: Tuple[int, int],
           interpolation: str = "linear") -> np.ndarray:
    """ Resize an image to ``(width, height)`` and return an RGB array. """
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    array = as_rgb(image)
    if cv2 is not None:
        flags = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "cubic": cv2.INTER_CUBIC,
        }.get(interpolation, cv2.INTER_LINEAR)
        return cv2.resize(array, (width, height), interpolation=flags)
    pil_image = Image.fromarray(array).resize((width, height), Image.BILINEAR)
    return np.asarray(pil_image)


def box_blur(image: Union[np.ndarray, Image.Image], ksize: int = 3) -> np.ndarray:
    """ Blur a grayscale or RGB image with a square box filter. """
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    array = as_gray(image) if as_array(image).shape[2] == 1 else as_rgb(image)
    if cv2 is not None:
        return cv2.blur(array, (ksize, ksize))
    return _numpy_blur(array.astype(np.float32), ksize).astype(np.uint8)


def gaussian_blur(image: Union[np.ndarray, Image.Image], ksize: int = 5,
                  sigma: float = 0.0) -> np.ndarray:
    """ Gaussian blur, using a box blur fallback when OpenCV is absent. """
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    array = as_gray(image) if as_array(image).shape[2] == 1 else as_rgb(image)
    if cv2 is not None:
        return cv2.GaussianBlur(array, (ksize, ksize), sigma)
    return _numpy_blur(array.astype(np.float32), ksize).astype(np.uint8)


def _numpy_blur(array: np.ndarray, ksize: int) -> np.ndarray:
    """ Separable box blur without OpenCV. """
    radius = ksize // 2
    if radius == 0:
        return array
    result = array
    for axis in (0, 1):
        pad = [(0, 0)] * result.ndim
        pad[axis] = (radius + 1, radius)
        padded = np.pad(result, pad, mode="edge")
        cumsum = np.cumsum(padded, axis=axis, dtype=np.float32)
        if axis == 0:
            result = (cumsum[2 * radius + 1:, :] - cumsum[:-2 * radius - 1, :]) / (2 * radius + 1)
        else:
            result = (cumsum[:, 2 * radius + 1:] - cumsum[:, :-2 * radius - 1]) / (2 * radius + 1)
    return result


def otsu_threshold(image: Union[np.ndarray, Image.Image]) -> int:
    """ Return an Otsu threshold for a grayscale image. """
    gray = as_gray(image)
    if cv2 is not None:
        value, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return int(value)
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total <= 0:
        return 0
    omegas = np.cumsum(histogram) / total
    mus = np.cumsum(histogram * np.arange(256)) / total
    mu_t = mus[-1]
    sigma_b = (mu_t * omegas - mus) ** 2 / np.maximum(omegas * (1.0 - omegas), 1e-12)
    sigma_b[~np.isfinite(sigma_b)] = -1.0
    return int(np.argmax(sigma_b))


def threshold(image: Union[np.ndarray, Image.Image], value: Optional[int] = None,
              inverse: bool = False) -> np.ndarray:
    """ Threshold a grayscale image and return a boolean mask.

    If ``value`` is ``None`` an Otsu threshold is calculated.
    """
    gray = as_gray(image)
    if value is None:
        value = otsu_threshold(gray)
    return gray <= int(value) if inverse else gray >= int(value)


def dilate(mask: np.ndarray, iterations: int = 1, kernel_size: int = 3) -> np.ndarray:
    """ Dilate a boolean mask. """
    mask = np.asarray(mask, dtype=np.bool_)
    iterations = max(0, int(iterations))
    if iterations == 0:
        return mask
    if cv2 is not None:
        kernel = np.ones((max(1, int(kernel_size)),) * 2, dtype=np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations).astype(np.bool_)
    result = mask.copy()
    for _ in range(iterations):
        result = _shift_or(result)
    return result


def erode(mask: np.ndarray, iterations: int = 1, kernel_size: int = 3) -> np.ndarray:
    """ Erode a boolean mask. """
    mask = np.asarray(mask, dtype=np.bool_)
    iterations = max(0, int(iterations))
    if iterations == 0:
        return mask
    if cv2 is not None:
        kernel = np.ones((max(1, int(kernel_size)),) * 2, dtype=np.uint8)
        return cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations).astype(np.bool_)
    result = mask.copy()
    for _ in range(iterations):
        result = _shift_and(result)
    return result


def _shift_or(mask: np.ndarray) -> np.ndarray:
    result = mask.copy()
    result[1:, :] |= mask[:-1, :]
    result[:-1, :] |= mask[1:, :]
    result[:, 1:] |= mask[:, :-1]
    result[:, :-1] |= mask[:, 1:]
    return result


def _shift_and(mask: np.ndarray) -> np.ndarray:
    result = mask.copy()
    result[1:, :] &= mask[:-1, :]
    result[:-1, :] &= mask[1:, :]
    result[:, 1:] &= mask[:, :-1]
    result[:, :-1] &= mask[:, 1:]
    return result


@dataclass
class Component:
    """ A connected component in a binary mask. """

    label: int
    x: int
    y: int
    width: int
    height: int
    area: int
    centroid: Tuple[float, float]
    mask: np.ndarray

    @property
    def bbox(self) -> "BoundingBox":
        from .types import BoundingBox
        return BoundingBox(self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def extent(self) -> float:
        area = self.width * self.height
        return float(self.area / area) if area > 0 else 0.0

    @property
    def aspect_ratio(self) -> float:
        return float(self.width / self.height) if self.height > 0 else 0.0


def connected_components(mask: np.ndarray, min_area: int = 1,
                         max_components: Optional[int] = None) -> List[Component]:
    """ Find 8-connected components in a boolean mask.

    Components smaller than ``min_area`` are discarded.  The returned
    component masks are cropped to their bounding boxes.
    """
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError("Expected a two-dimensional mask.")
    if cv2 is not None:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        components: List[Component] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            component_mask = labels[y:y + height, x:x + width] == label
            components.append(Component(label, x, y, width, height, area,
                                        (float(centroids[label][0]), float(centroids[label][1])),
                                        component_mask))
    else:
        components = _numpy_connected_components(mask.astype(np.bool_), min_area)

    if max_components is not None and len(components) > max_components:
        components.sort(key=lambda c: c.area, reverse=True)
        components = components[:max_components]
    return components


def _numpy_connected_components(mask: np.ndarray, min_area: int) -> List[Component]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=np.bool_)
    components: List[Component] = []
    label = 0
    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or visited[start_y, start_x]:
                continue
            label += 1
            queue = deque([(start_x, start_y)])
            visited[start_y, start_x] = True
            points = []
            while queue:
                x, y = queue.popleft()
                points.append((x, y))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((nx, ny))
            if len(points) < min_area:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            w, h = x2 - x1 + 1, y2 - y1 + 1
            component_mask = np.zeros((h, w), dtype=np.bool_)
            for x, y in points:
                component_mask[y - y1, x - x1] = True
            centroid = (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))
            components.append(Component(label, x1, y1, w, h, len(points), centroid, component_mask))
    components.sort(key=lambda c: c.area, reverse=True)
    return components


def component_mask(shape: Tuple[int, int], component: Component) -> np.ndarray:
    """ Create a full-size boolean mask for a component. """
    mask = np.zeros(shape, dtype=np.bool_)
    mask[component.y:component.y + component.height,
         component.x:component.x + component.width] = component.mask
    return mask


def nms(detections: Sequence, iou_threshold: float = 0.45,
        score_key: str = "confidence") -> List:
    """ Greedy non-maximum suppression.

    ``detections`` can be :class:`Detection` objects or any objects with a
    ``bbox`` attribute and sortable score attribute.
    """
    if not detections:
        return []
    ordered = sorted(detections, key=lambda d: float(getattr(d, score_key, 0.0)), reverse=True)
    kept = []
    while ordered:
        best = ordered.pop(0)
        kept.append(best)
        ordered = [d for d in ordered if best.bbox.iou(d.bbox) <= iou_threshold]
    return kept


def clip_box(box: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    """ Clip ``(x1, y1, x2, y2)`` to an image and return integers. """
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(width, round(x1))))
    x2 = int(max(0, min(width, round(x2))))
    y1 = int(max(0, min(height, round(y1))))
    y2 = int(max(0, min(height, round(y2))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def safe_crop(image: Union[np.ndarray, Image.Image], box: Tuple[float, float, float, float]) -> np.ndarray:
    """ Crop an RGB image to a box, returning an empty array if needed. """
    array = as_rgb(image)
    height, width = array.shape[:2]
    x1, y1, x2, y2 = clip_box(box, width, height)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return array[y1:y2, x1:x2]


def hsv_mask(image: Union[np.ndarray, Image.Image], lower: Sequence[int],
             upper: Sequence[int]) -> np.ndarray:
    """ Return a boolean mask for an inclusive HSV range. """
    hsv = as_hsv(image)
    lower_arr = np.asarray(lower, dtype=np.int16)
    upper_arr = np.asarray(upper, dtype=np.int16)
    if cv2 is not None:
        return cv2.inRange(hsv, lower_arr.astype(np.uint8), upper_arr.astype(np.uint8)).astype(np.bool_)
    mask = np.ones(hsv.shape[:2], dtype=np.bool_)
    for index in range(3):
        mask &= (hsv[:, :, index] >= lower_arr[index]) & (hsv[:, :, index] <= upper_arr[index])
    return mask


def dominant_color(image: Union[np.ndarray, Image.Image], mask: Optional[np.ndarray] = None) -> Tuple[int, int, int]:
    """ Return the mean RGB color of an image or a masked region. """
    rgb = as_rgb(image)
    if mask is None:
        pixels = rgb.reshape(-1, 3)
    else:
        mask = np.asarray(mask, dtype=np.bool_)
        if mask.shape != rgb.shape[:2]:
            raise ValueError("Mask must match the image dimensions.")
        pixels = rgb[mask]
        if pixels.size == 0:
            return (0, 0, 0)
    mean = pixels.astype(np.float64).mean(axis=0)
    return tuple(int(round(value)) for value in mean)  # type: ignore


_COLOR_NAMES = [
    ("black", (0, 0, 0)),
    ("white", (255, 255, 255)),
    ("gray", (128, 128, 128)),
    ("red", (220, 40, 40)),
    ("orange", (230, 140, 35)),
    ("yellow", (235, 220, 50)),
    ("green", (40, 190, 80)),
    ("cyan", (40, 210, 210)),
    ("blue", (45, 80, 210)),
    ("purple", (150, 60, 190)),
    ("pink", (235, 130, 170)),
    ("brown", (125, 75, 40)),
]


def color_name(rgb: Sequence[int]) -> str:
    """ Return a coarse color name for an RGB triple. """
    values = np.asarray(rgb, dtype=np.float64)
    best_name = "unknown"
    best_distance = float("inf")
    for name, reference in _COLOR_NAMES:
        distance = float(np.linalg.norm(values - np.asarray(reference, dtype=np.float64)))
        if distance < best_distance:
            best_name = name
            best_distance = distance
    return best_name
