"""

Occupancy grids and map building.

The grid stores log-odds occupancy values.  Positive values are occupied,
negative values are free and zero is unknown.  World coordinates and
resolutions are expressed in millimetres, matching the Cozmo SDK.

"""

from __future__ import annotations

import math
import os
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .planner import plan_path


__all__ = [
    "OccupancyGrid",
    "MapBuilder",
]


def _pose2d(pose: Any):
    """ Convert a pose-like object without importing vision at module load. """
    if pose is None:
        from ..vision.types import Pose2D
        return Pose2D()
    if isinstance(pose, (tuple, list)):
        values = list(pose) + [0.0] * (3 - len(pose))
        from ..vision.types import Pose2D
        return Pose2D(*values[:3])
    if hasattr(pose, "x") and hasattr(pose, "y"):
        if not hasattr(pose, "yaw"):
            from ..vision.types import Pose2D
            return Pose2D(pose.x, pose.y, getattr(pose, "heading", 0.0))
        return pose
    from ..vision.types import Pose2D
    return Pose2D.from_pose(pose)


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> Iterable[Tuple[int, int]]:
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx - dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            return
        doubled = 2 * error
        if doubled > -dy:
            error -= dy
            x0 += sx
        if doubled < dx:
            error += dx
            y0 += sy


class OccupancyGrid:
    """ A 2D log-odds occupancy grid.

    Args:
        width: Grid width in cells.
        height: Grid height in cells.
        resolution: Cell size in millimetres.
        origin: World coordinate of the grid's lower-left corner.
        occupied_log_odds: Log-odds value written by :meth:`mark_occupied`.
        free_log_odds: Log-odds value written by :meth:`mark_free`.
        occupied_threshold: Log-odds threshold for classifying occupied.
        free_threshold: Log-odds threshold for classifying free.
    """

    def __init__(self,
                 width: int = 160,
                 height: int = 160,
                 resolution: float = 25.0,
                 origin: Tuple[float, float] = (-2000.0, -2000.0),
                 occupied_log_odds: float = 1.6,
                 free_log_odds: float = 0.7,
                 occupied_threshold: float = 0.8,
                 free_threshold: float = -0.4) -> None:
        self.width = int(width)
        self.height = int(height)
        self.resolution = float(resolution)
        self.origin = (float(origin[0]), float(origin[1]))
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Grid dimensions must be positive.")
        if self.resolution <= 0.0:
            raise ValueError("Grid resolution must be positive.")
        self.occupied_log_odds = float(abs(occupied_log_odds))
        self.free_log_odds = float(abs(free_log_odds))
        self.occupied_threshold = float(occupied_threshold)
        self.free_threshold = float(free_threshold)
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------
    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """ Convert world millimetres to ``(column, row)``. """
        column = int(math.floor((float(x) - self.origin[0]) / self.resolution))
        row = int(math.floor((float(y) - self.origin[1]) / self.resolution))
        return column, row

    def cell_to_world(self, column: int, row: int) -> Tuple[float, float]:
        """ Return the world coordinate of a cell centre. """
        return (self.origin[0] + (float(column) + 0.5) * self.resolution,
                self.origin[1] + (float(row) + 0.5) * self.resolution)

    def cell_to_index(self, column: int, row: int) -> Tuple[int, int]:
        return int(row), int(column)

    def index_to_cell(self, row: int, column: int) -> Tuple[int, int]:
        return int(column), int(row)

    def contains_cell(self, column: int, row: int) -> bool:
        return 0 <= column < self.width and 0 <= row < self.height

    def contains(self, x: float, y: float) -> bool:
        column, row = self.world_to_cell(x, y)
        return self.contains_cell(column, row)

    @property
    def world_bounds(self) -> Tuple[float, float, float, float]:
        return (self.origin[0], self.origin[1],
                self.origin[0] + self.width * self.resolution,
                self.origin[1] + self.height * self.resolution)

    # ------------------------------------------------------------------
    # Cell updates
    # ------------------------------------------------------------------
    def _update(self, column: int, row: int, delta: float) -> None:
        if self.contains_cell(column, row):
            current = float(self.log_odds[row, column])
            self.log_odds[row, column] = np.clip(current + delta, -8.0, 8.0)

    def update_free_cell(self, column: int, row: int, confidence: float = 1.0) -> None:
        """ Add negative log-odds to a cell. """
        self._update(column, row, -self.free_log_odds * max(0.0, float(confidence)))

    def update_occupied_cell(self, column: int, row: int, confidence: float = 1.0) -> None:
        """ Add positive log-odds to a cell. """
        self._update(column, row, self.occupied_log_odds * max(0.0, float(confidence)))

    def set_free_cell(self, column: int, row: int) -> None:
        if self.contains_cell(column, row):
            self.log_odds[row, column] = min(
                float(self.log_odds[row, column]), -2.0 * self.free_log_odds)

    def set_occupied_cell(self, column: int, row: int) -> None:
        if self.contains_cell(column, row):
            self.log_odds[row, column] = max(
                float(self.log_odds[row, column]), 2.0 * self.occupied_log_odds)

    def mark_free(self, x: float, y: float, radius: float = 0.0) -> None:
        """ Mark the world point (and optional radius) as free. """
        self._mark_disk(x, y, radius, occupied=False)

    def mark_occupied(self, x: float, y: float, radius: float = 0.0) -> None:
        """ Mark the world point (and optional radius) as occupied. """
        self._mark_disk(x, y, radius, occupied=True)

    def _mark_disk(self, x: float, y: float, radius: float, occupied: bool) -> None:
        column, row = self.world_to_cell(x, y)
        if not self.contains_cell(column, row):
            return
        if radius <= 0.0:
            cells = [(column, row)]
        else:
            cell_radius = int(math.ceil(float(radius) / self.resolution))
            cells = []
            for drow in range(-cell_radius, cell_radius + 1):
                for dcolumn in range(-cell_radius, cell_radius + 1):
                    if (drow * self.resolution) ** 2 + (dcolumn * self.resolution) ** 2 <= radius ** 2:
                        cells.append((column + dcolumn, row + drow))
        for cell_column, cell_row in cells:
            if occupied:
                self.set_occupied_cell(cell_column, cell_row)
            else:
                self.set_free_cell(cell_column, cell_row)

    def mark_free_ray(self, x1: float, y1: float, x2: float, y2: float,
                      update: bool = True) -> None:
        """ Mark cells along a world-space line as free. """
        start = self.world_to_cell(x1, y1)
        end = self.world_to_cell(x2, y2)
        for column, row in _bresenham(start[0], start[1], end[0], end[1]):
            if not self.contains_cell(column, row):
                continue
            if update:
                self.update_free_cell(column, row)
            else:
                self.set_free_cell(column, row)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def occupied_mask(self) -> np.ndarray:
        return self.log_odds >= self.occupied_threshold

    def free_mask(self) -> np.ndarray:
        return self.log_odds <= self.free_threshold

    def known_mask(self) -> np.ndarray:
        return np.abs(self.log_odds) > 1e-6

    def unknown_mask(self) -> np.ndarray:
        return ~self.known_mask()

    def is_occupied_cell(self, column: int, row: int) -> bool:
        return (self.contains_cell(column, row) and
                bool(self.log_odds[row, column] >= self.occupied_threshold))

    def is_free_cell(self, column: int, row: int) -> bool:
        return (self.contains_cell(column, row) and
                bool(self.log_odds[row, column] <= self.free_threshold))

    def is_known_cell(self, column: int, row: int) -> bool:
        return (self.contains_cell(column, row) and
                bool(abs(float(self.log_odds[row, column])) > 1e-6))

    def is_occupied(self, x: float, y: float) -> bool:
        return self.is_occupied_cell(*self.world_to_cell(x, y))

    def is_free(self, x: float, y: float) -> bool:
        return self.is_free_cell(*self.world_to_cell(x, y))

    def traversable_mask(self, allow_unknown: bool = False) -> np.ndarray:
        """ Boolean mask of cells the planner may enter. """
        occupied = self.occupied_mask()
        if allow_unknown:
            return ~occupied
        return self.free_mask() & ~occupied

    def inflated_occupied_mask(self, radius: float) -> np.ndarray:
        """ Occupied cells dilated by a disk of ``radius`` millimetres. """
        occupied = self.occupied_mask()
        if radius <= 0.0 or not occupied.any():
            return occupied
        cell_radius = int(math.ceil(float(radius) / self.resolution))
        if cell_radius <= 0:
            return occupied
        padded = np.pad(occupied, cell_radius, mode="constant", constant_values=False)
        result = np.zeros_like(occupied)
        height, width = occupied.shape
        for drow in range(-cell_radius, cell_radius + 1):
            for dcolumn in range(-cell_radius, cell_radius + 1):
                if (drow * self.resolution) ** 2 + (dcolumn * self.resolution) ** 2 > radius ** 2:
                    continue
                result |= padded[cell_radius + drow:cell_radius + drow + height,
                                 cell_radius + dcolumn:cell_radius + dcolumn + width]
        return result

    # ------------------------------------------------------------------
    # Planning and exploration
    # ------------------------------------------------------------------
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float],
             allow_unknown: bool = False, robot_radius: float = 0.0,
             unknown_cost: float = 2.0) -> List[Tuple[float, float]]:
        """ Plan a path of world waypoints.  See :func:`plan_path`. """
        return plan_path(self, start, goal, allow_unknown=allow_unknown,
                         robot_radius=robot_radius, unknown_cost=unknown_cost)

    def frontiers(self, min_unknown_neighbors: int = 1) -> List[Tuple[float, float]]:
        """ World positions of unknown cells adjacent to free cells. """
        free = self.free_mask()
        if not free.any():
            return []
        unknown = self.unknown_mask()
        padded_free = np.pad(free, 1, mode="constant", constant_values=False)
        neighbor_count = np.zeros((self.height, self.width), dtype=np.uint8)
        for drow in (0, 1, 2):
            for dcolumn in (0, 1, 2):
                if drow == 1 and dcolumn == 1:
                    continue
                neighbor_count += padded_free[drow:drow + self.height,
                                              dcolumn:dcolumn + self.width].astype(np.uint8)
        frontier_rows, frontier_columns = np.nonzero(unknown & (neighbor_count >= min_unknown_neighbors))
        return [self.cell_to_world(int(column), int(row))
                for row, column in zip(frontier_rows, frontier_columns)]

    def nearest_free(self, x: float, y: float, max_radius: float = 500.0
                     ) -> Optional[Tuple[int, int]]:
        """ Return the nearest free cell as ``(column, row)``. """
        start = self.world_to_cell(x, y)
        if self.is_free_cell(*start):
            return start
        max_cells = int(math.ceil(max_radius / self.resolution))
        best = None
        best_distance = float("inf")
        for row in range(max(0, start[1] - max_cells), min(self.height, start[1] + max_cells + 1)):
            for column in range(max(0, start[0] - max_cells), min(self.width, start[0] + max_cells + 1)):
                if not self.is_free_cell(column, row):
                    continue
                distance = math.hypot(column - start[0], row - start[1]) * self.resolution
                if distance <= max_radius and distance < best_distance:
                    best_distance = distance
                    best = (column, row)
        return best

    # ------------------------------------------------------------------
    # Rendering and persistence
    # ------------------------------------------------------------------
    def to_image(self, scale: int = 1) -> np.ndarray:
        """ Render the grid as an RGB image. """
        image = np.empty((self.height, self.width, 3), dtype=np.uint8)
        image[:] = (128, 128, 128)
        image[self.free_mask()] = (245, 245, 240)
        image[self.occupied_mask()] = (30, 30, 35)
        scale = max(1, int(scale))
        if scale > 1:
            image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
        return image

    def save(self, path: Union[str, os.PathLike]) -> None:
        np.savez_compressed(
            path,
            log_odds=self.log_odds,
            width=np.asarray([self.width]),
            height=np.asarray([self.height]),
            resolution=np.asarray([self.resolution]),
            origin=np.asarray(self.origin),
            occupied_log_odds=np.asarray([self.occupied_log_odds]),
            free_log_odds=np.asarray([self.free_log_odds]),
            occupied_threshold=np.asarray([self.occupied_threshold]),
            free_threshold=np.asarray([self.free_threshold]),
        )

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "OccupancyGrid":
        data = np.load(path, allow_pickle=False)
        grid = cls(
            width=int(data["width"][0]),
            height=int(data["height"][0]),
            resolution=float(data["resolution"][0]),
            origin=tuple(float(value) for value in data["origin"]),
            occupied_log_odds=float(data["occupied_log_odds"][0]),
            free_log_odds=float(data["free_log_odds"][0]),
            occupied_threshold=float(data["occupied_threshold"][0]),
            free_threshold=float(data["free_threshold"][0]),
        )
        grid.log_odds = np.asarray(data["log_odds"], dtype=np.float32)
        return grid

    def copy(self) -> "OccupancyGrid":
        grid = OccupancyGrid(
            width=self.width, height=self.height, resolution=self.resolution,
            origin=self.origin, occupied_log_odds=self.occupied_log_odds,
            free_log_odds=self.free_log_odds,
            occupied_threshold=self.occupied_threshold,
            free_threshold=self.free_threshold)
        grid.log_odds = self.log_odds.copy()
        return grid

    def clear(self) -> None:
        self.log_odds.fill(0.0)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "origin": list(self.origin),
            "occupied_cells": int(self.occupied_mask().sum()),
            "free_cells": int(self.free_mask().sum()),
            "known_cells": int(self.known_mask().sum()),
        }

    def __repr__(self) -> str:
        return "OccupancyGrid({}x{}, resolution={:.1f}mm, origin={})".format(
            self.width, self.height, self.resolution, self.origin)


class MapBuilder:
    """ Integrate robot scans and detections into an occupancy grid.

    Args:
        grid: Grid to update.  A default 4x4 metre grid is created when None.
        pose: Current robot pose.
        obstacle_radius: Radius in millimetres used when marking detections.
    """

    def __init__(self, grid: Optional[OccupancyGrid] = None, pose: Any = None,
                 obstacle_radius: float = 30.0) -> None:
        self.grid = grid if grid is not None else OccupancyGrid()
        self.pose = _pose2d(pose)
        self.obstacle_radius = float(obstacle_radius)

    def update_pose(self, pose: Any) -> None:
        self.pose = _pose2d(pose)

    def set_pose(self, pose: Any) -> None:
        self.update_pose(pose)

    def add_scan(self, ranges: Sequence[float],
                 angle_min: float = -math.pi,
                 angle_increment: Optional[float] = None,
                 max_range: Optional[float] = None,
                 min_range: float = 0.0) -> int:
        """ Integrate a polar range scan.

        ``ranges`` are in millimetres.  Missing or infinite readings can be
        passed as ``None`` or ``inf``; when ``max_range`` is known they mark
        free space up to that range without creating an obstacle.
        """
        if angle_increment is None:
            angle_increment = (2.0 * math.pi / len(ranges)) if ranges else 0.0
        angle_increment = float(angle_increment)
        pose = self.pose
        origin_x, origin_y = pose.x, pose.y
        self.grid.mark_free(origin_x, origin_y)
        updated = 1
        for index, value in enumerate(ranges):
            angle = pose.yaw + float(angle_min) + index * angle_increment
            bearing = math.cos(angle)
            sin_bearing = math.sin(angle)
            if value is None or not math.isfinite(float(value)):
                if max_range is None:
                    continue
                distance = float(max_range)
                hit = False
            else:
                distance = float(value)
                hit = distance <= float(max_range) if max_range is not None else math.isfinite(distance)
                if max_range is not None and distance > max_range:
                    distance = float(max_range)
                    hit = False
            if distance < min_range:
                continue
            endpoint_x = origin_x + bearing * distance
            endpoint_y = origin_y + sin_bearing * distance
            if hit:
                free_distance = max(0.0, distance - self.grid.resolution * 0.5)
                self.grid.mark_free_ray(origin_x, origin_y,
                                        origin_x + bearing * free_distance,
                                        origin_y + sin_bearing * free_distance)
                column, row = self.grid.world_to_cell(endpoint_x, endpoint_y)
                self.grid.update_occupied_cell(column, row)
            else:
                self.grid.mark_free_ray(origin_x, origin_y, endpoint_x, endpoint_y)
            updated += 1
        return updated

    def add_contact(self, contact: Any) -> Optional[Tuple[float, float]]:
        """ Mark a ground contact (or any object with ``world``) as occupied. """
        world = getattr(contact, "world", None)
        if world is None and hasattr(contact, "bbox"):
            return None
        if world is None:
            return None
        self.grid.mark_occupied(float(world[0]), float(world[1]), self.obstacle_radius)
        return (float(world[0]), float(world[1]))

    def add_contacts(self, contacts: Iterable[Any]) -> int:
        added = 0
        for contact in contacts:
            if self.add_contact(contact) is not None:
                added += 1
        return added

    def add_detection(self, detection: Any, camera: Any = None,
                      radius: Optional[float] = None) -> Optional[Tuple[float, float]]:
        """ Mark a detection's ground position as occupied.

        The detection must either carry a world position in its attributes or
        a bounding box plus a camera model for ground projection.
        """
        world = getattr(detection, "world", None)
        if world is not None and len(world) >= 2:
            point = (float(world[0]), float(world[1]))
        elif camera is not None and hasattr(detection, "bbox"):
            bbox = detection.bbox
            point_3d = camera.ground_intersection(bbox.center_x, bbox.y2 - 1)
            if point_3d is None:
                return None
            point = (float(point_3d[0]), float(point_3d[1]))
        else:
            return None
        self.grid.mark_occupied(point[0], point[1],
                                self.obstacle_radius if radius is None else radius)
        return point

    def add_detections(self, detections: Iterable[Any], camera: Any = None) -> int:
        added = 0
        for detection in detections:
            if self.add_detection(detection, camera=camera) is not None:
                added += 1
        return added

    def add_ground_observation(self, drivable: Any, camera: Any,
                               step: int = 4, max_samples: int = 20000) -> int:
        """ Integrate a :class:`DrivableArea` result into the map.

        Free image pixels are projected onto the ground plane and marked
        free.  Obstacle contacts are marked occupied.  This is a simple and
        surprisingly effective way to build a map while driving.
        """
        mask = getattr(drivable, "mask", None)
        if mask is None:
            return 0
        height, width = mask.shape[:2]
        floor_row = int(max(0, getattr(drivable, "floor_row", 0.0)))
        step = max(1, int(step))
        free_cells = set()
        samples = 0
        for row in range(floor_row, height, step):
            for column in range(0, width, step):
                if not mask[row, column]:
                    continue
                point = camera.ground_intersection(column, row)
                if point is None:
                    continue
                cell = self.grid.world_to_cell(point[0], point[1])
                if self.grid.contains_cell(*cell):
                    free_cells.add(cell)
                samples += 1
                if samples >= max_samples:
                    break
            if samples >= max_samples:
                break
        for column, row in free_cells:
            self.grid.set_free_cell(column, row)
        self.add_contacts(getattr(drivable, "contacts", []))
        return len(free_cells)

    def to_dict(self) -> dict:
        data = self.grid.to_dict()
        data["pose"] = {"x": self.pose.x, "y": self.pose.y, "yaw": self.pose.yaw}
        return data

    def __repr__(self) -> str:
        return "MapBuilder(grid={!r}, pose={!r})".format(self.grid, self.pose)
