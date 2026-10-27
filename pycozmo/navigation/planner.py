"""

A* path planning on an :class:`~pycozmo.navigation.mapping.OccupancyGrid`.

The planner works in world millimetres and supports unknown cells, obstacle
inflation for the robot footprint and line-of-sight path simplification.

"""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from .mapping import OccupancyGrid


__all__ = [
    "AStarPlanner",
    "plan_path",
    "simplify_path",
    "line_of_sight",
    "path_length",
]


Cell = Tuple[int, int]
Point = Tuple[float, float]


def _neighbors(cell: Cell) -> Iterable[Tuple[Cell, float, bool]]:
    row, column = cell
    for drow, dcolumn in ((-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)):
        diagonal = drow != 0 and dcolumn != 0
        yield (row + drow, column + dcolumn), (math.sqrt(2.0) if diagonal else 1.0), diagonal


def _line_cells(start: Cell, end: Cell) -> Iterable[Cell]:
    """ Bresenham line between two cells, including both endpoints. """
    row0, column0 = start
    row1, column1 = end
    drow = abs(row1 - row0)
    dcolumn = abs(column1 - column0)
    step_row = 1 if row0 < row1 else -1
    step_column = 1 if column0 < column1 else -1
    error = dcolumn - drow
    while True:
        yield row0, column0
        if row0 == row1 and column0 == column1:
            return
        doubled = 2 * error
        if doubled > -drow:
            error -= drow
            column0 += step_column
        if doubled < dcolumn:
            error += dcolumn
            row0 += step_row


def line_of_sight(grid: "OccupancyGrid", start: Cell, end: Cell,
                  traversable: Optional[Sequence] = None) -> bool:
    """ Return whether a straight grid line is traversable.

    Diagonal moves may not cut between two blocked orthogonal neighbours.
    """
    if traversable is None:
        traversable = grid.traversable_mask()
    height, width = grid.log_odds.shape

    def allowed(cell: Cell) -> bool:
        row, column = cell
        return 0 <= row < height and 0 <= column < width and bool(traversable[row, column])

    for cell in _line_cells(start, end):
        if not allowed(cell):
            return False
    # Check the orthogonal neighbours of each diagonal step.  This prevents
    # paths that squeeze through the corner of two occupied cells.
    previous = start
    for cell in _line_cells(start, end):
        row, column = cell
        prow, pcolumn = previous
        if row != prow and column != pcolumn:
            if not (allowed((prow, column)) and allowed((row, pcolumn))):
                return False
        previous = cell
    return True


def simplify_path(grid: "OccupancyGrid", cells: List[Cell],
                  traversable: Optional[Sequence] = None) -> List[Cell]:
    """ Greedily remove intermediate waypoints that have line of sight. """
    if len(cells) <= 2:
        return list(cells)
    if traversable is None:
        traversable = grid.traversable_mask()
    simplified = [cells[0]]
    anchor = 0
    index = 2
    while index < len(cells):
        if line_of_sight(grid, cells[anchor], cells[index], traversable):
            index += 1
            continue
        simplified.append(cells[index - 1])
        anchor = index - 1
        index += 1
    simplified.append(cells[-1])
    return simplified


def _octile(dx: int, dy: int) -> float:
    dx, dy = abs(dx), abs(dy)
    return (dx + dy) + (math.sqrt(2.0) - 2.0) * min(dx, dy)


def plan_path(grid: "OccupancyGrid", start: Point, goal: Point,
              allow_unknown: bool = False,
              robot_radius: float = 0.0,
              unknown_cost: float = 2.0,
              max_expansions: int = 500000) -> List[Point]:
    """ Plan a path between two world points.

    Args:
        grid: Occupancy grid to search.
        start: Start ``(x, y)`` in millimetres.
        goal: Goal ``(x, y)`` in millimetres.
        allow_unknown: Treat unknown cells as traversable but more expensive.
        robot_radius: Inflate occupied cells by this many millimetres.
        unknown_cost: Cost multiplier applied to unknown cells.
        max_expansions: Safety limit for the search.

    Returns:
        A list of world ``(x, y)`` waypoints, including ``start`` and
        ``goal``.  An empty list means no path was found.
    """
    traversable = grid.traversable_mask(allow_unknown=allow_unknown)
    if robot_radius > 0.0:
        traversable = traversable & ~grid.inflated_occupied_mask(robot_radius)

    # Planner cells are (row, column); world_to_cell returns (column, row).
    start_column, start_row = grid.world_to_cell(start[0], start[1])
    goal_column, goal_row = grid.world_to_cell(goal[0], goal[1])
    start_cell = (start_row, start_column)
    goal_cell = (goal_row, goal_column)
    height, width = grid.log_odds.shape

    def inside(cell: Cell) -> bool:
        return 0 <= cell[0] < height and 0 <= cell[1] < width

    if not inside(start_cell) or not inside(goal_cell):
        return []
    if not traversable[goal_cell[0], goal_cell[1]]:
        return []
    # The robot may start next to an inflated obstacle.  The start cell is
    # allowed as long as it is not actually occupied.
    if not traversable[start_cell[0], start_cell[1]]:
        if grid.is_occupied(*start):
            return []
        traversable = traversable.copy()
        traversable[start_cell[0], start_cell[1]] = True

    resolution = float(grid.resolution)
    open_heap: List[Tuple[float, float, Cell]] = [
        (_octile(goal_cell[0] - start_cell[0], goal_cell[1] - start_cell[1]) * resolution,
         0.0, start_cell)]
    came_from = {}
    costs = {start_cell: 0.0}
    closed = set()
    expanded = 0

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_cell:
            cells = [current]
            while cells[-1] in came_from:
                cells.append(came_from[cells[-1]])
            cells.reverse()
            cells = simplify_path(grid, cells, traversable)
            points: List[Point] = [start]
            for row, column in cells[1:-1]:
                points.append(grid.cell_to_world(column, row))
            if goal != start:
                points.append(goal)
            return _deduplicate(points)

        closed.add(current)
        expanded += 1
        if expanded > max_expansions:
            return []

        for neighbor, step_cost, diagonal in _neighbors(current):
            if not inside(neighbor) or neighbor in closed:
                continue
            row, column = neighbor
            if not traversable[row, column]:
                continue
            if diagonal:
                if not traversable[current[0], column] or not traversable[row, current[1]]:
                    continue
            multiplier = 1.0
            if not grid.is_known_cell(column, row):
                multiplier = max(1.0, unknown_cost)
            tentative = current_cost + step_cost * resolution * multiplier
            if tentative >= costs.get(neighbor, float("inf")):
                continue
            costs[neighbor] = tentative
            came_from[neighbor] = current
            heuristic = _octile(goal_cell[0] - row, goal_cell[1] - column) * resolution
            heapq.heappush(open_heap, (tentative + heuristic, tentative, neighbor))
    return []


def _deduplicate(points: List[Point]) -> List[Point]:
    result: List[Point] = []
    for point in points:
        if not result or point[0] != result[-1][0] or point[1] != result[-1][1]:
            result.append(point)
    return result


def path_length(points: Sequence[Point]) -> float:
    """ Total Euclidean length of a path in millimetres. """
    total = 0.0
    for first, second in zip(points, points[1:]):
        total += math.hypot(second[0] - first[0], second[1] - first[1])
    return total


class AStarPlanner:
    """ Reusable planner with default search options. """

    def __init__(self, grid: Optional["OccupancyGrid"] = None,
                 allow_unknown: bool = False,
                 robot_radius: float = 0.0,
                 unknown_cost: float = 2.0) -> None:
        self.grid = grid
        self.allow_unknown = bool(allow_unknown)
        self.robot_radius = float(robot_radius)
        self.unknown_cost = float(unknown_cost)

    def plan(self, start: Point, goal: Point, grid: Optional["OccupancyGrid"] = None,
             **kwargs) -> List[Point]:
        target = grid or self.grid
        if target is None:
            raise ValueError("A grid is required.")
        options = {
            "allow_unknown": self.allow_unknown,
            "robot_radius": self.robot_radius,
            "unknown_cost": self.unknown_cost,
        }
        options.update(kwargs)
        return plan_path(target, start, goal, **options)
