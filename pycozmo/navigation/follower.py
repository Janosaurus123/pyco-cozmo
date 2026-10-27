"""

Path following for differential-drive robots such as Cozmo and Vector.

The follower is deliberately independent of the robot client: it converts a
pose and a path into linear/angular velocity commands, which the caller can
send to the robot using :meth:`pycozmo.client.Client.drive_wheels`.

"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, List, Optional, Sequence, Tuple


__all__ = [
    "DriveCommand",
    "PathFollower",
]


def _wrap(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _pose_values(pose: Any) -> Tuple[float, float, float]:
    if isinstance(pose, (tuple, list)):
        values = list(pose) + [0.0] * (3 - len(pose))
        return float(values[0]), float(values[1]), float(values[2])
    return (float(getattr(pose, "x", 0.0)),
            float(getattr(pose, "y", 0.0)),
            float(getattr(pose, "yaw", getattr(pose, "heading", 0.0))))


@dataclass
class DriveCommand:
    """ A differential-drive velocity command. """

    linear: float = 0.0
    angular: float = 0.0
    target: Optional[Tuple[float, float]] = None
    waypoint_index: int = 0
    distance_to_goal: float = 0.0
    done: bool = False

    def wheel_speeds(self, track_width: float = 60.0) -> Tuple[float, float]:
        """ Convert to left/right wheel speeds in mm/s. """
        half = float(track_width) / 2.0
        return self.linear - self.angular * half, self.linear + self.angular * half


class PathFollower:
    """ Pure-pursuit style path follower.

    Args:
        lookahead_mm: Distance ahead along the path used as the target.
        linear_speed: Maximum forward speed in mm/s.
        angular_gain: Proportional gain for heading correction.
        max_angular: Maximum turn rate in rad/s.
        position_tolerance: Goal tolerance in millimetres.
        waypoint_tolerance: Distance at which an intermediate waypoint is
            considered reached.
    """

    def __init__(self,
                 lookahead_mm: float = 120.0,
                 linear_speed: float = 80.0,
                 angular_gain: float = 2.0,
                 max_angular: float = 2.0,
                 position_tolerance: float = 40.0,
                 waypoint_tolerance: float = 60.0) -> None:
        self.lookahead_mm = float(lookahead_mm)
        self.linear_speed = float(linear_speed)
        self.angular_gain = float(angular_gain)
        self.max_angular = float(max_angular)
        self.position_tolerance = float(position_tolerance)
        self.waypoint_tolerance = float(waypoint_tolerance)

    def closest_point(self, pose: Any, path: Sequence[Tuple[float, float]]
                      ) -> Tuple[int, float, float]:
        """ Return ``(segment_index, fraction, distance)`` of the closest point. """
        x, y, _ = _pose_values(pose)
        if not path:
            return 0, 0.0, 0.0
        if len(path) == 1:
            return 0, 0.0, math.hypot(path[0][0] - x, path[0][1] - y)
        best = (0, 0.0, float("inf"))
        for index, (first, second) in enumerate(zip(path, path[1:])):
            dx = second[0] - first[0]
            dy = second[1] - first[1]
            length_squared = dx * dx + dy * dy
            if length_squared <= 1e-9:
                fraction = 0.0
            else:
                fraction = ((x - first[0]) * dx + (y - first[1]) * dy) / length_squared
                fraction = max(0.0, min(1.0, fraction))
            point_x = first[0] + fraction * dx
            point_y = first[1] + fraction * dy
            distance = math.hypot(point_x - x, point_y - y)
            if distance < best[2]:
                best = (index, fraction, distance)
        return best

    def _lookahead_target(self, path: Sequence[Tuple[float, float]],
                          segment_index: int, fraction: float) -> Tuple[int, Tuple[float, float]]:
        remaining = self.lookahead_mm
        index = min(segment_index, len(path) - 2) if len(path) > 1 else 0
        if len(path) == 1:
            return 0, path[0]
        first = path[index]
        second = path[index + 1]
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            return index, second
        start_fraction = fraction if index == segment_index else 0.0
        remaining *= 1.0 - start_fraction
        while index < len(path) - 1:
            first = path[index]
            second = path[index + 1]
            dx = second[0] - first[0]
            dy = second[1] - first[1]
            length = math.hypot(dx, dy)
            if length >= remaining and length > 1e-9:
                target_fraction = remaining / length
                return index, (first[0] + dx * target_fraction,
                               first[1] + dy * target_fraction)
            remaining -= length
            index += 1
        return len(path) - 1, path[-1]

    def compute(self, pose: Any, path: Sequence[Tuple[float, float]]) -> DriveCommand:
        """ Compute a velocity command for the current pose. """
        x, y, yaw = _pose_values(pose)
        if not path:
            return DriveCommand(target=None, done=True)
        goal_x, goal_y = path[-1]
        distance_to_goal = math.hypot(goal_x - x, goal_y - y)
        if distance_to_goal <= self.position_tolerance:
            return DriveCommand(target=(goal_x, goal_y), waypoint_index=len(path) - 1,
                                distance_to_goal=distance_to_goal, done=True)

        segment_index, fraction, _ = self.closest_point((x, y, yaw), path)
        target_index, target = self._lookahead_target(path, segment_index, fraction)
        # Advance to the next waypoint once the robot is close to it.
        while (target_index < len(path) - 1 and
               math.hypot(path[target_index][0] - x, path[target_index][1] - y)
               <= self.waypoint_tolerance):
            target_index += 1
            target = path[target_index]

        heading = math.atan2(target[1] - y, target[0] - x)
        error = _wrap(heading - yaw)
        angular = float(max(-self.max_angular, min(self.max_angular,
                                                   self.angular_gain * error)))
        alignment = max(0.0, math.cos(error))
        distance_factor = min(1.0, distance_to_goal / max(1.0, self.lookahead_mm * 2.0))
        linear = self.linear_speed * alignment * max(0.25, distance_factor)
        if abs(error) > 0.9:
            linear = 0.0
        return DriveCommand(linear=linear, angular=angular, target=target,
                            waypoint_index=target_index,
                            distance_to_goal=distance_to_goal, done=False)

    def plan_and_follow(self, grid: Any, start: Any, goal: Tuple[float, float],
                        **plan_kwargs) -> Tuple[List[Tuple[float, float]], DriveCommand]:
        """ Convenience helper that plans a path and returns its first command. """
        pose = start if not isinstance(start, (tuple, list)) else start
        if hasattr(start, "x"):
            path = grid.plan((start.x, start.y), goal, **plan_kwargs)
        else:
            path = grid.plan((float(start[0]), float(start[1])), goal, **plan_kwargs)
        return path, self.compute(pose, path)
