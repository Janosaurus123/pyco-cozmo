#!/usr/bin/env python

"""Build an occupancy grid, plan a path and print drive commands.

This example is offline.  Feed real ``Pose2D`` values and detections from a
robot into ``MapBuilder`` to build a map from live camera frames.
"""

import math

from pycozmo import navigation
from pycozmo.vision import Pose2D


def main() -> int:
    grid = navigation.OccupancyGrid(width=48, height=48, resolution=25.0,
                                    origin=(0.0, 0.0))
    # Free the corridor and place a couple of obstacles.
    grid.mark_free(100, 100, radius=80)
    grid.mark_free(800, 800, radius=80)
    grid.mark_occupied(500, 500, radius=80)
    grid.mark_occupied(500, 650, radius=80)

    path = navigation.plan_path(grid, start=(100, 100), goal=(800, 800),
                                allow_unknown=True)
    if not path:
        print("No path found.")
        return 1

    print("Path ({} waypoints, {:.0f} mm):".format(
        len(path), navigation.path_length(path)))

    follower = navigation.PathFollower()
    pose = Pose2D(100.0, 100.0, math.radians(45.0))
    for waypoint in path:
        command = follower.compute(pose, [waypoint])
        print("  waypoint={} linear={:.1f} mm/s angular={:.2f} rad/s".format(
            waypoint, command.linear, command.angular))
        pose = Pose2D(waypoint[0], waypoint[1], pose.yaw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
