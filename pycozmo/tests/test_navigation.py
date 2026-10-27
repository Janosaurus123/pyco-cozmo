
import math
import os
import tempfile
import unittest

import numpy as np

from pycozmo.navigation import (AStarPlanner, DriveCommand, MapBuilder,
                                OccupancyGrid, PathFollower, line_of_sight,
                                path_length, plan_path, simplify_path)
from pycozmo.vision.types import Pose2D


def make_open_grid(size=50, resolution=10.0):
    grid = OccupancyGrid(width=size, height=size, resolution=resolution,
                         origin=(0.0, 0.0))
    for row in range(size):
        for column in range(size):
            grid.set_free_cell(column, row)
    return grid


class TestOccupancyGrid(unittest.TestCase):

    def test_coordinates(self):
        grid = OccupancyGrid(width=10, height=20, resolution=25.0,
                             origin=(-100.0, -200.0))
        self.assertEqual(grid.world_to_cell(-100.0, -200.0), (0, 0))
        self.assertEqual(grid.world_to_cell(-76.0, -176.0), (0, 0))
        self.assertEqual(grid.world_to_cell(-75.0, -175.0), (1, 1))
        self.assertEqual(grid.cell_to_world(0, 0), (-87.5, -187.5))
        self.assertTrue(grid.contains(-1.0, -1.0))
        self.assertFalse(grid.contains(1000.0, 1000.0))
        self.assertEqual(grid.world_bounds, (-100.0, -200.0, 150.0, 300.0))

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            OccupancyGrid(width=0)
        with self.assertRaises(ValueError):
            OccupancyGrid(resolution=0.0)

    def test_marking(self):
        grid = OccupancyGrid(width=10, height=10, resolution=10.0, origin=(0, 0))
        grid.mark_free(50.0, 50.0)
        self.assertTrue(grid.is_free(50.0, 50.0))
        self.assertFalse(grid.is_occupied(50.0, 50.0))
        grid.mark_occupied(50.0, 50.0)
        self.assertTrue(grid.is_occupied(50.0, 50.0))
        grid.clear()
        self.assertEqual(int(grid.known_mask().sum()), 0)

    def test_radius_and_rays(self):
        grid = OccupancyGrid(width=20, height=20, resolution=10.0, origin=(0, 0))
        grid.mark_free(50.0, 50.0, radius=25.0)
        self.assertTrue(grid.is_free(60.0, 50.0))
        self.assertTrue(grid.is_free(50.0, 60.0))
        grid.mark_occupied(150.0, 50.0, radius=10.0)
        self.assertTrue(grid.is_occupied(150.0, 50.0))
        grid.mark_free_ray(0.0, 0.0, 190.0, 190.0)
        self.assertTrue(grid.is_free(100.0, 100.0))
        self.assertTrue(grid.is_free(190.0, 190.0))
        self.assertFalse(grid.is_free(100.0, 150.0))

    def test_masks_and_traversable(self):
        grid = OccupancyGrid(width=10, height=10, resolution=10.0, origin=(0, 0))
        grid.mark_free(10.0, 10.0)
        grid.mark_occupied(90.0, 90.0)
        self.assertTrue(grid.free_mask()[1, 1])
        self.assertTrue(grid.occupied_mask()[9, 9])
        self.assertFalse(grid.traversable_mask()[0, 0])
        self.assertTrue(grid.traversable_mask(allow_unknown=True)[0, 0])
        self.assertFalse(grid.traversable_mask(allow_unknown=True)[9, 9])

    def test_inflation(self):
        grid = OccupancyGrid(width=11, height=11, resolution=10.0, origin=(0, 0))
        grid.mark_occupied(50.0, 50.0)
        inflated = grid.inflated_occupied_mask(15.0)
        self.assertTrue(inflated[5, 5])
        self.assertTrue(inflated[5, 6])
        self.assertFalse(inflated[5, 9])

    def test_nearest_free_and_frontiers(self):
        grid = OccupancyGrid(width=10, height=10, resolution=10.0, origin=(0, 0))
        grid.mark_occupied(50.0, 50.0)
        grid.mark_free(10.0, 10.0)
        self.assertEqual(grid.nearest_free(50.0, 50.0), (1, 1))
        self.assertIsNone(grid.nearest_free(50.0, 50.0, max_radius=5.0))
        frontiers = grid.frontiers()
        self.assertTrue(frontiers)

    def test_persistence_and_copy(self):
        grid = make_open_grid(20)
        grid.mark_occupied(50.0, 50.0)
        path = os.path.join(tempfile.mkdtemp(), "grid.npz")
        grid.save(path)
        loaded = OccupancyGrid.load(path)
        self.assertEqual(loaded.width, grid.width)
        self.assertEqual(loaded.height, grid.height)
        self.assertTrue(np.allclose(loaded.log_odds, grid.log_odds))
        os.unlink(path)
        copied = grid.copy()
        copied.mark_free(50.0, 50.0)
        self.assertTrue(grid.is_occupied(50.0, 50.0))
        self.assertFalse(copied.is_occupied(50.0, 50.0))
        self.assertIn("occupied_cells", grid.to_dict())
        self.assertIn("OccupancyGrid", repr(grid))

    def test_render(self):
        grid = make_open_grid(10)
        grid.mark_occupied(50.0, 50.0)
        image = grid.to_image()
        self.assertEqual(image.shape, (10, 10, 3))
        self.assertEqual(tuple(image[5, 5]), (30, 30, 35))
        scaled = grid.to_image(scale=2)
        self.assertEqual(scaled.shape, (20, 20, 3))


class TestPlanner(unittest.TestCase):

    def setUp(self):
        self.grid = make_open_grid()
        # A wall with a gap near the bottom.
        for row in range(0, 35):
            self.grid.set_occupied_cell(25, row)

    def test_line_of_sight(self):
        traversable = self.grid.traversable_mask()
        self.assertFalse(line_of_sight(self.grid, (5, 5), (5, 45), traversable))
        self.assertTrue(line_of_sight(self.grid, (5, 5), (10, 5), traversable))
        self.assertTrue(line_of_sight(self.grid, (5, 5), (45, 5), traversable))
        self.assertFalse(line_of_sight(self.grid, (5, 5), (30, 45), traversable))

    def test_plan_around_wall(self):
        path = plan_path(self.grid, (50.0, 50.0), (450.0, 50.0))
        self.assertGreater(len(path), 2)
        self.assertEqual(path[0], (50.0, 50.0))
        self.assertEqual(path[-1], (450.0, 50.0))
        self.assertGreater(path_length(path), 400.0)
        for point in path:
            self.assertFalse(self.grid.is_occupied(*point))

    def test_no_path(self):
        # Block the gap.
        for row in range(0, 50):
            self.grid.set_occupied_cell(25, row)
        self.assertEqual(plan_path(self.grid, (50.0, 50.0), (450.0, 50.0)), [])

    def test_goal_outside_or_occupied(self):
        self.assertEqual(plan_path(self.grid, (50.0, 50.0), (10000.0, 50.0)), [])
        self.assertEqual(plan_path(self.grid, (50.0, 50.0), (250.0, 50.0)), [])

    def test_unknown_cells(self):
        grid = OccupancyGrid(width=20, height=20, resolution=10.0, origin=(0, 0))
        self.assertEqual(plan_path(grid, (50.0, 50.0), (150.0, 50.0)), [])
        path = plan_path(grid, (50.0, 50.0), (150.0, 50.0), allow_unknown=True)
        self.assertGreater(len(path), 0)

    def test_robot_radius_inflation(self):
        path = plan_path(self.grid, (50.0, 50.0), (450.0, 50.0),
                         robot_radius=20.0)
        self.assertGreater(len(path), 0)
        for point in path:
            column, row = self.grid.world_to_cell(*point)
            self.assertFalse(self.grid.inflated_occupied_mask(20.0)[row, column])

    def test_simplify_path(self):
        cells = [(0, column) for column in range(10)]
        simplified = simplify_path(self.grid, cells)
        self.assertEqual(simplified, [cells[0], cells[-1]])
        self.assertEqual(simplify_path(self.grid, []), [])
        self.assertEqual(simplify_path(self.grid, [(0, 0)]), [(0, 0)])

    def test_astar_planner_class(self):
        planner = AStarPlanner(self.grid)
        path = planner.plan((50.0, 50.0), (450.0, 50.0))
        self.assertGreater(len(path), 0)
        with self.assertRaises(ValueError):
            AStarPlanner().plan((0.0, 0.0), (1.0, 1.0))
        self.assertEqual(path_length([]), 0.0)


class TestMapBuilder(unittest.TestCase):

    def test_scan(self):
        grid = OccupancyGrid(width=100, height=100, resolution=25.0,
                             origin=(-1250.0, -1250.0))
        builder = MapBuilder(grid=grid, pose=Pose2D(0.0, 0.0, 0.0))
        updated = builder.add_scan([800.0] * 9, angle_min=-math.pi / 4,
                                   angle_increment=math.pi / 16)
        self.assertEqual(updated, 10)
        self.assertGreater(int(grid.known_mask().sum()), 0)
        self.assertGreater(int(grid.occupied_mask().sum()), 0)
        self.assertEqual(builder.to_dict()["pose"]["x"], 0.0)

    def test_contacts_and_detections(self):
        class Contact:
            def __init__(self, x, y):
                self.world = (x, y)

        grid = OccupancyGrid(width=100, height=100, resolution=25.0,
                             origin=(-1250.0, -1250.0))
        builder = MapBuilder(grid)
        contact = Contact(200.0, 300.0)
        self.assertEqual(builder.add_contact(contact), (200.0, 300.0))
        self.assertIsNone(builder.add_contact(object()))
        self.assertEqual(builder.add_contacts([contact, contact]), 2)
        detection = Contact(-200.0, 100.0)
        self.assertEqual(builder.add_detection(detection), (-200.0, 100.0))
        self.assertIsNone(builder.add_detection(object()))
        self.assertEqual(builder.add_detections([detection]), 1)
        self.assertTrue(grid.is_occupied(200.0, 300.0))

    def test_update_pose(self):
        builder = MapBuilder()
        builder.update_pose((100.0, 200.0, 0.5))
        self.assertEqual(builder.pose.x, 100.0)
        self.assertEqual(builder.pose.y, 200.0)
        self.assertAlmostEqual(builder.pose.yaw, 0.5)
        builder.set_pose(Pose2D(1.0, 2.0, 3.0))
        self.assertEqual(builder.pose.x, 1.0)
        self.assertIn("MapBuilder", repr(builder))

    def test_ground_observation(self):
        from pycozmo.vision import CameraModel, DrivableAreaEstimator
        image = np.full((240, 320, 3), (190, 180, 160), dtype=np.uint8)
        camera = CameraModel.cozmo()
        camera.set_pose(pitch_deg=15.0)
        drivable = DrivableAreaEstimator(max_distance_mm=2000).estimate(
            image, camera=camera)
        grid = OccupancyGrid(width=160, height=160, resolution=25.0,
                             origin=(-2000.0, -2000.0))
        builder = MapBuilder(grid)
        count = builder.add_ground_observation(drivable, camera, step=8)
        self.assertGreater(count, 0)
        self.assertGreater(int(grid.free_mask().sum()), 0)


class TestPathFollower(unittest.TestCase):

    def setUp(self):
        self.follower = PathFollower(lookahead_mm=100.0, linear_speed=80.0)
        self.path = [(0.0, 0.0), (1000.0, 0.0)]

    def test_wheel_speeds(self):
        command = DriveCommand(linear=100.0, angular=1.0)
        left, right = command.wheel_speeds(track_width=60.0)
        self.assertAlmostEqual(left, 70.0)
        self.assertAlmostEqual(right, 130.0)

    def test_compute_forward(self):
        command = self.follower.compute((0.0, 0.0, 0.0), self.path)
        self.assertFalse(command.done)
        self.assertGreater(command.linear, 0.0)
        self.assertAlmostEqual(command.angular, 0.0)
        self.assertGreater(command.distance_to_goal, 0.0)
        self.assertIsNotNone(command.target)

    def test_compute_done(self):
        command = self.follower.compute((1000.0, 0.0, 0.0), self.path)
        self.assertTrue(command.done)
        self.assertEqual(command.waypoint_index, 1)
        self.assertEqual(command.linear, 0.0)
        self.assertTrue(self.follower.compute((0.0, 0.0, 0.0), []).done)

    def test_compute_wrong_heading(self):
        command = self.follower.compute((0.0, 0.0, math.pi), self.path)
        self.assertAlmostEqual(command.angular, -2.0)
        self.assertEqual(command.linear, 0.0)

    def test_closest_point_and_lookahead(self):
        index, fraction, distance = self.follower.closest_point(
            (250.0, 10.0, 0.0), self.path)
        self.assertEqual(index, 0)
        self.assertAlmostEqual(fraction, 0.25)
        self.assertAlmostEqual(distance, 10.0)
        self.assertEqual(self.follower.closest_point((0.0, 0.0, 0.0), []),
                         (0, 0.0, 0.0))
        index, target = self.follower._lookahead_target(self.path, 0, 0.25)
        self.assertEqual(index, 0)
        self.assertAlmostEqual(target[0], 75.0)

    def test_plan_and_follow(self):
        grid = make_open_grid()
        path, command = self.follower.plan_and_follow(
            grid, (50.0, 50.0), (450.0, 50.0))
        self.assertGreater(len(path), 0)
        self.assertFalse(command.done)


if __name__ == "__main__":
    unittest.main()
