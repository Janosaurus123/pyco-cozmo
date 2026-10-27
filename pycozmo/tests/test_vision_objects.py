import os
import tempfile
import unittest

import numpy as np

from pycozmo.vision import (BoundingBox, CameraModel, CalibrationResult,
                            CustomMarkerDetector,
                            CUSTOM_MARKER_TYPES, CV2_AVAILABLE, Detection,
                            DrivableAreaEstimator, ExpressionEstimator,
                            Face, FaceDetector, FaceRecognizer, FaceTracker,
                            MotionDetector, MotionTracker, Pet, PetDetector,
                            Pose2D, Robot, RobotDetector, VisionConfig,
                            VisionPipeline, align_face, detect_faces,
                            detect_motion, detect_pets, detect_robots,
                            estimate_homography, apply_homography,
                            load_cube_face_templates, CubeFaceDetector,
                            ArucoDetector, render_custom_marker)
from pycozmo.vision.utils import cv2


def make_scene(width=320, height=240):
    """ A synthetic room with a floor, wall, obstacle, pet and robot. """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (190, 180, 160)
    image[:110] = (90, 110, 150)
    image[150:205, 150:190] = (180, 40, 35)
    yy, xx = np.mgrid[0:height, 0:width]
    pet = ((xx - 70) / 26) ** 2 + ((yy - 175) / 20) ** 2 <= 1
    image[pet] = (120, 70, 35)
    image[130:155, 235:275] = (30, 220, 90)
    image[155:205, 230:280] = (35, 35, 40)
    image[138:148, 245:252] = (250, 250, 250)
    return image


def make_face(smile=0.0, open_mouth=False, blink=False, skin=(224, 172, 140),
              shift=(0, 0), size=(240, 240), gaze=(0, 0), eye_sep=26,
              eye_ry=6.0, mouth_w=22, brow_dy=45):
    image = np.full((size[1], size[0], 3), 40, dtype=np.uint8)
    yy, xx = np.mgrid[0:size[1], 0:size[0]]
    cx, cy = size[0] // 2 + shift[0], size[1] // 2 + shift[1]
    rx, ry = 62, 82
    mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
    image[mask] = skin
    for ex in (-eye_sep, eye_sep):
        ey = cy - 22
        if blink:
            image[ey - 1:ey + 2, cx + ex - 11:cx + ex + 11] = (60, 50, 45)
        else:
            eye = ((xx - (cx + ex)) / 10.0) ** 2 + ((yy - ey) / eye_ry) ** 2 <= 1
            image[eye] = (35, 30, 28)
            px = cx + ex + int(gaze[0] * 6)
            py = ey + int(gaze[1] * 3)
            pupil = ((xx - px) / 4.0) ** 2 + ((yy - py) / 4.0) ** 2 <= 1
            image[pupil] = (15, 15, 15)
        image[cy - brow_dy:cy - brow_dy + 3,
              cx + ex - 14:cx + ex + 14] = (70, 55, 45)
    if open_mouth:
        mouth = ((xx - cx) / float(mouth_w)) ** 2 + ((yy - (cy + 38)) / 16.0) ** 2 <= 1
        image[mouth] = (90, 40, 40)
    else:
        ym = cy + 36 - (smile * 9.0 * np.clip(((xx - cx) / float(mouth_w)) ** 2, 0, 1)).astype(int)
        band = (np.abs(yy - ym) <= 2) & (np.abs(xx - cx) <= 22)
        image[band] = (100, 45, 45)
    return image


def cozmo_camera(width=320, height=240):
    camera = CameraModel.cozmo(width=width, height=height)
    camera.set_pose(position=(0.0, 0.0, 40.0), yaw_deg=0.0, pitch_deg=15.0)
    return camera


class TestTypes(unittest.TestCase):

    def test_bounding_box(self):
        box = BoundingBox.from_xywh(10, 20, 30, 40)
        self.assertEqual(box.xyxy, (10.0, 20.0, 40.0, 60.0))
        self.assertEqual(box.xywh, (10.0, 20.0, 30.0, 40.0))
        self.assertEqual(box.center, (25.0, 40.0))
        self.assertEqual(box.area, 1200.0)
        self.assertEqual(tuple(box), box.xyxy)
        self.assertEqual(box.to_int(), (10, 20, 40, 60))
        self.assertTrue(box.contains(25, 40))
        self.assertFalse(box.contains(0, 0))
        self.assertEqual(BoundingBox.from_center(25, 40, 30, 40), box)
        self.assertEqual(BoundingBox.from_iterable((10, 20, 40, 60)), box)
        with self.assertRaises(ValueError):
            BoundingBox.from_iterable((1, 2, 3))

    def test_bounding_box_iou_and_clip(self):
        first = BoundingBox(0, 0, 10, 10)
        second = BoundingBox(5, 5, 15, 15)
        self.assertAlmostEqual(first.iou(second), 25.0 / 175.0)
        self.assertEqual(first.intersection(BoundingBox(20, 20, 30, 30)).area, 0.0)
        self.assertEqual(first.expand(2).xyxy, (-2.0, -2.0, 12.0, 12.0))
        self.assertEqual(first.expand(20, clip_size=(20, 20)).xyxy,
                         (0.0, 0.0, 20.0, 20.0))
        self.assertEqual(BoundingBox(-5, -5, 30, 30).clip(20, 20).xyxy,
                         (0.0, 0.0, 20.0, 20.0))
        self.assertEqual(first.to_dict(),
                         {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0})

    def test_detection(self):
        detection = Detection(kind="cube:face", bbox=(1, 2, 3, 4),
                              confidence=2.0, label=None,
                              attributes={"cube_id": 2})
        self.assertEqual(detection.confidence, 1.0)
        self.assertEqual(detection.label, "")
        self.assertEqual(detection.kind_name, "cube")
        self.assertEqual(detection.cube_id, 2)
        self.assertEqual(detection.center, (2.0, 3.0))
        with self.assertRaises(AttributeError):
            _ = detection.no_such_attribute
        data = detection.to_dict()
        self.assertEqual(data["kind"], "cube:face")
        self.assertEqual(data["attributes"]["cube_id"], 2)

    def test_pose(self):
        pose = Pose2D(10.0, 20.0, 0.5)
        moved = pose.moved(10.0)
        self.assertAlmostEqual(moved.x, 10.0 + 10.0 * np.cos(0.5))
        self.assertAlmostEqual(moved.y, 20.0 + 10.0 * np.sin(0.5))
        other = Pose2D(20.0, 20.0, 0.0)
        self.assertAlmostEqual(pose.distance_to(other), 10.0)
        self.assertAlmostEqual(pose.angle_to(other), 0.0)
        self.assertAlmostEqual(pose.relative_angle_to(other), -0.5)
        composed = pose.transform(Pose2D(1.0, 0.0, 0.1))
        inverse = pose.inverse().transform(composed)
        self.assertAlmostEqual(inverse.x, 1.0)
        self.assertAlmostEqual(inverse.y, 0.0)
        self.assertAlmostEqual(inverse.yaw, 0.1)
        self.assertEqual(tuple(pose), (10.0, 20.0, 0.5))


class TestCameraModel(unittest.TestCase):

    def test_defaults_and_projection(self):
        camera = CameraModel.cozmo()
        self.assertEqual((camera.width, camera.height), (320, 240))
        self.assertAlmostEqual(np.degrees(camera.hfov), 60.0, places=4)
        camera.set_pose(pitch_deg=15.0)
        self.assertLess(camera.horizon_row(), camera.cy)
        point = camera.ground_intersection(160, 239)
        self.assertIsNotNone(point)
        self.assertAlmostEqual(point[2], 0.0)
        projected = camera.project(np.asarray([point]))[0]
        self.assertTrue(np.allclose(projected, (160.0, 239.0), atol=1e-6))
        distance = camera.ground_distance(160, 239)
        self.assertAlmostEqual(distance, np.hypot(point[0], point[1]), places=5)
        row = camera.row_for_ground_distance(distance)
        self.assertAlmostEqual(row, 239.0, places=4)
        self.assertIsNone(camera.ground_intersection(160, 0))
        self.assertAlmostEqual(camera.ground_distance_for_row(camera.horizon_row()),
                               float("inf"))

    def test_scaled_and_serialization(self):
        camera = CameraModel.cozmo()
        scaled = camera.scaled(640, 480)
        self.assertEqual((scaled.width, scaled.height), (640, 480))
        self.assertAlmostEqual(scaled.fx, camera.fx * 2.0)
        self.assertAlmostEqual(scaled.cx, camera.cx * 2.0)
        filename = os.path.join(tempfile.mkdtemp(), "camera.json")
        camera.save(filename)
        loaded = CameraModel.load(filename)
        self.assertEqual(loaded.to_dict(), camera.to_dict())
        self.assertEqual(CameraModel.from_dict(camera.to_dict()).to_dict(),
                         camera.to_dict())
        os.unlink(filename)

    def test_ground_point_to_pixel(self):
        camera = cozmo_camera()
        pixel = camera.ground_point_to_pixel(100.0, 0.0)
        self.assertIsNotNone(pixel)
        self.assertAlmostEqual(pixel[0], camera.cx)
        left = camera.ground_point_to_pixel(100.0, 50.0)
        self.assertIsNotNone(left)
        self.assertLess(left[0], camera.cx)
        right = camera.ground_point_to_pixel(100.0, -50.0)
        self.assertIsNotNone(right)
        self.assertGreater(right[0], camera.cx)
        self.assertIsNone(camera.ground_point_to_pixel(-10000.0, 0.0))

    def test_homography(self):
        source = [[0, 0], [1, 0], [1, 1], [0, 1]]
        destination = [[10, 10], [20, 10], [20, 20], [10, 20]]
        homography = estimate_homography(source, destination)
        mapped = apply_homography(homography, source)
        self.assertTrue(np.allclose(mapped, destination, atol=1e-6))
        with self.assertRaises(ValueError):
            estimate_homography([[0, 0]], [[1, 1]])

    def test_calibration_result_roundtrip(self):
        result = CalibrationResult(CameraModel.cozmo(), rms=0.5, image_count=4,
                                   per_image_errors=[0.4, 0.6])
        restored = CalibrationResult.from_dict(result.to_dict())
        self.assertAlmostEqual(restored.rms, 0.5)
        self.assertEqual(restored.image_count, 4)
        self.assertEqual(restored.per_image_errors, [0.4, 0.6])


class TestFaces(unittest.TestCase):

    def setUp(self):
        self.detector = FaceDetector(use_opencv=False)
        self.estimator = ExpressionEstimator()

    def detect(self, image):
        faces = self.detector.detect(image)
        self.assertTrue(faces)
        face = max(faces, key=lambda item: item.bbox.area)
        self.estimator.estimate(image, face)
        return face

    def test_detects_face_and_expression(self):
        neutral = self.detect(make_face())
        self.assertIsInstance(neutral, Face)
        self.assertEqual(neutral.kind, "face")
        self.assertEqual(neutral.label, "face")
        self.assertEqual(neutral.expression, "neutral")
        self.assertGreaterEqual(len(neutral.eyes), 2)
        self.assertIsNotNone(neutral.mouth_bbox)
        self.assertGreater(neutral.confidence, 0.0)

        happy = self.detect(make_face(smile=1.0))
        self.assertEqual(happy.expression, "happy")
        self.assertGreater(happy.smile, 0.35)
        sad = self.detect(make_face(smile=-1.0))
        self.assertEqual(sad.expression, "sad")
        self.assertLess(sad.smile, -0.35)
        surprised = self.detect(make_face(open_mouth=True))
        self.assertEqual(surprised.expression, "surprised")
        blinking = self.detect(make_face(blink=True))
        self.assertEqual(blinking.expression, "blinking")
        self.assertGreater(blinking.blink, 0.65)

    def test_face_serialization_and_alignment(self):
        image = make_face()
        face = self.detect(image)
        patch = align_face(image, face, (64, 64))
        self.assertEqual(patch.shape, (64, 64, 3))
        self.assertEqual(patch.dtype, np.uint8)
        data = face.to_dict()
        self.assertEqual(data["expression"], "neutral")
        self.assertIn("eyes", data)
        self.assertIn("mouth_bbox", data)

    def test_recognizer(self):
        alice = dict(eye_sep=26, eye_ry=6.0, mouth_w=22, brow_dy=45)
        bob = dict(eye_sep=19, eye_ry=8.0, mouth_w=30, brow_dy=36)
        carol = dict(eye_sep=32, eye_ry=4.0, mouth_w=16, brow_dy=52)
        images = [make_face(shift=(dx, 0), **alice) for dx in (-4, 0, 4)]
        labels = ["alice"] * 3
        images += [make_face(shift=(dx, 0), **bob) for dx in (-4, 0, 4)]
        labels += ["bob"] * 3
        recognizer = FaceRecognizer(n_components=5, threshold=0.35)
        recognizer.fit(images, labels)
        self.assertEqual(recognizer.predict(make_face(shift=(2, 1), **alice)), "alice")
        self.assertEqual(recognizer.predict(make_face(shift=(-2, 1), **bob)), "bob")
        self.assertIsNone(recognizer.predict(make_face(skin=(90, 60, 40), **carol)))

        filename = os.path.join(tempfile.mkdtemp(), "faces.npz")
        recognizer.save(filename)
        loaded = FaceRecognizer.load(filename)
        self.assertEqual(loaded.predict(make_face(shift=(2, 1), **alice)), "alice")
        os.unlink(filename)

        with self.assertRaises(ValueError):
            FaceRecognizer().fit([make_face()], [])
        with self.assertRaises(ValueError):
            FaceRecognizer().fit([], [])
        with self.assertRaises(RuntimeError):
            FaceRecognizer().predict(make_face())

    def test_tracker(self):
        first = self.detector.detect(make_face())[0]
        second = self.detector.detect(make_face(shift=(5, 3)))[0]
        tracker = FaceTracker()
        tracked_first = tracker.update([first])[0]
        tracked_second = tracker.update([second])[0]
        self.assertIsNotNone(tracked_first.track_id)
        self.assertEqual(tracked_first.track_id, tracked_second.track_id)
        tracker.reset()
        self.assertEqual(tracker.tracks, [])

    def test_detect_faces_wrapper(self):
        faces = detect_faces(make_face(), use_opencv=False)
        self.assertTrue(faces)


class TestGround(unittest.TestCase):

    def test_drivable_area(self):
        image = make_scene()
        camera = cozmo_camera()
        area = DrivableAreaEstimator(max_distance_mm=2000).estimate(
            image, camera=camera)
        self.assertGreater(area.free_ratio, 0.1)
        self.assertTrue(area.mask[220, 40])
        self.assertFalse(area.mask[170, 170])
        self.assertGreaterEqual(len(area.contacts), 1)
        self.assertGreater(area.clearance_mm, 0.0)
        self.assertLessEqual(abs(area.best_heading), camera.hfov / 2.0)
        self.assertTrue(area.obstacle_mask[170, 170])
        data = area.to_dict()
        self.assertIn("contacts", data)
        annotated = area.annotate(image)
        self.assertEqual(annotated.size, (image.shape[1], image.shape[0]))
        self.assertEqual(annotated.mode, "RGB")
        self.assertIs(area.free, area.mask)
        self.assertAlmostEqual(area.heading_degrees,
                               np.degrees(area.best_heading))

    def test_plain_floor_has_no_contacts(self):
        image = np.full((240, 320, 3), (190, 180, 160), dtype=np.uint8)
        area = DrivableAreaEstimator(max_distance_mm=2000).estimate(
            image, camera=cozmo_camera())
        self.assertGreater(area.free_ratio, 0.5)
        self.assertEqual(area.contacts, [])


class TestPetsAndRobots(unittest.TestCase):

    def test_pet_detection(self):
        image = make_scene()
        pets = PetDetector().detect(image, camera=cozmo_camera())
        self.assertEqual(len(pets), 1)
        pet = pets[0]
        self.assertIsInstance(pet, Pet)
        self.assertEqual(pet.kind, "pet")
        self.assertEqual(pet.fur_color, "brown")
        self.assertGreater(pet.confidence, 0.35)
        self.assertIsNotNone(pet.distance_mm)
        self.assertGreater(pet.bbox.iou(BoundingBox(44, 155, 97, 196)), 0.5)
        self.assertEqual(pet.to_dict()["fur_color"], "brown")
        self.assertEqual(detect_pets(image, camera=cozmo_camera())[0].kind, "pet")

    def test_robot_detection(self):
        image = make_scene()
        robots = RobotDetector().detect(image, camera=cozmo_camera())
        self.assertEqual(len(robots), 1)
        robot = robots[0]
        self.assertIsInstance(robot, Robot)
        self.assertEqual(robot.kind, "cozmo")
        self.assertEqual(robot.screen_color, "green")
        self.assertIsNotNone(robot.screen_bbox)
        self.assertGreater(robot.bbox.iou(BoundingBox(230, 130, 280, 205)), 0.5)
        self.assertEqual(robot.to_dict()["screen_color"], "green")
        self.assertEqual(detect_robots(image, camera=cozmo_camera())[0].kind,
                         "cozmo")

    def test_vector_robot(self):
        image = np.full((240, 320, 3), (190, 180, 160), dtype=np.uint8)
        image[100:130, 100:200] = (30, 220, 90)
        image[130:200, 95:205] = (35, 35, 40)
        robots = RobotDetector().detect(image, camera=cozmo_camera())
        self.assertEqual(len(robots), 1)
        self.assertEqual(robots[0].kind, "vector")

    def test_blank_scene(self):
        image = np.full((240, 320, 3), (190, 180, 160), dtype=np.uint8)
        self.assertEqual(PetDetector().detect(image, camera=cozmo_camera()), [])
        self.assertEqual(RobotDetector().detect(image, camera=cozmo_camera()), [])


class TestMotion(unittest.TestCase):

    @staticmethod
    def frames():
        first = np.full((240, 320, 3), 100, dtype=np.uint8)
        second = first.copy()
        first[40:100, 30:90] = 20
        second[40:100, 80:140] = 20
        return first, second

    def test_one_shot(self):
        first, second = self.frames()
        result = detect_motion(first, second, min_area=100)
        self.assertTrue(result.changed)
        self.assertTrue(result.motion_detected)
        self.assertGreater(result.score, 0.0)
        self.assertTrue(result.detections)
        self.assertIn("area", result.detections[0].attributes)
        self.assertTrue(result.regions)
        with self.assertRaises(ValueError):
            detect_motion(first, second[:100])

    def test_detector_and_tracker(self):
        first, second = self.frames()
        detector = MotionDetector(min_area=100)
        empty = detector.update(first)
        self.assertFalse(empty.changed)
        result = detector.update(second)
        self.assertTrue(result.changed)
        self.assertEqual(detector.frame_count, 2)
        detector.reset()
        self.assertIsNone(detector.background)

        tracker = MotionTracker(min_frames=2)
        self.assertFalse(tracker.update(first).changed)
        self.assertFalse(tracker.update(second).changed)
        self.assertTrue(tracker.update(second).changed)
        tracker.reset()

        tracker = MotionTracker(min_frames=1)
        tracker.update(first)
        self.assertTrue(tracker.update(second).changed)

    def test_global_change(self):
        first = np.full((240, 320, 3), 30, dtype=np.uint8)
        second = np.full((240, 320, 3), 220, dtype=np.uint8)
        detector = MotionDetector()
        detector.update(first)
        result = detector.update(second)
        self.assertTrue(result.global_change)
        self.assertFalse(result.detections)


class TestPipeline(unittest.TestCase):

    def test_scene_pipeline(self):
        image = make_scene()
        camera = cozmo_camera()
        config = VisionConfig(faces=False, ground=True, pets=True,
                              robots=True, track=True)
        pipeline = VisionPipeline(camera=camera, config=config)
        result = pipeline.process(image, timestamp=1.0)
        self.assertEqual(result.timestamp, 1.0)
        self.assertEqual(len(result.pets), 1)
        self.assertEqual(len(result.robots), 1)
        self.assertIsNotNone(result.ground)
        self.assertEqual(len(result.by_kind("pet")), 1)
        self.assertEqual(len(result.by_kind("cozmo")), 1)
        self.assertGreaterEqual(len(result.detections), 2)
        self.assertEqual(result.detections[0].confidence,
                         max(d.confidence for d in result.detections))
        data = result.to_dict()
        self.assertIn("ground", data)
        annotated = result.annotate(image)
        self.assertEqual(annotated.size, (image.shape[1], image.shape[0]))
        self.assertEqual(pipeline.frame_count, 1)
        pipeline.reset()
        self.assertEqual(pipeline.frame_count, 0)

    def test_face_pipeline(self):
        image = make_face()
        config = VisionConfig(faces=True, expressions=True, track=True)
        pipeline = VisionPipeline(config=config,
                                  face_detector=FaceDetector(use_opencv=False))
        result = pipeline.process(image)
        self.assertEqual(len(result.faces), 1)
        self.assertEqual(result.faces[0].expression, "neutral")
        self.assertTrue(result.detections)
        plain = result.annotate(image, show_ground=False)
        self.assertEqual(plain.mode, "RGB")

    def test_empty_result(self):
        config = VisionConfig(faces=False, track=False)
        pipeline = VisionPipeline(config=config)
        result = pipeline.process(np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertEqual(result.detections, [])
        self.assertIsNone(result.ground)
        self.assertIsNone(result.timestamp)


@unittest.skipUnless(CV2_AVAILABLE, "OpenCV is required for template detection.")
class TestCubesMarkersAruco(unittest.TestCase):

    def test_cube_detection(self):
        templates = load_cube_face_templates()
        self.assertEqual(len(templates), 18)
        template = max(templates, key=lambda item: item.image.std())
        image = np.full((240, 320, 3), 128, dtype=np.uint8)
        image[60:161, 60:161] = np.repeat(template.image[:, :, None], 3, axis=2)
        detector = CubeFaceDetector(min_confidence=0.3, scales=(1.0,),
                                    rotations=(0,), max_image_width=640)
        detections = detector.detect(image)
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection.kind, "cube")
        self.assertEqual(detection.label, "cube{}".format(template.cube_id))
        self.assertEqual(detection.cube_id, template.cube_id)
        self.assertEqual(detection.face, template.face)
        self.assertGreater(detection.confidence, 0.9)
        self.assertGreater(detection.bbox.iou(BoundingBox(60, 60, 161, 161)), 0.9)

    def test_custom_marker_render_and_detect(self):
        marker_type = CUSTOM_MARKER_TYPES[0]
        self.assertEqual(len(CUSTOM_MARKER_TYPES), 16)
        self.assertEqual(CUSTOM_MARKER_TYPES[1].marker_id, 1)
        self.assertEqual(marker_type.name, "Circle2")
        image = render_custom_marker(marker_type, 384)
        self.assertEqual(image.size, (384, 384))
        self.assertEqual(image.mode, "RGB")
        with self.assertRaises(ValueError):
            render_custom_marker(marker_type, 32)

        scene = np.full((480, 640, 3), 255, dtype=np.uint8)
        scene[48:432, 128:512] = np.asarray(image)
        detections = CustomMarkerDetector(min_confidence=0.1).detect(scene)
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection.kind, "marker")
        self.assertEqual(detection.label, marker_type.name)
        self.assertEqual(detection.marker_id, marker_type.marker_id)
        self.assertEqual(detection.count, marker_type.count)
        self.assertGreater(detection.confidence, 0.5)
        self.assertEqual(CustomMarkerDetector(use_opencv=False).detect(scene), [])

    @unittest.skipUnless(cv2 is not None and hasattr(cv2, "aruco"),
                         "cv2.aruco is required for ArUco detection.")
    def test_aruco_detection(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, "generateImageMarker"):
            marker = cv2.aruco.generateImageMarker(dictionary, 7, 200)
        else:  # pragma: no cover - older OpenCV
            marker = np.zeros((200, 200), dtype=np.uint8)
            cv2.aruco.drawMarker(dictionary, 7, 200, marker, 1)
        image = np.full((300, 300, 3), 255, dtype=np.uint8)
        image[50:250, 50:250, :] = marker[:, :, None]
        detector = ArucoDetector()
        self.assertTrue(detector.enabled)
        detections = detector.detect(image)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].kind, "aruco")
        self.assertEqual(detections[0].label, "7")
        self.assertEqual(detections[0].marker_id, 7)
        self.assertEqual(ArucoDetector(use_opencv=False).detect(image), [])


if __name__ == "__main__":
    unittest.main()
