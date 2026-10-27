
import unittest

import numpy as np
from PIL import Image

from pycozmo.vision import (BoundingBox, CameraModel, ChargerDetector,
                            Platform, PlatformDetector, detect_platforms)


def make_charger_image(width=320, height=240):
    image = np.full((height, width, 3), (140, 140, 140), dtype=np.uint8)
    image[150:200, 120:220] = (50, 50, 50)
    image[175:180, 140:150] = (0, 0, 255)
    image[175:180, 190:200] = (0, 0, 255)
    return image


def make_mat_image(width=320, height=240):
    image = np.full((height, width, 3), (140, 140, 140), dtype=np.uint8)
    image[90:170, 40:280] = (200, 30, 30)
    return image


class TestPlatformDetector(unittest.TestCase):

    def test_detects_charger(self):
        image = make_charger_image()
        detections = PlatformDetector().detect(image)
        self.assertEqual(len(detections), 1)
        platform = detections[0]
        self.assertIsInstance(platform, Platform)
        self.assertEqual(platform.kind, "platform")
        self.assertEqual(platform.platform_type, "charger")
        self.assertEqual(platform.led_count, 2)
        self.assertGreaterEqual(platform.confidence, 0.35)
        self.assertLessEqual(platform.bbox.x1, 140)
        self.assertGreaterEqual(platform.bbox.x2, 200)
        self.assertEqual(platform.led_color, (0, 0, 255))
        self.assertEqual(len(platform.led_boxes), 2)
        self.assertIsNotNone(platform.distance_mm)

    def test_detects_with_pillow_and_camera(self):
        image = Image.fromarray(make_charger_image())
        camera = CameraModel.cozmo()
        detections = PlatformDetector().detect(image, camera=camera,
                                               on_charger=True)
        self.assertEqual(len(detections), 1)
        self.assertTrue(detections[0].on_charger)

    def test_blank_image(self):
        image = np.full((240, 320, 3), (140, 140, 140), dtype=np.uint8)
        self.assertEqual(PlatformDetector().detect(image), [])

    def test_small_isolated_reflection_is_rejected(self):
        image = np.full((240, 320, 3), (140, 140, 140), dtype=np.uint8)
        image[175:178, 140:143] = (0, 0, 255)
        self.assertEqual(PlatformDetector().detect(image), [])

    def test_detect_platforms_wrapper(self):
        detections = detect_platforms(make_charger_image())
        self.assertEqual(len(detections), 1)
        self.assertEqual(ChargerDetector, PlatformDetector)

    def test_play_mat(self):
        image = make_mat_image()
        self.assertEqual(PlatformDetector().detect(image), [])
        detections = PlatformDetector(detect_mats=True).detect(image)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].platform_type, "play_mat")
        self.assertEqual(detections[0].led_count, 0)

    def test_downscaling(self):
        # Scale the synthetic charger to a larger frame.
        image = np.full((600, 800, 3), (140, 140, 140), dtype=np.uint8)
        image[375:500, 300:550] = (50, 50, 50)
        image[437:450, 350:375] = (0, 0, 255)
        image[437:450, 475:500] = (0, 0, 255)
        detector = PlatformDetector(max_image_width=400)
        detections = detector.detect(image)
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detector._scale, 0.5)

    def test_platform_serialization(self):
        platform = Platform(kind="platform", label="charger",
                            platform_type="charger", confidence=0.8,
                            bbox=BoundingBox(120, 150, 220, 200),
                            led_count=2, led_color=(0, 0, 255),
                            on_charger=False)
        data = platform.to_dict()
        self.assertEqual(data["platform_type"], "charger")
        self.assertEqual(data["led_color"], [0, 0, 255])
        self.assertEqual(data["led_count"], 2)
        self.assertFalse(data["on_charger"])


if __name__ == "__main__":
    unittest.main()
