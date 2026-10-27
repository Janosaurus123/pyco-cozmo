#!/usr/bin/env python

"""Run the off-board vision pipeline on one Cozmo camera frame."""

import argparse
import time

import numpy as np

import pycozmo
from pycozmo import vision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="vision.png")
    parser.add_argument("--no-faces", action="store_true")
    parser.add_argument("--cubes", action="store_true")
    parser.add_argument("--markers", action="store_true")
    parser.add_argument("--aruco", action="store_true")
    args = parser.parse_args()

    config = vision.VisionConfig(
        faces=not args.no_faces,
        cubes=args.cubes,
        markers=args.markers,
        aruco=args.aruco,
        platforms=True,
        ground=True,
    )
    pipeline = vision.VisionPipeline(config=config)

    with pycozmo.connect() as cli:
        cli.set_head_angle(angle=0.3)
        cli.enable_camera(enable=True, color=True)
        time.sleep(2.0)
        image = cli.latest_image
        if image is None:
            print("No camera frame received.")
            return 1
        result = pipeline.process(np.array(image))
        result.annotate(image).save(args.output)
        print("Saved {} with {} detections.".format(args.output, len(result.detections)))
        for detection in result.detections:
            print("  {} {:.2f} {}".format(
                detection.kind_name, detection.confidence, detection.bbox))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
