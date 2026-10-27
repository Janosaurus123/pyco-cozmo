#!/usr/bin/env python

"""Load a Vector ``.bin`` animation and play one clip.

Without ``--client`` the clip is played through handler callbacks, which is
useful for inspecting animations offline.  With ``--client`` the common tracks
are sent to a connected Cozmo using Cozmo-compatible client methods.
"""

import argparse

import pycozmo
from pycozmo.vector import VectorAnimator, is_vector_animation, load_vector_animations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", help="Vector animation .bin or JSON file.")
    parser.add_argument("--clip", default=None, help="Clip name (default: first).")
    parser.add_argument("--list", action="store_true", help="List clips and exit.")
    parser.add_argument("--client", action="store_true", help="Play on Cozmo.")
    args = parser.parse_args()

    if not is_vector_animation(args.filename):
        print("This does not look like a Vector animation file.")
        return 1

    animations = load_vector_animations(args.filename)
    if args.list:
        for clip in animations:
            print("{:<40} {:>7.2f} s  {} keyframes".format(
                clip.name, clip.duration, len(clip)))
        return 0

    clip = animations.get(args.clip) if args.clip else animations.clips[0]
    if clip is None:
        print("Clip {!r} not found.".format(args.clip))
        return 1

    handlers = {
        "event": lambda event_id: print("event:", event_id),
        "sprite": lambda keyframe: print("sprite:", keyframe.asset_name),
        "audio": lambda keyframe: print("audio groups:", len(keyframe.event_groups)),
        "keyframe": lambda keyframe: print(type(keyframe).__name__,
                                           keyframe.trigger_time_ms),
    }

    if args.client:
        with pycozmo.connect() as cli:
            VectorAnimator(clip).play(client=cli)
    else:
        VectorAnimator(clip).play(handlers=handlers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
