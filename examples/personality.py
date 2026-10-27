#!/usr/bin/env python

"""Demonstrate the personality engine without a robot."""

import tempfile

from pycozmo import personality


def main() -> int:
    mood = personality.Personality.playful()

    # Simulate ten seconds of time passing, then a few events.
    mood.update(dt=10.0)
    mood.react("EvtFaceObserved")
    mood.react("EvtCubeTapped")
    mood.satisfy("social", 0.2)

    print("mood:", round(mood.mood, 3))
    print("dominant drive:", mood.dominant_drive)
    print("dominant emotion:", mood.dominant_emotion)
    print("drives:", {name: round(drive.value, 2)
                      for name, drive in mood.drives.items()})

    activities = ["Freeplay", "Socialize", "Rest", "Explore"]
    print("chosen activity:", mood.choose_activity(activities))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as stream:
        filename = stream.name
    try:
        mood.save(filename)
        restored = personality.Personality.load(filename)
        print("restored mood:", round(restored.mood, 3))
    finally:
        import os
        os.unlink(filename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
