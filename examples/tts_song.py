#!/usr/bin/env python

"""Speak text and play a generated song on Cozmo."""

import time

import pycozmo
from pycozmo import songs, tts


def main() -> int:
    speech = tts.TextToSpeech()
    print("Available TTS backends:", speech.backends)

    with pycozmo.connect() as cli:
        speech.say(cli, "Hello from PyCozmo.")
        time.sleep(1.0)
        songs.play(cli, "C4/4 D4/4 E4/4 G4/4 C5/2")
        time.sleep(3.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
