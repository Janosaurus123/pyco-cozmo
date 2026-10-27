#!/usr/bin/env python

"""Parse and execute typed voice commands.

Pass ``--listen`` to use a microphone when the optional ``SpeechRecognition``
package and a microphone are available.
"""

import argparse

import pycozmo
from pycozmo.voice import CommandExecutor, VoiceController, parse_command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="*", help="Command text to execute.")
    parser.add_argument("--listen", action="store_true", help="Use the microphone.")
    args = parser.parse_args()

    with pycozmo.connect() as cli:
        if args.listen:
            VoiceController(cli).run_forever()
            return 0

        text = " ".join(args.command) or "turn left 90 degrees"
        command = parse_command(text)
        if command is None:
            print("Could not parse: {!r}".format(text))
            return 1
        print("Parsed {} {}".format(command.name, command.parameters))
        result = CommandExecutor(cli).execute(command)
        print(result.message if result.success else result.error)
        return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
