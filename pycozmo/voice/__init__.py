"""

Voice command support.

The command parser and executor have no mandatory third-party dependencies.
Speech recognition is optional and uses the ``SpeechRecognition`` package::

    from pycozmo.voice import VoiceController

    controller = VoiceController(cli)
    controller.run_forever()

"""

from .commands import (
    Command,
    CommandExecutor,
    CommandParser,
    CommandResult,
    CommandRule,
    VoiceCommandError,
    execute_command,
    parse_command,
    parse_number,
)
from .recognition import SpeechRecognizer, VoiceRecognitionError, is_available
from .controller import VoiceController


__all__ = [
    "Command",
    "CommandExecutor",
    "CommandParser",
    "CommandResult",
    "CommandRule",
    "VoiceCommandError",
    "SpeechRecognizer",
    "VoiceRecognitionError",
    "VoiceController",
    "execute_command",
    "parse_command",
    "parse_number",
    "is_available",
]
