"""

Voice controller: speech recognition, command parsing and execution.

"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from .commands import Command, CommandExecutor, CommandParser, CommandResult
from .recognition import SpeechRecognizer


__all__ = [
    "VoiceController",
]


class VoiceController:
    """ Connect a microphone, parser and robot client.

    Args:
        client: Cozmo-like client used to execute commands.
        parser: Optional custom :class:`CommandParser`.
        recognizer: Optional :class:`SpeechRecognizer`; when omitted, speech
            recognition is created on first use.
        on_command: Optional callback receiving ``(command, result)``.
        on_text: Optional callback receiving raw recognized text.
        sleep: Sleep function passed to the command executor (useful in tests).
    """

    def __init__(self,
                 client: Any,
                 parser: Optional[CommandParser] = None,
                 recognizer: Optional[SpeechRecognizer] = None,
                 on_command: Optional[Callable[[Command, CommandResult], Any]] = None,
                 on_text: Optional[Callable[[str], Any]] = None,
                 sleep: Optional[Callable[[float], None]] = None) -> None:
        self.client = client
        self.parser = parser or CommandParser()
        self.recognizer = recognizer
        self.on_command = on_command
        self.on_text = on_text
        if sleep is None:
            import time
            sleep = time.sleep
        self.executor = CommandExecutor(client, sleep=sleep)
        self._stop = threading.Event()

    def ensure_recognizer(self, **kwargs: Any) -> SpeechRecognizer:
        if self.recognizer is None:
            self.recognizer = SpeechRecognizer(**kwargs)
        return self.recognizer

    def handle_text(self, text: str, confidence: float = 1.0) -> Optional[CommandResult]:
        """ Parse and execute text.  Returns ``None`` when nothing matched. """
        command = self.parser.parse(text, confidence=confidence)
        if command is None:
            return None
        result = self.executor.execute(command)
        if self.on_command is not None:
            self.on_command(command, result)
        # Notify listeners on the client when it supports event dispatch.
        dispatch = getattr(self.client, "dispatch", None)
        if callable(dispatch):
            from .. import event
            dispatch(event.EvtVoiceCommand, self.client, command, result)
        return result

    def listen_once(self, **kwargs: Any) -> Optional[CommandResult]:
        """ Recognize one phrase and execute it. """
        recognizer = self.ensure_recognizer()
        text = recognizer.listen(**kwargs)
        if not text:
            return None
        if self.on_text is not None:
            self.on_text(text)
        return self.handle_text(text)

    def run_forever(self,
                    stop_event: Optional[threading.Event] = None,
                    **kwargs: Any) -> None:
        """ Listen and execute commands until ``stop_event`` is set. """
        stop_event = stop_event or self._stop
        recognizer = self.ensure_recognizer()

        def callback(text: Optional[str]) -> None:
            if not text:
                return
            if self.on_text is not None:
                self.on_text(text)
            self.handle_text(text)

        recognizer.listen_forever(callback, stop_event=stop_event, **kwargs)

    def stop(self) -> None:
        self._stop.set()
