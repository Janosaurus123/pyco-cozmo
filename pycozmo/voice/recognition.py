"""

Optional speech recognition backend.

This module imports the third-party ``SpeechRecognition`` package lazily, so
the base PyCozmo package remains importable without a microphone stack.  The
Google recognizer used by default requires network access; other recognizer
methods supported by ``SpeechRecognition`` can be selected by name.

"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional


__all__ = [
    "VoiceRecognitionError",
    "SpeechRecognizer",
    "is_available",
]


class VoiceRecognitionError(RuntimeError):
    """ Raised when speech recognition is unavailable or fails. """


def is_available() -> bool:
    """ Return whether the optional ``SpeechRecognition`` package is installed. """
    try:
        import speech_recognition  # noqa: F401
        return True
    except ImportError:
        return False


class SpeechRecognizer:
    """ Thin wrapper around :mod:`speech_recognition`.

    Args:
        language: BCP-47 language tag, for example ``"en-US"``.
        backend: Recognizer method suffix, for example ``"google"``,
            ``"sphinx"``, ``"whisper"`` or ``"vosk"``.
        energy_threshold: Initial microphone energy threshold.
        pause_threshold: Seconds of silence that end a phrase.
        recognizer: Optional pre-configured ``speech_recognition.Recognizer``.
    """

    def __init__(self,
                 language: str = "en-US",
                 backend: str = "google",
                 energy_threshold: int = 300,
                 pause_threshold: float = 0.8,
                 recognizer: Optional[Any] = None) -> None:
        try:
            import speech_recognition as sr
        except ImportError as exc:
            raise VoiceRecognitionError(
                "Speech recognition requires the optional 'SpeechRecognition' "
                "package. Install it with: pip install pycozmo[voice]"
            ) from exc
        self._sr = sr
        self.language = str(language)
        self.backend = str(backend)
        self.recognizer = recognizer or sr.Recognizer()
        self.recognizer.energy_threshold = int(energy_threshold)
        self.recognizer.pause_threshold = float(pause_threshold)

    @property
    def microphone_available(self) -> bool:
        try:
            self._sr.Microphone.list_microphone_names()
            return True
        except Exception:
            return False

    def _recognize(self, audio: Any) -> str:
        method = getattr(self.recognizer, "recognize_{}".format(self.backend), None)
        if method is None:
            raise VoiceRecognitionError(
                "Unknown speech recognizer backend '{}'.".format(self.backend))
        try:
            return str(method(audio, language=self.language))
        except self._sr.UnknownValueError:
            return ""
        except self._sr.RequestError as exc:
            raise VoiceRecognitionError("Speech recognizer request failed: {}".format(exc)) from exc

    def listen(self,
               timeout: float = 5.0,
               phrase_time_limit: Optional[float] = 5.0,
               adjust_for_ambient_noise: bool = True,
               microphone: Optional[Any] = None) -> Optional[str]:
        """ Record one phrase and return recognized text, or ``None``.

        ``None`` means that no speech was heard; an empty string means that
        speech was heard but could not be recognized.
        """
        if microphone is None:
            microphone = self._sr.Microphone()
        with microphone as source:
            if adjust_for_ambient_noise:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
            try:
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            except self._sr.WaitTimeoutError:
                return None
        text = self._recognize(audio).strip()
        return text or ""

    def recognize_file(self, filename: str, **kwargs: Any) -> Optional[str]:
        """ Recognize speech in an audio file supported by SpeechRecognition. """
        with self._sr.AudioFile(filename) as source:
            audio = self.recognizer.record(source, **kwargs)
        text = self._recognize(audio).strip()
        return text or ""

    def recognize_audio(self, audio: Any) -> Optional[str]:
        """ Recognize an existing ``AudioData`` object. """
        text = self._recognize(audio).strip()
        return text or ""

    def listen_forever(self,
                       callback: Callable[[Optional[str]], Any],
                       stop_event: Optional[threading.Event] = None,
                       timeout: float = 5.0,
                       phrase_time_limit: Optional[float] = 5.0,
                       adjust_for_ambient_noise: bool = True,
                       microphone: Optional[Any] = None) -> None:
        """ Call ``callback`` with each recognized phrase until stopped. """
        stop_event = stop_event or threading.Event()
        microphone = microphone or self._sr.Microphone()
        first = True
        while not stop_event.is_set():
            try:
                text = self.listen(timeout=timeout,
                                   phrase_time_limit=phrase_time_limit,
                                   adjust_for_ambient_noise=adjust_for_ambient_noise and first,
                                   microphone=microphone)
                first = False
            except VoiceRecognitionError:
                callback(None)
                if stop_event.wait(0.5):
                    break
                continue
            if text:
                callback(text)
