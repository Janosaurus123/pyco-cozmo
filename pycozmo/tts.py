"""

Cross-platform text-to-speech support for PyCozmo.

The Cozmo speaker accepts 22 kHz (or 48 kHz), 16-bit, mono WAVE audio.  This
module turns text into WAVE bytes using whichever speech engine is available on
the host:

* ``espeak-ng`` / ``espeak`` on Linux, macOS and Windows (if installed)
* Windows SAPI through PowerShell
* macOS ``say``
* ``pyttsx3`` (optional Python package)

The generated audio can be passed directly to
:meth:`pycozmo.client.Client.play_audio` or saved to a file.

"""

from typing import Callable, List, Optional
import io
import os
import shutil
import subprocess
import tempfile
import wave

from . import logger


__all__ = [
    "AudioBackendError",
    "available_backends",
    "synthesize_wav",
    "TextToSpeech",
]


COZMO_RATE = 22050


class AudioBackendError(RuntimeError):
    """Raised when no usable text-to-speech backend is available."""


Backend = Callable[[str, str, Optional[str], Optional[int]], None]


def _run(cmd: List[str]) -> None:
    logger.debug("Running TTS command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _espeak_backend(executable: str) -> Backend:
    def backend(text: str, out_path: str, voice: Optional[str], rate: Optional[int]) -> None:
        cmd = [executable, "-w", out_path]
        if voice:
            cmd += ["-v", voice]
        if rate:
            cmd += ["-s", str(rate)]
        cmd.append(text)
        _run(cmd)

    return backend


def _windows_sapi_backend(text: str, out_path: str,
                          voice: Optional[str], rate: Optional[int]) -> None:
    rate_value = 0 if rate is None else max(-10, min(10, int((rate - 175) / 20)))
    voice_select = ""
    if voice:
        safe_voice = voice.replace("'", "''")
        voice_select = f"$s.SelectVoice('{safe_voice}');"
    safe_path = out_path.replace("'", "''")
    safe_text = text.replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.Rate = {rate_value};"
        f"{voice_select}"
        f"$s.SetOutputToWaveFile('{safe_path}');"
        f"$s.Speak('{safe_text}');"
        "$s.Dispose();"
    )
    powershell = shutil.which("powershell") or shutil.which("powershell.exe")
    if not powershell:
        raise AudioBackendError("PowerShell is required for the Windows SAPI backend.")
    _run([powershell, "-NoProfile", "-NonInteractive", "-Command", script])


def _macos_say_backend(text: str, out_path: str,
                       voice: Optional[str], rate: Optional[int]) -> None:
    cmd = ["say", "-o", out_path, "--data-format=LEI16@22050"]
    if voice:
        cmd += ["-v", voice]
    if rate:
        cmd += ["-r", str(rate)]
    cmd.append(text)
    _run(cmd)


def _pyttsx3_backend(text: str, out_path: str,
                     voice: Optional[str], rate: Optional[int]) -> None:
    try:
        import pyttsx3
    except ImportError as e:
        raise AudioBackendError("pyttsx3 is not installed.") from e
    engine = pyttsx3.init()
    if voice:
        for candidate in engine.getProperty("voices"):
            if voice.lower() in (candidate.id or "").lower() or \
                    voice.lower() in (candidate.name or "").lower():
                engine.setProperty("voice", candidate.id)
                break
    if rate:
        engine.setProperty("rate", rate)
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    engine.stop()


def _backend_map():
    backends = {}
    espeak_ng = shutil.which("espeak-ng")
    espeak = shutil.which("espeak")
    if espeak_ng:
        backends["espeak-ng"] = _espeak_backend(espeak_ng)
    if espeak:
        backends["espeak"] = _espeak_backend(espeak)
    if os.name == "nt":
        backends["sapi"] = _windows_sapi_backend
    if shutil.which("say"):
        backends["say"] = _macos_say_backend
    try:
        import pyttsx3  # noqa: F401
    except ImportError:
        pass
    else:
        backends["pyttsx3"] = _pyttsx3_backend
    return backends


def available_backends() -> List[str]:
    """Return the names of usable TTS backends."""
    return list(_backend_map().keys())


def _read_wav(data: bytes) -> tuple:
    with wave.open(io.BytesIO(data), "rb") as w:
        params = (w.getnchannels(), w.getsampwidth(), w.getframerate())
        frames = w.readframes(w.getnframes())
    return params, frames


def _write_wav(frames: bytes, channels: int, sampwidth: int, rate: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(frames)
    return out.getvalue()


def normalize_wav(data: bytes, rate: int = COZMO_RATE) -> bytes:
    """Normalize WAVE data to 16-bit mono at a Cozmo-compatible rate.

    Cozmo accepts 22.05 kHz and 48 kHz.  Input with other rates is linearly
    resampled to 22.05 kHz.  Stereo input is downmixed to mono.
    """
    import struct

    (channels, sampwidth, source_rate), frames = _read_wav(data)
    if sampwidth != 2:
        raise ValueError("Only 16-bit WAVE audio is supported.")

    samples = list(struct.unpack("<{}h".format(len(frames) // 2), frames))
    if channels > 1:
        samples = [
            int(sum(samples[i:i + channels]) / channels)
            for i in range(0, len(samples), channels)
        ]
    if source_rate not in (COZMO_RATE, 48000):
        if not samples:
            return _write_wav(b"", 1, 2, rate)
        ratio = rate / float(source_rate)
        count = max(1, int(len(samples) * ratio))
        resampled = []
        for i in range(count):
            pos = i / ratio
            idx = int(pos)
            frac = pos - idx
            a = samples[min(idx, len(samples) - 1)]
            b = samples[min(idx + 1, len(samples) - 1)]
            resampled.append(int(a + (b - a) * frac))
        samples = resampled
        source_rate = rate

    if not samples:
        return _write_wav(b"", 1, 2, source_rate)
    out_frames = struct.pack("<{}h".format(len(samples)), *samples)
    return _write_wav(out_frames, 1, 2, source_rate)


def synthesize_wav(text: str,
                   voice: Optional[str] = None,
                   rate: Optional[int] = None,
                   backend: Optional[str] = None,
                   normalize: bool = True) -> bytes:
    """Synthesize ``text`` and return Cozmo-compatible WAVE bytes.

    Args:
        text: Text to speak.
        voice: Optional voice name or identifier.
        rate: Optional speech rate (backend specific; words per minute for
            most engines).
        backend: Force a backend by name.  By default the first available
            backend is used.
        normalize: Convert the output to 16-bit mono at 22.05 kHz.
    """
    if not text or not text.strip():
        raise ValueError("Text must not be empty.")

    backends = _backend_map()
    if not backends:
        raise AudioBackendError(
            "No text-to-speech backend found. Install espeak-ng, pyttsx3, or "
            "use a platform with SAPI/say support.")

    if backend is not None:
        if backend not in backends:
            raise AudioBackendError(
                "Unknown or unavailable TTS backend '{}'. Available: {}".format(
                    backend, ", ".join(sorted(backends))))
        selected = [(backend, backends[backend])]
    else:
        selected = list(backends.items())

    last_error = None
    for name, func in selected:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            func(text, path, voice, rate)
            if os.path.getsize(path) == 0:
                raise AudioBackendError("Backend '{}' produced no audio.".format(name))
            with open(path, "rb") as f:
                data = f.read()
            return normalize_wav(data) if normalize else data
        except Exception as e:  # Try the next backend.
            last_error = e
            logger.debug("TTS backend '%s' failed: %s", name, e)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    raise AudioBackendError("All TTS backends failed: {}".format(last_error))


class TextToSpeech:
    """Convenience wrapper around :func:`synthesize_wav`.

    Args:
        voice: Default voice name.
        rate: Default speech rate.
        backend: Default backend name.
    """

    def __init__(self, voice: Optional[str] = None,
                 rate: Optional[int] = None,
                 backend: Optional[str] = None) -> None:
        self.voice = voice
        self.rate = rate
        self.backend = backend

    @property
    def backends(self) -> List[str]:
        return available_backends()

    def synthesize(self, text: str, **kwargs) -> bytes:
        options = {
            "voice": self.voice,
            "rate": self.rate,
            "backend": self.backend,
        }
        options.update(kwargs)
        return synthesize_wav(text, **options)

    def save(self, text: str, path: str, **kwargs) -> str:
        data = self.synthesize(text, **kwargs)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def say(self, cli, text: str, **kwargs) -> None:
        """Synthesize and play ``text`` on a connected client."""
        cli.play_audio(self.synthesize(text, **kwargs))

    def speak(self, cli, text: str, **kwargs) -> None:
        """Alias for :meth:`say`."""
        self.say(cli, text, **kwargs)
