"""

Simple songs and tone synthesis for the Cozmo speaker.

Cozmo plays 22 kHz, 16-bit, mono WAVE audio.  This module provides a tiny
notation language and a pure-Python synthesizer so applications can play
melodies without shipping external audio files.

Example::

    from pycozmo import songs

    song = songs.parse("C4/4 D4/4 E4/4 G4/4 C5/2")
    songs.play(cli, song)

Notation
--------
``NOTE/LENGTH`` where ``NOTE`` is a scientific pitch name (``C4``, ``F#5``,
``Bb3``) and ``LENGTH`` is a duration in beats (``1``, ``0.5``, ``1/4``).
``R`` or ``-`` is a rest.  A trailing ``~`` marks a tie with the next note.
"""

from typing import Iterable, List, Optional, Union
import io
import re
import wave

import numpy as np


__all__ = [
    "Note",
    "Song",
    "parse",
    "synthesize",
    "play",
    "note_frequency",
]


_NOTE_RE = re.compile(r"^(?P<name>[A-Ga-g])(?P<acc>[#b]?)(?P<octave>-?\d)$")
_TOKEN_RE = re.compile(r"^(?P<name>R|-|[A-Ga-g][#b]?-?\d)(?:/(?P<len>\d+(?:\.\d+)?))?(?P<tie>~?)$")


class Note:
    """A single note or rest.

    Args:
        name: Scientific pitch name, or ``None``/``"R"`` for a rest.
        beats: Duration in beats.
        tie: Whether the note is tied to the following note.
    """

    __slots__ = ("name", "beats", "tie")

    def __init__(self, name: Optional[str], beats: float = 1.0, tie: bool = False):
        self.name = None if name in (None, "", "R", "-") else str(name)
        self.beats = float(beats)
        self.tie = bool(tie)

    def __repr__(self):
        return "Note({!r}, beats={}, tie={})".format(self.name, self.beats, self.tie)

    @property
    def frequency(self) -> Optional[float]:
        return None if self.name is None else note_frequency(self.name)


class Song:
    """A sequence of notes.

    Args:
        notes: Notes in the song.
        tempo: Beats per minute.
        waveform: ``sine``, ``square``, ``triangle`` or ``saw``.
        volume: Linear volume in the range 0..1.
    """

    def __init__(self, notes: Iterable[Note], tempo: float = 120.0,
                 waveform: str = "sine", volume: float = 0.35):
        self.notes = list(notes)
        self.tempo = float(tempo)
        self.waveform = str(waveform)
        self.volume = max(0.0, min(1.0, float(volume)))

    def __iter__(self):
        return iter(self.notes)

    def __len__(self):
        return len(self.notes)

    @property
    def duration(self) -> float:
        return sum(n.beats for n in self.notes) * 60.0 / self.tempo


def note_frequency(name: str) -> float:
    """Return the frequency of a scientific pitch name in Hz."""
    match = _NOTE_RE.match(name)
    if not match:
        raise ValueError("Invalid note name: {!r}".format(name))
    letter = match.group("name").upper()
    accidental = match.group("acc")
    octave = int(match.group("octave"))
    semitones = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
    semitones += {"#": 1, "b": -1, "": 0}[accidental]
    midi = (octave + 1) * 12 + semitones
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _duration_beats(value: str) -> float:
    if "/" in value:
        num, den = value.split("/", 1)
        return float(num) / float(den)
    return float(value)


def parse(notation: str, tempo: float = 120.0, **kwargs) -> Song:
    """Parse a whitespace separated melody into a :class:`Song`."""
    notes: List[Note] = []
    for token in notation.split():
        match = _TOKEN_RE.match(token)
        if not match:
            raise ValueError("Invalid note token: {!r}".format(token))
        name = match.group("name")
        length = 1.0 if match.group("len") is None else _duration_beats(match.group("len"))
        notes.append(Note(name, length, tie=bool(match.group("tie"))))
    return Song(notes, tempo=tempo, **kwargs)


def _waveform(phase: np.ndarray, waveform: str) -> np.ndarray:
    if waveform == "sine":
        return np.sin(phase)
    if waveform == "square":
        return np.sign(np.sin(phase))
    if waveform == "triangle":
        return 2.0 / np.pi * np.arcsin(np.sin(phase))
    if waveform == "saw":
        return 2.0 * (phase / (2.0 * np.pi) - np.floor(phase / (2.0 * np.pi) + 0.5))
    raise ValueError("Unknown waveform: {!r}".format(waveform))


def synthesize(song: Song, rate: int = 22050) -> bytes:
    """Render a song to 16-bit mono WAVE bytes."""
    total_beats = sum(n.beats for n in song.notes)
    total_samples = max(1, int(round(total_beats * 60.0 / song.tempo * rate)))
    samples = np.zeros(total_samples, dtype=np.float64)

    pos = 0
    i = 0
    notes = song.notes
    while i < len(notes):
        note = notes[i]
        beats = note.beats
        # Combine tied notes into a single sustained tone.
        while note.tie and i + 1 < len(notes) and notes[i + 1].name == note.name:
            i += 1
            beats += notes[i].beats
        count = max(0, int(round(beats * 60.0 / song.tempo * rate)))
        if note.frequency is not None and count:
            t = np.arange(count, dtype=np.float64) / rate
            phase = 2.0 * np.pi * note.frequency * t
            tone = _waveform(phase, song.waveform)
            # Short attack/release envelope to avoid clicks.
            envelope = np.ones(count, dtype=np.float64)
            fade = min(int(rate * 0.01), count // 2)
            if fade > 0:
                envelope[:fade] = np.linspace(0.0, 1.0, fade)
                envelope[-fade:] = np.linspace(1.0, 0.0, fade)
            end = min(total_samples, pos + count)
            samples[pos:end] += tone[:end - pos] * envelope[:end - pos]
        pos += count
        i += 1

    samples = np.clip(samples * song.volume, -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return out.getvalue()


def play(cli, song: Union[Song, str], tempo: float = 120.0) -> None:
    """Synthesize and play a song on a connected client."""
    if isinstance(song, str):
        song = parse(song, tempo=tempo)
    cli.play_audio(synthesize(song))
