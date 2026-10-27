
import io
import math
import os
import struct
import tempfile
import unittest
import wave
from unittest import mock

from pycozmo import songs, tts


def _wav_bytes(samples, rate=22050, channels=1, sampwidth=2):
    out = io.BytesIO()
    with wave.open(out, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(sampwidth)
        stream.setframerate(rate)
        if sampwidth == 2:
            stream.writeframes(struct.pack("<{}h".format(len(samples)), *samples))
        else:
            stream.writeframes(bytes(samples))
    return out.getvalue()


class TestSongs(unittest.TestCase):

    def test_note_frequency(self):
        self.assertAlmostEqual(songs.note_frequency("A4"), 440.0)
        self.assertAlmostEqual(songs.note_frequency("C4"), 261.625565, places=4)
        self.assertAlmostEqual(songs.note_frequency("C#4"),
                               songs.note_frequency("Db4"))
        self.assertAlmostEqual(songs.note_frequency("A5"), 880.0)
        with self.assertRaises(ValueError):
            songs.note_frequency("H4")

    def test_parse(self):
        song = songs.parse("C4/4 D4/4 E4/4 G4/4 C5/2")
        self.assertEqual(len(song), 5)
        self.assertAlmostEqual(song.duration, 9.0)
        self.assertEqual(song.notes[0].name, "C4")
        self.assertAlmostEqual(song.notes[0].beats, 4.0)
        self.assertAlmostEqual(song.notes[-1].frequency, songs.note_frequency("C5"))

    def test_parse_rest_and_tie(self):
        song = songs.parse("C4/1~ C4/1 R/2 -/2")
        self.assertEqual(len(song), 4)
        self.assertTrue(song.notes[0].tie)
        self.assertIsNone(song.notes[2].name)
        self.assertIsNone(song.notes[3].frequency)

    def test_parse_invalid(self):
        with self.assertRaises(ValueError):
            songs.parse("H4/1")
        with self.assertRaises(ValueError):
            songs.parse("C4/x")

    def test_synthesize(self):
        song = songs.parse("C4/4 D4/4", waveform="sine", volume=0.5)
        data = songs.synthesize(song)
        with wave.open(io.BytesIO(data), "rb") as stream:
            self.assertEqual(stream.getnchannels(), 1)
            self.assertEqual(stream.getsampwidth(), 2)
            self.assertEqual(stream.getframerate(), 22050)
            self.assertEqual(stream.getnframes(), 88200)

    def test_synthesize_tie_holds_note(self):
        tied = songs.parse("C4/1~ C4/1")
        separate = songs.parse("C4/1 C4/1")
        self.assertEqual(len(songs.synthesize(tied)), len(songs.synthesize(separate)))
        # The tied version has no gap/zero crossing in the middle; compare
        # the middle sample energy.
        tied_data = songs.synthesize(tied)
        separate_data = songs.synthesize(separate)
        with wave.open(io.BytesIO(tied_data), "rb") as stream:
            tied_frames = struct.unpack("<{}h".format(stream.getnframes()),
                                        stream.readframes(stream.getnframes()))
        with wave.open(io.BytesIO(separate_data), "rb") as stream:
            separate_frames = struct.unpack("<{}h".format(stream.getnframes()),
                                            stream.readframes(stream.getnframes()))
        middle = len(tied_frames) // 2
        self.assertGreater(abs(tied_frames[middle]), 1000)
        self.assertEqual(len(tied_frames), len(separate_frames))

    def test_all_waveforms(self):
        for waveform in ("sine", "square", "triangle", "saw"):
            data = songs.synthesize(songs.parse("C4/1", waveform=waveform))
            self.assertTrue(data.startswith(b"RIFF"))
        with self.assertRaises(ValueError):
            songs.synthesize(songs.parse("C4/1", waveform="noise"))

    def test_play(self):
        class Client:
            def __init__(self):
                self.audio = []

            def play_audio(self, data):
                self.audio.append(data)

        client = Client()
        songs.play(client, "C4/1 D4/1")
        self.assertEqual(len(client.audio), 1)
        self.assertTrue(client.audio[0].startswith(b"RIFF"))
        self.assertGreater(len(client.audio[0]), 1000)


class TestTTS(unittest.TestCase):

    def test_normalize_resamples_stereo(self):
        # 1000 stereo samples at 44.1 kHz.
        samples = []
        for index in range(1000):
            value = int(20000 * math.sin(index / 10.0))
            samples.extend([value, value])
        data = _wav_bytes(samples, rate=44100, channels=2)
        normalized = tts.normalize_wav(data)
        with wave.open(io.BytesIO(normalized), "rb") as stream:
            self.assertEqual(stream.getnchannels(), 1)
            self.assertEqual(stream.getsampwidth(), 2)
            self.assertEqual(stream.getframerate(), tts.COZMO_RATE)
            self.assertEqual(stream.getnframes(), 500)

    def test_normalize_preserves_cozmo_rate(self):
        data = _wav_bytes([100, -100, 200, -200])
        normalized = tts.normalize_wav(data)
        with wave.open(io.BytesIO(normalized), "rb") as stream:
            self.assertEqual(stream.getframerate(), tts.COZMO_RATE)
            self.assertEqual(stream.getnframes(), 4)

    def test_normalize_rejects_non_16_bit(self):
        data = _wav_bytes([0, 1, 2, 3], sampwidth=1)
        with self.assertRaises(ValueError):
            tts.normalize_wav(data)

    def test_synthesize_with_fake_backend(self):
        def fake_backend(text, out_path, voice, rate):
            self.assertEqual(text, "hello")
            with open(out_path, "wb") as stream:
                stream.write(_wav_bytes([1000] * 100))

        with mock.patch("pycozmo.tts._backend_map",
                        return_value={"fake": fake_backend}):
            self.assertEqual(tts.available_backends(), ["fake"])
            data = tts.synthesize_wav("hello")
            self.assertTrue(data.startswith(b"RIFF"))
            with wave.open(io.BytesIO(data), "rb") as stream:
                self.assertEqual(stream.getnframes(), 100)

    def test_synthesize_errors(self):
        with self.assertRaises(ValueError):
            tts.synthesize_wav("   ")
        with mock.patch("pycozmo.tts._backend_map",
                        return_value={"fake": lambda *args: None}):
            with self.assertRaises(tts.AudioBackendError):
                tts.synthesize_wav("hello", backend="missing")

    def test_all_backends_fail(self):
        def failing_backend(text, out_path, voice, rate):
            raise RuntimeError("backend failure")

        with mock.patch("pycozmo.tts._backend_map",
                        return_value={"fake": failing_backend}):
            with self.assertRaises(tts.AudioBackendError):
                tts.synthesize_wav("hello")

    def test_text_to_speech_wrapper(self):
        def fake_backend(text, out_path, voice, rate):
            self.assertEqual(voice, "test")
            self.assertEqual(rate, 200)
            with open(out_path, "wb") as stream:
                stream.write(_wav_bytes([500] * 50))

        with mock.patch("pycozmo.tts._backend_map",
                        return_value={"fake": fake_backend}):
            speech = tts.TextToSpeech(voice="test", rate=200, backend="fake")
            self.assertEqual(speech.backends, ["fake"])
            path = os.path.join(tempfile.mkdtemp(), "speech.wav")
            self.assertEqual(speech.save("hello", path), path)
            self.assertTrue(os.path.getsize(path) > 0)
            os.unlink(path)

            class Client:
                def __init__(self):
                    self.audio = []

                def play_audio(self, data):
                    self.audio.append(data)

            client = Client()
            speech.say(client, "hello")
            speech.speak(client, "again")
            self.assertEqual(len(client.audio), 2)


if __name__ == "__main__":
    unittest.main()
