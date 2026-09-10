import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
from app.asr_engine import AudioFormat
from app.debug_audio import AudioCapture

class CaptureTests(unittest.TestCase):
    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(AudioCapture.from_environment(AudioFormat(16000, 2, 1), 16000))

    def test_raw_format_limit_and_empty_asr(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = AudioCapture(directory, AudioFormat(8000, 2, 2), 16000)
            capture.receive(b"abcd" * 240001)
            self.assertTrue(capture.truncated)
            capture.save(b"", "audio-limit", "")
            with wave.open(str(next(Path(directory).glob("*-received.wav"))), "rb") as wav:
                self.assertEqual(wav.getparams()[:3], (2, 2, 8000))
                self.assertEqual(wav.getnframes(), 240000)
            with wave.open(str(next(Path(directory).glob("*-asr-input.wav"))), "rb") as wav:
                self.assertEqual(wav.getnframes(), 0)

    def test_write_error_does_not_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file"
            path.write_text("occupied")
            with self.assertLogs("wyoming-sherpa-onnx", level="ERROR"):
                AudioCapture(path, AudioFormat(16000, 2, 1), 16000).save(b"", "audio-stop", "")
