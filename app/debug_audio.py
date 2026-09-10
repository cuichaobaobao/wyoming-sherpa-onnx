"""Optional, bounded diagnostic WAV capture; never changes inference samples."""
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import uuid
import wave

LOGGER = logging.getLogger("wyoming-sherpa-onnx")


class AudioCapture:
    def __init__(self, directory, fmt, model_rate):
        self.directory = Path(directory)
        self.fmt = fmt
        self.model_rate = model_rate
        self.started = datetime.now(timezone.utc)
        self.name = self.started.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
        self.raw = bytearray()
        self.limit = 30 * fmt.rate * fmt.width * fmt.channels
        self.truncated = False

    @classmethod
    def from_environment(cls, fmt, model_rate):
        directory = os.environ.get("DEBUG_AUDIO_DIR", "").strip()
        return cls(directory, fmt, model_rate) if directory else None

    def receive(self, payload):
        remaining = max(0, self.limit - len(self.raw))
        self.raw.extend(payload[:remaining])
        self.truncated |= len(payload) > remaining

    def save(self, pcm, reason, text):
        try:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            # Bound disk usage without deleting the user's diagnostic recordings.
            if sum(1 for _ in self.directory.glob("*-received.wav")) >= 100:
                LOGGER.warning("Debug audio directory has 100 sessions; capture skipped: %s", self.directory)
                return
            self._wav("received", self.raw, self.fmt.rate, self.fmt.width, self.fmt.channels)
            self._wav("asr-input", pcm, self.model_rate, 2, 1)
            metadata = {
                "started_utc": self.started.isoformat(), "reason": reason, "text": text,
                "received_seconds": len(self.raw) / (self.fmt.rate * self.fmt.width * self.fmt.channels),
                "asr_input_seconds": len(pcm) / (self.model_rate * 2),
                "received_truncated_at_30s": self.truncated,
                "note": "Received audio before server processing, up to session finish; ASR audio has removed intervals concatenated. Disconnect/error ASR input may be unflushed and not decoded.",
            }
            with (self.directory / (self.name + ".json")).open("x", encoding="utf-8") as output:
                json.dump(metadata, output, ensure_ascii=False, indent=2)
            LOGGER.info("Debug audio saved: %s reason=%s", self.directory / self.name, reason)
        except Exception:
            LOGGER.exception("Debug audio save failed; recognition is unaffected")

    def _wav(self, suffix, pcm, rate, width, channels):
        path = self.directory / (self.name + "-" + suffix + ".wav")
        with path.open("xb") as output:
            with wave.open(output, "wb") as wav:
                wav.setparams((channels, width, rate, 0, "NONE", "not compressed"))
                wav.writeframes(pcm)
