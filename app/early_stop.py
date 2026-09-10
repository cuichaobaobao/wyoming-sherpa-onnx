"""Streaming VAD and speaker policy. No HA/device control or model downloads."""

from __future__ import annotations

import logging
import math

import numpy as np

LOGGER = logging.getLogger("wyoming-sherpa-onnx")


class StreamingVad:
    """Read active VAD segments incrementally, without waiting for silence.

    sherpa-onnx 1.13.7 exposes current_segment in its Python bindings. Using
    only front/empty would wait for speech to end and defeat speaker early stop.
    Each sample is emitted once; None marks a completed speech region.
    """

    def __init__(self, cfg, detector=None):
        self.frame_size = 512  # Silero VAD at 16 kHz
        if detector is None:
            import sherpa_onnx

            config = sherpa_onnx.VadModelConfig()
            config.silero_vad.model = str(cfg.vad_model)
            config.silero_vad.threshold = cfg.vad_threshold
            config.silero_vad.window_size = self.frame_size
            config.silero_vad.min_speech_duration = 0.25
            config.silero_vad.min_silence_duration = 0.2
            config.silero_vad.max_speech_duration = 60
            config.sample_rate = cfg.sample_rate
            config.num_threads = max(1, cfg.num_threads)
            detector = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=35)
        self.detector = detector
        self.pending = np.empty(0, dtype=np.float32)
        self.received = 0
        self.emitted_end = 0

    def _new_samples(self, segment):
        start = int(segment.start)
        if start < 0:
            return None
        samples = np.asarray(segment.samples, dtype=np.float32)
        end = min(start + samples.size, self.received)
        begin = max(start, self.emitted_end)
        if end <= begin:
            return None
        self.emitted_end = end
        return samples[begin - start : end - start].copy()

    def _drain(self):
        while not self.detector.empty():
            samples = self._new_samples(self.detector.front)
            self.detector.pop()
            if samples is not None:
                yield samples
            yield None
        if self.detector.is_speech_detected():
            samples = self._new_samples(self.detector.current_segment)
            if samples is not None:
                yield samples

    def accept(self, waveform):
        self.received += waveform.size
        self.pending = np.concatenate((self.pending, waveform))
        while self.pending.size >= self.frame_size:
            frame = self.pending[:self.frame_size]
            self.pending = self.pending[self.frame_size:]
            self.detector.accept_waveform(frame)
            yield from self._drain()

    def finish(self):
        if self.pending.size:
            frame = np.zeros(self.frame_size, dtype=np.float32)
            frame[:self.pending.size] = self.pending
            self.pending = np.empty(0, dtype=np.float32)
            self.detector.accept_waveform(frame)
            yield from self._drain()
        self.detector.flush()
        yield from self._drain()


class SpeakerEndpoint:
    """High: keep; borderline: one-window lookahead; low: discard and count.

    Borderline audio is retained only between two high windows of the locked
    speaker. Silence pauses the low-score count; high/borderline/unknown scores
    break it. Unknown/too-short embeddings must never count as a rejection.
    """

    def __init__(self, cfg, scorer, vad, feed):
        self.rate = cfg.sample_rate
        self.high = cfg.speaker_threshold
        self.low = cfg.speaker_low_threshold
        self.window = max(1, round(cfg.speaker_window_seconds * self.rate))
        self.reject_limit = max(1, math.ceil(cfg.speaker_reject_seconds * self.rate))
        self.min_samples = round(0.4 * self.rate)
        self.scorer, self.vad, self.feed = scorer, vad, feed
        self.pending = np.empty(0, dtype=np.float32)
        self.borderline = None
        self.previous_high = False
        self.speaker_id = None
        self.rejected_samples = 0
        self.stopped = False

    def accept(self, waveform):
        if self.stopped:
            return
        for region in self.vad.accept(waveform):
            self._region(region)
            if self.stopped:
                break

    def _region(self, region):
        if region is None:
            self._tail()
            self.borderline = None
            self.previous_high = False
            return
        self.pending = np.concatenate((self.pending, region))
        while self.pending.size >= self.window and not self.stopped:
            window = self.pending[:self.window].copy()
            self.pending = self.pending[self.window:]
            self._score(window)
        if self.stopped:
            self.pending = np.empty(0, dtype=np.float32)

    def _score(self, window):
        if window.size < self.min_samples:
            self.borderline = None
            self.previous_high = False
            self.rejected_samples = 0
            return
        score, speaker = self.scorer.score_waveform(window, self.rate, self.speaker_id)
        if score is None or not math.isfinite(score) or speaker is None:
            tier = "unknown"
        elif self.speaker_id is not None and speaker != self.speaker_id:
            tier = "low"
        elif score >= self.high:
            tier = "high"
        elif score < self.low:
            tier = "low"
        else:
            tier = "borderline"

        if tier == "high":
            self.speaker_id = speaker
            if self.borderline is not None:
                self.feed(self.borderline)
            self.feed(window)
            self.borderline = None
            self.previous_high = True
            self.rejected_samples = 0
        elif tier == "borderline":
            # Keep only one ambiguous window, and only after a high window.
            self.borderline = window if self.previous_high else None
            self.previous_high = False
            self.rejected_samples = 0
        else:
            self.borderline = None
            self.previous_high = False
            self.rejected_samples = self.rejected_samples + window.size if tier == "low" else 0
            if self.rejected_samples >= self.reject_limit:
                self.stopped = True
        LOGGER.debug("Speaker window: tier=%s score=%s locked=%s rejected=%.3fs stop=%s",
                     tier, score, self.speaker_id, self.rejected_samples / self.rate, self.stopped)

    def _tail(self):
        if self.pending.size and not self.stopped:
            self._score(self.pending.copy())
        self.pending = np.empty(0, dtype=np.float32)

    def finish(self):
        if not self.stopped:
            for region in self.vad.finish():
                self._region(region)
                if self.stopped:
                    break
            self._tail()
        # Unresolved borderline audio is discarded at termination.
        self.borderline = None
        self.pending = np.empty(0, dtype=np.float32)
