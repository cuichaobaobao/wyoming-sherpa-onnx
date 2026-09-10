"""Buffered speaker decisions with overlapping context and disjoint output hops."""
from __future__ import annotations

import logging
import math
import numpy as np

LOGGER = logging.getLogger("wyoming-sherpa-onnx")


class ContextSpeakerEndpoint:
    def __init__(self, cfg, scorer, vad, feed):
        self.rate = cfg.sample_rate
        self.refine_tail = getattr(cfg, "speaker_boundary_refine", False)
        self.selected_runs: list[tuple[int, int]] = []
        self.previous_high = False
        self.borderline: tuple[int, int] | None = None
        self.context = round(cfg.speaker_context_seconds * self.rate)
        self.hop = round(cfg.speaker_window_seconds * self.rate)
        self.minimum = round(0.4 * self.rate)
        self.high = cfg.speaker_threshold
        self.low = cfg.speaker_low_threshold
        self.short_high = cfg.speaker_short_threshold
        self.reject_limit = math.ceil(cfg.speaker_reject_seconds * self.rate)
        self.scorer, self.vad, self.feed = scorer, vad, feed
        self.samples = np.empty(0, dtype=np.float32)
        self.cursor = 0
        self.rejected_samples = 0
        self.speaker_id = None
        self.stopped = False
        self.finished = False
        self._cached_bounds = None
        self._cached_score = None

    def accept(self, waveform):
        if self.stopped or self.finished:
            return
        for region in self.vad.accept(waveform):
            self._region(region)
            if self.stopped:
                break

    def _region(self, region):
        if self.stopped:
            return
        if region is None:
            self._drain(final=True)
            self._flush_selected()
            self.borderline = None
            self.previous_high = False
            self.samples = np.empty(0, dtype=np.float32)
            self.cursor = 0
            self._cached_bounds = None
            return
        self.samples = np.concatenate((self.samples, region))
        self._drain(final=False)

    def _drain(self, final):
        size = self.samples.size
        if size < self.context:
            if not final or size == 0:
                return
            if size < self.minimum:
                self.rejected_samples = 0
                return
            # A complete short VAD region cannot provide full context. Never
            # wait beyond audio-stop or count uncertain short audio as rejection.
            score, speaker = self.scorer.score_waveform(self.samples, self.rate, self.speaker_id)
            valid = score is not None and math.isfinite(score) and speaker is not None
            accepted = valid and score >= self.short_high and (self.speaker_id is None or speaker == self.speaker_id)
            if accepted:
                self.speaker_id = speaker
                self.feed(self.samples.copy())
            self.rejected_samples = 0
            LOGGER.info("Context speaker short region: seconds=%.3f score=%s accepted=%s", size / self.rate, score, accepted)
            self.cursor = size
            return
        while self.cursor < size and not self.stopped:
            end = min(self.cursor + self.hop, size)
            if not final and end - self.cursor < self.hop:
                break
            # Delay a hop until its centered context is available. At region
            # boundaries use the first/last full context; no cross-silence mixing.
            left = max(0, self.cursor + self.hop // 2 - self.context // 2)
            right = left + self.context
            if right > size:
                if not final:
                    break
                right, left = size, size - self.context
            bounds = (left, right)
            if bounds != self._cached_bounds:
                self._cached_score = self.scorer.score_waveform(self.samples[left:right], self.rate, self.speaker_id)
                self._cached_bounds = bounds
            score, speaker = self._cached_score
            valid = score is not None and math.isfinite(score) and speaker is not None
            same = self.speaker_id is None or speaker == self.speaker_id
            tier = "unknown" if not valid else ("high" if same and score >= self.high else ("low" if not same or score < self.low else "borderline"))
            if tier == "high":
                if self.speaker_id is None:
                    self.speaker_id = speaker
                    self._cached_bounds = None
                if self.borderline is not None:
                    left, right = self.borderline
                    self._select(left, right)
                    LOGGER.info("Context speaker borderline restored: start=%.3f end=%.3f speaker=%s",
                                left / self.rate, right / self.rate, self.speaker_id)
                self._select(self.cursor, end)
                self.borderline = None
                self.previous_high = True
                self.rejected_samples = 0
            elif tier == "borderline":
                # Exactly one uncertain hop may bridge two adjacent high hops
                # of the locked speaker. A second uncertain hop cancels it.
                self.borderline = (self.cursor, end) if self.previous_high else None
                self.previous_high = False
                self.rejected_samples = 0
            else:
                self.borderline = None
                self.previous_high = False
                if tier == "low":
                    # Count only disjoint output samples, never context overlap.
                    self.rejected_samples += end - self.cursor
                    self.stopped = self.rejected_samples >= self.reject_limit
                else:
                    self.rejected_samples = 0
            LOGGER.info("Context speaker hop: start=%.3f end=%.3f score=%s tier=%s rejected=%.3f stop=%s", self.cursor / self.rate, end / self.rate, score, tier, self.rejected_samples / self.rate, self.stopped)
            self.cursor = end

    def _select(self, left, right):
        if self.refine_tail:
            if self.selected_runs and self.selected_runs[-1][1] == left:
                start, _ = self.selected_runs[-1]
                self.selected_runs[-1] = (start, right)
            else:
                self.selected_runs.append((left, right))
        else:
            self.feed(self.samples[left:right].copy())

    def _refined_end(self, start, original_end):
        # Only revisit the final accepted run followed by rejected/uncertain
        # audio in this VAD region. Short commands and interior gaps stay intact.
        if original_end - start < self.context or self.cursor <= original_end:
            return original_end
        window = round(0.8 * self.rate)
        half = window // 2
        step = round(0.1 * self.rate)
        radius = round(0.8 * self.rate)
        first = max(start + half, original_end - radius)
        last = min(self.samples.size - half, original_end + radius)
        seen_high = False
        first_low = None
        for center in range(first, last + 1, step):
            score, speaker = self.scorer.score_waveform(
                self.samples[center - half:center + half], self.rate, self.speaker_id
            )
            if score is None or not math.isfinite(score) or speaker != self.speaker_id:
                return original_end
            # This shorter context has its own threshold; it does not alter
            # the long-context decisions or the early-stop rejection counter.
            if score >= 0.60:
                seen_high = True
                first_low = None
            elif seen_high:
                if first_low is not None:
                    return first_low
                first_low = center
        # No confirmed local transition: keep the original decision.
        return original_end

    def _flush_selected(self):
        if not self.selected_runs:
            return
        start, old_end = self.selected_runs[-1]
        new_end = self._refined_end(start, old_end)
        self.selected_runs[-1] = (start, new_end)
        LOGGER.info(
            "Context speaker boundary: original=%.3f refined=%.3f delta=%.3f (VAD region seconds)",
            old_end / self.rate, new_end / self.rate, (new_end - old_end) / self.rate,
        )
        for left, right in self.selected_runs:
            self.feed(self.samples[left:right].copy())
        self.selected_runs.clear()

    def finish(self):
        if self.finished:
            return
        if not self.stopped:
            for region in self.vad.finish():
                self._region(region)
                if self.stopped:
                    break
            if not self.stopped:
                self._drain(final=True)
        self._flush_selected()
        self.borderline = None
        self.previous_high = False
        self.finished = True
        self.samples = np.empty(0, dtype=np.float32)
