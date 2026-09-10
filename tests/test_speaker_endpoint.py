"""Real policy/PCM arrays with scripted model decisions; no model or device."""

import unittest
from types import SimpleNamespace

import numpy as np

from app.early_stop import SpeakerEndpoint, StreamingVad


def config(**changes):
    values = dict(sample_rate=16000, speaker_threshold=0.4, speaker_low_threshold=0.3,
                  speaker_window_seconds=0.8, speaker_reject_seconds=1.6)
    values.update(changes)
    return SimpleNamespace(**values)


class SpeechVad:
    def accept(self, samples):
        yield samples

    def finish(self):
        return iter(())


class Scorer:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.targets = []

    def score_waveform(self, samples, rate, speaker_id):
        self.targets.append(speaker_id)
        value = next(self.scores)
        return (value, "alice") if not isinstance(value, tuple) else value


class PolicyTests(unittest.TestCase):
    def make(self, scores):
        kept = []
        scorer = Scorer(scores)
        endpoint = SpeakerEndpoint(config(), scorer, SpeechVad(), lambda x: kept.append(x.copy()))
        return endpoint, kept, scorer

    def push(self, endpoint, value=1, size=12800):
        endpoint.accept(np.full(size, value, dtype=np.float32))

    def test_keep_high_drop_low_and_stop_without_waiting_for_silence(self):
        endpoint, kept, scorer = self.make([0.8, 0.1, 0.2])
        self.push(endpoint, 1)
        self.push(endpoint, 2)
        self.assertFalse(endpoint.stopped)
        self.push(endpoint, 3)
        self.assertTrue(endpoint.stopped)
        self.push(endpoint, 4)  # Late audio must not even be scored.
        endpoint.finish()
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.all(kept[0] == 1))
        self.assertEqual(scorer.targets, [None, "alice", "alice"])

    def test_high_resets_rejection_streak(self):
        endpoint, _, _ = self.make([0.1, 0.8, 0.1, 0.1])
        for _ in range(3):
            self.push(endpoint)
        self.assertFalse(endpoint.stopped)
        self.push(endpoint)
        self.assertTrue(endpoint.stopped)

    def test_borderline_between_highs_is_kept_in_order(self):
        endpoint, kept, _ = self.make([0.8, 0.35, 0.8])
        for value in range(1, 4):
            self.push(endpoint, value)
        self.assertEqual([x[0] for x in kept], [1, 2, 3])

    def test_borderline_before_low_or_at_end_is_discarded(self):
        for scores in ([0.8, 0.35, 0.1], [0.8, 0.35]):
            endpoint, kept, _ = self.make(scores)
            for value in range(len(scores)):
                self.push(endpoint, value)
            endpoint.finish()
            self.assertEqual([x[0] for x in kept], [0])

    def test_low_between_highs_is_never_reintroduced(self):
        endpoint, kept, _ = self.make([0.8, 0.1, 0.8])
        for value in range(3):
            self.push(endpoint, value)
        self.assertEqual([x[0] for x in kept], [0, 2])

    def test_multiple_borderlines_and_leading_borderline_are_discarded(self):
        endpoint, kept, _ = self.make([0.35, 0.8, 0.35, 0.35, 0.8])
        for value in range(5):
            self.push(endpoint, value)
        self.assertEqual([x[0] for x in kept], [1, 4])

    def test_unknown_and_borderline_break_low_streak(self):
        for middle in [None, float("nan"), 0.35]:
            endpoint, _, _ = self.make([0.1, middle, 0.1])
            for _ in range(3):
                self.push(endpoint)
            self.assertFalse(endpoint.stopped)

    def test_thresholds_are_not_relaxed_for_first_window(self):
        endpoint, kept, _ = self.make([0.30, 0.40])
        self.push(endpoint, 1)
        self.push(endpoint, 2)
        self.assertEqual([x[0] for x in kept], [2])

    def test_invalid_short_tail_cannot_count_as_rejection(self):
        endpoint, _, scorer = self.make([0.1])
        self.push(endpoint)
        self.push(endpoint, size=2000)
        endpoint.finish()
        self.assertFalse(endpoint.stopped)
        self.assertEqual(len(scorer.targets), 1)

    def test_fragmented_and_large_network_chunks_give_same_result(self):
        waveform = np.repeat(np.array([1, 2, 3, 4], dtype=np.float32), 12800)
        for chunk_size in (320, 1111, waveform.size):
            endpoint, kept, scorer = self.make([0.8, 0.1, 0.1])
            for i in range(0, waveform.size, chunk_size):
                endpoint.accept(waveform[i:i + chunk_size])
            self.assertTrue(endpoint.stopped)
            self.assertEqual(len(scorer.targets), 3)
            np.testing.assert_array_equal(np.concatenate(kept), waveform[:12800])

    def test_silence_not_scored_and_does_not_count_as_rejected_audio(self):
        endpoint, kept, scorer = self.make([])
        class SilenceVad(SpeechVad):
            def accept(self, samples):
                return iter(())
        endpoint.vad = SilenceVad()
        self.push(endpoint, size=160000)
        endpoint.finish()
        self.assertFalse(endpoint.stopped)
        self.assertEqual(scorer.targets, [])
        self.assertEqual(kept, [])

    def test_completed_region_does_not_bridge_borderlines_across_silence(self):
        endpoint, kept, _ = self.make([0.8, 0.35, 0.8])
        self.push(endpoint, 1)
        self.push(endpoint, 2)
        endpoint._region(None)
        self.push(endpoint, 3)
        self.assertEqual([x[0] for x in kept], [1, 3])

    def test_separate_sessions_lock_separate_speakers(self):
        a, _, _ = self.make([(0.8, "alice")])
        b, _, _ = self.make([(0.8, "bob")])
        self.push(a)
        self.push(b)
        self.assertEqual((a.speaker_id, b.speaker_id), ("alice", "bob"))


class Detector:
    """Mimics sherpa segment offsets, active snapshots and completed queue."""
    def __init__(self):
        self.samples = np.empty(0, dtype=np.float32)
        self.queue = []
        self.active = True

    def accept_waveform(self, frame):
        self.samples = np.concatenate((self.samples, frame))

    def empty(self):
        return not self.queue

    def pop(self):
        self.queue.pop(0)

    @property
    def front(self):
        return self.queue[0]

    def is_speech_detected(self):
        return self.active

    @property
    def current_segment(self):
        # Real sherpa current_segment may lag by one sample.
        return SimpleNamespace(start=0, samples=self.samples[:-1])

    def flush(self):
        self.queue.append(SimpleNamespace(start=0, samples=self.samples.copy()))
        self.active = False


class VadAdapterTests(unittest.TestCase):
    def test_active_audio_emitted_before_end_without_duplicates_or_padding(self):
        vad = StreamingVad(config(), Detector())
        source = np.arange(1700, dtype=np.float32)
        regions = []
        for start in range(0, source.size, 333):
            regions.extend(vad.accept(source[start:start + 333]))
        self.assertGreater(sum(x.size for x in regions if x is not None), 1000)
        regions.extend(vad.finish())
        np.testing.assert_array_equal(np.concatenate([x for x in regions if x is not None]), source)
        self.assertEqual(sum(x is None for x in regions), 1)

    def test_completed_then_active_regions_preserve_offsets(self):
        detector = Detector()
        vad = StreamingVad(config(), detector)
        vad.received = 3000
        detector.queue = [SimpleNamespace(start=200, samples=np.arange(1000, dtype=np.float32))]
        detector.active = False
        emitted = list(vad._drain())
        self.assertEqual(emitted[0].size, 1000)
        self.assertIsNone(emitted[1])
        detector.queue = [SimpleNamespace(start=2000, samples=np.ones(500, dtype=np.float32))]
        self.assertEqual(list(vad._drain())[0].size, 500)


if __name__ == "__main__":
    unittest.main()
