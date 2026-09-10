import unittest
from types import SimpleNamespace
import numpy as np
from app.speaker_context import ContextSpeakerEndpoint

class Vad:
    def accept(self, x):
        yield x
    def finish(self):
        return iter(())

class Scorer:
    def __init__(self, score): self.score = score
    def score_waveform(self, x, rate, speaker_id=None):
        return self.score, "lichao"

def endpoint(score=0.8):
    cfg = SimpleNamespace(sample_rate=16000, speaker_context_seconds=2.0,
        speaker_window_seconds=0.4, speaker_threshold=0.63,
        speaker_low_threshold=0.60, speaker_short_threshold=0.40,
        speaker_reject_seconds=1.6)
    output=[]
    return ContextSpeakerEndpoint(cfg, Scorer(score), Vad(), lambda x: output.append(x.copy())), output

class ContextTests(unittest.TestCase):
    def test_buffers_context_then_flushes_without_loss_or_duplicate(self):
        e, output = endpoint()
        raw=np.arange(72123, dtype=np.float32)
        for start in range(0, 30000, 1000): e.accept(raw[start:start+1000])
        self.assertFalse(output)
        for start in range(30000, len(raw), 1000): e.accept(raw[start:start+1000])
        e.finish()
        np.testing.assert_array_equal(np.concatenate(output), raw)
        count=len(output)
        e.finish(); e.accept(raw)
        self.assertEqual(len(output), count)

    def test_overlapping_context_does_not_count_two_seconds_each_time(self):
        e, output=endpoint(0.1)
        e.accept(np.ones(32000,dtype=np.float32))
        self.assertFalse(e.stopped)
        self.assertEqual(e.rejected_samples,19200)
        e.accept(np.ones(6400,dtype=np.float32))
        self.assertTrue(e.stopped)
        self.assertEqual(e.rejected_samples,25600)
        self.assertFalse(output)
        e.accept(np.ones(32000,dtype=np.float32)); e.finish()
        self.assertFalse(output)

    def test_short_command_flushes_on_audio_stop(self):
        e, output=endpoint(0.5)
        raw=np.ones(16000,dtype=np.float32)
        e.accept(raw)
        self.assertFalse(output)
        e.finish()
        np.testing.assert_array_equal(np.concatenate(output),raw)
        self.assertFalse(e.stopped)

    def test_too_short_and_unknown_do_not_count_as_rejection(self):
        for score, size in [(None,48000),(float("nan"),48000),(0.1,3200)]:
            e, output=endpoint(score)
            e.accept(np.ones(size,dtype=np.float32));e.finish()
            self.assertFalse(output)
            self.assertFalse(e.stopped)
            self.assertEqual(e.rejected_samples,0)

    def test_vad_boundary_does_not_mix_context(self):
        e, output=endpoint(0.5)
        a=np.ones(16000,dtype=np.float32)
        e._region(a);e._region(None)
        e._region(a*2);e.finish()
        self.assertEqual(len(output),2)
        np.testing.assert_array_equal(output[0],a)
        np.testing.assert_array_equal(output[1],a*2)

    def test_sessions_do_not_share_buffers(self):
        a, out_a=endpoint(); b, out_b=endpoint(.1)
        a.accept(np.ones(16000,dtype=np.float32))
        b.accept(np.ones(38400,dtype=np.float32))
        a.finish(); b.finish()
        self.assertEqual(sum(x.size for x in out_a),16000)
        self.assertFalse(out_b)
        self.assertTrue(b.stopped)
        self.assertFalse(a.stopped)


class BoundaryScorer:
    def __init__(self, transition, invalid=False):
        self.transition = transition
        self.invalid = invalid

    def score_waveform(self, x, rate, speaker_id=None):
        center = (float(x[0]) + float(x[-1]) + 1) / (2 * rate)
        if x.size == round(0.8 * rate):
            if self.invalid:
                return None, None
            boundary = self.transition
        else:
            boundary = 2.8
        return (0.8 if center < boundary else 0.2), "lichao"


class BoundaryTests(unittest.TestCase):
    def run_audio(self, boundary, enabled=True, invalid=False):
        e, output = endpoint()
        e.refine_tail = enabled
        e.scorer = BoundaryScorer(boundary, invalid)
        raw = np.arange(96000, dtype=np.float32)
        for i in range(0, raw.size, 640):
            e.accept(raw[i:i+640])
            if e.stopped:
                break
        rejected = e.rejected_samples
        e.finish()
        self.assertTrue(e.stopped)
        self.assertEqual(rejected, 25600)
        self.assertEqual(e.rejected_samples, rejected)
        result = np.concatenate(output)
        np.testing.assert_array_equal(result, raw[:result.size])
        count = len(output)
        e.finish(); e.accept(raw)
        self.assertEqual(len(output), count)
        return result.size

    def test_recovers_tail_without_changing_early_stop(self):
        self.assertEqual(self.run_audio(3.0), 48000)

    def test_removes_leaked_tail_without_changing_early_stop(self):
        self.assertEqual(self.run_audio(2.4), 38400)

    def test_disabled_unknown_and_no_transition_preserve_original(self):
        self.assertEqual(self.run_audio(3.0, enabled=False), 44800)
        self.assertEqual(self.run_audio(3.0, invalid=True), 44800)
        self.assertEqual(self.run_audio(100.0), 44800)
        self.assertEqual(self.run_audio(0.0), 44800)

    def test_only_last_run_changes_and_interior_gap_stays_rejected(self):
        e, output = endpoint()
        e.refine_tail = True
        e.samples = np.arange(128000, dtype=np.float32)
        e.speaker_id = "lichao"
        e.selected_runs = [(0, 16000), (32000, 76800)]
        e.cursor = 102400
        e.scorer = BoundaryScorer(5.0)
        e._flush_selected()
        np.testing.assert_array_equal(output[0], e.samples[:16000])
        np.testing.assert_array_equal(output[1], e.samples[32000:80000])
        self.assertFalse(e.selected_runs)

    def test_short_region_and_vad_region_isolation(self):
        e, output = endpoint()
        e.refine_tail = True
        first = np.arange(48000, dtype=np.float32)
        e._region(first); e._region(None)
        second = np.ones(16000, dtype=np.float32) * -1
        e._region(second); e.finish()
        np.testing.assert_array_equal(np.concatenate(output), np.concatenate([first, second]))
        self.assertFalse(e.stopped)


class HopScorer:
    def __init__(self, scores):
        self.scores = scores

    def score_waveform(self, x, rate, speaker_id=None):
        center = round((float(x[0]) + float(x[-1]) + 1) * 5 / rate)
        value = self.scores.get(center, 0.8)
        return value if isinstance(value, tuple) else (value, "lichao")


class BorderlineProtectionTests(unittest.TestCase):
    def render(self, scores, refine=False, size=96000):
        e, output = endpoint()
        e.refine_tail = refine
        e.scorer = HopScorer(scores)
        raw = np.arange(size, dtype=np.float32)
        for i in range(0, size, 640):
            e.accept(raw[i:i+640])
        e.finish()
        result = np.concatenate(output) if output else np.empty(0, dtype=np.float32)
        count = len(output)
        e.finish(); e.accept(raw)
        self.assertEqual(len(output), count)
        self.assertIsNone(e.borderline)
        return e, raw, result

    def test_single_borderline_between_highs_restored_with_and_without_refinement(self):
        for refine in (False, True):
            with self.subTest(refine=refine):
                e, raw, result = self.render({18: .639, 22: .628, 26: .669}, refine)
                np.testing.assert_array_equal(result, raw)
                self.assertFalse(e.stopped)
                self.assertEqual(e.rejected_samples, 0)

    def test_low_or_unknown_is_never_bridged(self):
        for score in (.59, float("nan"), None):
            e, raw, result = self.render({22: score}, True)
            np.testing.assert_array_equal(result, np.concatenate([raw[:32000], raw[38400:]]))

    def test_consecutive_borderlines_are_not_bridged(self):
        e, raw, result = self.render({22: .62, 26: .62}, True)
        np.testing.assert_array_equal(result, np.concatenate([raw[:32000], raw[44800:]]))

    def test_different_speaker_cannot_confirm_borderline(self):
        e, raw, result = self.render({22: .62, 26: (.9, "other")}, True)
        np.testing.assert_array_equal(result, np.concatenate([raw[:32000], raw[44800:]]))

    def test_leading_and_trailing_borderlines_not_bridged(self):
        e, raw, result = self.render({10: .62})
        np.testing.assert_array_equal(result, raw[19200:])
        e, raw, result = self.render({22: .62}, size=51200)
        np.testing.assert_array_equal(result, raw[:32000])

    def test_no_bridge_across_vad_regions(self):
        e, output = endpoint()
        e.scorer = HopScorer({22: .62})
        raw = np.arange(51200, dtype=np.float32)
        e._region(raw); e._region(None)
        self.assertIsNone(e.borderline)
        self.assertFalse(e.previous_high)
        second = np.arange(96000, 112000, dtype=np.float32)
        e._region(second); e.finish()
        np.testing.assert_array_equal(np.concatenate(output), np.concatenate([raw[:32000], second]))
