"""Exercise actual server, Wyoming byte framing and custom integration in memory.

Only native ASR/voiceprint/VAD inference is replaced. No listening sockets,
model files, microphone, Home Assistant installation or package installs.
"""

import asyncio
from dataclasses import replace
import os
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from app.config import parse_args
from app.protocol import read_message, write_message

# Import production modules without importing an unavailable native package.
# Constructors are never called; inference is explicitly injected below.
with patch.dict(sys.modules, {"sherpa_onnx": ModuleType("sherpa_onnx")}):
    from app.server import WyomingAsrServer
    from app.speaker_gate import SpeakerGate

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from custom_components.wyoming_speaker_stt.session import transcribe
    from custom_components.wyoming_speaker_stt.pipeline_bridge import PipelineBridge
except ModuleNotFoundError as err:
    if err.name not in ("custom_components", "custom_components.wyoming_speaker_stt"):
        raise
    # The standalone STT repository intentionally does not ship HA integration code.
    transcribe = PipelineBridge = None


class MemoryWriter:
    def __init__(self, reader):
        self.reader = reader
        self.closed = False

    def write(self, data):
        if self.closed:
            raise BrokenPipeError()
        self.reader.feed_data(data)

    async def drain(self):
        await asyncio.sleep(0)

    def close(self):
        if not self.closed:
            self.closed = True
            self.reader.feed_eof()

    async def wait_closed(self):
        pass

    def get_extra_info(self, name):
        return "in-memory"


class WireClient:
    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer
        self.sent, self.received = [], []

    async def write_event(self, event):
        self.sent.append(event)
        await write_message(self.writer, event["type"], event.get("data", {}), event.get("payload", b""))

    async def read_event(self):
        msg = await read_message(self.reader)
        event = {"type": msg.msg_type, "data": {**msg.data, **msg.extra_data}, "payload": msg.payload}
        self.received.append(event)
        return event


class FakeEngine:
    def __init__(self):
        self.streams = []
        self.decode_count = 0
        self.release = threading.Event()
        self.release.set()

    def create_stream(self):
        stream = []
        self.streams.append(stream)
        return stream

    def pcm_chunk_to_model_waveform(self, data, fmt):
        return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768

    def feed_waveform_to_stream(self, stream, samples):
        if samples.size:
            stream.append(samples.copy())

    def finish_stream(self, stream):
        self.decode_count += 1
        if not self.release.wait(3):
            raise RuntimeError("Test decoder was not released")
        return "打开灯" if stream else ""


class FakeSpeaker:
    def score_waveform(self, samples, rate, speaker_id=None):
        return (0.8 if samples.mean() > 0 else 0.1), speaker_id or "alice"

    def accepts_waveform(self, samples, rate, threshold=None):
        score, speaker = self.score_waveform(samples, rate)
        return score >= threshold, score, speaker


class FakeVad:
    def __init__(self, cfg):
        pass

    def accept(self, samples):
        yield samples

    def finish(self):
        return iter(())


def pcm(value, count=12800):
    return np.full(count, value, dtype="<i2").tobytes()


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, "argv", ["test"]):
            cfg = parse_args()
        self.server = WyomingAsrServer.__new__(WyomingAsrServer)
        self.server.cfg = replace(cfg, speaker_gate=True, speaker_early_stop=True, denoise_enabled=False)
        self.server.engine = FakeEngine()
        self.server.speaker_gate = FakeSpeaker()
        self.server._model_lock = asyncio.Lock()
        self.server._info_cache = None
        self.vad_patch = patch("app.server.StreamingVad", FakeVad)
        self.vad_patch.start()
        self.network_patch = patch("asyncio.start_server", side_effect=AssertionError("No network in tests"))
        self.network_patch.start()
        self.connections = []

    async def asyncTearDown(self):
        self.server.engine.release.set()
        for client, task in self.connections:
            client.writer.close()
            await asyncio.wait_for(task, 3)
        self.vad_patch.stop()
        self.network_patch.stop()

    def connect(self):
        server_reader, client_reader = asyncio.StreamReader(), asyncio.StreamReader()
        client = WireClient(client_reader, MemoryWriter(server_reader))
        task = asyncio.create_task(self.server.handle_client(server_reader, MemoryWriter(client_reader)))
        self.connections.append((client, task))
        return client

    async def start(self, client, negotiated=True):
        await client.write_event({"type": "transcribe", "data": {"speaker_early_stop": negotiated}})
        await client.write_event({"type": "audio-start", "data": {"rate": 16000, "width": 2, "channels": 1}})

    async def send(self, client, samples):
        await client.write_event({"type": "audio-chunk", "payload": samples})

    async def read(self, client):
        return await asyncio.wait_for(client.read_event(), 2)

    @unittest.skipIf(transcribe is None, "Optional sibling HA integration is not available")
    async def test_full_chain_stops_pipeline_before_decoder_returns(self):
        client = self.connect()
        stopped = asyncio.Event()
        events = []
        class Run:
            def __init__(self, provider):
                self.stt_provider = provider

            def process_event(self, event):
                events.append(event.type)
                stopped.set()

            async def speech_to_text(self, metadata, stream):
                async def audio():
                    async for chunk in stream:
                        yield chunk.audio
                return await self.stt_provider.process(audio())

        bridge = PipelineBridge(Run, lambda ts: SimpleNamespace(type="stt-vad-end", data={"timestamp": ts}), "stt-vad-end")
        class Provider:
            async def process(self, stream):
                return await transcribe(client, stream, lambda: bridge.stop_input(self), timeout=3)

        provider = Provider()
        bridge.install()
        bridge.register(provider)
        async def audio():
            for i, value in enumerate([1000, -1000, -1000]):
                yield SimpleNamespace(timestamp_ms=(i + 1) * 800, audio=pcm(value))
            await asyncio.Event().wait()  # Device would keep streaming indefinitely.
        self.server.engine.release.clear()
        task = asyncio.create_task(Run(provider).speech_to_text(None, audio()))
        try:
            await asyncio.wait_for(stopped.wait(), 2)
            self.assertFalse(task.done())
            self.assertEqual(client.received[0]["type"], "voice-stopped")
            self.assertEqual(client.received[0]["data"]["reason"], "speaker-rejected")
            self.server.engine.release.set()
            self.assertEqual(await asyncio.wait_for(task, 2), "打开灯")
            self.assertEqual(events, ["stt-vad-end"])
            self.assertEqual([x["type"] for x in client.received], ["voice-stopped", "transcript"])
            self.assertEqual(sum(x["type"] == "audio-stop" for x in client.sent), 1)
            self.assertTrue(client.sent[0]["data"]["speaker_early_stop"])
            self.assertEqual(sum(x.size for x in self.server.engine.streams[0]), 12800)
        finally:
            self.server.engine.release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            bridge.uninstall()

    async def test_late_chunks_and_duplicate_stops_do_not_decode_or_reply_twice(self):
        client = self.connect()
        await self.start(client)
        for value in [1000, -1000, -1000, 1000]:
            await self.send(client, pcm(value))
        self.assertEqual((await self.read(client))["type"], "voice-stopped")
        self.assertEqual((await self.read(client))["data"]["text"], "打开灯")
        for _ in range(2):
            await client.write_event({"type": "audio-stop"})
        await client.write_event({"type": "describe"})
        self.assertEqual((await self.read(client))["type"], "info")
        self.assertEqual(self.server.engine.decode_count, 1)
        self.assertEqual(sum(x.size for x in self.server.engine.streams[0]), 12800)

    async def test_rejected_only_returns_empty_after_stop(self):
        client = self.connect()
        await self.start(client)
        await self.send(client, pcm(-1000, 25600))
        self.assertEqual((await self.read(client))["type"], "voice-stopped")
        self.assertEqual((await self.read(client))["data"]["text"], "")

    async def test_legacy_client_waits_for_audio_stop(self):
        client = self.connect()
        await self.start(client, negotiated=False)
        await self.send(client, pcm(1000))
        await self.send(client, pcm(-1000, 25600))
        await client.write_event({"type": "describe"})
        self.assertEqual((await self.read(client))["type"], "info")
        await client.write_event({"type": "audio-stop"})
        self.assertEqual((await self.read(client))["type"], "transcript")

    async def test_server_feature_disabled_keeps_original_flow(self):
        self.server.cfg.speaker_early_stop = False
        client = self.connect()
        await self.start(client, negotiated=True)
        await self.send(client, pcm(-1000, 25600))
        await client.write_event({"type": "describe"})
        self.assertEqual((await self.read(client))["type"], "info")
        await client_stop(client)
        self.assertEqual((await self.read(client))["type"], "transcript")

    async def test_second_request_on_same_connection_starts_clean(self):
        client = self.connect()
        for _ in range(2):
            await self.start(client)
            await self.send(client, pcm(-1000, 25600))
            self.assertEqual((await self.read(client))["type"], "voice-stopped")
            self.assertEqual((await self.read(client))["type"], "transcript")
            await client.write_event({"type": "audio-stop"})
        self.assertEqual(self.server.engine.decode_count, 2)

    async def test_normal_audio_stop_flushes_partial_valid_window(self):
        client = self.connect()
        await self.start(client)
        await self.send(client, pcm(1000, 8000))
        await client.write_event({"type": "audio-stop"})
        self.assertEqual((await self.read(client))["data"]["text"], "打开灯")
        self.assertEqual(sum(x.size for x in self.server.engine.streams[0]), 8000)

    async def test_duration_limit_notifies_then_finishes_empty(self):
        client = self.connect()
        await self.start(client)
        await self.send(client, pcm(1000, 480001))
        self.assertEqual((await self.read(client))["data"]["reason"], "audio-limit")
        self.assertEqual((await self.read(client))["data"]["text"], "")

    async def test_concurrent_sessions_do_not_share_selected_audio(self):
        a, b = self.connect(), self.connect()
        await self.start(a)
        await self.start(b)
        await self.send(a, pcm(1000))
        await self.send(b, pcm(-1000, 25600))
        self.assertEqual((await self.read(b))["type"], "voice-stopped")
        self.assertEqual((await self.read(b))["data"]["text"], "")
        await client_stop(a)
        self.assertEqual((await self.read(a))["data"]["text"], "打开灯")

    async def test_native_worker_finishes_before_cancel_releases_model_lock(self):
        entered, release = threading.Event(), threading.Event()
        def worker():
            entered.set()
            release.wait(2)
        task = asyncio.create_task(self.server._model_call(worker))
        while not entered.is_set():
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        self.assertTrue(self.server._model_lock.locked())
        task.cancel()  # Repeated cancellation must not release a running model.
        await asyncio.sleep(0)
        self.assertTrue(self.server._model_lock.locked())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.server._model_lock.locked())

    async def test_denoiser_only_receives_selected_audio_and_is_per_session(self):
        instances = []
        class Denoiser:
            def __init__(self, **kwargs):
                self.seen = []
                self.flushed = False
                instances.append(self)

            def enhance(self, samples, rate):
                self.seen.append(samples.copy())
                return samples

            def flush(self):
                self.flushed = True
                return np.zeros(1, dtype=np.float32)

        self.server.cfg.denoise_enabled = True
        with patch("app.server.GtcrnEnhancer", Denoiser):
            a, b = self.connect(), self.connect()
            await self.start(a)
            await self.start(b)
            await self.send(a, pcm(1000))
            await self.send(a, pcm(-1000, 25600))
            await self.send(b, pcm(-1000, 25600))
            for client in (a, b):
                self.assertEqual((await self.read(client))["type"], "voice-stopped")
                await self.read(client)
        self.assertEqual(len(instances), 2)
        self.assertEqual(sorted(sum(x.size for x in d.seen) for d in instances), [0, 12800])
        self.assertEqual(sum(d.flushed for d in instances), 1)


async def client_stop(client):
    await client.write_event({"type": "audio-stop"})


class VoiceprintTests(unittest.TestCase):
    def test_locked_speaker_is_scored_even_if_another_enrolled_speaker_matches(self):
        gate = SpeakerGate.__new__(SpeakerGate)
        gate.speaker_embeddings = {"alice": np.array([1., 0.]), "bob": np.array([0., 1.])}
        gate._compute_embedding = lambda *args: np.array([0., 1.])
        waveform = np.ones(12800)
        self.assertEqual(gate.score_waveform(waveform, 16000), (1., "bob"))
        self.assertEqual(gate.score_waveform(waveform, 16000, "alice"), (0., "alice"))

    def test_unavailable_embedding_is_unknown(self):
        gate = SpeakerGate.__new__(SpeakerGate)
        gate._compute_embedding = lambda *args: np.zeros(2)
        self.assertEqual(gate.score_waveform(np.ones(12800), 16000), (None, None))


class ConfigTests(unittest.TestCase):
    def test_feature_is_off_by_default(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, "argv", ["test"]):
            self.assertFalse(parse_args().speaker_early_stop)

    def test_invalid_enabled_configuration_fails_before_model_loading(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, "argv", ["test"]):
            cfg = replace(parse_args(), speaker_gate=True, speaker_early_stop=True)
        for fields in [{"speaker_gate": False}, {"sample_rate": 8000},
                       {"speaker_low_threshold": 0.5}, {"speaker_threshold": float("nan")},
                       {"speaker_window_seconds": 0}, {"speaker_reject_seconds": float("inf")},
                       {"vad_threshold": 1}, {"vad_model": None}]:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(cfg, **fields).validate_early_stop()


if __name__ == "__main__":
    unittest.main()
