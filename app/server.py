from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .asr_engine import AudioFormat, Qwen3AsrEngine
from .config import AppConfig
from .denoise import GtcrnEnhancer
from .debug_audio import AudioCapture
from .early_stop import SpeakerEndpoint, StreamingVad
from .speaker_context import ContextSpeakerEndpoint
from .protocol import read_message, write_message
from .speaker_gate import SpeakerGate

LOGGER = logging.getLogger("wyoming-sherpa-onnx")
_MAX_AUDIO_SECONDS = 30.0
_SPEAKER_GATE_WINDOW_SECONDS = 0.8
_SPEAKER_GATE_HYSTERESIS_DELTA = 0.06


def _is_disconnect_error(exc: BaseException) -> bool:
    return isinstance(exc, (BrokenPipeError, ConnectionResetError))


@dataclass(slots=True)
class SessionState:
    transcribe_opts: dict[str, Any]
    audio_format: AudioFormat | None
    stream: Any | None
    chunk_count: int
    total_bytes: int
    over_limit: bool
    detected_segments: int
    accepted_segments: int
    rejected_segments: int
    accepted_samples: int
    gate_pending_waveform: np.ndarray
    gate_pending_start_idx: int
    gate_segments: list["GateSegment"]
    gate_active: bool
    endpoint: SpeakerEndpoint | ContextSpeakerEndpoint | None = None
    denoiser: GtcrnEnhancer | None = None
    finished: bool = False
    stop_notified: bool = False
    capture: AudioCapture | None = None
    stop_reason: str = "audio-stop"


@dataclass(slots=True)
class GateSegment:
    start_idx: int
    end_idx: int
    waveform: np.ndarray
    accepted: bool
    similarity: float
    threshold: float
    speaker_id: str | None


class WyomingAsrServer:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        cfg.validate_early_stop()
        self.engine = Qwen3AsrEngine(
            model_dir=cfg.model_dir,
            sample_rate=cfg.sample_rate,
            num_threads=cfg.num_threads,
            hotwords=cfg.hotwords,
        )
        self.speaker_gate: SpeakerGate | None = None
        if cfg.speaker_gate:
            self.speaker_gate = SpeakerGate(
                model_path=cfg.speaker_model_dir / cfg.speaker_model_file,
                reference_wavs=cfg.speaker_reference_wavs,
                threshold=cfg.speaker_threshold,
                num_threads=max(1, cfg.num_threads),
                reference_root=cfg.speaker_reference_dir,
            )
        # Shared speaker extractor/recognizer calls are serialized off-loop.
        self._model_lock = asyncio.Lock()
        self._info_cache: dict[str, Any] | None = None
        self._server: asyncio.AbstractServer | None = None

    async def handle_client(self, reader, writer) -> None:
        peer = self._get_peername(writer)
        LOGGER.info("Client connected: %s", peer)
        state = SessionState(
            transcribe_opts={},
            audio_format=None,
            stream=None,
            chunk_count=0,
            total_bytes=0,
            over_limit=False,
            detected_segments=0,
            accepted_segments=0,
            rejected_segments=0,
            accepted_samples=0,
            gate_pending_waveform=np.empty((0,), dtype=np.float32),
            gate_pending_start_idx=0,
            gate_segments=[],
            gate_active=False,
        )

        try:
            while True:
                msg = await read_message(reader)
                data = {**msg.data, **msg.extra_data}
                LOGGER.debug(
                    "[%s] Received message type=%s data_keys=%s payload_bytes=%s",
                    peer,
                    msg.msg_type,
                    sorted(data.keys()),
                    len(msg.payload),
                )

                if msg.msg_type == "describe":
                    await write_message(writer, "info", self._get_info())
                elif msg.msg_type == "transcribe":
                    state.transcribe_opts = data
                elif msg.msg_type == "audio-start":
                    await self._save_capture(state, "replaced-audio-start")
                    state.stream = self.engine.create_stream()
                    state.audio_format = AudioFormat(
                        rate=int(data.get("rate", self.cfg.sample_rate)),
                        width=int(data.get("width", 2)),
                        channels=int(data.get("channels", 1)),
                    )
                    state.capture = AudioCapture.from_environment(state.audio_format, self.cfg.sample_rate)
                    state.stop_reason = "audio-stop"
                    state.chunk_count = 0
                    state.total_bytes = 0
                    state.over_limit = False
                    state.detected_segments = 0
                    state.accepted_segments = 0
                    state.rejected_segments = 0
                    state.accepted_samples = 0
                    state.gate_pending_waveform = np.empty((0,), dtype=np.float32)
                    state.gate_pending_start_idx = 0
                    state.gate_segments = []
                    state.gate_active = False
                    state.finished = False
                    state.stop_notified = False
                    state.endpoint = None
                    # GTCRN carries stream state, so never share it across clients.
                    state.denoiser = None
                    if self.cfg.denoise_enabled:
                        state.denoiser = await self._model_call(
                            lambda: GtcrnEnhancer(
                                model_path=self.cfg.denoise_model_dir / self.cfg.denoise_model_file,
                                num_threads=max(1, self.cfg.num_threads),
                            )
                        )
                    if (self.cfg.speaker_early_stop and
                            state.transcribe_opts.get("speaker_early_stop") is True):
                        vad = await self._model_call(lambda: StreamingVad(self.cfg))
                        endpoint_type = ContextSpeakerEndpoint if self.cfg.speaker_context_seconds > 0 else SpeakerEndpoint
                        state.endpoint = endpoint_type(
                            self.cfg, self.speaker_gate, vad,
                            lambda waveform: self._feed_selected(state, waveform),
                        )
                    LOGGER.debug(
                        "[%s] Audio stream started: %dHz, %d-bit, %d channels",
                        peer,
                        state.audio_format.rate,
                        state.audio_format.width * 8,
                        state.audio_format.channels,
                    )
                elif msg.msg_type == "audio-chunk":
                    if (msg.payload and not state.finished and state.stream is not None
                            and state.audio_format is not None):
                        if state.capture is not None:
                            state.capture.receive(msg.payload)
                        bytes_per_sample = state.audio_format.width * state.audio_format.channels
                        total_bytes = state.total_bytes + len(msg.payload)
                        audio_duration = total_bytes / (bytes_per_sample * state.audio_format.rate)
                        if audio_duration > _MAX_AUDIO_SECONDS:
                            state.over_limit = True
                            state.stream = None
                            LOGGER.warning(
                                "[%s] Audio exceeded max duration %.2fs, dropping request at %.2fs",
                                peer,
                                _MAX_AUDIO_SECONDS,
                                audio_duration,
                            )
                            if state.endpoint is not None:
                                await self._notify_stop(writer, state, "audio-limit")
                                await self._finish_session(peer, writer, state)
                            continue
                        waveform = self.engine.pcm_chunk_to_model_waveform(
                            msg.payload, state.audio_format
                        )
                        if state.endpoint is not None:
                            # VAD and voiceprint see unmodified samples; optional
                            # denoising runs only on accepted audio for ASR.
                            await self._model_call(lambda: state.endpoint.accept(waveform))
                        else:
                            await self._model_call(lambda: self._process_legacy(peer, state, waveform))
                        state.chunk_count += 1
                        state.total_bytes += len(msg.payload)
                        LOGGER.debug(
                            "[%s] Audio chunk #%d received (%.1f KB)",
                            peer,
                            state.chunk_count,
                            len(msg.payload) / 1024,
                        )
                        if state.endpoint is not None and state.endpoint.stopped:
                            await self._notify_stop(writer, state, "speaker-rejected")
                            await self._finish_session(peer, writer, state)
                elif msg.msg_type == "audio-stop":
                    await self._finish_session(peer, writer, state)
                else:
                    LOGGER.debug("Ignoring unsupported message type: %s", msg.msg_type)
        except EOFError:
            LOGGER.info("Client disconnected: %s", peer)
        except (BrokenPipeError, ConnectionResetError) as exc:
            LOGGER.info("Client connection closed while streaming: %s (%s)", peer, exc)
        except asyncio.CancelledError:
            LOGGER.info("Client session cancelled: %s", peer)
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Client session error: %s", exc)
        finally:
            await self._save_capture(state, "disconnect-or-error")
            try:
                writer.close()
            except Exception:
                pass

            try:
                await writer.wait_closed()
            except Exception as exc:  # noqa: BLE001
                if not _is_disconnect_error(exc):
                    LOGGER.debug("Error while closing client stream %s: %s", peer, exc)

    async def _model_call(self, function):
        async with self._model_lock:
            task = asyncio.create_task(asyncio.to_thread(function))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # A native inference call cannot be cancelled. Drain it before
                # releasing the shared model lock or discarding session state.
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if not task.cancelled():
                    task.exception()  # Retrieve any worker error during cancellation.
                raise

    def _process_legacy(self, peer, state, waveform):
        if state.denoiser is not None:
            waveform = state.denoiser.enhance(waveform, self.cfg.sample_rate)
        self._process_segments(peer, state, [waveform])

    def _feed_selected(self, state, waveform):
        state.accepted_segments += 1
        state.accepted_samples += waveform.size
        if state.denoiser is not None:
            waveform = state.denoiser.enhance(waveform, self.cfg.sample_rate)
        self.engine.feed_waveform_to_stream(state.stream, waveform)

    async def _notify_stop(self, writer, state, reason):
        if state.stop_notified:
            return
        state.stop_notified = True
        state.stop_reason = reason
        fmt = state.audio_format
        timestamp = 0 if fmt is None else round(
            1000 * state.total_bytes / (fmt.rate * fmt.width * fmt.channels)
        )
        await write_message(writer, "voice-stopped", {"timestamp": timestamp, "reason": reason})
        LOGGER.info("Early input stop: reason=%s timestamp=%dms", reason, timestamp)

    def _finish_audio(self, peer, state):
        if state.endpoint is not None:
            state.endpoint.finish()
            if state.denoiser is not None and state.accepted_samples > 0:
                self.engine.feed_waveform_to_stream(state.stream, state.denoiser.flush())
        else:
            if state.denoiser is not None:
                self._process_segments(peer, state, [state.denoiser.flush()])
            self._flush_pending_speaker_gate(peer, state, force=True)

    async def _finish_session(self, peer, writer, state):
        if state.finished:
            return  # Includes duplicate/late audio-stop and in-flight chunks.
        state.finished = True
        text = ""
        start_time = time.monotonic()
        if not state.over_limit and state.stream is not None:
            await self._model_call(lambda: self._finish_audio(peer, state))
            # Endpoint can also be reached while flushing a final partial window.
            if state.endpoint is not None and state.endpoint.stopped:
                await self._notify_stop(writer, state, "speaker-rejected")
            text = await self._model_call(lambda: self.engine.finish_stream(state.stream))
        await self._save_capture(state, "audio-limit" if state.over_limit else state.stop_reason, text)
        state.stream = None
        state.denoiser = None
        state.endpoint = None
        state.gate_pending_waveform = np.empty(0, dtype=np.float32)
        state.gate_segments = []
        await write_message(writer, "transcript", {
            "text": text, "language": state.transcribe_opts.get("language", "zh"),
        })
        LOGGER.info("[%s] Recognition completed: finish=%.3fs accepted=%.2fs text=%r",
                    peer, time.monotonic() - start_time,
                    state.accepted_samples / self.cfg.sample_rate, text)

    async def _save_capture(self, state, reason, text=""):
        capture = state.capture
        if capture is None:
            return
        state.capture = None
        pcm = bytes(getattr(state.stream, "pcm_buffer", b""))
        await asyncio.to_thread(capture.save, pcm, reason, text)

    def _get_peername(self, writer) -> str:
        """获取客户端地址，安全处理异常。"""
        try:
            peer = writer.get_extra_info("peername")
            return str(peer) if peer else "unknown"
        except Exception:
            return "unknown"

    def _get_info(self) -> dict[str, Any]:
        """获取 Wyoming 协议 info 响应（带缓存）。"""
        if self._info_cache is None:
            self._info_cache = {
                "asr": [
                    {
                        "name": self.cfg.model_name,
                        "attribution": {
                            "name": "k2-fsa/sherpa-onnx",
                            "url": "https://github.com/k2-fsa/sherpa-onnx",
                        },
                        "installed": True,
                        "models": [
                            {
                                "name": self.cfg.model_name,
                                "languages": ["zh", "en"],
                                "attribution": {
                                    "name": "k2-fsa/sherpa-onnx",
                                    "url": "https://github.com/k2-fsa/sherpa-onnx",
                                },
                                "installed": True,
                                "description": "Qwen3-ASR via sherpa-onnx",
                            }
                        ],
                        "supports_transcript_streaming": False,
                    }
                ],
            }
        return self._info_cache

    def _process_segments(
        self,
        peer: str,
        state: SessionState,
        segments: list,
    ) -> None:
        if state.stream is None:
            return
        for segment in segments:
            if segment.size == 0:
                continue
            state.detected_segments += 1
            if self.speaker_gate is None:
                self.engine.feed_waveform_to_stream(state.stream, segment)
                state.accepted_segments += 1
                state.accepted_samples += len(segment)
                seg_sec = len(segment) / float(self.cfg.sample_rate)
                LOGGER.debug(
                    "[%s] Segment #%d fed to ASR: dur=%.2fs samples=%d accepted=%d rejected=%d asr_fed=%.2fs",
                    peer,
                    state.detected_segments,
                    seg_sec,
                    len(segment),
                    state.accepted_segments,
                    state.rejected_segments,
                    state.accepted_samples / float(self.cfg.sample_rate),
                )
                continue

            if state.gate_pending_waveform.size == 0:
                state.gate_pending_start_idx = state.detected_segments
                state.gate_pending_waveform = np.ascontiguousarray(segment, dtype=np.float32)
            else:
                state.gate_pending_waveform = np.concatenate(
                    (state.gate_pending_waveform, segment)
                )
            self._flush_pending_speaker_gate(peer, state, force=False)

    def _flush_pending_speaker_gate(
        self, peer: str, state: SessionState, force: bool
    ) -> None:
        if self.speaker_gate is None:
            return
        if state.stream is None:
            return
        if state.gate_pending_waveform.size == 0:
            return

        gate_window = max(1, int(self.cfg.sample_rate * _SPEAKER_GATE_WINDOW_SECONDS))

        while state.gate_pending_waveform.size >= gate_window:
            seg = np.ascontiguousarray(state.gate_pending_waveform[:gate_window], dtype=np.float32)
            self._eval_gate_segment(peer, state, seg, float(self.cfg.speaker_threshold))
            state.gate_pending_waveform = state.gate_pending_waveform[gate_window:]
            state.gate_pending_start_idx = state.detected_segments

        if force and state.gate_pending_waveform.size > 0:
            seg = np.ascontiguousarray(state.gate_pending_waveform, dtype=np.float32)
            seg_sec = seg.size / float(self.cfg.sample_rate)
            effective_threshold = float(self.cfg.speaker_threshold)
            if seg_sec < _SPEAKER_GATE_WINDOW_SECONDS:
                # Tail segment is shorter than fixed window; slightly relax threshold.
                relax = (1.0 - max(0.0, seg_sec / _SPEAKER_GATE_WINDOW_SECONDS)) * 0.10
                effective_threshold = max(0.20, effective_threshold - relax)
            self._eval_gate_segment(peer, state, seg, effective_threshold)
            state.gate_pending_waveform = np.empty((0,), dtype=np.float32)
            state.gate_pending_start_idx = 0
        if force:
            self._flush_selected_gate_segments(peer, state)

    def _flush_selected_gate_segments(self, peer: str, state: SessionState) -> None:
        if state.stream is None or self.speaker_gate is None:
            return
        if not state.gate_segments:
            return

        accepted_positions = [i for i, item in enumerate(state.gate_segments) if item.accepted]
        if not accepted_positions:
            LOGGER.info(
                "[%s] Speaker gate selection: no accepted fixed segments; nothing fed to ASR.",
                peer,
            )
            state.gate_segments = []
            return

        first_pos = accepted_positions[0]
        last_pos = accepted_positions[-1]
        span_start = max(0, first_pos - 1)
        span_end = min(len(state.gate_segments) - 1, last_pos + 1)

        selected_positions = set(accepted_positions)
        selected_positions.update(range(span_start, span_end + 1))
        ordered_positions = sorted(selected_positions)

        fed_samples = 0
        for pos in ordered_positions:
            item = state.gate_segments[pos]
            self.engine.feed_waveform_to_stream(state.stream, item.waveform)
            fed_samples += int(item.waveform.size)
            state.accepted_samples += int(item.waveform.size)
            LOGGER.info(
                "[%s] Fixed segment #%d-%d selected for ASR: dur=%.2fs samples=%d accepted=%d "
                "similarity=%.3f threshold=%.3f matched=%s asr_fed=%.2fs",
                peer,
                item.start_idx,
                item.end_idx,
                item.waveform.size / float(self.cfg.sample_rate),
                item.waveform.size,
                1 if item.accepted else 0,
                item.similarity,
                item.threshold,
                item.speaker_id or "-",
                state.accepted_samples / float(self.cfg.sample_rate),
            )

        LOGGER.info(
            "[%s] Speaker gate selection summary: total=%d accepted=%d selected=%d "
            "first_accept_pos=%d last_accept_pos=%d span=[%d,%d] fed=%.2fs",
            peer,
            len(state.gate_segments),
            len(accepted_positions),
            len(ordered_positions),
            first_pos + 1,
            last_pos + 1,
            span_start + 1,
            span_end + 1,
            fed_samples / float(self.cfg.sample_rate),
        )

        state.gate_segments = []

    def _eval_gate_segment(
        self, peer: str, state: SessionState, segment: np.ndarray, threshold: float
    ) -> None:
        if self.speaker_gate is None or state.stream is None:
            return

        start_idx = state.gate_pending_start_idx or 1
        end_idx = state.detected_segments
        enter_threshold = float(threshold)
        if start_idx <= 1:
            # Relax first segment slightly to avoid missing weak wake-up/command onset.
            enter_threshold = max(0.20, enter_threshold - 0.05)
        keep_threshold = max(0.20, enter_threshold - _SPEAKER_GATE_HYSTERESIS_DELTA)
        use_keep = state.gate_active
        effective_threshold = keep_threshold if use_keep else enter_threshold

        accepted, similarity, speaker_id = self.speaker_gate.accepts_waveform(
            segment,
            self.cfg.sample_rate,
            threshold=effective_threshold,
        )
        state.gate_active = accepted
        samples = int(segment.size)
        seg_sec = samples / float(self.cfg.sample_rate)

        if not accepted:
            state.rejected_segments += 1
            LOGGER.info(
                "[%s] Fixed segment #%d-%d rejected: dur=%.2fs samples=%d similarity=%.3f threshold=%.3f "
                "mode=%s matched=%s accepted=%d rejected=%d",
                peer,
                start_idx,
                end_idx,
                seg_sec,
                samples,
                similarity,
                effective_threshold,
                "keep" if use_keep else "enter",
                speaker_id or "-",
                state.accepted_segments,
                state.rejected_segments,
            )
            state.gate_segments.append(
                GateSegment(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    waveform=np.ascontiguousarray(segment, dtype=np.float32),
                    accepted=False,
                    similarity=float(similarity),
                    threshold=effective_threshold,
                    speaker_id=speaker_id,
                )
            )
            return

        state.accepted_segments += 1
        LOGGER.info(
            "[%s] Fixed segment #%d-%d accepted by gate: dur=%.2fs samples=%d similarity=%.3f "
            "threshold=%.3f mode=%s matched=%s accepted=%d rejected=%d",
            peer,
            start_idx,
            end_idx,
            seg_sec,
            samples,
            similarity,
            effective_threshold,
            "keep" if use_keep else "enter",
            speaker_id or "-",
            state.accepted_segments,
            state.rejected_segments,
        )
        state.gate_segments.append(
            GateSegment(
                start_idx=start_idx,
                end_idx=end_idx,
                waveform=np.ascontiguousarray(segment, dtype=np.float32),
                accepted=True,
                similarity=float(similarity),
                threshold=effective_threshold,
                speaker_id=speaker_id,
            )
        )

    async def run(self) -> None:
        self._server = await asyncio.start_server(self.handle_client, self.cfg.host, self.cfg.port)
        addrs = ", ".join(str(sock.getsockname()) for sock in self._server.sockets or [])
        LOGGER.info("Wyoming server listening on %s", addrs)
        LOGGER.info("Wyoming Qwen3-ASR service started successfully")
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        if self._server is None:
            return

        self._server.close()
        await self._server.wait_closed()
        self._server = None
        LOGGER.info("Wyoming server stopped")
