# 连续声纹拒绝结束（工作区版本）

本地 STT 仍叫 `wyoming-sherpa-onnx`；HA 端使用工作区中独立的
`custom_components/wyoming_speaker_stt/`。本功能已完成源码及模拟测试，
未部署、未加载真实模型、未在真实 HA / Voice PE 验证。

## 音频如何处理

`未增强音频 → Silero VAD 标记人声 → 短窗口声纹评分 → 选择有效音频 → 可选 GTCRN → Qwen3-ASR`

- VAD 使用项目已有 sherpa-onnx 1.13.3 的 `VoiceActivityDetector`，按 512 点读取。
  通过 `current_segment` 增量读取尚未结束的人声，不等完整静音后再评分。
  模型有 0.25 秒起声确认、0.2 秒静音确认；VAD 区域可能含边缘静音。
- 默认每 0.8 秒 VAD 人声做一次声纹评分。首次高分锁定本轮注册说话人，之后
  对该声纹评分；其他已注册说话人不能自动接管本轮。下一轮重新选择。
- 高分（默认 ≥ 0.40）：保留。
- 临界分（默认 0.30 ≤ 分数 < 0.40）：最多暂存一窗；仅当前后紧邻窗口均为
  本轮说话人的高分时保留。开头、结尾、连续临界窗口、跨 VAD 段的临界片段丢弃。
- 低分（默认 < 0.30）：直接丢弃，绝不因为后续高分重新补入。
- 连续低分人声累计到 1.6 秒即结束。高分、临界分或无效声纹打断计数；
  无人声不计时。两个低分片段之间的静音仅暂停计数，不视为一次新评分。
- 小于 0.4 秒的尾段、全零/无效声纹向量不作为“低分证据”，丢弃而不触发结束。
- 达到结束条件后冻结有效音频，后续在途音频丢弃；通知发出后再完成识别。
  只有拒绝音频则返回空文字，不编造指令。30 秒总音频上限保留，超限则通知结束并返回空文字。

这些阈值是可调初值，不是已经验证的声纹准确率。短窗、重叠说话及 VAD 漏检仍可能
造成误判或漏词；声纹分数不是概率，也没有加入语义端点模型或说话人分离模型。
只有原说话人停止、另一个人继续说话的场景，才有清晰的连续拒绝窗口；同时说话不能保证分离。

## 双方约定

为了保持旧客户端行为，必须同时满足服务端 `SPEAKER_EARLY_STOP=true` 和客户端
`transcribe.data.speaker_early_stop=true`，才启用上述新处理链路。
自定义集成已发送这个字段；原生 HA Wyoming 集成不发送，仍走原门控和 `audio-stop` 流程。
服务端未开启时，自定义集成仍能走普通识别，但没有连续声纹拒绝停麦能力。

1. 服务发送一次 `voice-stopped`，附音频毫秒时间戳和 `reason=speaker-rejected`（或 `audio-limit`）。
2. 自定义集成并发接收，触发当前 Pipeline 的 `STT_VAD_END`，停止音频发送并回一次 `audio-stop`。
3. 服务不等待这个确认才解码；保留连接并发送一次最终 `transcript`。
4. 重复 `audio-stop`、迟到 `audio-chunk` 不会重复识别或返回第二份文字。

`voice-stopped` 是 Wyoming 已有 VAD 事件，但本 ASR 用法、`speaker_early_stop` 字段和
`reason` 是双方扩展约定。它不是原生 Wyoming STT 已支持的官方结束握手。
HA 端通过内部 Pipeline 方法的运行时包装发出事件，不是稳定的官方扩展 API。

自定义集成按 Cloud STT v2 的音频偏好请求关闭增益/降噪，在支持的 Voice PE 上选择
未增强声道，同时保留 HA 外部 VAD 作为静音结束条件。停的是本轮对话音频输入，
不是关闭本地唤醒词功能或硬件静音开关。

## 将来启用时需要的参数（本次未运行）

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SPEAKER_GATE` | `false` | 必须开启且已有注册声纹 |
| `SPEAKER_EARLY_STOP` | `false` | 新功能开关 |
| `SPEAKER_THRESHOLD` | `0.40` | 高分阈值 |
| `SPEAKER_LOW_THRESHOLD` | `0.30` | 低分边界，必须低于高分阈值 |
| `SPEAKER_WINDOW_SECONDS` | `0.8` | 评分窗，允许 0.4–3 秒 |
| `SPEAKER_REJECT_SECONDS` | `1.6` | 连续低分人声时长 |
| `VAD_MODEL` | `data/models/vad/silero_vad.onnx` | 按项目目录解析的默认模型路径 |
| `VAD_THRESHOLD` | `0.5` | Silero VAD 阈值 |

对应 CLI 参数为 `--speaker-early-stop`、`--speaker-low-threshold`、
`--speaker-window-seconds`、`--speaker-reject-seconds`、`--vad-model`、`--vad-threshold`。
新模式只支持 16 kHz 模型输入。VAD 模型必须提前存在；缺失则明确报错，不自动下载。
Docker 用户需要把新增环境变量显式写入 Compose 的 `environment`，仅设置宿主机环境变量不会自动传入容器。
现有 Compose 未被修改或启动。

每个请求有独立 VAD、GTCRN 和计数状态。模型调用在线程中进行，共享声纹/ASR 通过锁串行使用；
单次识别不会阻塞 asyncio 接收方，但繁忙时其他请求的模型调用需要排队。未做吞吐/延迟实测。

## 工作区验证

在本 STT 仓库根目录运行（Python 3.10+，需要 NumPy）：

```sh
PYTHONPATH=. python3 -B -m unittest discover -s tests -p 'test_*.py' -v
```

本仓库只包含 STT 服务，不包含 HA 自定义集成。独立运行时，29 项服务端/策略测试执行，
1 项依赖配套 HA 源码的跨项目测试明确跳过；如需执行该项，请通过 PYTHONPATH 提供配套集成的父目录。
完整工作区曾通过 30 项服务端/两端测试及另外 27 项 HA 集成测试，不能将其当成独立仓库的 57 项测试。
测试使用真实服务端处理函数、协议编解码、声纹选择逻辑；有配套源码时还使用自定义集成会话与 Pipeline 适配器；
传输通过内存字节流，模型推理和 HA/设备回调用明确替身。覆盖先结束后出文字、拒绝片段不回填、
未知声纹、临界分、重复消息、并发请求、旧客户端、超限和取消清理。
这不证明真实模型识别率，也不等同于已验证 Voice PE 实机停麦。

## 源码依据

- [sherpa-onnx 1.13.3 VAD Python API](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.3/sherpa-onnx/python/csrc/voice-activity-detector.cc)
- [HA 2026.9.0 Cloud 音频偏好](https://github.com/home-assistant/core/blob/2026.9.0/homeassistant/components/cloud/stt.py#L42)
- [HA ESPHome 事件转发](https://github.com/home-assistant/core/blob/2026.9.0/homeassistant/components/esphome/assist_satellite.py#L343)
- [ESPHome 2026.8.0 收到结束事件](https://github.com/esphome/esphome/blob/2026.8.0/esphome/components/voice_assistant/voice_assistant.cpp#L979)
- [ESPHome 停止本轮音频源](https://github.com/esphome/esphome/blob/2026.8.0/esphome/components/voice_assistant/voice_assistant.cpp#L406)
