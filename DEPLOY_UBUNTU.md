# Ubuntu Docker + HA 部署

使用本 Fork 的 `main` 分支。上游原版不包含本次改动；HA 自定义集成另行分发。
当前 Docker/代码未启用 CUDA，ASR 使用默认 CPU，声纹/降噪显式使用 CPU。
RTX 3060 不会被此配置自动使用；没有加入 GPU 依赖或 NVIDIA 容器配置。

## 1. Ubuntu：克隆修改分支

在你的业务目录下执行（例如先进入 `/data`，确保不存在同名项目目录）：

```sh
git clone https://github.com/cuichaobaobao/wyoming-sherpa-onnx.git
cd wyoming-sherpa-onnx
docker compose version
```

部署只使用 `docker-compose.yml`，数据均位于项目内 `data/`。
当前文件包含本机参数，迁移主机时应调整端口绑定的 `192.168.50.20`。

```sh
mkdir -p data/models/vad data/speaker_refs/lichao
```

## 2. Ubuntu：准备 VAD 和注册录音

VAD 不自动下载。执行下面命令下载到容器映射的模型目录：

```sh
curl -fL --retry 3 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx \
  -o "data/models/vad/silero_vad.onnx"
```

来源：[sherpa-onnx 官方下载说明](https://github.com/k2-fsa/sherpa-onnx/blob/master/wasm/vad/assets/README.md)。

准备几段只有你本人清楚说话的录音，保存为真正的 PCM WAV（建议 16 kHz、16-bit、单声道），放到：

```text
data/speaker_refs/lichao/01.wav
data/speaker_refs/lichao/02.wav
```

不能把 MP3/M4A 直接改扩展名。录音必须放在说话人子目录；不能直接放 `speaker_refs/` 根目录。
首次测试只注册你本人，便于观察拒绝其他说话人的效果。服务启动时加载注册文件，更换录音后需重启服务。
没有有效录音或 VAD 文件时不要启动，否则服务会报错退出。

## 3. Ubuntu：构建并启动

```sh
# 在你实际克隆的 wyoming-sherpa-onnx 目录执行
docker compose config
docker compose up -d --build
docker compose logs -f --tail=100
```

当前声纹配置通过 `SPEAKER_MODEL_FILE` 指定文件，须先准备
`data/models/speaker/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx`。本机已经具备该文件。
首次构建需要下载 Python 依赖；启动后自动下载缺少的 Qwen3-ASR 模型。GTCRN 当前关闭，
需要 Ubuntu 能访问相关下载站。看到 `Wyoming server listening` 才表示服务开始监听。
日志若反复报错退出，不要继续配置 HA，先处理具体报错。

模型保存在 `data/models/`，注册音频在 `data/speaker_refs/`。
HA 需要能通过局域网访问 Ubuntu 的 TCP 10300；如有防火墙，仅允许所需局域网来源访问。
不需要公网端口映射。修改代码后使用相同 `up -d --build` 命令重新构建。

## 4. HA：安装独立自定义集成

本集成按 HA Core **2026.9.0** 内部接口设计。安装前核对实际 HA 版本；其他版本兼容性未验证。

取得单独分发的配套 HA 自定义集成后，将其目录放到 HA **配置目录**（本仓库不包含该集成）：

```text
/config/custom_components/wyoming_speaker_stt/manifest.json
/config/custom_components/wyoming_speaker_stt/__init__.py
/config/custom_components/wyoming_speaker_stt/stt.py
...
```

对于 HA Container，这里的 `/config` 是 HA 容器的配置目录，需要复制到其宿主机挂载目录。
对于 HA OS，可使用现有的配置文件访问方式上传。不要把整个 ZIP 再套一层同名目录。
不覆盖 `/config/custom_components/` 中的其他集成，也不改 HA 官方 Wyoming 文件。

1. 重启 Home Assistant。
2. 设置 → 设备与服务 → 添加集成 → 搜索 **Wyoming Speaker STT**。
3. 主机填 Ubuntu 的局域网 IP，端口填 `10300`（不是 HA IP，也不是 Voice PE IP）。
4. 设置 → 语音助手，编辑 Voice PE 正在使用的助手，将语音转文字改成本集成提供的实体。

不需要在自定义集成中另行绑定 Voice PE，也不需要刷 Voice PE 固件。
HA 首次加载会按 manifest 解析/安装 `wyoming==1.10.0` 依赖。

## 5. 验证真实停麦

先确认只有你本人说指令时能正常识别。然后测试你说完指令、另一人接着说话的录音/现场场景。
观察服务日志是否出现 `Early input stop: reason=speaker-rejected`，并核对 Voice PE 是否
先停止本轮输入、进入等待状态，之后 HA 收到之前有效音频的识别文字。
有服务器日志不等于实机已停麦；需要同时查看 HA 流水线与设备行为。

当前 2 秒评分上下文、0.4 秒步长、1.6 秒连续低分人声阈值不代表精确的实际停麦延迟，仍受模型速度、
VAD 起止确认、网络和队列影响。声纹效果需结合实际录音判断。

## 停止和回退

Ubuntu 停止服务（保留模型及注册录音）：

```sh
# 在你实际克隆的 wyoming-sherpa-onnx 目录执行
docker compose down
```

HA 端将语音助手切回原 STT，再移除自定义集成；如需彻底移除，删除本集成目录并重启 HA。

本机已构建并运行容器，使用 sherpa-onnx 1.13.7；这不代表所有声纹场景均能准确处理。
详细处理规则见 [EARLY_STOP.md](EARLY_STOP.md)。
