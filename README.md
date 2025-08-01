# Whisper

[[博客]](https://openai.com/blog/whisper)
[[论文]](https://arxiv.org/abs/2212.04356)
[[模型卡片]](https://github.com/openai/whisper/blob/main/model-card.md)
[[Colab 示例]](https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb)

Whisper 是一个通用的语音识别模型。它在大量多样化音频数据集上训练，也是一个多任务模型，可以执行多语言语音识别、语音翻译和语言识别。

## 🚀 共绩算力平台 - 一键部署 Whisper

### 平台优势

**🎯 一站式解决方案**
- 预构建的 Whisper 容器镜像，开箱即用
- 无需复杂的环境配置和依赖安装
- 支持 GPU 加速，提供强大的计算能力

**💰 成本效益**
- 按需付费，只为使用时间付费
- 弹性算力资源，快速扩容缩容
- 支持多种 GPU 配置（4090、A100 等）

**🔧 企业级服务**
- 高可用性部署，99.9% 服务可用性
- 专业的技术支持和文档
- 支持大规模并发处理

**🌐 便捷访问**
- 自动分配公网域名
- 标准 HTTP API 接口
- 支持多种音频/视频格式

### 快速开始

#### 1. 平台注册
访问 [共绩算力控制台](https://console.suanli.cn) 完成注册

#### 2. 一键部署
1. 进入服务列表，点击"新增部署服务"
2. 选择 GPU 配置（推荐：单卡 4090）
3. 选择"预制镜像"中的 Whisper 镜像
4. 点击部署，等待服务启动

#### 3. 立即使用
- 服务启动后，点击 9000 端口链接访问
- 通过 Web 界面或 API 接口使用
- 支持语音识别、语言检测等功能

### 应用场景

**🎬 媒体制作**
- 视频字幕自动生成
- 播客内容转写
- 会议记录整理

**📱 移动应用**
- 语音输入转文字
- 实时语音翻译
- 语音助手开发

**🏢 企业服务**
- 客服录音分析
- 培训内容整理
- 多语言会议记录

**🎓 教育培训**
- 在线课程字幕
- 语音学习辅助
- 多语言教学支持

### 技术特性

**🔊 多格式支持**
- 音频：MP3, WAV, FLAC, M4A
- 视频：MP4, MOV, AVI, MKV
- 实时流：支持音频流处理

**🌍 多语言识别**
- 支持 99+ 种语言
- 自动语言检测
- 高精度语音转文字

**⚡ 高性能处理**
- GPU 加速推理
- 批量处理支持
- 低延迟实时处理

**🛡️ 安全可靠**
- HTTPS 加密传输
- 数据隐私保护
- 企业级安全标准

## 方法

![方法](https://raw.githubusercontent.com/openai/whisper/main/approach.png)

一个 Transformer 序列到序列模型在各种语音处理任务上训练，包括多语言语音识别、语音翻译、口语语言识别和语音活动检测。这些任务被联合表示为解码器要预测的标记序列，允许单个模型替换传统语音处理管道的多个阶段。多任务训练格式使用一组特殊标记作为任务说明符或分类目标。

## 环境配置

我们使用 Python 3.9.9 和 [PyTorch](https://pytorch.org/) 1.10.1 来训练和测试我们的模型，但代码库预计与 Python 3.8-3.11 和最近的 PyTorch 版本兼容。代码库还依赖于几个 Python 包，最值得注意的是 [OpenAI 的 tiktoken](https://github.com/openai/tiktoken) 用于其快速分词器实现。您可以使用以下命令下载并安装（或更新到）Whisper 的最新版本：

    pip install -U openai-whisper

或者，以下命令将从该仓库拉取并安装最新提交，以及其 Python 依赖项：

    pip install git+https://github.com/openai/whisper.git 

要将包更新到此仓库的最新版本，请运行：

    pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git

它还需要在您的系统上安装命令行工具 [`ffmpeg`](https://ffmpeg.org/)，这可以从大多数包管理器获得：

```bash
# 在 Ubuntu 或 Debian 上
sudo apt update && sudo apt install ffmpeg

# 在 Arch Linux 上
sudo pacman -S ffmpeg

# 在 MacOS 上使用 Homebrew (https://brew.sh/)
brew install ffmpeg

# 在 Windows 上使用 Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# 在 Windows 上使用 Scoop (https://scoop.sh/)
scoop install ffmpeg
```

如果 [tiktoken](https://github.com/openai/tiktoken) 没有为您的平台提供预构建的轮子，您可能还需要安装 [`rust`](http://rust-lang.org)。如果您在上述 `pip install` 命令期间看到安装错误，请按照 [入门页面](https://www.rust-lang.org/learn/get-started) 安装 Rust 开发环境。此外，您可能需要配置 `PATH` 环境变量，例如 `export PATH="$HOME/.cargo/bin:$PATH"`。如果安装失败并显示 `No module named 'setuptools_rust'`，您需要安装 `setuptools_rust`，例如通过运行：

```bash
pip install setuptools-rust
```

## 可用模型和语言

有六种模型大小，其中四种有仅英语版本，提供速度和准确性的权衡。
以下是可用模型的名称及其相对于大型模型的近似内存需求和推理速度。
下面的相对速度是通过在 A100 上转录英语语音来测量的，实际速度可能因许多因素而显著不同，包括语言、说话速度和可用硬件。

|  大小  | 参数数量 | 仅英语模型 | 多语言模型 | 所需显存 | 相对速度 |
|:------:|:--------:|:----------:|:----------:|:--------:|:--------:|
|  tiny  |    39 M  |   `tiny.en` |    `tiny`  |   ~1 GB  |   ~10x   |
|  base  |    74 M  |   `base.en` |    `base`  |   ~1 GB  |    ~7x   |
| small  |   244 M  |  `small.en` |   `small`  |   ~2 GB  |    ~4x   |
| medium |   769 M  | `medium.en` |  `medium`  |   ~5 GB  |    ~2x   |
| large  |  1550 M  |     N/A     |   `large`  |  ~10 GB  |    1x    |
| turbo  |   809 M  |     N/A     |   `turbo`  |   ~6 GB  |    ~8x   |

仅英语应用程序的 `.en` 模型往往表现更好，特别是对于 `tiny.en` 和 `base.en` 模型。我们观察到，对于 `small.en` 和 `medium.en` 模型，差异变得不那么显著。
此外，`turbo` 模型是 `large-v3` 的优化版本，提供更快的转录速度，准确性下降最小。

Whisper 的性能因语言而异。下图显示了 `large-v3` 和 `large-v2` 模型按语言的性能分解，使用在 Common Voice 15 和 Fleurs 数据集上评估的 WER（词错误率）或 CER（字符错误率，以*斜体*显示）。其他模型和数据集对应的额外 WER/CER 指标可以在[论文](https://arxiv.org/abs/2212.04356)的附录 D.1、D.2 和 D.4 中找到，以及附录 D.3 中翻译的 BLEU（双语评估研究）分数。

![按语言的 WER 分解](https://github.com/openai/whisper/assets/266841/f4619d66-1058-4005-8f67-a9d811b77c62)

## 命令行使用

以下命令将使用 `turbo` 模型转录音频文件中的语音：

```bash
whisper audio.flac audio.mp3 audio.wav --model turbo
```

默认设置（选择 `turbo` 模型）对于转录英语效果很好。但是，**`turbo` 模型没有针对翻译任务进行训练**。如果您需要**将非英语语音翻译成英语**，请使用**多语言模型**（`tiny`、`base`、`small`、`medium`、`large`）而不是 `turbo`。

例如，要转录包含非英语语音的音频文件，您可以指定语言：

```bash
whisper japanese.wav --language Japanese
```

要**翻译**语音成英语，请使用：

```bash
whisper japanese.wav --model medium --language Japanese --task translate
```

> **注意：** 即使指定了 `--task translate`，`turbo` 模型也会返回原始语言。使用 `medium` 或 `large` 获得最佳翻译结果。

运行以下命令查看所有可用选项：

```bash
whisper --help
```

有关所有可用语言的列表，请参见 [tokenizer.py](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py)。

## Python 使用

也可以在 Python 中执行转录：

```python
import whisper

model = whisper.load_model("turbo")
result = model.transcribe("audio.mp3")
print(result["text"])
```

在内部，`transcribe()` 方法读取整个文件并使用滑动 30 秒窗口处理音频，对每个窗口执行自回归序列到序列预测。

以下是 `whisper.detect_language()` 和 `whisper.decode()` 的使用示例，它们提供对模型的低级访问。

```python
import whisper

model = whisper.load_model("turbo")

# 加载音频并填充/修剪以适合 30 秒
audio = whisper.load_audio("audio.mp3")
audio = whisper.pad_or_trim(audio)

# 制作对数梅尔频谱图并移动到与模型相同的设备
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# 检测口语语言
_, probs = model.detect_language(mel)
print(f"检测到的语言: {max(probs, key=probs.get)}")

# 解码音频
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# 打印识别的文本
print(result.text)
```

## 更多示例

请在讨论中使用 [🙌 展示和讲述](https://github.com/openai/whisper/discussions/categories/show-and-tell) 类别来分享更多 Whisper 示例用法和第三方扩展，如 Web 演示、与其他工具的集成、不同平台的端口等。

## 📖 详细使用文档

### API 接口说明

部署完成后，您可以通过以下 API 接口使用 Whisper 服务：

#### 语音识别接口（/asr）

**请求方式**：POST

**核心参数**：
- `file`：必填，上传音频/视频文件（.mp3, .wav, .mp4, .mov 等）
- `language`：可选，指定音频语言（如 en、zh），留空自动检测
- `encode`：可选，是否用 ffmpeg 预处理，推荐 true
- `task`：可选，transcribe（转写，默认）/translate（翻译为英文）
- `output`：可选，输出格式 txt/srt/json/vtt

**请求示例**：
```bash
curl -X POST "http://api.example.com/asr?language=zh&encode=true&output=json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test.wav"
```

**返回示例**：
```json
{
  "status": "success",
  "text": "这是识别出来的文本内容。"
}
```

#### 语言检测接口（/detect-language）

**请求方式**：POST

**参数**：
- `file`：必填，上传音频/视频文件

**返回示例**：
```json
{
  "status": "success",
  "language": "zh"
}
```

### Python 调用示例

```python
import requests

# 语音识别
def transcribe_audio(file_path, language="zh"):
    url = "http://your-whisper-service:9000/asr"
    params = {
        "language": language,
        "encode": "true",
        "output": "json"
    }
    
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, params=params, files=files)
    
    return response.json()

# 语言检测
def detect_language(file_path):
    url = "http://your-whisper-service:9000/detect-language"
    
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
    
    return response.json()
```

### 完整部署文档

更多详细信息请参考：[共绩算力 Whisper 部署文档](https://www.gongjiyun.com/docs/y/ofl0wheysi5kwhkh2nfcownhnxf/h8ouwznhgiy9dfkylxdcydmrnqb/)

## 许可证

Whisper 的代码和模型权重在 MIT 许可证下发布。有关更多详细信息，请参见 [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE)。
