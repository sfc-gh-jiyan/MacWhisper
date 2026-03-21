# MacWhisper

A lightweight macOS menu bar app for real-time voice transcription and translation, powered by [OpenAI Whisper](https://github.com/openai/whisper) running locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

一款轻量级 macOS 菜单栏语音转录工具，基于 [OpenAI Whisper](https://github.com/openai/whisper)，通过 [MLX](https://github.com/ml-explore/mlx) 在 Apple Silicon 上本地运行。

**Hold a key, speak, release — your words appear as text instantly.** No cloud API, no subscription, everything runs on-device.

**按住快捷键，说话，松开 — 文字立即出现。** 无需云端 API，无需订阅，一切在本地完成。

## Features / 功能

- **Hold-to-record / 按住录音** — Hold Right Option key to record, release to transcribe and auto-paste / 按住右 Option 键录音，松开自动转录并粘贴
- **Bilingual support / 双语支持** — Chinese, English, and mixed Chinese-English speech / 中文、英文、中英混合语音
- **Translate mode / 翻译模式** — Toggle translation to English for any spoken language / 一键切换，将任何语言翻译成英文
- **3 model sizes / 三种模型** — Switch between Small, Medium, and Large models on the fly / 随时切换 Small、Medium、Large 模型
- **Menu bar status / 菜单栏状态** — 🎙 Ready / 🔴 Recording / 💬 Transcribing / 🌐 Translate mode
- **Keyboard shortcuts / 快捷键** — `Ctrl+Shift+M` to cycle models, `Ctrl+Shift+T` to toggle translation
- **Persistent settings / 设置持久化** — Model choice and translate mode survive app restarts / 模型和翻译模式重启后保留
- **100% local / 完全本地** — All processing on Apple Silicon GPU, no data leaves your machine / 全部在 Apple Silicon GPU 上处理，数据不离开你的电脑

## Requirements / 系统要求

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- Microphone permission / 麦克风权限
- Accessibility permission (for global hotkey) / 辅助功能权限（全局快捷键）

## Installation / 安装

```bash
git clone https://github.com/YOUR_USERNAME/MacWhisper.git
cd MacWhisper

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage / 使用

```bash
source venv/bin/activate
python3 app.py
```

A microphone icon (🎙) appears in your menu bar. That's it — you're ready.

菜单栏出现麦克风图标（🎙），即可使用。

### Controls / 操作方式

| Action / 操作 | How / 方式 |
|---|---|
| **Record & transcribe / 录音转录** | Hold **Right Option** / 按住右 Option 键 |
| **Switch model / 切换模型** | Click menu bar icon, or **Ctrl+Shift+M** / 点菜单栏图标，或 Ctrl+Shift+M |
| **Toggle translation / 翻译开关** | Click menu bar icon, or **Ctrl+Shift+T** / 点菜单栏图标，或 Ctrl+Shift+T |
| **Quit / 退出** | Click menu bar icon → Quit / 点菜单栏图标 → Quit |

### Menu Bar Icons / 菜单栏图标

| Icon / 图标 | State / 状态 |
|---|---|
| 🎙 | Ready (transcribe mode) / 就绪（转录模式） |
| 🌐 | Ready (translate mode) / 就绪（翻译模式） |
| 🔴 | Recording / 录音中 |
| 💬 | Transcribing / 转录中 |

## Model Comparison / 模型对比

We tested all three models with the same bilingual Chinese-English test script.

我们用同一段中英混合的测试台词，对比了三个模型的准确率。

**Test script (spoken aloud) / 测试台词（朗读）：**

> 我上周在San Francisco参加了一个conference，speaker是一个叫做Andrew Ng的professor。他讲了about artificial intelligence and machine learning。会议是在March 15th，大概有2000 people参加。The ticket price was $299 per person。

### Results / 结果

| Test Item / 测试项 | Expected / 预期 | Small | Medium | Large |
|---|---|---|---|---|
| San Francisco | San Francisco | ❌ 三方西斯与庆祝 | ✅ San Francisco | ✅ San Francisco |
| conference | conference | ❌ 订阅会 | ✅ conference | ✅ conference |
| Andrew Ng | Andrew Ng | ❌ AndroidNG | ❌ Andre NG | ✅ Andrew N.G. |
| professor | professor | ❌ 教授 | ✅ professor | ✅ professor |
| March 15th | March 15th | ⚠️ march 15 | ⚠️ March 15 | ✅ March 15th |
| 2000 people | 2000 people | ❌ 2000批会 | ⚠️ 2000人 | ✅ 2000 people |
| ticket price | ticket price | ❌ taking surprise | ✅ ticket price | ✅ ticket price |
| $299 | $299 | ❌ 299 | ⚠️ 299 | ✅ $299 |

### Summary / 总结

| Model / 模型 | Size / 大小 | First Load / 首次加载 | Accuracy / 准确率 | Best For / 适用场景 |
|---|---|---|---|---|
| **Small** | ~460 MB | ~5s | Low / 低 — struggles with English in Chinese speech / 中文中夹杂英文时容易出错 | Quick drafts / 快速草稿 |
| **Medium** | ~1.5 GB | ~10s | Good / 良好 — handles bilingual well / 双语处理好，偶尔专有名词出错 | Daily use (recommended) / 日常使用（推荐） |
| **Large** | ~3 GB | ~30s | Excellent / 优秀 — proper nouns, numbers, formatting all correct / 专有名词、数字、格式全对 | Important meetings / 重要会议、正式文档 |

> **Note / 注意：** "First Load" refers to the initial JIT compilation time when a model is used for the first time after app launch. Subsequent transcriptions are near-instant.
>
> "首次加载"指的是 app 启动后第一次使用某个模型的 JIT 编译时间，之后同一模型的转录几乎是即时的。

## Configuration / 配置

Settings are stored in / 设置保存在 `~/.macwhisper_config.json`:

```json
{
  "translate_mode": false,
  "current_model": "mlx-community/whisper-medium-mlx"
}
```

## Creating a .app Bundle (Optional) / 创建 .app（可选）

To launch MacWhisper from your Applications folder:

从"应用程序"文件夹启动 MacWhisper：

```bash
mkdir -p /Applications/MacWhisper.app/Contents/MacOS

cat > /Applications/MacWhisper.app/Contents/MacOS/MacWhisper << 'EOF'
#!/bin/bash
cd /path/to/MacWhisper
exec ./venv/bin/python3 app.py
EOF
chmod +x /Applications/MacWhisper.app/Contents/MacOS/MacWhisper

cat > /Applications/MacWhisper.app/Contents/Info.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>MacWhisper</string>
  <key>CFBundleDisplayName</key>
  <string>MacWhisper</string>
  <key>CFBundleIdentifier</key>
  <string>com.macwhisper.app</string>
  <key>CFBundleExecutable</key>
  <string>MacWhisper</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
</dict>
</plist>
EOF
```

Replace `/path/to/MacWhisper` with the actual path to your cloned repository.

将 `/path/to/MacWhisper` 替换为你的实际项目路径。

## Tech Stack / 技术栈

- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon ML framework / Apple Silicon 机器学习框架
- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — Whisper optimized for Apple GPU / 针对 Apple GPU 优化的 Whisper
- [rumps](https://github.com/jaredks/rumps) — macOS menu bar apps in Python / Python macOS 菜单栏应用框架
- [pynput](https://github.com/moses-palmer/pynput) — Global keyboard listener / 全局键盘监听
- [sounddevice](https://python-sounddevice.readthedocs.io/) — Audio input via PortAudio / 通过 PortAudio 进行音频输入

## License

MIT
