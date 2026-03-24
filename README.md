# MacWhisper

[English](README.md) | [中文](README_zh.md)

A lightweight macOS menu bar app for real-time voice transcription and translation, powered by [OpenAI Whisper](https://github.com/openai/whisper) running locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

**Hold a key, speak, release — your words appear as text instantly.** No cloud API, no subscription, everything runs on-device.

## Features

- **Hold-to-record** — Hold Right Option key to record, release to transcribe and auto-paste
- **Bilingual support** — Chinese, English, and mixed Chinese-English speech
- **Translate mode** — Toggle translation to English for any spoken language
- **3 model sizes** — Switch between Small, Medium, and Large models on the fly
- **Menu bar status** — 🎙 Ready / 🔴 Recording / 💬 Transcribing / 🌐 Translate mode
- **Keyboard shortcuts** — `Ctrl+Shift+M` to cycle models, `Ctrl+Shift+T` to toggle translation
- **Persistent settings** — Model choice and translate mode survive app restarts
- **100% local** — All processing on Apple Silicon GPU, no data leaves your machine

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+

## Installation

### Option 1: Homebrew (Recommended)

```bash
# Step 1 — Add the tap (one time only)
brew tap sfc-gh-jiyan/macwhisper

# Step 2 — Install
brew install macwhisper-mlx

# Step 3 — Create MacWhisper.app in /Applications
macwhisper-mlx-install
```

### Option 2: From Source

```bash
git clone https://github.com/sfc-gh-jiyan/MacWhisper.git
cd MacWhisper
./install.sh
```

The install script will:
1. Create a Python virtual environment and install dependencies
2. Build a `MacWhisper.app` in `/Applications`
3. Pre-download the default Whisper model (~1.5 GB)
4. Verify all dependencies are working

### Required Permissions (Important!)

MacWhisper needs **three** macOS permissions to function. **The app will not work without these.**

Go to **System Settings → Privacy & Security**:

**1. Microphone**
- Navigate to: Privacy & Security → Microphone
- Toggle **ON** for `MacWhisper`
- *Why: needed to record audio from your microphone*

**2. Accessibility**
- Navigate to: Privacy & Security → Accessibility
- Click the **+** button, find and add `MacWhisper` from `/Applications`
- Toggle **ON** for `MacWhisper`
- *Why: needed to simulate Cmd+V to auto-paste transcribed text*

**3. Input Monitoring**
- Navigate to: Privacy & Security → Input Monitoring
- Click the **+** button, find and add `MacWhisper` from `/Applications`
- Toggle **ON** for `MacWhisper`
- *Why: needed to detect the Right Option key as a global hotkey*

> **Running from Terminal?** If you use `macwhisper-mlx` or `./run.sh` instead of the .app, you must also add your **terminal app** (Terminal, iTerm, Warp, etc.) to **all three** permission categories above.

> **Permissions not taking effect?** After granting permissions, you may need to quit and relaunch MacWhisper. In some cases, a system restart is needed for Input Monitoring to activate.

## Usage

**Option A** — Open `MacWhisper` from your Applications folder (Launchpad or Spotlight)

**Option B** — Run from terminal:
```bash
./run.sh
```

A microphone icon (🎙) appears in your menu bar. That's it — you're ready.

### Controls

| Action | How |
|---|---|
| **Record & transcribe** | Hold **Right Option**, speak, release |
| **Switch model** | Click menu bar icon, or **Ctrl+Shift+M** |
| **Toggle translation** | Click menu bar icon, or **Ctrl+Shift+T** |
| **Quit** | Click menu bar icon → Quit |

### Menu Bar Icons

| Icon | State |
|---|---|
| 🎙 | Ready (transcribe mode) |
| 🌐 | Ready (translate mode) |
| 🔴 | Recording |
| 💬 | Transcribing |

## Model Comparison

All three models were tested with the same bilingual Chinese-English script:

> 我上周在San Francisco参加了一个conference，speaker是一个叫做Andrew Ng的professor。他讲了about artificial intelligence and machine learning。会议是在March 15th，大概有2000 people参加。The ticket price was $299 per person。

### Results

| Test Item | Expected | Small | Medium | Large |
|---|---|---|---|---|
| San Francisco | San Francisco | ❌ 三方西斯与庆祝 | ✅ San Francisco | ✅ San Francisco |
| conference | conference | ❌ 订阅会 | ✅ conference | ✅ conference |
| Andrew Ng | Andrew Ng | ❌ AndroidNG | ❌ Andre NG | ✅ Andrew N.G. |
| professor | professor | ❌ 教授 | ✅ professor | ✅ professor |
| March 15th | March 15th | ⚠️ march 15 | ⚠️ March 15 | ✅ March 15th |
| 2000 people | 2000 people | ❌ 2000批会 | ⚠️ 2000人 | ✅ 2000 people |
| ticket price | ticket price | ❌ taking surprise | ✅ ticket price | ✅ ticket price |
| $299 | $299 | ❌ 299 | ⚠️ 299 | ✅ $299 |

### Summary

| Model | Size | First Load | Accuracy | Best For |
|---|---|---|---|---|
| **Small** | ~460 MB | ~5s | Low — struggles with English in Chinese speech | Quick drafts, single-language |
| **Medium** | ~1.5 GB | ~10s | Good — handles bilingual well, occasional proper noun errors | Daily use (recommended) |
| **Large** | ~3 GB | ~30s | Excellent — proper nouns, numbers, formatting all correct | Important meetings, formal documents |

> **Note:** "First Load" is the JIT compilation time on first use after app launch. Subsequent transcriptions with the same model are near-instant.

## Configuration

Settings are stored in `~/.macwhisper_config.json`:

```json
{
  "translate_mode": false,
  "current_model": "mlx-community/whisper-medium-mlx"
}
```

## Tech Stack

- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon ML framework
- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — Whisper optimized for Apple GPU
- [rumps](https://github.com/jaredks/rumps) — macOS menu bar apps in Python
- [pynput](https://github.com/moses-palmer/pynput) — Global keyboard listener
- [sounddevice](https://python-sounddevice.readthedocs.io/) — Audio input via PortAudio

## License

MIT
