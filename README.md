# MacWhisper

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
- Microphone permission
- Accessibility permission (for global hotkey)

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/MacWhisper.git
cd MacWhisper

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
source venv/bin/activate
python3 app.py
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
| 🔴 | Recording... |
| 💬 | Transcribing... |

## Model Comparison

We tested all three models with the same bilingual Chinese-English test script. Results below.

**Test script (spoken aloud):**

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
| **Small** | ~460 MB | ~5s | Low — struggles with English words in Chinese speech | Quick drafts, pure single-language use |
| **Medium** | ~1.5 GB | ~10s | Good — handles bilingual well, occasional proper noun errors | Daily use (recommended) |
| **Large** | ~3 GB | ~30s | Excellent — proper nouns, numbers, formatting all correct | Important meetings, formal documents |

> **Note:** "First Load" refers to the initial JIT compilation time when a model is used for the first time after app launch. Subsequent transcriptions with the same model are near-instant.

## Configuration

Settings are stored in `~/.macwhisper_config.json`:

```json
{
  "translate_mode": false,
  "current_model": "mlx-community/whisper-medium-mlx"
}
```

## Creating a .app Bundle (Optional)

To launch MacWhisper from your Applications folder:

```bash
# Create the app structure
mkdir -p /Applications/MacWhisper.app/Contents/MacOS

# Create the launcher script
cat > /Applications/MacWhisper.app/Contents/MacOS/MacWhisper << 'EOF'
#!/bin/bash
cd /path/to/MacWhisper
exec ./venv/bin/python3 app.py
EOF
chmod +x /Applications/MacWhisper.app/Contents/MacOS/MacWhisper

# Create Info.plist
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

## Tech Stack

- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon ML framework
- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — Whisper optimized for Apple GPU
- [rumps](https://github.com/jaredks/rumps) — macOS menu bar apps in Python
- [pynput](https://github.com/moses-palmer/pynput) — Global keyboard listener
- [sounddevice](https://python-sounddevice.readthedocs.io/) — Audio input via PortAudio

## License

MIT
