# MacWhisper

[English](README.md) | [中文](README_zh.md)

A lightweight macOS menu bar app for dictation and meeting transcripts, powered by [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) running locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). Strong on mixed Chinese and English speech.

**Hold a key, speak, release — your words appear as text instantly.** No cloud API, no subscription, everything runs on-device.

## Features

- **Hold-to-record** — Hold Right Option key to record, release to transcribe and auto-paste
- **Live Subtitles** — Real-time floating overlay showing confirmed (white) and unconfirmed (gray) text as you speak
- **Meeting Mode** — Continuous recording with automatic segmentation, dual-channel mic + system audio capture, Markdown/JSON/SRT export
- **Bilingual support** — Chinese, English, and mixed Chinese-English speech
- **Translate mode** — Toggle translation to English for any spoken language
- **Qwen3-ASR engine (v0.7)** — Qwen3-ASR 0.6B / 1.7B: much more accurate on mixed Chinese-English speech, 5.6× (0.6B) / 2.6× (1.7B) faster than Whisper Large, keeps English verbatim (never translates). Whisper stays loaded as an automatic fallback
- **5 models** — Whisper Small / Medium / Large and Qwen3 0.6B / 1.7B, switchable on the fly
- **Menu bar status** — 🎙 Ready / 🔴 Recording / 💬 Transcribing / 🌐 Translate mode
- **Customizable shortcuts** — defaults: hold Right ⌥ to talk, `⌃⌥M` or `⌃⇧F8` meeting, `⌃⇧M` model, `⌃⇧T` translate, `⌃⇧S` live subtitles; change any of them in **Settings…**
- **Settings window** — every option on one page (menu bar → Settings…, `⌘,`)
- **Persistent settings** — Model choice and all toggles survive app restarts
- **100% local** — All processing on Apple Silicon GPU, no data leaves your machine

## Requirements

- Mac with Apple Silicon (M1 or later). Intel Macs are not supported.
- macOS 14 Sonoma or later
- Python 3.10+ for the app, plus [uv](https://docs.astral.sh/uv/) or Python 3.12 for the speech engine
- Command Line Tools: `xcode-select --install`
- About 4 GB free disk (Qwen3-ASR 0.6B model ~2.9 GB). The 1.7B model needs ~7.6 GB more and downloads only when you select it.

## Installation

> **Install from source.** The Homebrew formula still points to v0.6.1, the older Whisper-based release, and will be updated with the next release.

### 1. Get the code and run the installer

```bash
brew install uv                 # if you have neither uv nor Python 3.12
git clone https://github.com/sfc-gh-jiyan/MacWhisper.git
cd MacWhisper
./install.sh
```

`install.sh` does five things:
1. Creates the app environment `venv` and installs the dependencies
2. Builds `/Applications/MacWhisper.app`, a native launcher that runs the code in this folder
3. Sets up the Qwen3-ASR speech engine in its own environment `.venv-qwen` and downloads the model (the longest step on first install)
4. Builds the system-audio helper, which lets Meeting Mode hear the other people
5. Checks that everything imports

If step 3 or 4 fails, the installer goes on and tells you; run `make qwen-setup` or `make helper` afterwards. Without the helper, meetings record your microphone only.

### 2. Grant permissions, then launch

Open **MacWhisper** from `/Applications`, then grant the permissions below. Relaunch after granting — macOS only applies them at startup.

### Required Permissions (Important!)

Go to **System Settings → Privacy & Security** and turn on `MacWhisper` in:

| Permission | Needed for |
|---|---|
| **Microphone** | Recording your voice (macOS asks on first use) |
| **Accessibility** | Pasting the text into the app you are typing in |
| **Input Monitoring** | Detecting the Right Option key and other shortcuts |
| **Screen & System Audio Recording** | Meeting Mode only: hearing the other people |

For Accessibility and Input Monitoring, click **+**, choose `MacWhisper` in `/Applications`, then turn it on.

> **Running from Terminal?** If you use `./run.sh` instead of the app, your **terminal app** (Terminal, iTerm, Warp, …) needs these permissions instead.

> **Permissions not taking effect?** Quit and relaunch MacWhisper after granting. Check `~/.macwhisper/logs/macwhisper.log` for `Permissions (...): accessibility=True input_monitoring=True`. If a switch looks on but the log says `False`, quit MacWhisper, run `make reset-permissions`, and grant both again.
>
> **No "Meeting saved" banner?** The notification goes to Notification Center. To see a banner, open System Settings → Notifications → MacWhisper and choose **Banners** or **Alerts**. If you use a Focus mode, add MacWhisper to its allowed apps.

### Updating

```bash
git pull
```

Then quit and relaunch MacWhisper — the app always runs the code in this folder. Run `make app` only if you move the folder or recreate `venv`, and run `make helper` if `swift/` changed.

### Optional: local AI text cleanup

**Settings → What gets pasted** can tidy the text with a local model (Qwen3-4B, ~4.2 GB, downloaded on first use). It is off by default. To use it:

```bash
make llm-setup
```

### Uninstall

```bash
rm -rf /Applications/MacWhisper.app     # the app
rm -rf ~/.macwhisper                    # settings, logs, vocabulary, meeting transcripts
```

Then delete the cloned folder. Downloaded models stay in `~/.cache/huggingface/` until you remove them.

## Usage

**Option A** — Double-click `MacWhisper` in Applications (or open it from Launchpad / Spotlight). To start it automatically, turn on **Settings… → General → Launch MacWhisper at login**.

**Option B** — Run from terminal (for development; Terminal then needs the permissions):
```bash
./run.sh
```

A microphone icon (🎙) appears in your menu bar. That's it — you're ready.

Only one MacWhisper runs at a time. Opening it again while it is running just shows a notification. Quit the running one first (menu bar → Quit) if you want to switch between the app and `./run.sh`; `./run.sh` stops a running one for you.

### Controls

| Action | How |
|---|---|
| **Record & transcribe** | Hold **Right ⌥**, speak, release |
| **Meeting Mode** | **⌃⌥M** or **⌃⇧F8**, the menu, or **■ Stop** in the meeting subtitle window |
| **Switch model** | **⌃⇧M** or the menu |
| **Live Subtitles** | **⌃⇧S** or Settings |
| **Toggle translation** | **⌃⇧T** or Settings |
| **All options / change shortcuts** | Menu bar → **Settings…** (`⌘,`) |
| **Quit** | Click menu bar icon → Quit |

All shortcuts are defaults and can be changed in Settings: click a shortcut, press the new keys (Esc cancels). Push to Talk can be Right ⌥/⌘/⌃/⇧ or F1–F20. Meeting Mode moved from ⌃⇧R (v0.7.0 and earlier) to ⌃⌥M / ⌃⇧F8.

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

### Qwen3-ASR vs Whisper (v0.7)

Measured on real recordings (mixed Chinese-English):

| | Whisper Large | Qwen3 0.6B | Qwen3 1.7B |
|---|---|---|---|
| Meeting, mixed error rate (2 recordings, lower is better) | 0.627 / 0.203 real-time, 0.729 / 0.359 full pass | **0.085 / 0.188** | **0.119 / 0.156** |
| Push to Talk speed (221 clips, total time) | 1× (523 s) | **5.6× faster** (94 s) | **2.6× faster** (201 s) |
| English inside Chinese speech | sometimes translated | kept verbatim (0 / 221 translated) | kept verbatim |
| Live subtitles vs final text (14 clips, simulated) | live text ~49% different from final | first text ~1 s; live ≈ final (~1% apart) | not measured |
| Translate to English | ✅ | ❌ (Whisper is used automatically) | ❌ |
| Word timestamps | ✅ | ❌ | ❌ |

**What uses which engine** when a Qwen3 model is selected:

| | Engine |
|---|---|
| Push to Talk — live subtitles and final text | Qwen3 |
| Push to Talk in Translate mode | Whisper |
| Meeting Mode — real-time subtitles | Qwen3, one sentence at a time (v0.7.3); Whisper takes over if Qwen3 fails mid-meeting |
| Meeting Mode — final transcript after Stop | Qwen3, from in-memory audio (no audio file written) |
| Any Qwen3 failure (not installed, crash, timeout) | Whisper on the same audio + one red alert |

**Going back to Whisper:** pick any Whisper model in **Push to Talk → Model**. That is saved as `"engine": "whisper"` and is kept across restarts.

## Configuration

Settings are stored in `~/.macwhisper/config.json`:

```json
{
  "translate_mode": false,
  "current_model": "mlx-community/whisper-medium-mlx",
  "live_mode": false,
  "save_audio": false,
  "log_level": "WARNING",
  "engine": "qwen",
  "qwen_model": "mlx-community/Qwen3-ASR-0.6B-bf16",
  "qwen_strip_fillers": true,
  "qwen_tight_spacing": true,
  "meeting_final_pass": true
}
```

- `engine` — `"qwen"` or `"whisper"`. `current_model` always names the Whisper model, which stays loaded as the fallback.
- `qwen_strip_fillers` — remove standalone 嗯 / 呃 / um from Qwen3 output (never inside words such as 额外).
- `qwen_tight_spacing` — no space between Chinese and English (`这个commit的情况`), matching Whisper's style.
- `meeting_final_pass` — re-transcribe a meeting with Qwen3 after Stop (in memory only).

## Tech Stack

- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon ML framework
- [Qwen3-ASR](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16) via [mlx-audio](https://github.com/Blaizzy/mlx-audio) — speech recognition on Apple GPU
- [rumps](https://github.com/jaredks/rumps) — macOS menu bar apps in Python
- [pynput](https://github.com/moses-palmer/pynput) — Global keyboard listener
- [sounddevice](https://python-sounddevice.readthedocs.io/) — Audio input via PortAudio

## License

MIT
