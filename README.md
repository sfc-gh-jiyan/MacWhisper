# MacWhisper

[English](README.md) | [中文](README_zh.md)

A lightweight macOS menu bar app for dictation and meeting transcripts, powered by [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) running locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). Strong on mixed Chinese and English speech.

**Hold a key, speak, release — your words appear as text instantly.** No cloud API, no subscription, everything runs on-device.

## Features

- **Hold-to-record** — Hold Right Option key to record, release to transcribe and auto-paste
- **Live Subtitles** — Real-time floating overlay showing confirmed (white) and unconfirmed (gray) text as you speak
- **Meeting Mode** — Continuous recording with automatic segmentation, mic + system audio capture, Markdown/JSON/SRT export. Each line is labelled **Mic** (you) or **Remote** (the other side of the call); speaker echo is removed and the overlay shows one line per side. An optional final pass re-transcribes each track after Stop (`meeting_final_pass`, on by default)
- **Bilingual support** — Chinese, English, and mixed Chinese-English speech; English is kept verbatim (never translated)
- **Qwen3-ASR engine** — Qwen3-ASR 0.6B (default, fast) or 1.7B (best), switchable on the fly
- **Self-correction** — a vocabulary of your terms (sent to Qwen3 as hotwords), correction rules, teach by voice (⇧ + Push to Talk), daily vocabulary learning with a local LLM, **Review Suggestions…**
- **Optional AI style** — a local LLM can fix typos, smooth the text, tidy the format, or apply your own rules (`make llm-setup`)
- **Never lose a dictation** — failed pastes are reported and **Copy Last Transcript** restores the newest dictation
- **Permission setup** — step-by-step dialogs for Microphone, Accessibility and Input Monitoring (at startup when one is missing, or menu → **Permissions…**)
- **Menu bar status** — 🎙 Ready / 🔴 Recording / 💬 Transcribing
- **Customizable shortcuts** — defaults: hold Right ⌥ to talk, `⌃⌥M` or `⌃⇧F8` meeting, `⌃⇧M` model, `⌃⇧S` live subtitles; change any of them in **Settings…**
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

### 1. Check the requirements

- Apple Silicon Mac (`install.sh` stops on any other architecture)
- `python3` 3.10 or newer on your PATH (`python3 --version`)
- [uv](https://docs.astral.sh/uv/) **or** `python3.12` on your PATH, for the speech engine (`brew install uv`)
- Command Line Tools (`xcode-select --install`): `clang` builds the app launcher, `swift` builds the Meeting Mode helper
- About 4 GB free disk (see [Requirements](#requirements))

### 2. Get the code and run the installer

```bash
git clone https://github.com/sfc-gh-jiyan/MacWhisper.git
cd MacWhisper
bash install.sh
```

`install.sh` prints six steps:

| Step | What it does | Where it lands |
|---|---|---|
| 1/6 | Creates the app's Python environment (skipped if it exists) | `venv/` in this folder |
| 2/6 | Installs `requirements.txt` (takes a few minutes) | `venv/` |
| 3/6 | Builds the native launcher app (`scripts/build_app.sh`) and creates the data folders | `/Applications/MacWhisper.app`, `~/.macwhisper/logs`, `~/.macwhisper/audio` |
| 4/6 | Sets up the Qwen3-ASR speech engine in its own environment and downloads the 0.6B model (~2.9 GB, the longest step) | `.venv-qwen/`; model in `~/.cache/huggingface/` |
| 5/6 | Builds the system-audio helper that lets Meeting Mode hear the other people | `swift/SystemAudioHelper/.build/release/SystemAudioHelper` |
| 6/6 | Checks that the app's dependencies import | — |

If step 4 or 5 fails, the installer goes on and prints a warning. Fix the cause (uv / Python 3.12, or the Command Line Tools), then run `make qwen-setup` or `make helper`. Without the helper, meetings record your microphone only.

`MacWhisper.app` is only a launcher: it runs the code in this folder with `venv`. Do not move or delete the folder after installing (if you move it, run `make app`).

### 3. First launch

1. Open **MacWhisper** from `/Applications` (or Spotlight).
2. If a permission is missing, the setup opens by itself and goes through them one by one — see [Permissions](#permissions).
3. Click **Restart Now** at the end, so Accessibility and Input Monitoring take effect.

### 4. Check that it works

1. A microphone icon (🎙) is in the menu bar.
2. Click into any text field (e.g. Notes), hold **Right ⌥**, say a sentence, release. The icon turns 🔴 while recording, 💬 while transcribing, and the text is pasted.
3. If nothing happens, see [Troubleshooting](#troubleshooting).

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

## Permissions

MacWhisper needs three macOS permissions for Push to Talk, and a fourth for Meeting Mode. All are in **System Settings → Privacy & Security**.

| Permission | Section in Privacy & Security | Used for | Without it |
|---|---|---|---|
| **Microphone** | Microphone | Recording your voice | Nothing is recorded |
| **Accessibility** | Accessibility | Pasting the text into the app you are typing in | Text is not pasted |
| **Input Monitoring** | Input Monitoring | Detecting Right ⌥ and the other shortcuts | Holding Right ⌥ does nothing |
| **Screen & System Audio Recording** (Meeting Mode only) | Screen & System Audio Recording (older macOS: Screen Recording) | Recording the other people in a meeting (system audio) | The helper gets silence; no Remote lines |

### The setup dialogs

At launch, MacWhisper checks the three required permissions. If one is missing, a step-by-step setup opens by itself. You can open it any time from the menu bar: **Permissions…**. For each permission:

1. It explains why it is needed. Click **Allow** (macOS shows its own prompt) or **Open Settings** (opens the right System Settings page). **Skip** moves on.
2. In System Settings, turn on **MacWhisper**.
3. If it is still off, the dialog offers **Check Again**, **Open Settings** and **Skip**.
4. At the end, if you just granted Accessibility or Input Monitoring, click **Restart Now**: macOS applies those two to a running app only after it restarts.

To do it by hand: for Accessibility and Input Monitoring, click **+**, choose `MacWhisper` in `/Applications`, then turn it on.

### Which entry to turn on

- Started from `/Applications/MacWhisper.app`: turn on **MacWhisper**. The launcher embeds Python, so the permissions belong to the app, not to Python or Terminal.
- Started with `./run.sh`: your **terminal app** (Terminal, iTerm, Warp, …) needs the permissions instead.

To check the result, look for this line in `~/.macwhisper/logs/macwhisper.log`: `Permissions (pid …): accessibility=True input_monitoring=True`. `False` means that permission is still off. (The path shown in that line is the embedded Python, even when started from the app; turn on **MacWhisper**, not Python.)

## Troubleshooting

**Holding Right ⌥ does nothing, or the text is not pasted.** Open **Permissions…** from the menu and check Input Monitoring (key) and Accessibility (paste), then restart MacWhisper. If a switch looks on but the log says `False`: rebuilding the app can leave a permission entry that no longer matches it. Quit MacWhisper, run `make reset-permissions`, open it again and grant the permissions once more.

**A meeting has no Remote lines.** Check that the helper exists (`make helper`; needs `xcode-select --install`) and that **Screen & System Audio Recording** is on for MacWhisper, then restart it.

**No "Meeting saved" banner.** The notification goes to Notification Center. To see a banner, open System Settings → Notifications → MacWhisper and choose **Banners** or **Alerts**. If you use a Focus mode, add MacWhisper to its allowed apps.

**The app does not open on a company Mac.** `MacWhisper.app` is built on your Mac and not signed by an Apple developer ID. Security software on managed Macs may block it. Ask your IT team.

**Logs.** `~/.macwhisper/logs/macwhisper.log`.

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
| **All options / change shortcuts** | Menu bar → **Settings…** (`⌘,`) |
| **Quit** | Click menu bar icon → Quit |

All shortcuts are defaults and can be changed in Settings: click a shortcut, press the new keys (Esc cancels). Push to Talk can be Right ⌥/⌘/⌃/⇧ or F1–F20. Meeting Mode moved from ⌃⇧R (v0.7.0 and earlier) to ⌃⌥M / ⌃⇧F8.

### Menu Bar Icons

| Icon | State |
|---|---|
| 🎙 | Ready |
| 🔴 | Recording |
| 💬 | Transcribing |

## Models

| Menu name | Model | Notes |
|---|---|---|
| **Qwen3 0.6B (Fast, zh+en)** | `mlx-community/Qwen3-ASR-0.6B-bf16` | Default |
| **Qwen3 1.7B (Best, zh+en)** | `mlx-community/Qwen3-ASR-1.7B-bf16` | More accurate, slower |

**Pinned versions.** Each model above, and the Qwen3-4B LLM, loads a fixed Hugging Face commit listed in `model_revisions.py`, so an upstream change never reaches you silently. A model you set yourself (`MACWHISPER_QWEN_MODEL` or `qwen_model`) loads the latest files and logs a warning. To update a pin, copy the new commit from `~/.cache/huggingface/hub/models--<org>--<name>/refs/main` into `MODEL_REVISIONS`.

Switch with **⌃⇧M** or the menu. Qwen3 runs in its own process (`.venv-qwen`); the optional AI style and vocabulary learning use a separate local LLM (`.venv-llm`).

Whisper was removed in 0.8.0. An old `"engine": "whisper"` setting is migrated automatically (Whisper Large → Qwen3 1.7B, otherwise 0.6B). Some earlier Qwen3 vs Whisper measurements are in the [changelog](CHANGELOG.md) under 0.7.3 (historical).

## Configuration

Settings are stored in `~/.macwhisper/config.json` (most are also in **Settings…**). Some keys:

```json
{
  "qwen_model": "mlx-community/Qwen3-ASR-0.6B-bf16",
  "live_mode": false,
  "save_audio": false,
  "log_level": "WARNING",
  "qwen_strip_fillers": true,
  "qwen_tight_spacing": true,
  "collapse_repeats": true,
  "paste_style": "original",
  "meeting_final_pass": true
}
```

- `qwen_strip_fillers` — remove standalone 嗯 / 呃 / um (never inside words such as 额外).
- `qwen_tight_spacing` — no space between Chinese and English (`这个commit的情况`).
- `collapse_repeats` — remove stutter repeats; intentional repeats are kept, only runaway loops (6+ repeats) are trimmed.
- `paste_style` — `original`, or an AI style: `light`, `medium`, `heavy`, `custom`.
- `meeting_final_pass` — re-transcribe each meeting track after Stop (in memory only). Push to Talk is paused while it runs.

## Tech Stack

- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon ML framework
- [Qwen3-ASR](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16) via [mlx-audio](https://github.com/Blaizzy/mlx-audio) — speech recognition on Apple GPU
- [rumps](https://github.com/jaredks/rumps) — macOS menu bar apps in Python
- [pynput](https://github.com/moses-palmer/pynput) — Global keyboard listener
- [sounddevice](https://python-sounddevice.readthedocs.io/) — Audio input via PortAudio

## License

MIT
