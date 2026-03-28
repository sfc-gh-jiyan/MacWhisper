# Changelog

All notable changes to MacWhisper will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.6.0] — 2026-03-27

### Added
- Meeting Mode: mic + system audio dual-channel capture via ScreenCaptureKit
- Overlap handling: RMS energy comparison selects the louder source when both channels are active
- Dual-mode overlay: Push to Talk (orange stripe) vs Meeting Recording (red stripe)
- Overlay shows mode label immediately on panel creation (no waiting for first transcription)
- Structured debug logging across entire transcription pipeline (online_processor, meeting, app)
- `log_level` configurable via `~/.macwhisper/config.json` (not exposed in menu UI)
- Three-level test Makefile (unit / integration / all)
- Dual-channel A/B comparison test coverage

### Fixed
- Meeting SRT/words history redirected to meetings directory
- Test output isolation to prevent pollution of `~/.macwhisper/`
- OnlineASRProcessor post-commit echo duplication

## [0.5.2] — 2026-03-25

### Fixed
- .app launcher preserves bundle identity for menu bar icon and hotkeys
- NSBundle swizzle for Homebrew Python menu bar icon

## [0.5.1] — 2026-03-24

### Fixed
- Add LSUIElement to fix missing menu bar icon from .app launch
- Memory leaks, context-aware punctuation, launcher reliability

## [0.5.0] — 2026-03-24

### Added
- Anti-hallucination defense (excessive word count detection + filtering)
- Instant overlay display
- Save Audio toggle in menu

### Fixed
- Stop-recording button not working

## [0.4.6] — 2026-03-24

### Fixed
- Crash on Option key release
- Display latency optimization
- Punctuation consistency

## [0.4.5] — 2026-03-24

### Changed
- LocalAgreement architecture refactor
- Enhanced replay evaluation system

## [0.4.0] — 2025-03-23

### Added
- Crash handler: uncaught exceptions logged to `~/.macwhisper/logs/crash.log`
- Audio device startup check with user-friendly alert if no microphone found
- `bump_version.sh` for automated version bumps (patch/minor/major)
- `build.sh` for source tarball + SHA256 generation
- `release.sh` for GitHub Release automation via `gh` CLI
- `test_install.sh` for clean-install verification
- GitHub Actions CI pipeline (`.github/workflows/ci.yml`)
- Homebrew formula template (`homebrew/macwhisper.rb`)

### Changed
- `install.sh` rewritten: Apple Silicon + Python 3.10+ pre-flight checks, dynamic version
- `requirements.txt` pinned with `certifi`, added `opencc-python-reimplemented`

### Removed
- `test.py` (legacy faster-whisper prototype)
- `overlay-preview.html` (unused mockup)

## [0.3.1] — 2025-03-23

### Changed
- All user data consolidated under `~/.macwhisper/` (config, audio, logs, transcripts, subtitles)
- One-time migration from old locations (`~/.macwhisper_config.json`, `./history/`, `./logs/`)

## [0.3.0] — 2025-03-23

### Added
- Version tracking (`VERSION` file + `__version__` in app.py + git tags)
- Bilingual prompt to prevent Whisper translating English speech to Chinese
- Subtitle latency optimization: reduced from ~5s to ~3s
- Overlay skip: redundant overlay updates avoided via text comparison

### Fixed
- English speech being translated to Chinese due to pure-Chinese `initial_prompt`

## [0.2.0] — Pre-tracking

### Added
- Live subtitles with real-time overlay
- Pause detection and automatic segmentation
- Stability-based text commit for live display
- OpenCC Traditional → Simplified Chinese conversion
- Multi-model support (Small/Medium/Large)
- Translate mode (all speech → English)
- Global hotkey (Right Option hold-to-record)
- Auto-paste via Cmd+V simulation
- Menu bar app with rumps
