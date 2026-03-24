# Changelog

All notable changes to MacWhisper will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/).

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
