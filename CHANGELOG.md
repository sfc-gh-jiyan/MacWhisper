# Changelog

All notable changes to MacWhisper will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.7.2] — 2026-10-20

### Added
- **Double-click MacWhisper in /Applications.** `./install.sh` (or `make app`) now builds `/Applications/MacWhisper.app` with a native launcher that embeds Python and runs this repo's code.
  - Privacy permissions (Microphone, Accessibility, Input Monitoring) now belong to **MacWhisper** itself. Before, they went to Python or, with `./run.sh`, to Terminal.
  - The app always runs the current repo code; rebuild only after moving the repo or recreating `venv`.
  - It sets the same memory setting as `./run.sh` (`MallocNanoZone=0`) and logs to `~/.macwhisper/logs/macwhisper.log`.
- **Launch at login**: Settings → General. The switch shows the real state from macOS, including "approve in Login Items". It is unavailable (grayed out) when started with `./run.sh`.
- **One instance only**: opening MacWhisper while it is already running shows a notification instead of starting a second copy that would fight over the hotkeys and the microphone.

### Changed
- The previous `/Applications/MacWhisper.app` was a Python script. Homebrew installs still pointed it at their own old copy. It is replaced by the native app.
- **After updating, grant the permissions to MacWhisper once** (System Settings → Privacy & Security → Accessibility and Input Monitoring). The new app is a different program to macOS.

### Fixed
- **"MacWhisper crashed last time" after logout, shutdown or a `./run.sh` restart.** macOS (and `./run.sh`) stop the app with SIGTERM, which ended it without the normal quit steps. The next launch then showed a false crash alert, and the Qwen3 helper was not closed. SIGTERM now quits the same way as menu → Quit.
- Every warning in `macwhisper.log` was written twice when the app was started from /Applications.

## [0.7.1] — 2026-10-19

### Added
- **Stop button in the meeting subtitle window**: a red **■ Stop** pinned to the top-right corner. It asks "End the meeting?" first. Confirming does exactly what the menu Stop does. Focus then returns to the app you were using.
- **Settings window** (menu bar → **Settings…**, `⌘,`): every option on one page.
  - Model.
  - Push to Talk live subtitles, translate and save audio.
  - Meeting subtitles, save meeting audio and the final Qwen3 transcript.
  - Qwen3 filler removal and Chinese/English spacing.
  - All keyboard shortcuts.
  - Changes apply immediately and are saved.
- **Customizable shortcuts**: click a shortcut in Settings and press the new keys. Esc cancels. Invalid keys or conflicts are explained, and you can try again. **Default** resets it. Each action can have a second shortcut.
  - Push to Talk can be Right ⌥/⌘/⌃/⇧ or F1–F20.
  - Global shortcuts are paused while one is being recorded.

### Changed
- **Default Qwen3 model is now 1.7B** (was 0.6B). This only applies when the config has no `qwen_model`; a saved choice is kept.
- **Meeting shortcut: ⌃⇧R → ⌃⌥M and ⌃⇧F8**, because ⌃⇧R conflicted with CoCo. The meeting window label shows the current shortcut.
- **Slim menu**: status, the five models, Start/Stop Meeting and Settings…. The Push to Talk / Meeting / Shortcuts submenus are gone.
- **Push to Talk ignores the key while another modifier is held** (⌃, ⇧ or ⌘), so ⌃⌥M never starts a recording. If Right ⌥ was pressed first and ⌃⌥M follows, the accidental recording is discarded: nothing is transcribed or pasted.
- With Push to Talk on Right ⌘/⌃/⇧, using that modifier for another shortcut (⌘C…) discards the recording. Right ⌥ behaves as before.

### Fixed
- **Meeting shortcut started a second meeting.** Holding the shortcut produced macOS key auto-repeat, and each repeat toggled the meeting (start, stop, start within one second). The same shortcut now starts and stops a meeting:
  - Auto-repeat is ignored for every shortcut.
  - Start/stop ignores further toggles for 1 s.
  - A second meeting can never start while one is running.
- A modifier "stuck" after a lost key release (for example in a password field) could silently disable Push to Talk, or turn typing into a shortcut. Modifier state now resyncs from the modifier flags of each key event. These are the flags at the moment the key was pressed, so a fast ⌃⌥M whose keys are already up when it is processed still works.
- Tests could overwrite `~/.macwhisper/config.json`. Every test now uses temp files, and the test run fails if the real config changes.

## [0.7.0] — 2026-10-19

### Added
- **Qwen3-ASR engine (optional, default after upgrade)**: Qwen3-ASR 0.6B and 1.7B join the model menu next to Whisper Small/Medium/Large. Measured on the user's own recordings:
  - Meeting (2 recordings, mixed error rate, lower is better): Qwen3 1.7B 0.119 / 0.156 and 0.6B 0.085 / 0.188, vs Whisper real-time 0.627 / 0.203 and Whisper full pass 0.729 / 0.359.
  - Push to Talk (221 clips, total time): 0.6B 5.6× and 1.7B 2.6× faster than Whisper Large; English inside Chinese speech is kept verbatim (0 translations).
- **Isolated sidecar**: Qwen3 runs in its own process and venv (`.venv-qwen`, mlx-audio 0.5.9 / mlx 0.32.4). The app venv, mlx 0.31.1 and mlx-whisper are untouched. Install with `make qwen-setup`; the first run after the env exists selects Qwen3 0.6B automatically.
- **Qwen3 live subtitles (Push to Talk)**: the utterance so far is re-transcribed every second; text both decodes agree on turns white, the rest stays gray. First text after about 1 s; live text ends within about 1% of the final text. If nothing appears after 4 s, the overlay says it is slow and that the final transcription still happens on release.
- **Meeting final transcript**: with Qwen3 selected, after a meeting stops the whole meeting is re-transcribed from in-memory audio. The same `.md` is rewritten with "Final transcript" first and the live transcript below it; the `.json` gets `final_transcript`/`final_engine`. Audio is never written to disk by this pass (Save Meeting Audio still controls WAV files). Config: `meeting_final_pass`.
- **Output style matching Whisper**: standalone fillers (嗯/呃/um) removed, never inside words (额外) or particles (好啊); no space between Chinese and English (`这个commit的情况`). Config: `qwen_strip_fillers`, `qwen_tight_spacing`.

### Safety / revertability
- **Whisper is always loaded** as the fallback and still handles translate mode and Meeting real-time subtitles.
- **Automatic fallback**: if Qwen3 fails to start, times out or crashes, the same audio is transcribed by Whisper (nothing lost), one red alert explains it, and the menu marks Qwen3 "unavailable (using Whisper)". A crashed sidecar is restarted automatically (up to 2 times); after it recovers the alert re-arms.
- **No leftover processes**: the sidecar closes on Quit and exits by itself if the app dies (measured: 0.5 s after the app was force-killed). `run.sh` also stops a stale sidecar.
- **To go back to Whisper**: pick any Whisper model in Push to Talk → Model (saved as `"engine": "whisper"`). Code-level rollback: tag/branch `whisper-stable-0.6.2`.

### Known issues
- Quitting during the first seconds after launch, while the Whisper model is still warming up, can segfault on exit. This is pre-existing: v0.6.2 does the same. No data is lost; the next launch reports the crash.
- Qwen3 gives no word timestamps; history entries made with Qwen3 have text only.

## [0.6.2] — 2026-10-18

### Fixed
- **Overlay jumping between displays**: content updates no longer re-center the panel on `NSScreen.mainScreen()` (the focused display). `update_overlay()` now only changes text — the window never moves or resizes on its own.
- **Option key silently stopped working**: menu-bar updates ran on the pynput listener thread (`NSInternalInconsistencyException`), and pynput stops the whole listener on any callback exception. All menu/status/icon writes and Ctrl+Shift shortcuts now run on the main thread; callbacks are guarded and a watchdog restarts the listener if it dies.
- **Mic stream race**: fast Option press/release could leak a running PortAudio stream; open/close is now locked and late streams are closed.
- **Meeting Mode dropped half the speech**: each chunk was fed to the stateful Silero VAD twice, corrupting its state (speech detected in 124/478 chunks instead of 294/478 on a real recording). Segments were cut mid-sentence and trailing sentences lost.
- **Overlay window level regression**: `setFloatingPanel_` silently reset the level to 3; level is now set after it.
- **Startup mic check never ran** (`DeviceList` has no `.get`).

### Added
- **Movable overlay**: drag anywhere on the subtitle window to move it, including to another display
- **Resizable overlay**: drag edges/corners to resize (min 240×60); text wraps and auto-scrolls to the newest line
- **Remembered position**: frame persisted per mode (Push to Talk / Meeting) in `~/.macwhisper/overlay.json`; falls back to the default position if the saved display is disconnected
- **Meeting Subtitles toggle**: new menu item to show/hide the Meeting Mode overlay (default On). Toggling mid-meeting shows/hides it immediately; recording, transcription, and export are unaffected. Saved as `meeting_subtitles` in config.

- **Never fail silently**: any error in any thread shows a persistent red alert ("please stop talking") until dismissed; hard crashes are captured to `crash.log` via `faulthandler` and reported on next launch.
- **Meeting transcript path on clipboard**: Meeting Mode copies the saved `.md` path (not the full text) and logs it with a `file://` link at the default log level. Falls back to copying text if saving failed.
- **Save Meeting Audio** toggle (default off — recording needs participants' consent): saves `meeting_<id>_mic.wav` / `_sys.wav` next to the transcript.
- Menu grouped into **Push to Talk / Meeting / Shortcuts** submenus.
- Evaluation tools (test-only, data stays in `~/.macwhisper/eval/`): `make eval-meeting`, `tests/compare_ptt.py`, `tests/simulate_stream.py` (Whisper vs Qwen3-ASR).

### Changed
- Overlay is a non-activating panel: clicking or dragging it never steals focus, so Hold-to-Record auto-paste still targets the active app

## [0.6.1] — 2026-03-30

### Fixed
- **Audio flood on first iteration**: reorder `_start_recording()` so `stream.start()` runs after processor/live_loop creation — prevents audio accumulating during backend warmup (buffer start: 8.0s → 2.5s, content retention: 30% → 75%)
- **Post-trim re-confirmation dedup**: check `committed_history[-50:]` in addition to `self.committed` so words trimmed by `_maybe_trim_buffer()` are detected as duplicates
- **Segment commit triple-fire**: 3-second cooldown guard prevents VAD from triggering `segment_close()` multiple times in rapid succession
- **Bilingual duplicate confirm**: intra-batch dedup detects repeated word prefix within a single `flush()` batch (e.g., two translation variants of the same phrase)
- **Tail CJK fragment filter**: strip trailing single-CJK-character fragments in `segment_close()` (e.g., "战。" from "挑战")
- **Language pinning**: auto-detect dominant language from first 2 consecutive same-language iterations, then lock `self.language` to prevent Whisper from oscillating between Chinese transcription and English translation
- **Defensive buffer guard**: `insert_audio_chunk()` trims buffer to `max_buffer_s` when no inference has run yet (iter_count == 0)

### Added
- **Session end logging**: capture complete `committed_history` text at recording stop → write to `subtitles.jsonl` as `session_end` entry + include as `realtime_text` in `transcripts.jsonl`

### Changed
- 14 ruff lint errors fixed (unused imports/variables, f-strings without placeholders)

## [0.6.0] — 2026-03-29

### Added
- **Meeting Mode**: continuous recording with automatic segmentation, real-time overlay, Markdown/JSON/SRT export
  - Mic + system audio dual-channel capture via ScreenCaptureKit
  - Overlap handling: RMS energy comparison selects the louder source
  - Dual-mode overlay: Push to Talk (orange) vs Meeting Recording (red)
  - VAD-driven paragraph breaks on extended silence (2s)
  - Max segment duration safety net (30s wall-clock)
- Word-level hallucination loop filter: collapses consecutive identical words (>=3) before LocalAgreement
- Overlap-aware prefix dedup: strips re-confirmed leading words from newly confirmed output
- `committed_history` (append-only) split from `committed` (trim-aware) for correct output across trim/segment boundaries
- Trim cooldown (2 iterations) after buffer trim to let LocalAgreement stabilize
- Structured debug logging across transcription pipeline
- `log_level` configurable via `~/.macwhisper/config.json`
- Three-level test Makefile (unit / integration / all)

### Changed
- Default model changed from Small to **Medium (Accurate)**
- Audio buffer trimmed after segment close — eliminates content loss from echo detection and near-duplicate text from re-transcription

### Fixed
- Meeting segment timestamps showing identical values (used committed_history instead of committed)
- Overlay not closing when Meeting Mode stops (reordered to destroy overlay before blocking stop)
- Trim loop: 9 trims to same position (committed cleanup after trim + cooldown)
- Adaptive trim over-reaction to GPU spikes removed — confirmed text +115% (Run 10: 297/318 chars)
- Dedup false positive after trim (use committed, not committed_history)
- Content loss after segment close (echo detection dropping legitimate re-confirmations)
- Sentence-level duplication (5 per session -> 1) via overlap prefix dedup
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
