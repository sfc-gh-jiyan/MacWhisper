"""
MacWhisper - macOS Menu Bar App  v0.4.6
Hold Right Option to record, release to transcribe
Ctrl + Shift + M  switch model
Ctrl + Shift + T  toggle translate mode (all speech -> English)
Ctrl + Shift + S  toggle live subtitles (show preview while recording)

v0.5: LocalAgreement architecture — word-level confirmation,
      dual-color overlay, Silero VAD, pluggable ASR backend.
"""

__version__ = "0.4.6"

import json
import threading
import os
import time
import queue
import subprocess
import sys
import wave
import datetime

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from AppKit import NSApplication, NSImage
from PyObjCTools import AppHelper

_app = NSApplication.sharedApplication()
_app.setActivationPolicy_(1)  # NSApplicationActivationPolicyAccessory

_bundle_icon = os.path.join(os.path.dirname(sys.argv[0]), "..", "Resources", "AppIcon.icns") if ".app" in sys.argv[0] else None
if _bundle_icon and os.path.exists(_bundle_icon):
    _app.setApplicationIconImage_(NSImage.alloc().initByReferencingFile_(_bundle_icon))

import numpy as np
import rumps
import pyperclip
import sounddevice as sd
from pynput import keyboard
from ApplicationServices import AXIsProcessTrusted

# ── New modular imports ──────────────────────────────────
from text_utils import (
    BILINGUAL_PROMPT, convert_t2s, strip_trailing_repetition,
    hallucination_reason, is_hallucination,
)
from asr_backend import MLXWhisperBackend, WordTimestamp
from vad import VoiceActivityDetector
from online_processor import OnlineASRProcessor
from overlay import create_overlay, update_overlay, destroy_overlay
from subtitle_export import export_srt, save_enhanced_history

# ── Constants ────────────────────────────────────────────
SAMPLE_RATE = 16000
MIN_CHUNK_SIZE = 0.7          # seconds — LocalAgreement confirmation latency ≈ 2x this
MAX_BUFFER_S = 20.0           # audio buffer cap (seconds)

_DATA_DIR = os.path.expanduser("~/.macwhisper")
CONFIG_PATH = os.path.join(_DATA_DIR, "config.json")
LOG_DIR = os.path.join(_DATA_DIR, "logs")

MODEL_OPTIONS = {
    "Small (Fast)":      "mlx-community/whisper-small-mlx",
    "Medium (Accurate)": "mlx-community/whisper-medium-mlx",
    "Large (Best)":      "mlx-community/whisper-large-v3-mlx",
}
MODEL_KEYS = list(MODEL_OPTIONS.keys())

HISTORY_DIR = _DATA_DIR
AUDIO_DIR = os.path.join(_DATA_DIR, "audio")
TRANSCRIPT_LOG = os.path.join(_DATA_DIR, "transcripts.jsonl")
SUBTITLE_LOG = os.path.join(_DATA_DIR, "subtitles.jsonl")


def _ensure_history_dirs():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def _migrate_old_data():
    """One-time migration from old locations to ~/.macwhisper/."""
    import shutil
    _ensure_history_dirs()

    old_config = os.path.expanduser("~/.macwhisper_config.json")
    if os.path.isfile(old_config) and not os.path.isfile(CONFIG_PATH):
        shutil.move(old_config, CONFIG_PATH)
        print(f"[MIGRATE] Moved config: {old_config} -> {CONFIG_PATH}")

    _project_dir = os.path.dirname(os.path.abspath(__file__))
    old_history = os.path.join(_project_dir, "history")

    old_audio = os.path.join(old_history, "audio")
    if os.path.isdir(old_audio):
        for fname in os.listdir(old_audio):
            src = os.path.join(old_audio, fname)
            dst = os.path.join(AUDIO_DIR, fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.move(src, dst)
        if os.path.isdir(old_audio) and not os.listdir(old_audio):
            os.rmdir(old_audio)
        print(f"[MIGRATE] Moved audio files: {old_audio} -> {AUDIO_DIR}")

    old_transcript = os.path.join(old_history, "transcripts.jsonl")
    if os.path.isfile(old_transcript) and not os.path.isfile(TRANSCRIPT_LOG):
        shutil.move(old_transcript, TRANSCRIPT_LOG)
        print(f"[MIGRATE] Moved transcripts: {old_transcript} -> {TRANSCRIPT_LOG}")
    elif os.path.isfile(old_transcript) and os.path.isfile(TRANSCRIPT_LOG):
        with open(old_transcript, "r", encoding="utf-8") as src:
            data = src.read()
        if data.strip():
            with open(TRANSCRIPT_LOG, "a", encoding="utf-8") as dst:
                dst.write(data)
        os.remove(old_transcript)
        print(f"[MIGRATE] Merged transcripts: {old_transcript} -> {TRANSCRIPT_LOG}")

    old_subtitle = os.path.join(old_history, "subtitles.jsonl")
    if os.path.isfile(old_subtitle) and not os.path.isfile(SUBTITLE_LOG):
        shutil.move(old_subtitle, SUBTITLE_LOG)
        print(f"[MIGRATE] Moved subtitles: {old_subtitle} -> {SUBTITLE_LOG}")
    elif os.path.isfile(old_subtitle) and os.path.isfile(SUBTITLE_LOG):
        with open(old_subtitle, "r", encoding="utf-8") as src:
            data = src.read()
        if data.strip():
            with open(SUBTITLE_LOG, "a", encoding="utf-8") as dst:
                dst.write(data)
        os.remove(old_subtitle)
        print(f"[MIGRATE] Merged subtitles: {old_subtitle} -> {SUBTITLE_LOG}")

    if os.path.isdir(old_history) and not os.listdir(old_history):
        os.rmdir(old_history)

    old_logs = os.path.join(_project_dir, "logs")
    if os.path.isdir(old_logs):
        for fname in os.listdir(old_logs):
            src = os.path.join(old_logs, fname)
            dst = os.path.join(LOG_DIR, fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.move(src, dst)
        if os.path.isdir(old_logs) and not os.listdir(old_logs):
            os.rmdir(old_logs)
        print(f"[MIGRATE] Moved logs: {old_logs} -> {LOG_DIR}")


class TranscriberApp(rumps.App):

    # ── Config ────────────────────────────────────────────────

    def _load_config(self):
        try:
            with open(CONFIG_PATH) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_config(self):
        cfg = {
            "translate_mode": self.translate_mode,
            "current_model": self.current_model,
            "live_mode": self.live_mode,
            "save_audio": self.save_audio,
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f)

    # ── Init ──────────────────────────────────────────────────

    def __init__(self):
        _migrate_old_data()
        cfg = self._load_config()
        self.translate_mode = cfg.get("translate_mode", False)
        self.current_model  = cfg.get("current_model", MODEL_OPTIONS["Small (Fast)"])
        self.live_mode      = cfg.get("live_mode", False)
        self.save_audio     = cfg.get("save_audio", False)

        idle_icon = "🌐" if self.translate_mode else "🎙"
        super().__init__(idle_icon, quit_button="Quit")

        # Model menu items
        self.model_items = {}
        model_menu_items = []
        for label, repo in MODEL_OPTIONS.items():
            prefix = "✅" if repo == self.current_model else "  "
            title = f"{prefix} {label}"
            item = rumps.MenuItem(title, callback=self._make_model_callback(label))
            self.model_items[label] = item
            model_menu_items.append(item)

        translate_prefix = "✅" if self.translate_mode else "  "
        live_prefix = "✅" if self.live_mode else "  "
        live_label = "On" if self.live_mode else "Off"

        self.item_live = rumps.MenuItem(f"{live_prefix} Live Subtitles: {live_label}", callback=self._toggle_live_mode)

        save_audio_prefix = "✅" if self.save_audio else "  "
        save_audio_label = "On" if self.save_audio else "Off"
        self.item_save_audio = rumps.MenuItem(f"{save_audio_prefix} Save Audio: {save_audio_label}", callback=self._toggle_save_audio)

        self.menu = [
            rumps.MenuItem("Status: Ready"),
            rumps.separator,
            *model_menu_items,
            rumps.separator,
            rumps.MenuItem(f"{translate_prefix} Translate to English", callback=self._toggle_translate),
            rumps.separator,
            self.item_live,
            self.item_save_audio,
            rumps.separator,
            rumps.MenuItem("Switch Model: Ctrl+Shift+M"),
            rumps.MenuItem("Toggle Translate: Ctrl+Shift+T"),
            rumps.MenuItem("Live Subtitles: Ctrl+Shift+S"),
            rumps.MenuItem("Hold Right Option to record"),
            rumps.separator,
        ]
        self.status_item    = self.menu["Status: Ready"]
        self.item_translate = self.menu[f"{translate_prefix} Translate to English"]

        # Hold-to-record state
        self.recording     = False
        self.frames        = []
        self.stream        = None
        self._idle_icon    = idle_icon

        # Hotkey modifier tracking
        self._ctrl_pressed  = False
        self._shift_pressed = False
        self._key_event_received = False

        # Live subtitle overlay state
        self._overlay_panel    = None
        self._overlay_text     = None
        self._last_overlay_key = None   # (confirmed, unconfirmed) dedup

        # Online processor (created per recording session)
        self._processor: OnlineASRProcessor | None = None
        self._live_loop_done = threading.Event()
        self._live_loop_done.set()  # initially "done" (no loop running)

        # Workers
        self.transcribe_queue = queue.Queue()
        threading.Thread(target=self._transcription_worker, daemon=True).start()
        threading.Thread(target=self._start_hotkey_listener, daemon=True).start()
        threading.Thread(target=self._check_permissions, daemon=True).start()

    def _set_status(self, text):
        self.status_item.title = f"Status: {text}"

    # ── Model switching ───────────────────────────────────────

    def _make_model_callback(self, label):
        def cb(_):
            self._set_model(label)
        return cb

    def _set_model(self, label):
        self.current_model = MODEL_OPTIONS[label]
        for k, item in self.model_items.items():
            prefix = "✅" if k == label else "  "
            item.title = f"{prefix} {k}"
        short_name = label.split(" ")[0]
        self._set_status(f"Model: {short_name}")
        self._save_config()

    def _toggle_translate(self, _):
        self.translate_mode = not self.translate_mode
        self.item_translate.title = "✅ Translate to English" if self.translate_mode else "   Translate to English"
        self._idle_icon = "🌐" if self.translate_mode else "🎙"
        if not self.recording:
            self.title = self._idle_icon
        mode_str = "Translate ON" if self.translate_mode else "Transcribe"
        self._set_status(mode_str)
        self._save_config()

    def _toggle_live_mode(self, _):
        self.live_mode = not self.live_mode
        if self.live_mode:
            self.item_live.title = "✅ Live Subtitles: On"
        else:
            self.item_live.title = "   Live Subtitles: Off"
        self._set_status("Subtitles ON" if self.live_mode else "Subtitles OFF")
        self._save_config()
        print(f"[INFO] Live subtitles {'enabled' if self.live_mode else 'disabled'}")

    def _toggle_save_audio(self, _):
        self.save_audio = not self.save_audio
        if self.save_audio:
            self.item_save_audio.title = "✅ Save Audio: On"
        else:
            self.item_save_audio.title = "   Save Audio: Off"
        self._set_status("Save Audio ON" if self.save_audio else "Save Audio OFF")
        self._save_config()
        print(f"[INFO] Save audio {'enabled' if self.save_audio else 'disabled'}")

    def _cycle_model(self):
        current_idx = next(
            i for i, v in enumerate(MODEL_OPTIONS.values()) if v == self.current_model
        )
        next_idx = (current_idx + 1) % len(MODEL_KEYS)
        self._set_model(MODEL_KEYS[next_idx])

    # ── Permissions ────────────────────────────────────────────

    def _notify(self, title, subtitle, message):
        try:
            rumps.notification(title, subtitle, message)
        except RuntimeError:
            pass

    def _check_permissions(self):
        if not AXIsProcessTrusted():
            print("[WARN] Accessibility permission NOT granted — auto-paste will fail")
            print("       → System Settings → Privacy & Security → Accessibility")
            self._notify(
                "MacWhisper — Permission Needed",
                "Accessibility access is required",
                "System Settings → Privacy & Security → Accessibility",
            )

        time.sleep(8)
        if not self._key_event_received:
            print("[WARN] No key events detected — Input Monitoring may not be granted")
            print("       → System Settings → Privacy & Security → Input Monitoring")
            self._notify(
                "MacWhisper — Permission Needed",
                "Input Monitoring access is required",
                "System Settings → Privacy & Security → Input Monitoring",
            )

    # ── Hotkeys ──────────────────────────────────────────────

    def _start_hotkey_listener(self):
        with keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        ) as listener:
            listener.join()

    def _on_press(self, key):
        self._key_event_received = True
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._ctrl_pressed = True
        elif key in (keyboard.Key.shift, keyboard.Key.shift_r):
            self._shift_pressed = True

        if self._ctrl_pressed and self._shift_pressed:
            vk = getattr(key, 'vk', None)
            if vk == 46:  # M key
                self._cycle_model()
                return
            if vk == 17:  # T key
                self._toggle_translate(None)
                return
            if vk == 1:   # S key
                self._toggle_live_mode(None)
                return

        if key == keyboard.Key.alt_r and not self.recording:
            threading.Thread(target=self._start_recording, daemon=True).start()

    def _on_release(self, key):
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._ctrl_pressed = False
        elif key in (keyboard.Key.shift, keyboard.Key.shift_r):
            self._shift_pressed = False
        elif key == keyboard.Key.alt_r and self.recording:
            threading.Thread(target=self._stop_and_transcribe, daemon=True).start()

    # ── Hold-to-Record ────────────────────────────────────────

    def _start_recording(self):
        self.frames    = []
        self.recording = True
        self._last_overlay_key = None
        self.title     = "🟠"
        self._set_status("Recording...")
        print("[INFO] Recording started")

        # Create OnlineASRProcessor for this session
        backend = MLXWhisperBackend(model_repo=self.current_model)
        self._processor = OnlineASRProcessor(
            backend=backend,
            vad=None,  # VAD used for segmentation in the live loop
            min_chunk_size=MIN_CHUNK_SIZE,
            max_buffer_s=MAX_BUFFER_S,
            language=None,  # auto-detect
        )

        if self.live_mode:
            AppHelper.callAfter(self._create_overlay)
            self._live_loop_done.clear()
            threading.Thread(target=self._live_loop, daemon=True).start()

        def callback(indata, frame_count, time_info, status):
            if self.recording:
                self.frames.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=1024, callback=callback
        )
        self.stream.start()

    def _stop_and_transcribe(self):
        self.recording = False
        print(f"[INFO] Recording stopped, frames: {len(self.frames)}")

        # Wait for _live_loop to finish (so we don't run two mlx inferences
        # concurrently, which causes Metal command buffer assertion failures)
        self._live_loop_done.wait(timeout=10)

        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            # Force-confirm remaining words from processor
            final_text = ""
            if self._processor:
                final_text = self._processor.segment_close()

            if self._overlay_panel:
                AppHelper.callAfter(self._destroy_overlay)
        except Exception as e:
            print(f"[ERROR] Cleanup failed: {e}")

        if not self.frames:
            self.title = self._idle_icon
            self._set_status("Ready")
            return

        self.title = "💬"
        self._set_status("Transcribing...")
        self.transcribe_queue.put(list(self.frames))

    # ── Live Subtitles — unified loop (replaces old chunk + worker) ──

    def _live_loop(self):
        """Main live subtitle loop using OnlineASRProcessor.

        Runs in a background thread while recording. Feeds audio chunks
        to the processor, which handles LocalAgreement internally.
        The overlay is updated with confirmed (white) + unconfirmed (gray) text.
        """
        proc = self._processor
        if not proc:
            self._live_loop_done.set()
            return

        try:
            self._live_loop_impl(proc)
        except Exception as e:
            print(f"[ERROR] Live loop crashed: {e}")
        finally:
            self._live_loop_done.set()

    def _live_loop_impl(self, proc):
        """Inner live loop implementation (separated for clean try/finally)."""
        # VAD for speech-end detection (segmentation)
        try:
            vad = VoiceActivityDetector(
                threshold=0.5,
                min_silence_ms=800,
                min_speech_ms=250,
            )
            use_vad = True
        except Exception:
            use_vad = False
            print("[WARN] VAD unavailable for live loop")

        last_frame_idx = 0

        while self.recording:
            # Brief yield — processor's internal throttle handles timing
            time.sleep(0.05)
            if not self.recording:
                break

            n = len(self.frames)
            if n <= last_frame_idx:
                continue

            # Collect new frames since last iteration
            new_frames = self.frames[last_frame_idx:n]
            last_frame_idx = n

            # Convert int16 frames to float32 and feed to processor
            audio_chunk = np.concatenate(new_frames, axis=0).squeeze()
            audio_float = audio_chunk.astype(np.float32) / 32768.0
            proc.insert_audio_chunk(audio_float)

            # VAD: check for speech end → segment commit
            if use_vad:
                vad.process_chunk(audio_chunk)
                seg_duration = len(proc.audio_buffer) / SAMPLE_RATE
                if vad.is_speech_end() and seg_duration > 5.0:
                    segment_text = proc.segment_close()
                    if segment_text:
                        print(f"[INFO] VAD segment committed: {len(segment_text)} chars")
                        self._log_subtitle(segment_text, seg_duration)
                    vad.reset()

            # Run one processing iteration (LocalAgreement)
            result = proc.process_iter()
            if result is None:
                continue

            confirmed, unconfirmed = result

            # Log debug info + subtitle text to console
            if proc.last_debug:
                debug = proc.last_debug
                conf_tail = confirmed[-40:] if confirmed else "(empty)"
                print(f"[LIVE] iter={debug.get('iter', '?')} "
                      f"buf={debug.get('buffer_s', '?')}s "
                      f"inf={debug.get('inference_ms', '?')}ms "
                      f"conf={debug.get('confirmed_words', 0)} "
                      f"new={debug.get('newly_confirmed', 0)} "
                      f"| {conf_tail}")

            # Update overlay (dual-color: white confirmed, gray unconfirmed)
            overlay_key = (confirmed, unconfirmed)
            if overlay_key != self._last_overlay_key and self._overlay_panel:
                self._last_overlay_key = overlay_key
                AppHelper.callAfter(
                    lambda c=confirmed, u=unconfirmed: update_overlay(
                        self._overlay_panel, self._overlay_text, c, u
                    )
                )

            # Log subtitle entry
            if confirmed or unconfirmed:
                self._log_subtitle_live(confirmed, unconfirmed, proc.last_debug)

    def _log_subtitle(self, text, duration_s):
        """Log a committed segment to subtitle log."""
        _ensure_history_dirs()
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "segment_commit",
            "audio_s": round(duration_s, 1),
            "text": text,
        }
        with open(SUBTITLE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _log_subtitle_live(self, confirmed, unconfirmed, debug):
        """Log a live subtitle iteration to subtitle log."""
        _ensure_history_dirs()
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "live_iter",
            "confirmed": confirmed[-100:] if confirmed else "",
            "unconfirmed": unconfirmed[-50:] if unconfirmed else "",
            "debug": debug,
        }
        with open(SUBTITLE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Transcription (hold-to-record final pass) ──────────────

    def _transcription_worker(self):
        while True:
            frames = self.transcribe_queue.get()
            try:
                self._do_transcribe(frames)
            except Exception as e:
                print(f"[ERROR] Transcription failed: {e}")
            finally:
                self.transcribe_queue.task_done()
                self.title = self._idle_icon
                self._set_status("Ready")

    def _do_transcribe(self, frames):
        """Final transcription pass using the ASR backend.

        This runs after recording stops. For short recordings or when
        live mode is off, this is the only transcription. For live mode,
        this provides a final high-quality pass over the full audio.
        """
        audio = np.concatenate(frames, axis=0).squeeze()
        audio_float = audio.astype(np.float32) / 32768.0
        duration = len(audio_float) / SAMPLE_RATE

        # Save audio WAV (only when save_audio is enabled)
        _ensure_history_dirs()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        wav_path = None
        if self.save_audio:
            wav_path = os.path.join(AUDIO_DIR, f"{timestamp}.wav")
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio.tobytes())
            print(f"[INFO] Saved audio: {wav_path}")

        # Create backend for final pass
        backend = MLXWhisperBackend(model_repo=self.current_model)

        if self.translate_mode:
            task, prompt = "translate", None
            print(f"[INFO] Translating {duration:.1f}s audio to English...")
        else:
            task, prompt = "transcribe", BILINGUAL_PROMPT
            print(f"[INFO] Transcribing {duration:.1f}s audio...")

        t0 = time.time()
        result = backend.transcribe(
            audio_float,
            language=None,
            initial_prompt=prompt,
            task=task,
        )
        elapsed = time.time() - t0

        text = result.text.strip()
        if prompt and text.startswith(prompt):
            text = text[len(prompt):].strip()

        text = convert_t2s(text)
        if prompt and (text in prompt or text.rstrip("。.") in prompt):
            text = ""
            print("[INFO] Result (prompt echo filtered)")
        text = strip_trailing_repetition(text)
        hall_reason = hallucination_reason(text) if text else None
        if hall_reason:
            print(f"[INFO] Result (hallucination:{hall_reason}): {text}")
            text = ""

        print(f"[INFO] Final transcription: {elapsed:.1f}s for {duration:.1f}s audio")
        print(f"[INFO] Result: {text}")

        # Save enhanced history with word-level timestamps
        all_words = result.all_words()
        save_enhanced_history(
            history_dir=_DATA_DIR,
            audio_file=os.path.basename(wav_path) if wav_path else None,
            words=all_words,
            text=text,
            model=self.current_model,
            duration_s=duration,
            language=result.language,
        )

        # Log transcription result (legacy format for compatibility)
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "audio_file": os.path.basename(wav_path) if wav_path else None,
            "model": self.current_model,
            "translate": self.translate_mode,
            "duration_s": round(duration, 1),
            "text": text,
        }
        with open(TRANSCRIPT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        print(f"[INFO] Logged transcript: {TRANSCRIPT_LOG}")

        if text:
            self._auto_paste(text + " ")

    # ── Live Subtitles — overlay ──────────────────────────────

    def _create_overlay(self):
        panel, text_view = create_overlay()
        if panel:
            self._overlay_panel = panel
            self._overlay_text = text_view
            print("[INFO] Overlay panel created")
        else:
            print("[ERROR] Failed to create overlay panel")

    def _destroy_overlay(self):
        if self._overlay_panel:
            destroy_overlay(self._overlay_panel)
            self._overlay_panel = None
            self._overlay_text = None
            print("[INFO] Overlay panel destroyed")

    # ── Auto-paste ───────────────────────────────────────────

    def _auto_paste(self, text):
        pyperclip.copy(text)
        time.sleep(0.2)
        subprocess.run([
            "osascript", "-e",
            'tell application "System Events" to keystroke "v" using command down'
        ])


def _check_audio_device():
    """Verify at least one audio input device is available."""
    try:
        devices = sd.query_devices()
        has_input = any(
            d.get("max_input_channels", 0) > 0
            for d in (devices if isinstance(devices, list) else [devices])
        )
        if not has_input:
            rumps.alert(
                title="MacWhisper – No Microphone",
                message="No audio input device found. Please connect a microphone and restart MacWhisper.",
            )
            sys.exit(1)
    except Exception as e:
        print(f"[WARN] Audio device check failed: {e}")


if __name__ == "__main__":
    _check_audio_device()
    try:
        TranscriberApp().run()
    except Exception:
        import traceback
        crash_log = os.path.join(LOG_DIR, "crash.log")
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(crash_log, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"MacWhisper v{__version__} crash at {datetime.datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n")
            traceback.print_exc(file=f)
        print(f"[FATAL] Crash logged to {crash_log}")
        traceback.print_exc()
        sys.exit(1)
