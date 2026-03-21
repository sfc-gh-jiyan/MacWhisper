"""
MacWhisper - macOS Menu Bar App
Hold Right Option to record, release to transcribe
Ctrl + Shift + M  switch model
Ctrl + Shift + T  toggle translate mode (all speech -> English)
"""

import json
import threading
import os
import time
import queue
import subprocess

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import numpy as np
import rumps
import pyperclip
import sounddevice as sd
from pynput import keyboard
import mlx_whisper
from ApplicationServices import AXIsProcessTrusted

SAMPLE_RATE = 16000
CONFIG_PATH = os.path.expanduser("~/.macwhisper_config.json")

MODEL_OPTIONS = {
    "Small (Fast)":      "mlx-community/whisper-small-mlx",
    "Medium (Accurate)": "mlx-community/whisper-medium-mlx",
    "Large (Best)":      "mlx-community/whisper-large-v3-mlx",
}
MODEL_KEYS = list(MODEL_OPTIONS.keys())


class TranscriberApp(rumps.App):
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
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f)

    def __init__(self):
        cfg = self._load_config()
        self.translate_mode = cfg.get("translate_mode", False)
        self.current_model  = cfg.get("current_model", MODEL_OPTIONS["Small (Fast)"])

        idle_icon = "🌐" if self.translate_mode else "🎙"
        super().__init__(idle_icon, quit_button="Quit")

        self.model_items = {}
        model_menu_items = []
        for label, repo in MODEL_OPTIONS.items():
            prefix = "✅" if repo == self.current_model else "  "
            title = f"{prefix} {label}"
            item = rumps.MenuItem(title, callback=self._make_model_callback(label))
            self.model_items[label] = item
            model_menu_items.append(item)

        translate_prefix = "✅" if self.translate_mode else "  "

        self.menu = [
            rumps.MenuItem("Status: Ready"),
            rumps.separator,
            *model_menu_items,
            rumps.separator,
            rumps.MenuItem(f"{translate_prefix} Translate to English", callback=self._toggle_translate),
            rumps.separator,
            rumps.MenuItem("Switch Model: Ctrl+Shift+M"),
            rumps.MenuItem("Toggle Translate: Ctrl+Shift+T"),
            rumps.MenuItem("Hold Right Option to record"),
            rumps.separator,
        ]
        self.status_item    = self.menu["Status: Ready"]
        self.item_translate = self.menu[f"{translate_prefix} Translate to English"]

        self.recording     = False
        self.frames        = []
        self.stream        = None
        self.model_ready   = True
        self._idle_icon    = idle_icon

        self._ctrl_pressed  = False
        self._shift_pressed = False

        self._key_event_received = False

        self.transcribe_queue = queue.Queue()
        threading.Thread(target=self._transcription_worker, daemon=True).start()
        threading.Thread(target=self._start_hotkey_listener, daemon=True).start()
        threading.Thread(target=self._check_permissions, daemon=True).start()

    def _set_status(self, text):
        self.status_item.title = f"Status: {text}"

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

    def _cycle_model(self):
        """Cycle to the next model: Small -> Medium -> Large -> Small"""
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

        if key == keyboard.Key.alt_r and not self.recording:
            threading.Thread(target=self._start_recording, daemon=True).start()

    def _on_release(self, key):
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._ctrl_pressed = False
        elif key in (keyboard.Key.shift, keyboard.Key.shift_r):
            self._shift_pressed = False
        elif key == keyboard.Key.alt_r and self.recording:
            threading.Thread(target=self._stop_and_transcribe, daemon=True).start()

    # ── Recording ────────────────────────────────────────────

    def _start_recording(self):
        self.frames    = []
        self.recording = True
        self.title     = "🔴"
        self._set_status("Recording...")
        print("[INFO] Recording started")

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

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if not self.frames:
            self.title = self._idle_icon
            self._set_status("Ready")
            return

        self.title = "💬"
        self._set_status("Transcribing...")
        self.transcribe_queue.put(list(self.frames))

    # ── Transcription ────────────────────────────────────────

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
        audio = np.concatenate(frames, axis=0).squeeze()
        audio_float = audio.astype(np.float32) / 32768.0

        if self.translate_mode:
            task, prompt = "translate", None
            print(f"[INFO] Translating {len(audio_float)/SAMPLE_RATE:.1f}s audio to English...")
        else:
            task, prompt = "transcribe", "以下是普通话与英语的混合对话。"
            print(f"[INFO] Transcribing {len(audio_float)/SAMPLE_RATE:.1f}s audio...")

        kwargs = dict(path_or_hf_repo=self.current_model, task=task)
        if prompt:
            kwargs["initial_prompt"] = prompt

        result = mlx_whisper.transcribe(audio_float, **kwargs)
        text = result["text"].strip()
        print(f"[INFO] Result: {text}")

        if text:
            self._auto_paste(text + " ")

    # ── Auto-paste ───────────────────────────────────────────

    def _auto_paste(self, text):
        pyperclip.copy(text)
        time.sleep(0.2)
        subprocess.run([
            "osascript", "-e",
            'tell application "System Events" to keystroke "v" using command down'
        ])


if __name__ == "__main__":
    TranscriberApp().run()
