"""
MacWhisper - macOS Menu Bar App
长按右 Option 录音，松开自动转录并打出文字
Ctrl + Shift + M  弹出模型切换对话框
Ctrl + Shift + T  开关翻译模式（所有语音 → 英文）
"""

import json
import threading
import os
import time
import queue
import subprocess

import numpy as np
import rumps
import pyperclip
import sounddevice as sd
from pynput import keyboard
import mlx_whisper
from AppKit import (NSApplication, NSImage, NSAttributedString, NSFont,
                    NSFontAttributeName)

SAMPLE_RATE = 16000
CONFIG_PATH = os.path.expanduser("~/.macwhisper_config.json")

MODEL_OPTIONS = {
    "Small (快速)":  "mlx-community/whisper-small-mlx",
    "Medium (准确)": "mlx-community/whisper-medium-mlx",
}
MODEL_KEYS = list(MODEL_OPTIONS.keys())


def make_dock_icon(emoji, size=256):
    image = NSImage.alloc().initWithSize_((size, size))
    image.lockFocus()
    font = NSFont.systemFontOfSize_(size * 0.72)
    ns_str = NSAttributedString.alloc().initWithString_attributes_(
        emoji, {NSFontAttributeName: font}
    )
    w, h = ns_str.size()
    ns_str.drawAtPoint_(((size - w) / 2, (size - h) / 2))
    image.unlockFocus()
    return image


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
        self.current_model  = cfg.get("current_model", MODEL_OPTIONS["Small (快速)"])

        idle_icon = "🌐" if self.translate_mode else "🎙"
        super().__init__(idle_icon, quit_button="退出")

        small_prefix  = "✅" if self.current_model == MODEL_OPTIONS["Small (快速)"] else "  "
        medium_prefix = "✅" if self.current_model == MODEL_OPTIONS["Medium (准确)"] else "  "
        translate_prefix = "✅" if self.translate_mode else "  "

        self.menu = [
            rumps.MenuItem("状态: 就绪 ✓"),
            rumps.separator,
            rumps.MenuItem(f"{small_prefix} Small (快速)",  callback=self._set_model_small),
            rumps.MenuItem(f"{medium_prefix} Medium (准确)", callback=self._set_model_medium),
            rumps.separator,
            rumps.MenuItem(f"{translate_prefix} 翻译成英文", callback=self._toggle_translate),
            rumps.separator,
            rumps.MenuItem("切换模型: Ctrl+Shift+M"),
            rumps.MenuItem("翻译开关: Ctrl+Shift+T"),
            rumps.MenuItem("长按右 Option 录音，松开转录"),
            rumps.separator,
        ]
        self.status_item    = self.menu["状态: 就绪 ✓"]
        self.item_small     = self.menu[f"{small_prefix} Small (快速)"]
        self.item_medium    = self.menu[f"{medium_prefix} Medium (准确)"]
        self.item_translate = self.menu[f"{translate_prefix} 翻译成英文"]

        self.recording     = False
        self.frames        = []
        self.stream        = None
        self.model_ready   = True
        self._idle_icon    = idle_icon
        self._pending_icon = idle_icon

        self._ctrl_pressed  = False
        self._shift_pressed = False

        self.transcribe_queue = queue.Queue()
        threading.Thread(target=self._transcription_worker, daemon=True).start()
        threading.Thread(target=self._start_hotkey_listener, daemon=True).start()

    def _set_status(self, text):
        self.status_item.title = f"状态: {text}"

    def _set_model_small(self, _):
        self.current_model     = MODEL_OPTIONS["Small (快速)"]
        self.item_small.title  = "✅ Small (快速)"
        self.item_medium.title = "   Medium (准确)"
        self._set_status("模型: Small ✓")
        self._save_config()

    def _set_model_medium(self, _):
        self.current_model     = MODEL_OPTIONS["Medium (准确)"]
        self.item_small.title  = "   Small (快速)"
        self.item_medium.title = "✅ Medium (准确)"
        self._set_status("模型: Medium ✓")
        self._save_config()

    def _toggle_translate(self, _):
        self.translate_mode = not self.translate_mode
        self.item_translate.title = "✅ 翻译成英文" if self.translate_mode else "   翻译成英文"
        self._idle_icon = "🌐" if self.translate_mode else "🎙"
        if not self.recording:
            self.title = self._idle_icon
            self._pending_icon = self._idle_icon
        mode_str = "翻译→英文 ON" if self.translate_mode else "正常转录"
        self._set_status(f"{mode_str} ✓")
        self._save_config()

    def _show_model_picker(self):
        """在主线程弹出模型选择对话框"""
        current_label = next(
            k for k, v in MODEL_OPTIONS.items() if v == self.current_model
        )
        other_label = next(k for k in MODEL_KEYS if k != current_label)
        response = rumps.alert(
            title="MacWhisper — 切换模型",
            message=f"当前模型: {current_label}\n\n点击要切换到的模型：",
            ok=other_label,
            cancel="保持不变",
        )
        if response == 1:  # ok was clicked
            if other_label == "Small (快速)":
                self._set_model_small(None)
            else:
                self._set_model_medium(None)

    @rumps.timer(0.12)
    def _ui_updater(self, _):
        if self._pending_icon:
            icon, self._pending_icon = self._pending_icon, None
            NSApplication.sharedApplication().setApplicationIconImage_(
                make_dock_icon(icon)
            )

    # ── 热键 ─────────────────────────────────────────────────

    def _start_hotkey_listener(self):
        with keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        ) as listener:
            listener.join()

    def _on_press(self, key):
        # 追踪修饰键
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._ctrl_pressed = True
        elif key in (keyboard.Key.shift, keyboard.Key.shift_r):
            self._shift_pressed = True

        # Ctrl + Shift + 字母键快捷操作
        try:
            if self._ctrl_pressed and self._shift_pressed:
                if key.char in ("m", "M", "µ"):
                    threading.Thread(target=self._show_model_picker, daemon=True).start()
                    return
                if key.char in ("t", "T", "†"):
                    self._toggle_translate(None)
                    return
        except AttributeError:
            pass

        # 右 Option → 开始录音
        if key == keyboard.Key.alt_r and not self.recording:
            threading.Thread(target=self._start_recording, daemon=True).start()

    def _on_release(self, key):
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._ctrl_pressed = False
        elif key in (keyboard.Key.shift, keyboard.Key.shift_r):
            self._shift_pressed = False
        elif key == keyboard.Key.alt_r and self.recording:
            threading.Thread(target=self._stop_and_transcribe, daemon=True).start()

    # ── 录音 ─────────────────────────────────────────────────

    def _start_recording(self):
        self.frames    = []
        self.recording = True
        self.title     = "🔴"
        self._pending_icon = "🔴"
        self._set_status("录音中...")
        print("[INFO] 开始录音")

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
        print(f"[INFO] 停止录音，帧数: {len(self.frames)}")

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if not self.frames:
            self.title = self._idle_icon
            self._pending_icon = self._idle_icon
            self._set_status("就绪 ✓")
            return

        self.title = "💬"
        self._pending_icon = "💬"
        self._set_status("转录中...")
        self.transcribe_queue.put(list(self.frames))

    # ── 转录 ─────────────────────────────────────────────────

    def _transcription_worker(self):
        while True:
            frames = self.transcribe_queue.get()
            try:
                self._do_transcribe(frames)
            except Exception as e:
                print(f"[ERROR] 转录失败: {e}")
            finally:
                self.transcribe_queue.task_done()
                self.title = self._idle_icon
                self._pending_icon = self._idle_icon
                self._set_status("就绪 ✓")

    def _do_transcribe(self, frames):
        audio = np.concatenate(frames, axis=0).squeeze()
        audio_float = audio.astype(np.float32) / 32768.0

        if self.translate_mode:
            task, lang, prompt = "translate", None, None
            print(f"[INFO] 翻译 {len(audio_float)/SAMPLE_RATE:.1f}s 音频→英文...")
        else:
            task, lang, prompt = "transcribe", "zh", "以下是普通话与英语的混合对话。"
            print(f"[INFO] 转录 {len(audio_float)/SAMPLE_RATE:.1f}s 音频...")

        kwargs = dict(path_or_hf_repo=self.current_model, task=task)
        if lang:
            kwargs["language"] = lang
        if prompt:
            kwargs["initial_prompt"] = prompt

        result = mlx_whisper.transcribe(audio_float, **kwargs)
        text = result["text"].strip()
        print(f"[INFO] 结果: {text}")

        if text:
            self._auto_paste(text + " ")

    # ── 自动粘贴 ──────────────────────────────────────────────

    def _auto_paste(self, text):
        pyperclip.copy(text)
        time.sleep(0.2)
        subprocess.run([
            "osascript", "-e",
            'tell application "System Events" to keystroke "v" using command down'
        ])


if __name__ == "__main__":
    TranscriberApp().run()
