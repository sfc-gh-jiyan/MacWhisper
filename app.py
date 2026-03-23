"""
MacWhisper - macOS Menu Bar App
Hold Right Option to record, release to transcribe
Ctrl + Shift + M  switch model
Ctrl + Shift + T  toggle translate mode (all speech -> English)
Ctrl + Shift + S  toggle live subtitles (show preview while recording)
"""

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

from AppKit import (
    NSApplication, NSImage, NSPanel, NSColor, NSTextField, NSFont,
    NSScreen, NSMakeRect, NSBackingStoreBuffered,
)
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
import mlx_whisper
import opencc
from ApplicationServices import AXIsProcessTrusted

SAMPLE_RATE = 16000
LIVE_CHUNK_SECONDS = 3
MAX_LIVE_WINDOW = 30          # seconds – cap to keep inference within budget
SILENCE_RMS_THRESHOLD = 300   # int16 RMS below this = silence, skip inference
PAUSE_RMS_THRESHOLD   = 100   # RMS below this = silence (for pause detection, stricter than inference skip)
PAUSE_MIN_DURATION    = 0.8   # seconds of continuous silence to trigger pause commit
PAUSE_MIN_SEGMENT     = 10.0  # minimum segment length before allowing pause commit
PAUSE_SAFETY_FALLBACK = 28.0  # force commit at 28s even without pause (under 30s limit)
NSWindowStyleMaskBorderless = 0

_t2s = opencc.OpenCC('t2s')

import re
import unicodedata

_PUNCT_NORMALIZE = str.maketrans('，。！？、', ',.!?,')
_OVERLAP_STRIP_CHARS = set(' \t\n，。！？、,.!?-\u3000')

_HALLUCINATION_PHRASES = {
    "thank you for watching", "thanks for watching", "thank you",
    "please subscribe", "中文字幕君", "字幕由amara", "字幕提供",
    "请不吝点赞", "订阅转发", "打赏支持", "明镜与点点栏目",
    "感谢观看", "欢迎订阅", "点赞关注", "支持明镜",
}

_HALLUCINATION_SUBSTRINGS = [
    "请不吝点赞", "打赏支持明镜", "字幕由amara", "字幕提供",
]

def _strip_trailing_repetition(text):
    """Remove repetitive tail that Whisper appends when decoder loops.

    Uses normalized matching (lowercase, strip punctuation/whitespace) so
    that repetitions with slightly different punctuation are still caught.
    A position map translates the cut point back to the original string.
    """
    stripped = text.rstrip()
    if len(stripped) < 4:
        return text

    raw_tail = stripped[-300:] if len(stripped) > 300 else stripped
    raw_tail_offset = len(stripped) - len(raw_tail)

    # Build normalized tail with position mapping
    norm_chars = []
    positions = []  # norm_index -> index in stripped
    for i, ch in enumerate(raw_tail):
        if ch not in _OVERLAP_STRIP_CHARS:
            norm_chars.append(ch.lower())
            positions.append(raw_tail_offset + i)
    norm_tail = ''.join(norm_chars)

    if len(norm_tail) < 4:
        return text

    best_cut = None
    for unit_len in range(1, min(81, len(norm_tail) // 2 + 1)):
        unit = norm_tail[-unit_len:]
        count = 0
        pos = len(norm_tail)
        while pos >= unit_len and norm_tail[pos - unit_len:pos] == unit:
            count += 1
            pos -= unit_len
        min_repeats = 2 if unit_len > 10 else 3
        if count >= min_repeats:
            candidate = positions[pos] if pos < len(positions) else len(stripped)
            if best_cut is None or candidate < best_cut:
                best_cut = candidate

    if best_cut is not None:
        cleaned = text[:best_cut].rstrip(' ,，。.!！?？')
        return cleaned if cleaned else ""

    return text


def _common_prefix_len(a, b):
    """Return the length of the longest common prefix, tolerating case/punctuation."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] == b[i]:
            continue
        ca = a[i].lower().translate(_PUNCT_NORMALIZE)
        cb = b[i].lower().translate(_PUNCT_NORMALIZE)
        if ca != cb:
            return i
    return n


def _prefix_overlap_ratio(a, b):
    """Ratio of common prefix after stripping punctuation/spaces.

    Used by continuity guards to tolerate Whisper's punctuation and
    whitespace changes (e.g. comma→period, space inserted) while still
    catching true content rewrites.  Returns overlap / len(stripped_b).
    """
    def _strip(s):
        return ''.join(c.lower() for c in s if c not in _OVERLAP_STRIP_CHARS)
    sa, sb = _strip(a), _strip(b)
    if not sb:
        return 1.0
    n = min(len(sa), len(sb))
    common = 0
    for i in range(n):
        if sa[i] != sb[i]:
            break
        common += 1
    return common / len(sb)


def _snap_to_boundary(text, pos):
    """Snap a position back to the nearest sentence-ending punctuation."""
    if pos <= 0:
        return 0
    end = min(pos, len(text))
    for i in range(end - 1, max(0, end - 40) - 1, -1):
        if text[i] in '。！？.!?\n':
            return i + 1
    return end


def _find_after_overlap(committed, raw, min_match=6):
    """Find genuinely new content in *raw* that comes after *committed*.

    Uses aggressive normalization (lowercase, strip all punctuation and
    whitespace) so that Whisper's per-cycle wording variations (case,
    punctuation style) don't break the overlap search.  A position map
    translates the match back to the original *raw* string.
    """
    if not committed or not raw:
        return raw

    # Build normalized raw with position map back to original indices
    norm_raw_chars = []
    raw_positions = []
    for i, ch in enumerate(raw):
        if ch not in _OVERLAP_STRIP_CHARS:
            norm_raw_chars.append(ch.lower())
            raw_positions.append(i)
    norm_raw = ''.join(norm_raw_chars)

    # Normalize committed tail
    norm_committed = ''.join(
        ch.lower() for ch in committed if ch not in _OVERLAP_STRIP_CHARS
    )

    max_search = min(60, len(norm_committed))
    for tail_len in range(max_search, min_match - 1, -1):
        tail = norm_committed[-tail_len:]
        idx = norm_raw.find(tail)
        if idx >= 0:
            norm_end = idx + tail_len
            if norm_end >= len(raw_positions):
                return ""
            raw_end = raw_positions[norm_end]
            # Strip leading punctuation/space at the seam
            result = raw[raw_end:].lstrip(' ,，.。!！?？、')
            return result
    return raw


def _find_after_sentence_overlap(committed, raw, min_sent_len=6):
    """Fallback overlap using individual sentence anchors from committed tail.

    When ``_find_after_overlap`` fails (committed tail is paraphrased),
    this splits committed into sentences and searches for each one in *raw*.
    If any sentence from committed's tail is found, everything in *raw*
    after that anchor is returned as new content.
    """
    if not committed or not raw:
        return None

    # Split committed into sentences (keep delimiters attached to preceding text)
    sentences = re.split(r'(?<=[。！？.!?\n])', committed)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= min_sent_len]
    if not sentences:
        return None

    # Normalize raw once (same pattern as _find_after_overlap)
    norm_raw_chars = []
    raw_positions = []
    for i, ch in enumerate(raw):
        if ch not in _OVERLAP_STRIP_CHARS:
            norm_raw_chars.append(ch.lower())
            raw_positions.append(i)
    norm_raw = ''.join(norm_raw_chars)

    # Try each sentence from committed's tail, last to first (max 5)
    for sent in reversed(sentences[-5:]):
        norm_sent = ''.join(
            ch.lower() for ch in sent if ch not in _OVERLAP_STRIP_CHARS
        )
        if len(norm_sent) < min_sent_len:
            continue
        idx = norm_raw.find(norm_sent)
        if idx >= 0:
            norm_end = idx + len(norm_sent)
            if norm_end >= len(raw_positions):
                return ""
            raw_end = raw_positions[norm_end]
            return raw[raw_end:].lstrip(' ,，.。!！?？、')

    return None


def _is_hallucination(text):
    lower = text.lower().strip(" .!,。，！")
    if lower in _HALLUCINATION_PHRASES:
        return True
    for sub in _HALLUCINATION_SUBSTRINGS:
        if sub in text:
            return True
    tokens = text.split()
    if len(tokens) >= 3:
        if len(set(tokens)) == 1:
            return True
        for i in range(len(tokens) - 2):
            if tokens[i] == tokens[i+1] == tokens[i+2]:
                return True
    if len(text) >= 6:
        for size in range(1, len(text) // 3 + 1):
            pat = text[:size]
            if pat * (len(text) // len(pat)) == text[:len(pat) * (len(text) // len(pat))] and len(text) // len(pat) >= 3:
                return True
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith('L'):
            block = ord(ch)
            is_latin = block < 0x0250
            is_cjk = 0x2E80 <= block <= 0x9FFF or 0xF900 <= block <= 0xFAFF
            is_cjk_ext = 0x20000 <= block <= 0x2FA1F
            is_kana = 0x3040 <= block <= 0x30FF
            if not (is_latin or is_cjk or is_cjk_ext or is_kana):
                return True
    return False

CONFIG_PATH = os.path.expanduser("~/.macwhisper_config.json")

MODEL_OPTIONS = {
    "Small (Fast)":      "mlx-community/whisper-small-mlx",
    "Medium (Accurate)": "mlx-community/whisper-medium-mlx",
    "Large (Best)":      "mlx-community/whisper-large-v3-mlx",
}
MODEL_KEYS = list(MODEL_OPTIONS.keys())

HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history")
AUDIO_DIR = os.path.join(HISTORY_DIR, "audio")
TRANSCRIPT_LOG = os.path.join(HISTORY_DIR, "transcripts.jsonl")
SUBTITLE_LOG = os.path.join(HISTORY_DIR, "subtitles.jsonl")

def _ensure_history_dirs():
    os.makedirs(AUDIO_DIR, exist_ok=True)


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
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f)

    # ── Init ──────────────────────────────────────────────────

    def __init__(self):
        cfg = self._load_config()
        self.translate_mode = cfg.get("translate_mode", False)
        self.current_model  = cfg.get("current_model", MODEL_OPTIONS["Small (Fast)"])
        self.live_mode      = cfg.get("live_mode", False)

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

        self.menu = [
            rumps.MenuItem("Status: Ready"),
            rumps.separator,
            *model_menu_items,
            rumps.separator,
            rumps.MenuItem(f"{translate_prefix} Translate to English", callback=self._toggle_translate),
            rumps.separator,
            self.item_live,
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
        self._overlay_panel     = None
        self._overlay_text      = None
        self._live_queue        = queue.Queue(maxsize=2)
        self._inference_lock    = threading.Lock()
        self._last_live_result  = ""

        # Committed text state (stability-based commit)
        self._committed_text    = ""
        self._prev_raw_text     = ""
        self._stable_prefix_len = 0
        self._stable_cycles     = 0

        # Workers
        self.transcribe_queue = queue.Queue()
        threading.Thread(target=self._transcription_worker, daemon=True).start()
        threading.Thread(target=self._live_transcription_worker, daemon=True).start()
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
        self._last_live_result = ""
        self._best_raw          = ""
        self._prev_raw          = ""
        self._frozen_prefix     = ""
        self._stale_count       = 0
        # Pause-based segmentation state
        self._segment_start_frame    = 0      # index into self.frames where current segment starts
        self._pause_silence_frames   = 0      # consecutive silent frames counter
        self._pause_detected         = False  # flag for chunk loop to commit
        self._segment_committed_text = ""     # accumulated text from completed segments
        self.title     = "🟠"
        self._set_status("Recording...")
        print("[INFO] Recording started")

        if self.live_mode:
            AppHelper.callAfter(self._create_overlay)
            threading.Thread(target=self._live_chunk_loop, daemon=True).start()

        _pause_silence_threshold = int(PAUSE_MIN_DURATION * SAMPLE_RATE / 1024)

        def callback(indata, frame_count, time_info, status):
            if self.recording:
                self.frames.append(indata.copy())
                # Pause detection: track consecutive silent frames
                rms = np.sqrt(np.mean(indata.astype(np.float64) ** 2))
                if rms < PAUSE_RMS_THRESHOLD:
                    self._pause_silence_frames += 1
                else:
                    self._pause_silence_frames = 0
                seg_frames = len(self.frames) - self._segment_start_frame
                seg_secs = seg_frames * 1024 / SAMPLE_RATE
                if (self._pause_silence_frames >= _pause_silence_threshold
                        and seg_secs >= PAUSE_MIN_SEGMENT
                        and not self._pause_detected):
                    self._pause_detected = True
                    print(f"[INFO] Pause detected at {seg_secs:.1f}s into segment")

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

        # Drain any pending live chunks BEFORE destroying overlay
        # so the worker won't schedule more _replace_overlay calls
        while not self._live_queue.empty():
            try:
                self._live_queue.get_nowait()
            except queue.Empty:
                break

        if self._overlay_panel:
            AppHelper.callAfter(self._destroy_overlay)

        if not self.frames:
            self.title = self._idle_icon
            self._set_status("Ready")
            return

        self.title = "💬"
        self._set_status("Transcribing...")
        self.transcribe_queue.put(list(self.frames))

    # ── Transcription (hold-to-record) ────────────────────────

    def _transcription_worker(self):
        while True:
            frames = self.transcribe_queue.get()
            try:
                with self._inference_lock:
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

        # Save audio WAV for test corpus
        _ensure_history_dirs()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        wav_path = os.path.join(AUDIO_DIR, f"{timestamp}.wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
        print(f"[INFO] Saved audio: {wav_path}")

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
        if prompt and text.startswith(prompt):
            text = text[len(prompt):].strip()
        text = _strip_trailing_repetition(text)
        print(f"[INFO] Result: {text}")

        # Log transcription result
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "audio_file": os.path.basename(wav_path),
            "model": self.current_model,
            "translate": self.translate_mode,
            "duration_s": round(len(audio_float) / SAMPLE_RATE, 1),
            "text": text,
        }
        with open(TRANSCRIPT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        print(f"[INFO] Logged transcript: {TRANSCRIPT_LOG}")

        if text:
            self._auto_paste(text + " ")

    # ── Live Subtitles — overlay (NSPanel) ────────────────────

    def _create_overlay(self):
        screen = NSScreen.mainScreen()
        if not screen:
            print("[ERROR] No main screen found, cannot create overlay")
            return False
        screen_frame = screen.frame()
        panel_w = min(screen_frame.size.width * 0.8, 960)
        panel_h = 60
        panel_x = (screen_frame.size.width - panel_w) / 2
        panel_y = 40

        rect = NSMakeRect(panel_x, panel_y, panel_w, panel_h)
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
        )
        panel.setLevel_(1000)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(0, 0, 0, 0.88))
        panel.setIgnoresMouseEvents_(True)
        panel.setHasShadow_(False)
        panel.setHidesOnDeactivate_(False)
        panel.setCollectionBehavior_(1 << 0 | 1 << 4)

        content = panel.contentView()
        content_frame = content.frame()

        text_field = NSTextField.alloc().initWithFrame_(
            NSMakeRect(20, 0, content_frame.size.width - 40, content_frame.size.height)
        )
        text_field.setStringValue_("")
        text_field.setEditable_(False)
        text_field.setSelectable_(False)
        text_field.setBordered_(False)
        text_field.setDrawsBackground_(False)
        text_field.setTextColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, 0.92))
        text_field.setFont_(NSFont.systemFontOfSize_(15))
        text_field.setMaximumNumberOfLines_(0)
        text_field.cell().setWraps_(True)

        content.addSubview_(text_field)

        panel.orderFrontRegardless()
        self._overlay_panel = panel
        self._overlay_text = text_field
        print("[INFO] Overlay panel created")
        return True

    def _destroy_overlay(self):
        if self._overlay_panel:
            self._overlay_panel.orderOut_(None)
            self._overlay_panel = None
            self._overlay_text = None
            print("[INFO] Overlay panel destroyed")

    def _replace_overlay(self, full_text):
        display_text = re.sub(r'([。！？])\s*', r'\1\n', full_text).rstrip('\n')

        def _do_update():
            if not self._overlay_text or not self._overlay_panel:
                return
            self._overlay_text.setStringValue_(display_text)

            screen = NSScreen.mainScreen()
            if not screen:
                return
            screen_frame = screen.frame()
            panel_w = self._overlay_panel.frame().size.width

            attr_str = self._overlay_text.attributedStringValue()
            text_w = panel_w - 40
            text_rect = attr_str.boundingRectWithSize_options_(
                NSMakeRect(0, 0, text_w, 10000).size,
                1 << 0 | 1 << 2,
            )
            needed_h = text_rect.size.height + 28
            max_h = screen_frame.size.height * 0.5
            panel_h = min(needed_h, max_h)

            panel_x = (screen_frame.size.width - panel_w) / 2
            panel_y = 40
            self._overlay_panel.setFrame_display_(
                NSMakeRect(panel_x, panel_y, panel_w, panel_h), True
            )
            content_frame = self._overlay_panel.contentView().frame()
            self._overlay_text.setFrame_(
                NSMakeRect(20, 0, content_frame.size.width - 40, content_frame.size.height)
            )

        AppHelper.callAfter(_do_update)

    # ── Live Subtitles — chunk timer & worker ─────────────────

    def _live_chunk_loop(self):
        max_frames = int(MAX_LIVE_WINDOW * SAMPLE_RATE / 1024)

        while self.recording:
            n = len(self.frames)
            seg_start = self._segment_start_frame
            seg_frames = n - seg_start
            seg_secs = seg_frames * 1024 / SAMPLE_RATE
            interval = min(LIVE_CHUNK_SECONDS, max(1.0, seg_secs / 5.0))
            time.sleep(interval)
            if not self.recording:
                break

            n = len(self.frames)
            seg_start = self._segment_start_frame
            if n <= seg_start:
                continue

            # Check for pause commit or safety fallback
            seg_secs = (n - seg_start) * 1024 / SAMPLE_RATE
            needs_commit = self._pause_detected or seg_secs >= PAUSE_SAFETY_FALLBACK

            if needs_commit and self._last_live_result:
                # Commit: _last_live_result already includes _segment_committed_text
                # as prefix (from _build_display_text), so direct assignment avoids
                # double-counting.
                self._segment_committed_text = self._last_live_result
                reason = "pause" if self._pause_detected else f"safety@{seg_secs:.0f}s"
                print(f"[INFO] Segment committed ({reason}): "
                      f"{len(self._segment_committed_text)} chars total")
                # Reset per-segment state
                self._segment_start_frame = n
                self._best_raw = ""
                self._prev_raw = ""
                self._frozen_prefix = ""
                self._stale_count = 0
                self._last_live_result = ""
                self._pause_detected = False
                self._pause_silence_frames = 0
                continue  # skip this cycle, next cycle starts fresh segment

            # Take snapshot from current segment only, capped at max_frames
            seg_end = n
            seg_begin = max(seg_start, seg_end - max_frames)
            snapshot = self.frames[seg_begin:seg_end]

            if not self._live_queue.empty():
                continue

            try:
                self._live_queue.put_nowait(snapshot)
            except queue.Full:
                pass

    def _live_transcription_worker(self):
        while True:
            snapshot = self._live_queue.get()
            if not self.recording:
                continue
            try:
                with self._inference_lock:
                    raw = self._do_live_transcribe(snapshot)
                if raw and self.recording:
                    display = self._build_display_text(raw)
                    self._last_live_result = display
                    _ensure_history_dirs()
                    entry = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "audio_s": round(len(snapshot) * 1024 / SAMPLE_RATE, 1),
                        "raw": raw,
                        "display": display,
                    }
                    with open(SUBTITLE_LOG, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    if self._overlay_panel and self.recording:
                        AppHelper.callAfter(lambda t=display: self._replace_overlay(t))
            except Exception as e:
                print(f"[ERROR] Live transcription failed: {e}")

    def _do_live_transcribe(self, frames):
        audio = np.concatenate(frames, axis=0).squeeze()

        tail_samples = min(len(audio), int(3 * SAMPLE_RATE))
        rms = np.sqrt(np.mean(audio[-tail_samples:].astype(np.float64) ** 2))
        if rms < SILENCE_RMS_THRESHOLD:
            print(f"[INFO] Live chunk silent (RMS={rms:.0f}), skipped")
            return ""

        audio_float = audio.astype(np.float32) / 32768.0
        duration = len(audio_float) / SAMPLE_RATE
        print(f"[INFO] Live transcribing {duration:.1f}s chunk (RMS={rms:.0f})...")

        prompt = "以下是普通话与英语的混合对话。"

        t0 = time.time()
        result = mlx_whisper.transcribe(
            audio_float,
            path_or_hf_repo=self.current_model,
            task="transcribe",
            condition_on_previous_text=False,
            initial_prompt=prompt,
        )
        elapsed = time.time() - t0
        text = result["text"].strip()
        print(f"[INFO] Live inference took {elapsed:.1f}s for {duration:.1f}s audio")

        if text.startswith(prompt):
            text = text[len(prompt):].strip()

        text = _t2s.convert(text)
        if text in prompt or text.rstrip("。.") in prompt:
            print(f"[INFO] Live result (prompt echo filtered): {text}")
            return ""
        text = _strip_trailing_repetition(text)
        if not text or _is_hallucination(text):
            print(f"[INFO] Live result (hallucination filtered): {text}")
            return ""
        print(f"[INFO] Live result: {text}")
        return text

    def _build_display_text(self, raw_text):
        """Stabilized display: ratchet growth + frozen prefix tracking.

        Within a segment audio only grows, so Whisper output should grow
        proportionally.  Shorter outputs are truncation regressions and
        are ignored (ratchet).  The frozen prefix tracks the longest
        common sentence-boundary-aligned prefix between consecutive
        accepted raws — it only grows, providing visual stability.

        Prefix continuity guard: even if the new raw is longer, reject it
        when it doesn't preserve the frozen prefix — that means Whisper
        rewrote the beginning (content rewrite), which would silently
        drop already-displayed text.
        """
        accept = False
        stale_override = self._stale_count >= 5
        if len(raw_text) >= len(self._best_raw):
            accept = True
            if not stale_override:
                # Guard 1: reject if new raw rewrites substantial best_raw
                if accept and len(self._best_raw) >= 15:
                    if _prefix_overlap_ratio(raw_text, self._best_raw) < 0.5:
                        accept = False
                # Guard 2: reject if new raw rewrites frozen prefix
                if accept and len(self._frozen_prefix) >= 4:
                    if _prefix_overlap_ratio(raw_text, self._frozen_prefix) < 0.5:
                        accept = False
        # else: shorter raw = regression, keep _best_raw

        if accept:
            if stale_override:
                # Reset frozen prefix on stale recovery so new content
                # isn't immediately blocked by the old frozen state
                self._frozen_prefix = ""
            # Grow frozen prefix from common prefix with previous accepted raw
            if self._prev_raw:
                pfx = _common_prefix_len(raw_text, self._prev_raw)
                snapped = _snap_to_boundary(raw_text, pfx)
                if snapped > len(self._frozen_prefix):
                    self._frozen_prefix = raw_text[:snapped]
            # Ratchet: accept longer/equal raw
            self._prev_raw = raw_text
            self._best_raw = raw_text
            self._stale_count = 0
        else:
            self._stale_count += 1

        display = self._best_raw
        # Prepend text from previously committed segments
        if self._segment_committed_text:
            display = self._segment_committed_text + display
        return display

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
