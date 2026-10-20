"""
MacWhisper - macOS Menu Bar App  v0.7.3
Hold Right Option to record, release to transcribe
Ctrl + Shift + M  switch model
Ctrl + Shift + T  toggle translate mode (all speech -> English)
Ctrl + Shift + S  toggle live subtitles (show preview while recording)

v0.5: LocalAgreement architecture — word-level confirmation,
      dual-color overlay, Silero VAD, pluggable ASR backend.
"""

__version__ = "0.7.3"

import json
import io
import logging
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

# ── NSBundle swizzle ───────────────────────────────────────
# Homebrew's framework Python contains Python.app which hijacks
# NSBundle.mainBundle(), making it return Python.app's bundle instead
# of MacWhisper.app's.  This breaks LSUIElement (menu-bar-only mode)
# and the status bar icon.  We fix it by swizzling mainBundle at the
# ObjC runtime level before any AppKit code reads it.
import ctypes, ctypes.util  # noqa: E401,E402
_objc_lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
_swizzle_prevent_gc = None  # prevent GC of ctypes callback

def _swizzle_main_bundle():
    """Override NSBundle.mainBundle() to return MacWhisper.app's bundle."""
    global _swizzle_prevent_gc
    import objc
    from Foundation import NSBundle as _NSB

    _app_path = "/Applications/MacWhisper.app"
    if not os.path.isdir(_app_path):
        return  # not installed as .app — nothing to fix

    _custom = _NSB.bundleWithPath_(_app_path)
    if _custom is None:
        return

    lib = _objc_lib  # noqa: F821 — defined at module level before this function
    for fn, res, args in [
        ("objc_getClass",         ctypes.c_void_p, [ctypes.c_char_p]),
        ("sel_registerName",      ctypes.c_void_p, [ctypes.c_char_p]),
        ("class_getClassMethod",  ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p]),
        ("method_setImplementation", ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p]),
    ]:
        f = getattr(lib, fn)
        f.restype, f.argtypes = res, args

    cls = lib.objc_getClass(b"NSBundle")
    sel = lib.sel_registerName(b"mainBundle")
    method = lib.class_getClassMethod(cls, sel)
    if not method:
        return

    _ptr = objc.pyobjc_id(_custom)
    _IMP = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)

    def _replacement(self, _cmd):
        return _ptr

    _swizzle_prevent_gc = _IMP(_replacement)
    lib.method_setImplementation(method, _swizzle_prevent_gc)

_swizzle_main_bundle()
del _swizzle_main_bundle, _objc_lib  # one-shot; clean up namespace
# ── end NSBundle swizzle ───────────────────────────────────

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
    hallucination_reason,
)
from asr_backend import MLXWhisperBackend
from vad import VoiceActivityDetector
from online_processor import OnlineASRProcessor
from overlay import create_overlay, update_overlay, destroy_overlay
from subtitle_export import save_enhanced_history
from meeting import MeetingSession, MEETINGS_DIR
import crash_guard
import error_alert
import qwen_backend
import qwen_engine
import settings_window
import shortcuts
import quit_guard
import single_instance
from qwen_backend import QwenUnavailable

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────
SAMPLE_RATE = 16000
MIN_CHUNK_SIZE = 1.0          # seconds — LocalAgreement confirmation latency ≈ 2x this
MAX_BUFFER_S = 8.0            # audio buffer cap (seconds) — balance inference speed vs context

_DATA_DIR = os.path.expanduser("~/.macwhisper")
CONFIG_PATH = os.path.join(_DATA_DIR, "config.json")
OVERLAY_FRAME_PATH = os.path.join(_DATA_DIR, "overlay.json")
LOG_DIR = os.path.join(_DATA_DIR, "logs")
CRASH_LOG_PATH = os.path.join(LOG_DIR, "crash.log")
RUN_MARKER_PATH = os.path.join(_DATA_DIR, ".running")
INSTANCE_LOCK_PATH = os.path.join(_DATA_DIR, ".instance.lock")
_INSTANCE_LOCK = None  # held for the whole process (GC would release the lock)


def _stream_is_file(stream, path: str) -> bool:
    """True if `stream` writes to the file at `path` (same device + inode)."""
    try:
        a = os.fstat(stream.fileno())
        b = os.stat(path)
    except (OSError, ValueError, AttributeError, io.UnsupportedOperation):
        return False
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def _input_monitoring_granted():
    """True/False from macOS (CGPreflightListenEventAccess); None if unknown."""
    try:
        import Quartz
        return bool(Quartz.CGPreflightListenEventAccess())
    except Exception:
        return None


def _permission_status() -> dict:
    """Ask macOS directly — the hotkeys need both, for THIS app identity."""
    return {"accessibility": bool(AXIsProcessTrusted()),
            "input_monitoring": _input_monitoring_granted()}


def _notify_already_running() -> None:
    """Tell the user why a second launch did nothing (no rumps app yet)."""
    import subprocess
    try:
        subprocess.run(
            ["osascript", "-e", 'display notification "MacWhisper is already running '
             '— look for it in the menu bar." with title "MacWhisper"'],
            timeout=5, check=False,
        )
    except Exception:
        pass


def _claim_single_instance() -> bool:
    """True if this process is the only MacWhisper; otherwise notify and return False."""
    global _INSTANCE_LOCK
    _INSTANCE_LOCK = single_instance.acquire(INSTANCE_LOCK_PATH)
    if _INSTANCE_LOCK is not None:
        return True
    logging.getLogger(__name__).warning(
        "MacWhisper is already running (pid %s) — exiting this launch",
        single_instance.holder_pid(INSTANCE_LOCK_PATH))
    _notify_already_running()
    return False
HOTKEY_RESTART_DELAY_S = 2.0   # backoff base when the pynput listener dies
# A combo pressed again without a release within this window is macOS key
# auto-repeat (holding the keys), not a new press.
COMBO_REPEAT_WINDOW_S = 1.5
# Meeting start/stop ignores further toggles for this long (double start/stop)
MEETING_TOGGLE_COOLDOWN_S = 1.0

# Set by __main__ when the previous run did not exit cleanly (crash.log tail)
_PREVIOUS_CRASH_TAIL: str | None = None

MODEL_OPTIONS = {
    "Small (Fast)":      "mlx-community/whisper-small-mlx",
    "Medium (Accurate)": "mlx-community/whisper-medium-mlx",
    "Large (Best)":      "mlx-community/whisper-large-v3-mlx",
}
MODEL_KEYS = list(MODEL_OPTIONS.keys())

# Qwen3-ASR models run in a separate sidecar process (.venv-qwen); Whisper
# stays loaded alongside as the instant fallback (translate mode, errors).
QWEN_MODEL_OPTIONS = {
    "Qwen3 0.6B (Fast, zh+en)": "mlx-community/Qwen3-ASR-0.6B-bf16",
    "Qwen3 1.7B (Best, zh+en)": "mlx-community/Qwen3-ASR-1.7B-bf16",
}
ALL_MODEL_OPTIONS = {**MODEL_OPTIONS, **QWEN_MODEL_OPTIONS}
ALL_MODEL_KEYS = list(ALL_MODEL_OPTIONS.keys())
DEFAULT_QWEN_MODEL = QWEN_MODEL_OPTIONS["Qwen3 0.6B (Fast, zh+en)"]


_flag_listener_cache: dict = {}


def _flag_listener_cls(base):
    """Subclass of `base` (pynput keyboard.Listener) that hands each event's
    own modifier flags to the app.

    The app resyncs its modifier tracking (a lost key release must not leave
    a modifier stuck). It must use the flags carried by the event being
    handled, not the keyboard state "now": by the time a fast ⌃⌥M is
    processed the keys may already be up, and the shortcut would be lost.
    The flags are stored on `sink._event_flags` right before pynput calls
    on_press/on_release, on the same thread.

    Built from the CURRENT keyboard.Listener at listener start, so tests that
    patch keyboard.Listener (the restart watchdog test) still take effect.
    """
    cls = _flag_listener_cache.get(base)
    if cls is not None:
        return cls

    class FlagListener(base):
        def __init__(self, sink, **kwargs):
            super().__init__(**kwargs)
            self._mw_sink = sink

        def _handle_message(self, proxy, event_type, event, refcon, injected):
            try:
                import Quartz
                self._mw_sink._event_flags = int(Quartz.CGEventGetFlags(event))
            except Exception:
                self._mw_sink._event_flags = None   # no resync for this event
            return super()._handle_message(proxy, event_type, event, refcon, injected)

    _flag_listener_cache[base] = cls = FlagListener
    return cls


def initial_engine(cfg: dict, qwen_env_exists: bool) -> str:
    """Engine on startup: the saved choice, else Qwen3 if its env is installed."""
    if cfg.get("engine") in ("whisper", "qwen"):
        return cfg["engine"]
    return "qwen" if qwen_env_exists else "whisper"

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
            "meeting_subtitles": self.meeting_subtitles,
            "save_meeting_audio": self.save_meeting_audio,
            "save_audio": self.save_audio,
            "log_level": self.log_level,
            "engine": getattr(self, "engine", "whisper"),
            "qwen_model": getattr(self, "qwen_model", DEFAULT_QWEN_MODEL),
            "qwen_strip_fillers": getattr(self, "qwen_strip_fillers", True),
            "qwen_tight_spacing": getattr(self, "qwen_tight_spacing", True),
            "meeting_final_pass": getattr(self, "meeting_final_pass", True),
            "shortcuts": getattr(self, "shortcuts", None) or shortcuts.merged({}),
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f)

    # ── Init ──────────────────────────────────────────────────

    def __init__(self):
        _migrate_old_data()
        cfg = self._load_config()
        self.translate_mode = cfg.get("translate_mode", False)
        self.current_model  = cfg.get("current_model", MODEL_OPTIONS["Medium (Accurate)"])
        self.live_mode      = cfg.get("live_mode", False)
        self.meeting_subtitles = cfg.get("meeting_subtitles", True)
        # Off by default: recording participants requires their consent
        self.save_meeting_audio = cfg.get("save_meeting_audio", False)
        self.save_audio     = cfg.get("save_audio", False)
        self.log_level      = cfg.get("log_level", "WARNING")
        # Qwen3-ASR (v0.7.0): "engine" selects Qwen3 or Whisper. current_model
        # always names the Whisper model, which stays loaded as the fallback.
        self.engine = initial_engine(cfg, os.path.isfile(qwen_backend.default_python_path()))
        self.qwen_model = cfg.get("qwen_model", DEFAULT_QWEN_MODEL)
        if self.qwen_model not in QWEN_MODEL_OPTIONS.values():
            self.qwen_model = DEFAULT_QWEN_MODEL
        self.qwen_strip_fillers = cfg.get("qwen_strip_fillers", True)
        self.qwen_tight_spacing = cfg.get("qwen_tight_spacing", True)
        # Re-transcribe meetings with Qwen3 from in-memory audio (never written to disk)
        self.meeting_final_pass = cfg.get("meeting_final_pass", True)
        # Keyboard shortcuts (v0.7.1): defaults fill anything missing/invalid
        self.shortcuts = shortcuts.merged(cfg.get("shortcuts"))
        self._apply_shortcut_hints()

        # Meeting mode state
        self._meeting_session: MeetingSession | None = None

        idle_icon = "🌐" if self.translate_mode else "🎙"
        super().__init__(idle_icon, quit_button="Quit")

        # Model menu items
        self.model_items = {}
        model_menu_items = []
        for label, repo in ALL_MODEL_OPTIONS.items():
            prefix = "✅" if label == self._active_model_label() else "  "
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

        self.item_meeting = rumps.MenuItem("   Meeting Mode: Off", callback=self._toggle_meeting_mode)

        msub_prefix = "✅" if self.meeting_subtitles else "  "
        msub_label = "On" if self.meeting_subtitles else "Off"
        self.item_meeting_subtitles = rumps.MenuItem(
            f"{msub_prefix} Meeting Subtitles: {msub_label}",
            callback=self._toggle_meeting_subtitles,
        )

        msa_prefix = "✅" if self.save_meeting_audio else "  "
        msa_label = "On" if self.save_meeting_audio else "Off"
        self.item_save_meeting_audio = rumps.MenuItem(
            f"{msa_prefix} Save Meeting Audio: {msa_label}",
            callback=self._toggle_save_meeting_audio,
        )
        self.item_translate = rumps.MenuItem(
            f"{translate_prefix} Translate to English", callback=self._toggle_translate
        )

        # v0.7.1: slim menu (status, models, meeting, Settings…). Every option
        # lives in the Settings window; the item_* objects are still kept so
        # the existing toggle handlers can update their titles unchanged.
        self._settings = None
        self.menu = self._build_menu()
        self.status_item    = self.menu["Status: Ready"]

        # Hold-to-record state
        self.recording     = False
        self.frames        = []
        self.stream        = None
        self._stream_lock  = threading.Lock()  # guards self.stream open/close
        self._rec_gen      = 0                 # recording generation token
        self._idle_icon    = idle_icon

        # Error reporting — every failure must reach the user, never silent
        self._reporter = crash_guard.ErrorReporter(alert=self._show_error)
        crash_guard.install_thread_excepthook(self._reporter)
        self._hotkey_stop = threading.Event()

        # Hotkey modifier tracking ({"ctrl", "alt", "shift", "cmd"} held now)
        self._held_mods: set = set()
        self._ptt_cancelled = False   # PTT key turned out to be part of a combo
        self._shortcuts_paused = False  # True while Settings records a shortcut
        self._combo_down: dict = {}     # combo spec -> last press time (until released)
        self._event_flags = None        # modifier flags of the event being handled
        self._key_event_received = False

        # Live subtitle overlay state
        self._overlay_panel    = None
        self._overlay_text     = None
        self._last_overlay_key = None   # (confirmed, unconfirmed) dedup

        # Online processor (created per recording session)
        self._processor: OnlineASRProcessor | None = None
        self._live_loop_done = threading.Event()
        self._live_loop_done.set()  # initially "done" (no loop running)

        # Pre-warmed ASR backend — created once, reused across recordings.
        # Warmup runs in a background thread so the app starts instantly.
        self._backend: MLXWhisperBackend | None = None
        self._backend_ready = threading.Event()
        self._warmup_model(self.current_model)

        # Qwen3 sidecar (optional engine). Whisper above stays the fallback.
        self._qwen: qwen_engine.QwenEngine | None = None
        self._qwen_alerted = False
        if self.engine == "qwen":
            self._start_qwen()

        # VAD — created once, reset() per recording (avoids reloading Silero model)
        self._vad: VoiceActivityDetector | None = None
        self._vad_available = False
        try:
            self._vad = VoiceActivityDetector(
                threshold=0.5, min_silence_ms=800, min_speech_ms=250,
            )
            self._vad_available = True
        except Exception:
            print("[WARN] VAD unavailable at startup")

        # Workers
        self.transcribe_queue = queue.Queue()
        threading.Thread(target=self._transcription_worker, daemon=True).start()
        threading.Thread(target=self._start_hotkey_listener, daemon=True).start()
        threading.Thread(target=self._check_permissions, daemon=True).start()

    @rumps.timer(2)
    def _report_previous_crash(self, sender):
        """One-shot: if the last run died, tell the user (needs a live event loop)."""
        sender.stop()
        tail = _PREVIOUS_CRASH_TAIL
        if tail is None:
            return
        last = "\n".join(tail.splitlines()[-6:]) or "(no details captured)"
        self._show_error(
            "⚠️ MacWhisper crashed last time",
            "The previous session ended unexpectedly. Anything you said right "
            "before that may not have been transcribed.\n\n"
            f"Last lines of crash.log:\n{last}",
        )

    # ── Main-thread UI helpers ────────────────────────────────
    # AppKit raises NSInternalInconsistencyException (or aborts the process)
    # when menu/window state is modified off the main thread. All UI writes
    # from worker/hotkey threads go through _on_main.

    @staticmethod
    def _on_main(fn):
        from Foundation import NSThread
        if NSThread.isMainThread():
            fn()
        else:
            AppHelper.callAfter(fn)

    def _set_status(self, text):
        item = self.status_item
        self._on_main(lambda: setattr(item, "title", f"Status: {text}"))

    def _set_title(self, icon):
        def apply():
            self.title = icon
        self._on_main(apply)

    def _show_error(self, title, message):
        """Persistent red alert + notification. Safe from any thread."""
        def apply():
            error_alert.show_error_alert(title, message, frame_path=OVERLAY_FRAME_PATH)
            try:
                rumps.notification(title, "", message.splitlines()[0])
            except Exception:
                logger.debug("notification failed", exc_info=True)
        self._on_main(apply)

    def _guarded(self, fn, where):
        """Wrap a callback so exceptions are reported instead of propagating.

        pynput stops the whole keyboard listener on any unhandled exception,
        which silently disables the Option key — never let one escape.

        pynput inspects the callback's signature and passes as many args as
        it accepts (a *args callback gets (key, injected)), so trim to the
        wrapped function's own arity.
        """
        import inspect
        try:
            arity = len(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            arity = None

        def inner(*args):
            if arity is not None:
                args = args[:arity]
            try:
                return fn(*args)
            except Exception as e:
                self._reporter.report(where, e, while_recording=getattr(self, "recording", False))
        return inner

    # ── Backend warmup ───────────────────────────────────────

    def _warmup_model(self, model_repo: str):
        """Create backend and warm up Metal shaders in a background thread."""
        self._backend_ready.clear()

        def _do_warmup():
            print(f"[INFO] Warming up model: {model_repo}")
            backend = MLXWhisperBackend(model_repo=model_repo)
            warmup_audio = np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32)
            backend.transcribe(warmup_audio, language=None, task="transcribe")
            self._backend = backend
            self._backend_ready.set()
            print("[INFO] Warmup complete — backend ready")

        threading.Thread(target=_do_warmup, daemon=True).start()

    # ── Model switching ───────────────────────────────────────

    def _make_model_callback(self, label):
        def cb(_):
            self._set_model(label)
        return cb

    def _build_menu(self) -> list:
        settings = rumps.MenuItem("Settings…", callback=self._open_settings, key=",")
        return [
            rumps.MenuItem("Status: Ready"),
            rumps.separator,
            *[self.model_items[label] for label in ALL_MODEL_KEYS],
            rumps.separator,
            self.item_meeting,
            rumps.separator,
            settings,
            rumps.separator,
        ]

    # ── Settings window ───────────────────────────────────────

    # Preferences without a dedicated toggle handler (set from Settings)
    _PREFS = ("meeting_final_pass", "qwen_strip_fillers", "qwen_tight_spacing")

    def _open_settings(self, _):
        if getattr(self, "login_item", None) is None:
            import login_item  # loads ServiceManagement only when Settings is opened
            self.login_item = login_item.LoginItem.for_this_app()
        if self._settings is None:
            self._settings = settings_window.SettingsWindow(self, ALL_MODEL_KEYS)
        self._settings.show()

    def _refresh_settings(self):
        if getattr(self, "_settings", None) is not None:
            self._settings.refresh()

    def _set_pref(self, name: str, value: bool) -> None:
        if name not in self._PREFS:
            raise ValueError(f"unknown preference: {name}")
        setattr(self, name, bool(value))
        qwen = getattr(self, "_qwen", None)
        if qwen is not None:   # apply to the running engine without a restart
            if name == "qwen_strip_fillers":
                qwen.strip_fillers = bool(value)
            elif name == "qwen_tight_spacing":
                qwen.tight = bool(value)
        logger.info("Preference %s = %s", name, value)
        self._save_config()

    def _set_shortcuts(self, cfg: dict) -> None:
        self.shortcuts = shortcuts.merged(cfg)
        self._apply_shortcut_hints()
        logger.info("Shortcuts updated: %s", self.shortcuts)
        self._save_config()

    def _apply_shortcut_hints(self) -> None:
        """Text that names shortcuts (meeting overlay label) follows the config."""
        import overlay as _overlay
        meeting = self.shortcuts.get("meeting") or []
        _overlay.meeting_stop_hint = shortcuts.display(meeting[0]) if meeting else ""

    def _active_model_label(self) -> str:
        """Menu label of the engine/model actually selected."""
        repo = self.qwen_model if self.engine == "qwen" else self.current_model
        for label, r in ALL_MODEL_OPTIONS.items():
            if r == repo:
                return label
        return MODEL_KEYS[0]

    def _refresh_model_menu(self):
        active = self._active_model_label()
        qwen = getattr(self, "_qwen", None)
        qwen_down = (self.engine == "qwen" and qwen is not None
                     and not qwen.is_ready and bool(qwen.last_error))
        for k, item in getattr(self, "model_items", {}).items():
            prefix = "✅" if k == active else "  "
            suffix = " — unavailable (using Whisper)" if (k == active and qwen_down) else ""
            item.title = f"{prefix} {k}{suffix}"

    def _set_model(self, label):
        if label in QWEN_MODEL_OPTIONS:
            self.engine = "qwen"
            self.qwen_model = QWEN_MODEL_OPTIONS[label]
            self._qwen_alerted = False
            self._start_qwen()
        else:
            self.engine = "whisper"
            if MODEL_OPTIONS[label] != self.current_model:
                self.current_model = MODEL_OPTIONS[label]
                # Re-warmup the new Whisper model in background
                self._warmup_model(self.current_model)
        logger.info("Model selected: %s (engine=%s)", label, self.engine)
        self._refresh_model_menu()
        self._refresh_settings()
        self._set_status(f"Model: {label.split(' (')[0]}")
        self._save_config()

    # ── Qwen3-ASR engine ──────────────────────────────────────

    def _start_qwen(self):
        """Start (or switch) the Qwen3 sidecar in the background."""
        if self._qwen is None:
            self._qwen = qwen_engine.QwenEngine(strip_fillers=self.qwen_strip_fillers,
                                                tight=self.qwen_tight_spacing)
        repo = self.qwen_model
        self._set_status("Loading Qwen3...")

        def done(ok, err):
            if self.engine != "qwen" or repo != self.qwen_model:
                return  # user switched again meanwhile
            if ok:
                logger.info("Qwen3 ready: %s", repo)
                self._set_status(f"Ready — Qwen3 {'1.7B' if '1.7B' in repo else '0.6B'}")
            else:
                self._qwen_failed(f"Qwen3 could not start ({err})")
            self._on_main(self._refresh_model_menu)

        self._qwen.start_async(repo, on_done=done)

    def _use_qwen(self) -> bool:
        """Qwen3 serves this final pass: selected, started once, not translating.

        Deliberately NOT gated on "ready right now": after a sidecar crash the
        request must reach Qwen3 so it restarts it — or fails loudly (alert +
        Whisper fallback) — instead of silently switching to Whisper. While
        Qwen3 is still loading (never started), Whisper is used quietly.
        """
        qwen = getattr(self, "_qwen", None)
        return (getattr(self, "engine", "whisper") == "qwen" and qwen is not None
                and getattr(qwen, "started_ok", False) and not self.translate_mode)

    def _qwen_live_ok(self) -> bool:
        """Live subtitles need a sidecar that is ready NOW (no restart delays)."""
        return self._use_qwen() and self._qwen.is_ready

    def _qwen_recovered(self):
        """Qwen3 served a request: re-arm the alert and clear the menu mark."""
        if self._qwen_alerted:
            logger.warning("Qwen3 recovered — using Qwen3 again")
            self._qwen_alerted = False
            self._on_main(self._refresh_model_menu)

    def _qwen_failed(self, reason: str, exc: Exception | None = None):
        """Qwen3 unusable: alert once per incident; Whisper takes over."""
        logger.error("Qwen3 unavailable, using Whisper: %s", reason)
        self._on_main(self._refresh_model_menu)
        if self._qwen_alerted:
            return
        self._qwen_alerted = True
        self._reporter.report(
            f"Qwen3 speech engine — switched to Whisper automatically, nothing was lost. "
            f"Reason: {reason}. Pick a Qwen3 model again in Push to Talk → Model to retry",
            exc or RuntimeError(reason),
        )

    def _toggle_translate(self, _):
        self.translate_mode = not self.translate_mode
        self.item_translate.title = "✅ Translate to English" if self.translate_mode else "   Translate to English"
        self._idle_icon = "🌐" if self.translate_mode else "🎙"
        if not self.recording:
            self._set_title(self._idle_icon)
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

    def _toggle_meeting_subtitles(self, _):
        """Show/hide the Meeting Mode overlay. Recording is unaffected."""
        self.meeting_subtitles = not self.meeting_subtitles
        if self.meeting_subtitles:
            self.item_meeting_subtitles.title = "✅ Meeting Subtitles: On"
        else:
            self.item_meeting_subtitles.title = "   Meeting Subtitles: Off"
        self._save_config()
        logger.info("Meeting subtitles %s (meeting active=%s)",
                    "enabled" if self.meeting_subtitles else "disabled",
                    self._meeting_session is not None)

        if self._meeting_session is None:
            return
        if self.meeting_subtitles and not self._overlay_panel:
            self._last_overlay_key = None  # force re-render on next update
            AppHelper.callAfter(lambda: self._create_overlay(mode="meeting"))
        elif not self.meeting_subtitles and self._overlay_panel:
            AppHelper.callAfter(self._destroy_overlay)

    def _toggle_save_meeting_audio(self, _):
        """Save raw meeting audio (WAV) next to the transcript. Default off."""
        self.save_meeting_audio = not self.save_meeting_audio
        if self.save_meeting_audio:
            self.item_save_meeting_audio.title = "✅ Save Meeting Audio: On"
        else:
            self.item_save_meeting_audio.title = "   Save Meeting Audio: Off"
        # Audio is written at stop, so a mid-meeting toggle applies to this meeting
        if self._meeting_session is not None:
            self._meeting_session.save_audio = self.save_meeting_audio
        self._save_config()
        logger.info("Save meeting audio %s", "enabled" if self.save_meeting_audio else "disabled")

    def _toggle_save_audio(self, _):
        self.save_audio = not self.save_audio
        if self.save_audio:
            self.item_save_audio.title = "✅ Save Audio: On"
        else:
            self.item_save_audio.title = "   Save Audio: Off"
        self._set_status("Save Audio ON" if self.save_audio else "Save Audio OFF")
        self._save_config()
        print(f"[INFO] Save audio {'enabled' if self.save_audio else 'disabled'}")

    def _quit_cleanup(self, marker) -> None:
        """Everything that must happen before the process ends (menu Quit,
        SIGTERM from logout/shutdown or ./run.sh). Each step is independent:
        one failing must not skip the others.

        An active meeting is saved (live transcript) with a short wait; the
        Qwen3 full pass is skipped — there is no time for it at quit.
        """
        session = getattr(self, "_meeting_session", None)
        if session is not None:
            try:
                session.stop(wait_s=2.0)
                logger.warning("Quit during a meeting — live transcript saved: %s",
                               getattr(session, "transcript_path", None))
            except Exception:
                logger.exception("Saving the meeting at quit failed")
            try:
                session.release_audio()
            except Exception:
                logger.exception("Releasing meeting audio at quit failed")
            self._meeting_session = None
        try:
            marker.clear()
        except Exception:
            logger.exception("Clearing the run marker at quit failed")
        qwen = getattr(self, "_qwen", None)
        if qwen is not None:
            try:
                qwen.close()   # never leave the sidecar holding GPU memory
            except Exception:
                logger.exception("Closing the Qwen3 sidecar at quit failed")

    def _stop_meeting_from_overlay(self):
        """Stop button in the meeting overlay (called after the user confirms).

        Never toggles: if the meeting already ended while the confirm dialog
        was open (e.g. stopped by shortcut), this must not start a new one.
        """
        if self._meeting_session and self._meeting_session.is_recording:
            logger.info("Meeting stopped from the overlay Stop button")
            self._overlay_stop_at = time.monotonic()
            self._stop_meeting()
        else:
            logger.info("Overlay Stop: meeting already ended, nothing to do")

    def _toggle_meeting_mode(self, _):
        """Start or stop Meeting Mode."""
        now = time.monotonic()
        last = max(getattr(self, "_meeting_toggled_at", -1e9),
                   getattr(self, "_overlay_stop_at", -1e9))
        if now - last < MEETING_TOGGLE_COOLDOWN_S:
            # Double press, key repeat, or a shortcut queued while the Stop
            # confirm dialog was open: never stop/start again right away
            logger.warning("Ignoring meeting toggle %.2fs after the last start/stop", now - last)
            return
        self._meeting_toggled_at = now
        if self._meeting_session and self._meeting_session.is_recording:
            self._stop_meeting()
        else:
            self._start_meeting()

    def _start_meeting(self):
        """Start a new meeting recording session."""
        if self._meeting_session and self._meeting_session.is_recording:
            logger.warning("Meeting already running — not starting a second one")
            return
        if self.recording:
            print("[WARN] Cannot start meeting while hold-to-record is active")
            return

        # Wait for backend
        if not self._backend_ready.wait(timeout=15):
            print("[WARN] Backend warmup timed out for meeting")
            self._backend = MLXWhisperBackend(model_repo=self.current_model)

        self._meeting_session = MeetingSession(
            backend=self._backend,
            vad=self._vad if self._vad_available else None,
            min_chunk_size=MIN_CHUNK_SIZE,
            max_buffer_s=MAX_BUFFER_S,
            language=None,
            on_update=self._meeting_overlay_update,
            capture_system_audio=True,
            save_audio=self.save_meeting_audio,
            **self._meeting_live_engine(),
        )

        # Show overlay (unless the user turned Meeting Subtitles off)
        if self.meeting_subtitles:
            AppHelper.callAfter(lambda: self._create_overlay(mode="meeting"))

        self._meeting_session.start()
        self._set_title("🔴")
        self.item_meeting.title = "✅ Meeting Mode: Recording"
        self._set_status("Meeting recording...")
        print("[INFO] Meeting Mode started")

    def _meeting_live_engine(self) -> dict:
        """MeetingSession kwargs for live subtitles (v0.7.3).

        Qwen3 when it is the selected engine and its sidecar is ready now:
        one segment at a time, closed after 0.8 s of silence (Whisper keeps
        2 s). If Qwen3 fails mid-meeting, Whisper takes over the rest of the
        meeting and the usual one-time alert is shown. Otherwise: Whisper,
        unchanged.
        """
        if not self._qwen_live_ok():
            logger.info("Meeting live subtitles: Whisper")
            return {}
        import qwen_meeting
        qwen = self._qwen
        strip = getattr(self, "qwen_strip_fillers", True)
        tight = getattr(self, "qwen_tight_spacing", True)
        logger.info("Meeting live subtitles: Qwen3 (segments, Whisper fallback)")
        return {
            "processor_factory": lambda: qwen_meeting.QwenSegmentProcessor(
                qwen.backend, strip_fillers=strip, tight=tight),
            "fallback_factory": None,     # MeetingSession builds its Whisper processor
            "on_engine_failed": lambda exc: self._qwen_failed(
                f"live meeting subtitles: {exc}", exc),
            "extended_silence_ms": 800,
        }

    def _stop_meeting(self):
        """Stop the current meeting session."""
        if not self._meeting_session:
            return

        # 1. Destroy overlay IMMEDIATELY so user sees instant response
        #    (same pattern as _stop_and_transcribe)
        if self._overlay_panel:
            AppHelper.callAfter(self._destroy_overlay)

        # 2. Stop session with exception protection
        try:
            transcript = self._meeting_session.stop()
        except Exception as e:
            logger.error("Meeting stop failed: %s", e)
            transcript = ""

        self._set_title(self._idle_icon)
        self.item_meeting.title = "   Meeting Mode: Off"

        seg_count = len(self._meeting_session.segments) if self._meeting_session else 0
        self._set_status(f"Meeting saved ({seg_count} segments)")
        print(f"[INFO] Meeting Mode stopped — {seg_count} segments")

        # Meetings are long (20-30 min): copy the transcript FILE PATH, not the
        # full text. Fall back to the text only if the file could not be saved.
        path = getattr(self._meeting_session, "transcript_path", None)
        if path:
            pyperclip.copy(path)
            # WARNING level on purpose: the default log_level is WARNING and the
            # user relies on this line to open the transcript from the log.
            import pathlib
            logger.warning("MEETING SAVED — transcript: %s | open: %s",
                           path, pathlib.Path(path).as_uri())
            self._notify("MacWhisper — Meeting saved",
                         "Transcript file path copied to clipboard", path)
        elif transcript:
            pyperclip.copy(transcript)
            self._reporter.report(
                "saving the meeting transcript file — the full text was copied "
                "to the clipboard instead, paste it somewhere now",
                OSError("meeting transcript file was not written"),
            )

        session = self._meeting_session
        self._meeting_session = None
        if path and getattr(self, "meeting_final_pass", True) and self._use_qwen():
            threading.Thread(target=self._meeting_final_pass, args=(session,),
                             daemon=True, name="meeting-final-pass").start()
        elif session is not None:
            session.release_audio()

    def _meeting_final_pass(self, session):
        """Re-transcribe the whole meeting with Qwen3 from in-memory audio.

        The live transcript is already saved (and its path copied) before
        this starts, so a failure here loses nothing. The .md is rewritten
        in place: final transcript first, live transcript second. Audio is
        never written to disk by this pass.
        """
        try:
            audio = session.mixed_audio()
            duration = len(audio) / SAMPLE_RATE
            if duration < 1.0:
                return
            self._set_status("Meeting: final transcript (Qwen3)...")
            logger.warning("Meeting final pass started: %.0fs audio (Qwen3)", duration)
            t0 = time.time()
            text = self._qwen.transcribe_final(audio)
            label = next((k for k, v in QWEN_MODEL_OPTIONS.items()
                          if v == self._qwen.model_repo), "Qwen3")
            path = session.write_final_transcript(text, f"{label.split(' (')[0]}, full pass")
            logger.warning("MEETING FINAL TRANSCRIPT — %s (%.0fs audio in %.1fs)",
                           path, duration, time.time() - t0)
            self._set_status("Meeting saved (final transcript ready)")
            self._notify("MacWhisper — Final meeting transcript ready",
                         "Same file, final version on top", path or "")
        except QwenUnavailable as e:
            self._qwen_failed(str(e), e)
            self._set_status("Meeting saved (live transcript only)")
        except Exception as e:
            self._reporter.report("meeting final transcript (the live transcript is saved)", e)
        finally:
            session.release_audio()

    def _meeting_overlay_update(self, confirmed, unconfirmed, segments, debug=None):
        """Callback from MeetingSession to update overlay."""
        # Prepend committed segment text for context
        seg_text = "\n".join(s.text for s in segments[-2:]) if segments else ""
        if seg_text:
            full_confirmed = seg_text + "\n" + confirmed
        else:
            full_confirmed = confirmed

        overlay_key = (full_confirmed, unconfirmed)
        if overlay_key != self._last_overlay_key and self._overlay_panel:
            self._last_overlay_key = overlay_key
            AppHelper.callAfter(
                lambda c=full_confirmed, u=unconfirmed: update_overlay(
                    self._overlay_panel, self._overlay_text, c, u, mode="meeting"
                )
            )

        # Log to subtitles.jsonl (same format as Live Subtitles)
        self._log_subtitle_live(confirmed, unconfirmed, debug or {}, mode="meeting")

    def _cycle_model(self):
        current_idx = ALL_MODEL_KEYS.index(self._active_model_label())
        next_idx = (current_idx + 1) % len(ALL_MODEL_KEYS)
        self._set_model(ALL_MODEL_KEYS[next_idx])

    # ── Permissions ────────────────────────────────────────────

    def _notify(self, title, subtitle, message):
        try:
            rumps.notification(title, subtitle, message)
        except RuntimeError:
            pass

    def _check_permissions(self):
        status = _permission_status()
        logger.warning("Permissions (pid %d, %s): accessibility=%s input_monitoring=%s",
                       os.getpid(), sys.executable, status["accessibility"],
                       status["input_monitoring"])
        if not status["accessibility"]:
            print("[WARN] Accessibility permission NOT granted — auto-paste will fail")
            print("       → System Settings → Privacy & Security → Accessibility")
            self._notify(
                "MacWhisper — Permission Needed",
                "Accessibility access is required",
                "System Settings → Privacy & Security → Accessibility",
            )
        if status["input_monitoring"] is False:
            logger.warning("Input Monitoring NOT granted — hotkeys (Right ⌥) will not work. "
                           "System Settings → Privacy & Security → Input Monitoring, then restart")
            self._notify(
                "MacWhisper — Permission Needed",
                "Input Monitoring is required for the hotkeys",
                "System Settings → Privacy & Security → Input Monitoring, then restart MacWhisper",
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
        """Run the pynput listener forever; restart it if it ever dies."""
        failures = 0
        while not self._hotkey_stop.is_set():
            try:
                logger.info("Hotkey listener starting (restart #%d)", failures)
                listener_cls = keyboard.Listener
                if hasattr(listener_cls, "_handle_message"):   # real pynput (darwin)
                    listener_cls = _flag_listener_cls(listener_cls)
                    kwargs = {"sink": self}
                else:
                    kwargs = {}
                with listener_cls(
                    **kwargs,
                    on_press=self._guarded(self._on_press, "hotkey (key press)"),
                    on_release=self._guarded(self._on_release, "hotkey (key release)"),
                ) as listener:
                    listener.join()
                if self._hotkey_stop.is_set():
                    return
                raise RuntimeError("keyboard listener stopped unexpectedly")
            except Exception as e:
                failures += 1
                self._reporter.report(
                    "hotkey listener — restarting it; the Option key may not "
                    "have worked until now", e,
                )
                self._hotkey_stop.wait(min(HOTKEY_RESTART_DELAY_S * failures, 30))

    # Combo action -> handler (menu actions touch AppKit: run on main thread)
    def _combo_handlers(self):
        return {
            "meeting": lambda: self._toggle_meeting_mode(None),
            "cycle_model": lambda: self._cycle_model(),
            "translate": lambda: self._toggle_translate(None),
            "live_subtitles": lambda: self._toggle_live_mode(None),
        }

    def _take_event_mods(self) -> set | None:
        """Modifiers carried by the event being handled (consumed once).

        None when unavailable (e.g. pynput internals changed): then the
        tracked state is used as before, without resync.
        """
        flags = getattr(self, "_event_flags", None)
        self._event_flags = None
        return None if flags is None else shortcuts.mods_from_flags(flags)

    def _ptt_spec(self) -> str:
        return (self.shortcuts.get("push_to_talk") or shortcuts.DEFAULTS["push_to_talk"])[0]

    def _on_press(self, key):
        self._key_event_received = True
        mod = shortcuts.modifier_of(key)
        if getattr(self, "_shortcuts_paused", False):
            # Settings is recording a shortcut: track modifiers, trigger nothing
            if mod:
                self._held_mods.add(mod)
            return
        ptt = self._ptt_spec()
        is_ptt_key = shortcuts.key_token(key) == ptt

        if is_ptt_key:
            real = self._take_event_mods()
            if real is not None and real != self._held_mods - {mod}:
                # Includes this key's own modifier (just pressed); drop stale ones
                if self._held_mods - real:
                    logger.warning("Resynced stuck modifiers %s (lost key release)",
                                   sorted(self._held_mods - real))
                self._held_mods = set(real)
            started = False
            if not self.recording and shortcuts.ptt_allowed(self._held_mods, ptt):
                # Block hold-to-record during active meeting
                if self._meeting_session and self._meeting_session.is_recording:
                    pass
                else:
                    self._ptt_cancelled = False
                    started = True
                    threading.Thread(target=self._start_recording, daemon=True).start()
            if mod:
                self._held_mods.add(mod)
                return
            if started or self.recording:
                return
            # Not dictation (e.g. ⌃⇧F8 with PTT = F8): match it as a combo below

        if mod:
            self._held_mods.add(mod)
            return

        if (self.recording and not self._ptt_cancelled and ptt in ("cmd_r", "ctrl_r", "shift_r")
                and shortcuts.modifier_name(ptt) in self._held_mods):
            # e.g. PTT = Right ⌘ and the user typed ⌘C: not dictation
            logger.info("Key pressed while holding %s — cancelling recording", ptt)
            self._ptt_cancelled = True
            threading.Thread(target=self._cancel_recording, daemon=True).start()

        real = self._take_event_mods()
        if real is not None and real != self._held_mods:
            # A lost release must not turn plain typing (⇧S) into a combo (⌃⇧S)
            logger.warning("Resynced modifiers %s -> %s before shortcut match",
                           sorted(self._held_mods), sorted(real))
            self._held_mods = set(real)
        spec = shortcuts.event_spec(key, self._held_mods)
        action = shortcuts.match_action(spec, self.shortcuts)
        if not action:
            return
        now = time.monotonic()
        down = getattr(self, "_combo_down", None)
        if down is None:
            down = self._combo_down = {}
        last = down.get(spec)
        down[spec] = now
        if last is not None and now - last < COMBO_REPEAT_WINDOW_S:
            logger.debug("shortcut %s auto-repeat ignored", spec)
            return
        if self.recording and shortcuts.combo_uses_ptt_key(spec, ptt):
            # e.g. ⌃⌥M pressed with Right ⌥ first: that "recording" was not speech
            logger.info("Shortcut %s uses the Push to Talk key — cancelling recording", spec)
            self._ptt_cancelled = True
            threading.Thread(target=self._cancel_recording, daemon=True).start()
        logger.debug("shortcut %s -> %s (main thread)", spec, action)
        handler = self._combo_handlers()[action]

        def run():
            handler()
            self._refresh_settings()   # keep an open Settings window in sync
        self._on_main(self._guarded(run, f"shortcut {shortcuts.display(spec)}"))

    def _on_release(self, key):
        mod = shortcuts.modifier_of(key)
        tok = shortcuts.key_token(key)
        down = getattr(self, "_combo_down", None)
        if down and tok and not mod:
            for spec in [s for s in down if s.rsplit("+", 1)[-1] == tok]:
                del down[spec]   # key released: the next press is a new press
        if getattr(self, "_shortcuts_paused", False):
            if mod:
                self._held_mods.discard(mod)
            return
        if shortcuts.key_token(key) == self._ptt_spec():
            if self._ptt_cancelled:
                self._ptt_cancelled = False
            elif self.recording:
                threading.Thread(target=self._stop_and_transcribe, daemon=True).start()
        if mod:
            self._held_mods.discard(mod)

    def _cancel_recording(self):
        """Stop a Push to Talk recording and discard it (no transcription, no paste)."""
        if not self.recording:
            return
        self.recording = False
        self._close_mic_stream()
        if self._overlay_panel:
            AppHelper.callAfter(self._destroy_overlay)
        self.frames = []   # new list: the Qwen live loop exits on identity change
        self._set_title(self._idle_icon)
        self._set_status("Ready")
        logger.info("Recording cancelled (shortcut used the Push to Talk key)")

    # ── Hold-to-Record ────────────────────────────────────────

    def _start_recording(self):
        self._rec_gen += 1
        gen = self._rec_gen
        self.frames    = []
        self.recording = True
        self._last_overlay_key = None
        self._set_title("🟠")
        self._set_status("Recording...")
        print("[INFO] Recording started")

        # ── 1. Show overlay IMMEDIATELY (before any blocking work) ──
        if self.live_mode:
            AppHelper.callAfter(self._create_overlay)

        # ── 2. Wait for pre-warmed backend (usually instant) ──
        # Backend was warmed up at app start or model switch.
        # If warmup is still running (rare — app just launched), wait briefly.
        if not self._backend_ready.wait(timeout=15):
            print("[WARN] Backend warmup timed out, creating fresh backend")
            self._backend = MLXWhisperBackend(model_repo=self.current_model)

        # ── 3. Create processor and start live loop ──
        self._processor = OnlineASRProcessor(
            backend=self._backend,
            vad=None,
            min_chunk_size=MIN_CHUNK_SIZE,
            max_buffer_s=MAX_BUFFER_S,
            language=None,
        )

        self._qwen_live_text = ""
        if self.live_mode:
            self._live_loop_done.clear()
            loop = self._qwen_live_loop if self._qwen_live_ok() else self._live_loop
            threading.Thread(target=loop, daemon=True).start()

        # ── 4. Start audio capture LAST ──
        # IMPORTANT: stream.start() must come AFTER processor creation and
        # live_loop launch. Previously it was step 2, before the blocking
        # backend_ready.wait(). This caused audio to accumulate in self.frames
        # during the wait, flooding the processor with 8s of audio on the
        # first iteration — pre-inference trim would discard the beginning,
        # and LocalAgreement couldn't confirm anything for 10+ seconds.
        self._open_mic_stream(gen)

    def _open_mic_stream(self, gen):
        """Open the mic stream unless the recording already ended.

        Hold-to-record start/stop run on separate threads. If Option is
        released before the stream exists, the old code left the stream
        running forever (leaked PortAudio callbacks). Under the lock we either
        hand the stream to self.stream or close it immediately.
        """
        def callback(indata, frame_count, time_info, status):
            # Runs on CoreAudio's real-time thread — must never raise.
            try:
                if self.recording:
                    self.frames.append(indata.copy())
            except Exception:
                pass

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=1024, callback=callback
        )
        with self._stream_lock:
            if not self.recording or gen != self._rec_gen:
                logger.warning("Mic stream opened after recording ended (gen %s/%s) — closing",
                               gen, self._rec_gen)
                stream.close()
                return
            self.stream = stream
            stream.start()
            logger.debug("Mic stream started (gen %s)", gen)

    def _close_mic_stream(self):
        """Stop and close the mic stream (idempotent, thread-safe)."""
        with self._stream_lock:
            stream, self.stream = self.stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception as e:
            self._reporter.report("closing the microphone", e)

    def _stop_and_transcribe(self):
        self.recording = False
        print(f"[INFO] Recording stopped, frames: {len(self.frames)}")

        # ── 1. Stop audio stream IMMEDIATELY (no more data for live loop) ──
        self._close_mic_stream()

        # ── 2. Destroy overlay IMMEDIATELY so user sees instant response ──
        if self._overlay_panel:
            AppHelper.callAfter(self._destroy_overlay)

        # ── 3. Wait briefly for live loop to finish current inference ──
        # If it's stuck in a long hallucination inference, don't block forever.
        # The live loop checks self.recording and will exit after its current
        # inference completes; the final transcription runs in a queue anyway.
        if not self._live_loop_done.wait(timeout=3):
            logger.warning("Live loop still running (mid-inference) — proceeding")

        # Force-confirm remaining words from processor and capture
        # the complete real-time text (concrete evidence of what the
        # subtitle window displayed during this session).
        realtime_text = getattr(self, "_qwen_live_text", "")
        try:
            if self._processor and not realtime_text:
                self._processor.segment_close()
                realtime_text = "".join(
                    w[2] for w in self._processor.committed_history
                )
        except Exception as e:
            logger.error("Processor cleanup failed: %s", e)

        # Log session_end with complete real-time text to subtitles.jsonl
        if realtime_text:
            self._log_subtitle_session_end(realtime_text)

        if not self.frames:
            self._set_title(self._idle_icon)
            self._set_status("Ready")
            return

        self._set_title("💬")
        self._set_status("Transcribing...")
        self.transcribe_queue.put((list(self.frames), realtime_text))
        self.frames = []  # release memory immediately

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
            self._reporter.report("live subtitles", e, while_recording=self.recording)
        finally:
            self._live_loop_done.set()

    def _qwen_live_loop(self):
        """Live subtitles with Qwen3: re-decode the whole utterance every 1 s.

        A failure here is logged only — the final pass (after release)
        falls back to Whisper and raises the alert if Qwen3 is really down.
        """
        frames_ref = self.frames  # list object for THIS recording

        def update(confirmed, tail):
            key = (confirmed, tail)
            if key == self._last_overlay_key or not self._overlay_panel:
                return
            self._last_overlay_key = key
            AppHelper.callAfter(
                lambda c=confirmed, u=tail: update_overlay(
                    self._overlay_panel, self._overlay_text, c, u
                )
            )
            if confirmed or tail:
                self._log_subtitle_live(confirmed, tail, {"engine": "qwen3"})

        try:
            self._qwen_live_text = qwen_engine.run_live_loop(
                self._qwen.backend,
                get_frames=lambda: list(frames_ref),
                is_recording=lambda: self.recording and self.frames is frames_ref,
                update=update,
                on_error=lambda e: logger.warning("Qwen3 live subtitles stopped: %s", e),
                use_timer=True,
            )
        except Exception as e:
            self._reporter.report("live subtitles (Qwen3)", e, while_recording=self.recording)
        finally:
            self._live_loop_done.set()

    def _live_loop_impl(self, proc):
        """Inner live loop implementation (separated for clean try/finally)."""
        # Reuse pre-created VAD instance (reset state for new recording)
        use_vad = self._vad_available
        if use_vad:
            self._vad.reset()

        last_frame_idx = 0
        last_segment_close_time = 0.0

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
                self._vad.process_chunk(audio_chunk)
                seg_duration = len(proc.audio_buffer) / SAMPLE_RATE
                time_since_close = time.time() - last_segment_close_time
                if self._vad.is_speech_end() and seg_duration > 5.0 and time_since_close > 3.0:
                    segment_text = proc.segment_close()
                    last_segment_close_time = time.time()
                    if segment_text:
                        logger.info("VAD segment committed: %d chars", len(segment_text))
                        self._log_subtitle(segment_text, seg_duration)
                    # Clear echo state — old segment text is committed, new words
                    # should not be echo-checked against it (same as Meeting Mode's
                    # _close_and_trim_segment).
                    proc._last_committed_raw = ""
                    self._vad.reset()

            # Show current state BEFORE blocking inference so overlay
            # stays responsive while Whisper is processing
            if self._overlay_panel and (proc.committed_history or proc.last_unconfirmed):
                pre_conf = "".join(w[2] for w in proc.committed_history)
                pre_unconf = "".join(w[2] for w in proc.last_unconfirmed)
                pre_key = (pre_conf, pre_unconf)
                if pre_key != self._last_overlay_key:
                    self._last_overlay_key = pre_key
                    AppHelper.callAfter(
                        lambda c=pre_conf, u=pre_unconf: update_overlay(
                            self._overlay_panel, self._overlay_text, c, u
                        )
                    )

            # Run one processing iteration (LocalAgreement)
            result = proc.process_iter()
            if result is None:
                continue

            confirmed, unconfirmed = result

            # Log debug info + subtitle text to console
            if proc.last_debug:
                debug = proc.last_debug
                conf_tail = confirmed[-40:] if confirmed else "(empty)"
                logger.debug(
                    "LIVE iter=%s buf=%ss inf=%sms conf=%s new=%s | %s",
                    debug.get('iter', '?'), debug.get('buffer_s', '?'),
                    debug.get('inference_ms', '?'),
                    debug.get('confirmed_words', 0),
                    debug.get('newly_confirmed', 0), conf_tail,
                )

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

    def _log_subtitle_live(self, confirmed, unconfirmed, debug, mode="live"):
        """Log a live subtitle iteration to subtitle log."""
        _ensure_history_dirs()
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": f"{mode}_iter",
            "confirmed": confirmed or "",
            "unconfirmed": unconfirmed or "",
            "debug": debug,
        }
        with open(SUBTITLE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _log_subtitle_session_end(self, realtime_text):
        """Log session_end with complete real-time text to subtitles.jsonl.

        This is the 'concrete evidence' — the full text that was displayed
        in the subtitle window across the entire Push-to-Talk session.
        """
        _ensure_history_dirs()
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "session_end",
            "realtime_text": realtime_text,
            "char_count": len(realtime_text),
        }
        with open(SUBTITLE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Session end logged: %d chars of real-time text", len(realtime_text))

    # ── Transcription (hold-to-record final pass) ──────────────

    def _transcription_worker(self):
        while True:
            item = self.transcribe_queue.get()
            # Unpack: new format is (frames, realtime_text), legacy is just frames
            if isinstance(item, tuple) and len(item) == 2:
                frames, realtime_text = item
            else:
                frames, realtime_text = item, ""
            try:
                self._do_transcribe(frames, realtime_text=realtime_text)
            except Exception as e:
                self._reporter.report("final transcription — text was not pasted", e)
            finally:
                self.transcribe_queue.task_done()
                # Release processor memory (audio_buffer, committed words, etc.)
                if self._processor:
                    self._processor.reset()
                    self._processor = None
                self._set_title(self._idle_icon)
                self._set_status("Ready")

    def _final_text(self, audio_float, audio_int16):
        """Final text for one utterance -> (text, model_used).

        Qwen3 when selected and ready (not in translate mode); on ANY Qwen3
        failure the SAME audio is transcribed by Whisper, so nothing is lost.
        model_used is the model repo that produced the text. The Whisper
        result (word timestamps for history) is kept in _last_whisper_result.
        """
        duration = len(audio_float) / SAMPLE_RATE
        self._last_whisper_result = None
        if self._use_qwen():
            t0 = time.time()
            try:
                text = self._qwen.transcribe_final(audio_float)
                self._qwen_recovered()
                print(f"[INFO] Final transcription (Qwen3): {time.time() - t0:.1f}s "
                      f"for {duration:.1f}s audio")
                print(f"[INFO] Result: {text}")
                return text, self._qwen.model_repo or self.qwen_model
            except QwenUnavailable as e:
                self._qwen_failed(str(e), e)
        text, result = self._whisper_final(audio_float, duration)
        self._last_whisper_result = result
        return text, self.current_model

    def _whisper_final(self, audio_float, duration):
        """Whisper final pass (unchanged v0.6 behavior) -> (text, result)."""
        # Reuse pre-warmed backend (avoids loading model again — saves ~500MB-3GB)
        backend = self._backend

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
        return text, result

    def _do_transcribe(self, frames, realtime_text=""):
        """Final transcription pass using the ASR backend.

        This runs after recording stops. For short recordings or when
        live mode is off, this is the only transcription. For live mode,
        this provides a final high-quality pass over the full audio.

        Args:
            frames: List of audio frames (int16 numpy arrays).
            realtime_text: Complete text from real-time subtitle display
                (committed_history). Saved alongside offline text for
                comparison / concrete evidence.
        """
        # Ensure live loop has fully exited before starting Metal inference
        # (concurrent Metal inferences cause assertion failures / crashes)
        done_event = getattr(self, "_live_loop_done", None)
        if done_event and not done_event.wait(timeout=15):
            logger.warning("Live loop did not exit in time — proceeding anyway")

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

        text, model_used = self._final_text(audio_float, audio)
        result = self._last_whisper_result

        # Save enhanced history (word timestamps: Whisper only)
        save_enhanced_history(
            history_dir=MEETINGS_DIR,
            audio_file=os.path.basename(wav_path) if wav_path else None,
            words=result.all_words() if result is not None else [],
            text=text,
            model=model_used,
            duration_s=duration,
            language=result.language if result is not None else "zh",
        )

        # Log transcription result with both offline and real-time text
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "audio_file": os.path.basename(wav_path) if wav_path else None,
            "model": model_used,
            "translate": self.translate_mode,
            "duration_s": round(duration, 1),
            "text": text,
        }
        if realtime_text:
            log_entry["realtime_text"] = realtime_text
        with open(TRANSCRIPT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        print(f"[INFO] Logged transcript: {TRANSCRIPT_LOG}")

        if text:
            self._auto_paste(text + " ")

    # ── Live Subtitles — overlay ──────────────────────────────

    def _create_overlay(self, mode: str = "talk"):
        if mode == "talk" and not getattr(self, "recording", True):
            # Queued creation that lost the race with a fast release/cancel
            logger.info("Skipping talk overlay: recording already ended")
            return
        on_stop = self._stop_meeting_from_overlay if mode == "meeting" else None
        panel, text_view = create_overlay(mode=mode, frame_path=OVERLAY_FRAME_PATH,
                                          on_stop=on_stop)
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
            for d in devices
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
    # Read log_level from config (before app init, since basicConfig must be early)
    _cfg_log_level = "WARNING"
    try:
        with open(CONFIG_PATH) as _f:
            _cfg_log_level = json.load(_f).get("log_level", "WARNING")
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    _file_level = getattr(logging, _cfg_log_level.upper(), logging.WARNING)

    # Configure logging — configurable level to file, WARNING to console
    _log_file = os.path.join(LOG_DIR, "macwhisper.log")
    os.makedirs(LOG_DIR, exist_ok=True)
    _handlers = [logging.FileHandler(_log_file, encoding="utf-8")]
    # WARNING+ also to the console (./run.sh). Not when stderr IS the log
    # (double-clicked app: the launcher redirects it there) — every line
    # would be written twice.
    if not _stream_is_file(sys.stderr, _log_file):
        _console = logging.StreamHandler()
        _console.setLevel(logging.WARNING)
        _handlers.append(_console)
    logging.basicConfig(
        level=_file_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=_handlers,
    )

    # One MacWhisper only (double-click + ./run.sh would fight over hotkeys
    # and the mic). Before the run marker, so a refused launch leaves the
    # running app's crash marker alone.
    if not _claim_single_instance():
        sys.exit(0)

    # ── Crash evidence: hard crashes (SIGABRT/SIGSEGV) bypass Python's
    #    exception handling entirely. faulthandler writes them to crash.log,
    #    and the run marker lets the next launch tell the user.
    import atexit
    crash_guard.enable_faulthandler(CRASH_LOG_PATH)
    _marker = crash_guard.RunMarker(RUN_MARKER_PATH)
    if _marker.previous_run_crashed():
        _PREVIOUS_CRASH_TAIL = crash_guard.read_tail(CRASH_LOG_PATH)
        logging.getLogger(__name__).error(
            "Previous MacWhisper run did not exit cleanly; crash.log tail:\n%s",
            _PREVIOUS_CRASH_TAIL,
        )
    _marker.create()
    atexit.register(_marker.clear)

    _check_audio_device()
    try:
        _app = TranscriberApp()

        # Menu Quit and SIGTERM (logout/shutdown, ./run.sh restart) share one
        # cleanup: save an active meeting, clear the run marker, close Qwen3.
        # SIGTERM first asks for a normal quit; if the main thread is stuck
        # (e.g. a modal alert), the guard cleans up and exits after 8 s.
        _guard = quit_guard.install(
            request_quit=lambda: AppHelper.callAfter(rumps.quit_application),
            cleanup=lambda: _app._quit_cleanup(_marker),
            deadline_s=8.0,
        )
        rumps.events.before_quit.register(_guard.run_cleanup)
        atexit.register(_guard.run_cleanup)
        _app.run()
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
