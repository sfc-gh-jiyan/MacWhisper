"""End-to-end verification of the Qwen3 engine inside a real TranscriberApp.

Loads the REAL Qwen3 sidecar model. Not part of pytest; run by hand:

    venv/bin/python tests/verify_app_qwen.py [--model mlx-community/Qwen3-ASR-0.6B-bf16]

Checks, with a time limit and progress output per stage:
  1. app constructs (menu, config) and Qwen3 becomes ready
  2. Push to Talk final pass on 3 recordings (short/medium/long)
  3. live subtitles loop against the real sidecar (first text latency)
  4. sidecar killed -> final pass keeps the audio (nothing pasted),
     exactly one alert
  5. meeting final pass rewrites the .md (final first, live second)

Side effects are contained: alerts/paste/notifications are stubbed, the
config file is restored afterwards, meeting output goes to a temp dir.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import tempfile
import threading
import time
import wave
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG = os.path.expanduser("~/.macwhisper/config.json")


def stage(name):
    print(f"\n=== {name}", flush=True)
    return time.time()


def done(t0, ok=True, note=""):
    print(f"    {'OK' if ok else 'FAIL'} ({time.time() - t0:.1f}s) {note}", flush=True)
    return ok


def wait_until(pred, timeout, what):
    t0 = time.time()
    while not pred():
        if time.time() - t0 > timeout:
            raise TimeoutError(f"{what} not ready after {timeout}s")
        time.sleep(0.2)


def load_wav(path):
    w = wave.open(path)
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return a


def pick_three():
    rows = []
    for p in glob.glob(os.path.expanduser("~/.macwhisper/audio/*.wav")):
        try:
            w = wave.open(p)
            rows.append((w.getnframes() / w.getframerate(), p))
        except Exception:
            pass
    return [min(rows, key=lambda r: abs(r[0] - t))[1] for t in (5, 20, 60)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-ASR-0.6B-bf16")
    args = ap.parse_args()

    backup = None
    if os.path.isfile(CONFIG):
        backup = tempfile.mktemp(suffix=".json")
        shutil.copy(CONFIG, backup)
    results = {}
    try:
        import app
        import meeting

        t0 = stage("1. construct TranscriberApp + engines ready")
        app.TranscriberApp._report_previous_crash = lambda self, _=None: None
        inst = app.TranscriberApp()
        inst._reporter = MagicMock()
        inst._auto_paste = MagicMock()
        inst._notify = MagicMock()
        if inst.engine != "qwen" or inst.qwen_model != args.model:
            inst.engine = "qwen"
            inst.qwen_model = args.model
            inst._start_qwen()
        wait_until(lambda: inst._qwen is not None and inst._qwen.is_ready, 240, "Qwen3")
        labels = [i.title for i in inst.model_items.values()]
        results["construct"] = done(t0, True, f"engine={inst.engine} menu={labels}")

        t0 = stage("2. Push to Talk final pass: 3 recordings, Qwen3")
        ok = True
        for p in pick_three():
            pcm = load_wav(p)
            f = pcm.astype(np.float32) / 32768.0
            ts = time.time()
            qtext, qmodel = inst._final_text(f, pcm)
            q_s = time.time() - ts
            print(f"  [{len(f) / 16000:4.1f}s] {os.path.basename(p)}")
            print(f"     Qwen3   ({q_s:4.2f}s, {qmodel.split('/')[-1]}): {qtext[:140]}")
            ok &= "Qwen3" in qmodel
        results["final_pass"] = done(t0, ok)

        t0 = stage("3. live subtitles loop against the real sidecar")
        pcm = load_wav(pick_three()[1])
        frames = []
        shown = []
        start = time.time()

        def feeder():
            for i in range(0, len(pcm), 1600):     # 100 ms chunks in real time
                frames.append(pcm[i:i + 1600].reshape(-1, 1))
                time.sleep(0.1)

        th = threading.Thread(target=feeder)
        th.start()
        from qwen_engine import run_live_loop
        last = run_live_loop(
            inst._qwen.backend, lambda: list(frames), th.is_alive,
            lambda c, u: shown.append((time.time() - start, c, u)), use_timer=True)
        th.join()
        first = next((t for t, c, u in shown if c or u), None)
        print(f"  updates={len(shown)} first_text_at={first and round(first, 2)}s")
        print(f"  last shown: {last[:140]}")
        results["live"] = done(t0, bool(shown) and first is not None and first < 4.0)

        t0 = stage("4. kill sidecar -> audio kept, nothing pasted, one alert")
        inst._qwen.backend.max_restarts = 0          # make the failure stick
        inst._qwen.backend._proc.kill()
        time.sleep(0.5)
        pcm = load_wav(pick_three()[0])
        f = pcm.astype(np.float32) / 32768.0
        t1, m1 = inst._final_text(f, pcm)
        t2, m2 = inst._final_text(f, pcm)
        alerts = inst._reporter.report.call_count
        menu = [i.title for i in inst.model_items.values() if "✅" in i.title]
        print(f"  1st: model={m1} text={t1[:80]}")
        print(f"  2nd: model={m2}  alerts={alerts}  menu={menu}")
        results["fallback"] = done(t0, t1 == "" and t2 == "" and alerts == 1)

        t0 = stage("4b. sidecar crash with restart budget -> Qwen3 recovers by itself")
        assert inst._qwen.start(args.model), inst._qwen.last_error   # fresh, default budget
        inst._reporter.report.reset_mock()
        inst._qwen_alerted = False
        inst._qwen.backend._proc.kill()
        time.sleep(0.5)
        t3, m3 = inst._final_text(f, pcm)
        print(f"  after crash: model={m3} restarts={inst._qwen.backend.restarts} "
              f"alerts={inst._reporter.report.call_count} text={t3[:60]}")
        results["recover"] = done(t0, "Qwen3" in m3 and inst._qwen.backend.restarts == 1
                                  and inst._reporter.report.call_count == 0)

        t0 = stage("5. meeting final pass (temp meetings dir)")
        tmp = tempfile.mkdtemp()
        orig = meeting.MEETINGS_DIR
        meeting.MEETINGS_DIR = tmp
        try:
            s = meeting.MeetingSession(backend=MagicMock(), audio_source=MagicMock())
            s._session_id = "verify"
            s._segments = [meeting.MeetingSegment(text="(live text)", start_time=0, end_time=1)]
            mic = load_wav(pick_three()[2])
            s._mic_frames = [mic.reshape(-1, 1)]
            s._sys_frames = []
            path = s._auto_save()
            inst._meeting_final_pass(s)
            md = open(path, encoding="utf-8").read()
            fi, li = md.find("## Final transcript"), md.find("## Live transcript")
            print(f"  {path}\n  final@{fi} live@{li} audio_freed={s._mic_frames == []}"
                  f" wav_files={glob.glob(os.path.join(tmp, '*.wav'))}")
            print("  " + md[fi:fi + 200].replace("\n", " | "))
            results["meeting"] = done(t0, 0 <= fi < li and s._mic_frames == [])
        finally:
            meeting.MEETINGS_DIR = orig
        inst._qwen.close()
    finally:
        if backup:
            shutil.move(backup, CONFIG)
            print(f"\n(config restored: {CONFIG})")

    print("\nSUMMARY:", " ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in results.items()))
    return 0 if results and all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
