"""Key probe — diagnose the double-clicked app's keyboard access.

Built by scripts/build_app.sh --probe as /Applications/MacWhisper Key Probe.app
(bundle id com.macwhisper.keyprobe). It uses the same native launcher as
MacWhisper.app, so it shows whether an app started that way gets
Accessibility, Input Monitoring and key events. It never records audio or
pastes; it logs only key NAMES (no text) to ~/.macwhisper/logs/key_probe.log.
"""

import datetime
import os
import threading

import rumps
from pynput import keyboard

LOG = os.path.expanduser("~/.macwhisper/logs/key_probe.log")


def log(msg: str) -> None:
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now():%H:%M:%S} {msg}\n")


def permissions() -> str:
    import HIServices
    import Quartz
    ax = bool(HIServices.AXIsProcessTrusted())
    try:
        listen = bool(Quartz.CGPreflightListenEventAccess())
    except Exception as e:  # older PyObjC
        listen = f"unknown ({e})"
    return f"accessibility={ax} input_monitoring={listen}"


class Probe(rumps.App):
    def __init__(self):
        super().__init__("⌨︎", quit_button="Quit Key Probe")
        self.count = 0
        self.status = rumps.MenuItem("Keys seen: 0")
        self.perm = rumps.MenuItem(permissions())
        self.menu = [self.status, self.perm]
        log(f"probe started pid={os.getpid()} {permissions()}")
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        def on_press(key, *_):
            self.count += 1
            name = getattr(key, "name", None) or f"vk{getattr(key, 'vk', '?')}"
            if self.count <= 20 or self.count % 20 == 0:
                log(f"key #{self.count}: {name}")

        while True:
            log("listener starting")
            with keyboard.Listener(on_press=on_press) as listener:
                listener.join()
            log("listener stopped (event tap could not be created?) — retry in 3 s")
            threading.Event().wait(3)

    @rumps.timer(2)
    def _tick(self, _):
        self.status.title = f"Keys seen: {self.count}"
        self.perm.title = permissions()


if __name__ == "__main__":
    Probe().run()
