"""Keep MacWhisper's data directory owner-only.

Dictations, meeting transcripts, audio and logs live under ~/.macwhisper.
`set_private_umask` makes everything created from then on 0700/0600, and
`tighten_tree` fixes what older versions already wrote with 0755/0644.
"""

from __future__ import annotations

import logging
import os
import stat

logger = logging.getLogger(__name__)

PRIVATE_UMASK = 0o077
_DIR_MODE = 0o700
_FILE_MODE = 0o600


def set_private_umask() -> int:
    """Set umask 077 for this process; return the previous umask."""
    old = os.umask(PRIVATE_UMASK)
    logger.debug("umask %03o -> %03o", old, PRIVATE_UMASK)
    return old


def _fix(path: str, want: int, counts: dict, key: str) -> None:
    try:
        st = os.lstat(path)
        if stat.S_IMODE(st.st_mode) == want:
            return
        os.chmod(path, want)
        counts[key] += 1
    except OSError as e:
        counts["errors"] += 1
        logger.debug("Could not tighten %s: %s", path, e)


def tighten_tree(root: str) -> dict:
    """chmod `root` and its subdirectories to 0700 and regular files to 0600.

    Symlinks are skipped and never followed, so nothing outside the tree is
    touched. Errors are counted per entry, never raised. Returns counts of
    changed dirs/files and errors.
    """
    counts = {"dirs": 0, "files": 0, "errors": 0}
    if not os.path.isdir(root) or os.path.islink(root):
        logger.debug("No data dir to tighten: %s", root)
        return counts
    _fix(root, _DIR_MODE, counts, "dirs")
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames:
            p = os.path.join(dirpath, name)
            if not os.path.islink(p):
                _fix(p, _DIR_MODE, counts, "dirs")
        for name in filenames:
            p = os.path.join(dirpath, name)
            try:
                if not stat.S_ISREG(os.lstat(p).st_mode):
                    continue  # symlinks, sockets, fifos
            except OSError as e:
                counts["errors"] += 1
                logger.debug("Could not stat %s: %s", p, e)
                continue
            _fix(p, _FILE_MODE, counts, "files")
    if counts["dirs"] or counts["files"] or counts["errors"]:
        logger.warning("Made %s private: %d dir(s) -> 0700, %d file(s) -> 0600, %d error(s)",
                       root, counts["dirs"], counts["files"], counts["errors"])
    else:
        logger.debug("Data dir already private: %s", root)
    return counts
