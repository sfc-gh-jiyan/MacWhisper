"""private_dirs: keep ~/.macwhisper owner-only (0700 dirs, 0600 files)."""

import os
import stat

import private_dirs


def _mode(p):
    return stat.S_IMODE(os.lstat(p).st_mode)


def _tree(tmp_path):
    root = tmp_path / "data"
    (root / "logs").mkdir(parents=True)
    (root / "meetings" / "2026").mkdir(parents=True)
    (root / "config.json").write_text("{}")
    (root / "logs" / "macwhisper.log").write_text("x")
    (root / "meetings" / "2026" / "m.md").write_text("x")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("x")
    os.chmod(outside, 0o755)
    os.chmod(outside / "keep.txt", 0o644)
    os.symlink(outside, root / "link_dir")
    os.symlink(outside / "keep.txt", root / "link_file")
    for d in (root, root / "logs", root / "meetings", root / "meetings" / "2026"):
        os.chmod(d, 0o755)
    for f in (root / "config.json", root / "logs" / "macwhisper.log", root / "meetings" / "2026" / "m.md"):
        os.chmod(f, 0o644)
    return root, outside


def test_tightens_dirs_and_files(tmp_path):
    root, _ = _tree(tmp_path)
    counts = private_dirs.tighten_tree(str(root))
    for d in (root, root / "logs", root / "meetings", root / "meetings" / "2026"):
        assert _mode(d) == 0o700
    for f in (root / "config.json", root / "logs" / "macwhisper.log", root / "meetings" / "2026" / "m.md"):
        assert _mode(f) == 0o600
    assert counts == {"dirs": 4, "files": 3, "errors": 0}


def test_symlinks_and_their_targets_untouched(tmp_path):
    root, outside = _tree(tmp_path)
    private_dirs.tighten_tree(str(root))
    assert _mode(outside) == 0o755
    assert _mode(outside / "keep.txt") == 0o644


def test_second_run_changes_nothing(tmp_path):
    root, _ = _tree(tmp_path)
    private_dirs.tighten_tree(str(root))
    assert private_dirs.tighten_tree(str(root)) == {"dirs": 0, "files": 0, "errors": 0}


def test_missing_root_is_fine(tmp_path):
    assert private_dirs.tighten_tree(str(tmp_path / "nope")) == {"dirs": 0, "files": 0, "errors": 0}


def test_chmod_error_is_counted_not_raised(tmp_path, monkeypatch):
    root, _ = _tree(tmp_path)
    real = os.chmod

    def flaky(p, m, **kw):
        if str(p).endswith("config.json"):
            raise PermissionError("nope")
        return real(p, m, **kw)

    monkeypatch.setattr(private_dirs.os, "chmod", flaky)
    counts = private_dirs.tighten_tree(str(root))
    assert counts["errors"] == 1 and counts["files"] == 2


def test_file_created_after_startup_is_owner_only(tmp_path):
    old = os.umask(0o022)
    try:
        private_dirs.set_private_umask()
        d = tmp_path / "new_dir"
        d.mkdir()
        f = d / "crash.log"
        with open(f, "w") as fh:
            fh.write("x")
        assert _mode(d) == 0o700
        assert _mode(f) == 0o600
    finally:
        os.umask(old)


def test_owner_executable_file_keeps_execute_bit(tmp_path):
    root, _ = _tree(tmp_path)
    script = root / "restart.sh"
    script.write_text("#!/bin/sh\n")
    os.chmod(script, 0o755)
    private_dirs.tighten_tree(str(root))
    assert _mode(script) == 0o700
    assert os.access(script, os.X_OK)
    assert _mode(root / "config.json") == 0o600
