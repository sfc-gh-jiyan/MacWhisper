"""Model revision pinning: known repos load a fixed commit, others unpinned + warning."""

import logging
import sys
import types

import model_revisions as mr
import llm_sidecar
import qwen_sidecar


def test_known_repos_are_pinned():
    assert mr.revision_for("mlx-community/Qwen3-ASR-0.6B-bf16") == "eae2b51f96265328f1e7beced788adb0e4536f92"
    assert mr.revision_for("mlx-community/Qwen3-ASR-1.7B-bf16") == "e1f6c266914abc5a46e8756e02580f834a6cf8a7"
    assert mr.revision_for("mlx-community/Qwen3-4B-Instruct-2507-4bit") == "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"


def test_unknown_repo_unpinned_with_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="model_revisions"):
        assert mr.revision_for("someone/custom-model") is None
    assert any("someone/custom-model" in r.getMessage() for r in caplog.records)


def _install_fake(monkeypatch, name, attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)


def test_qwen_sidecar_passes_revision(monkeypatch):
    calls = []
    _install_fake(monkeypatch, "mlx_audio", {})
    _install_fake(monkeypatch, "mlx_audio.stt", {})
    _install_fake(monkeypatch, "mlx_audio.stt.utils",
                  {"load": lambda repo, **kw: calls.append((repo, kw)) or object()})
    monkeypatch.setattr(qwen_sidecar, "make_run", lambda model: (lambda *a, **k: ""))
    qwen_sidecar.load_model_fn("mlx-community/Qwen3-ASR-1.7B-bf16")
    qwen_sidecar.load_model_fn("someone/custom-asr")
    assert calls[0] == ("mlx-community/Qwen3-ASR-1.7B-bf16",
                        {"revision": "e1f6c266914abc5a46e8756e02580f834a6cf8a7"})
    assert calls[1] == ("someone/custom-asr", {"revision": None})


def test_llm_sidecar_passes_revision_and_no_remote_code(monkeypatch):
    calls = []

    class Tok:
        def apply_chat_template(self, *a, **k):
            return ""

    _install_fake(monkeypatch, "mlx_lm", {
        "load": lambda repo, **kw: calls.append((repo, kw)) or (object(), Tok()),
        "generate": lambda *a, **k: "",
    })
    _install_fake(monkeypatch, "mlx_lm.sample_utils", {"make_sampler": lambda **k: None})
    llm_sidecar.load_complete_fn("mlx-community/Qwen3-4B-Instruct-2507-4bit")
    repo, kw = calls[0]
    assert kw["revision"] == "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
    assert kw["trust_remote_code"] is False
