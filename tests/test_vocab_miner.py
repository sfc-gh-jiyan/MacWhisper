"""vocab_miner: LLM-suggested vocabulary, validated against the real transcripts."""

import json

import vocab_miner as vm
from vocabulary import Entry, Vocabulary

MODEL = "mlx-community/Qwen3-ASR-1.7B-bf16"


def _log(tmp_path, texts, model=MODEL, start="2026-10-21T09:"):
    p = tmp_path / "t.jsonl"
    rows = [{"timestamp": f"{start}{i:02d}:00", "model": model, "text": t} for i, t in enumerate(texts)]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return str(p)


def _vocab(tmp_path, entries=()):
    return Vocabulary(path=str(tmp_path / "v.json"), entries=list(entries))


def test_parse_response_tolerates_fences_and_prose():
    text = 'Here you go:\n```json\n[{"heard": "哈什", "correct": "hash", "confidence": "high"}]\n```'
    assert vm.parse_response(text) == [{"heard": "哈什", "correct": "hash", "confidence": "high"}]
    assert vm.parse_response("no json here") == []
    assert vm.parse_response('[{"heard": 1}]') == []


def test_validate_requires_verbatim_evidence():
    corpus = "这个哈什会变吗。我在用Gira开票。"
    cands = [
        {"heard": "哈什", "correct": "hash", "confidence": "high"},
        {"heard": "不存在的词", "correct": "x", "confidence": "high"},     # invented
        {"heard": "Gira", "correct": "Gira", "confidence": "high"},        # identity
        {"heard": "三", "correct": "four", "confidence": "high"},          # too short / risky
        {"heard": "12", "correct": "13", "confidence": "high"},            # numeric
        {"heard": "Gira", "correct": "Jira", "confidence": "low"},         # low dropped
    ]
    out = vm.validate(cands, corpus, _vocab_entries=[])
    assert [(c.heard, c.correct, c.confidence) for c in out] == [("哈什", "hash", "high")]
    assert out[0].count == 1 and "哈什" in out[0].example


def test_validate_skips_known_aliases():
    out = vm.validate([{"heard": "CIC", "correct": "CSE", "confidence": "high"}], "我是CIC",
                      _vocab_entries=[Entry("CSE", aliases=("CIC",))])
    assert out == []


def test_run_applies_high_active_medium_pending_and_advances_state(tmp_path):
    log = _log(tmp_path, ["这个哈什会变吗 version", "我在用Gira开票", "哈什又变了"])
    v = _vocab(tmp_path)
    state = tmp_path / "state.json"
    reply = json.dumps([
        {"heard": "哈什", "correct": "hash", "confidence": "high", "reason": "tech term"},
        {"heard": "Gira", "correct": "Jira", "confidence": "medium", "reason": "tool"},
    ], ensure_ascii=False)
    prompts = []
    llm = lambda prompt, system="", max_tokens=0, timeout=0: prompts.append(prompt) or reply
    res = vm.run(llm, v, log, str(state), model=MODEL, min_new=1)
    assert [(c.heard, c.correct) for c in res.active] == [("哈什", "hash")]
    assert [(c.heard, c.correct) for c in res.pending] == [("Gira", "Jira")]
    assert v.find("hash").status == "active" and v.find("hash").source == "ai"
    assert v.find("Jira").status == "pending"
    assert "这个哈什会变吗" in prompts[0]
    # second run: nothing new -> no LLM call
    res2 = vm.run(llm, v, log, str(state), model=MODEL, min_new=1)
    assert res2.examined == 0 and len(prompts) == 1


def test_run_only_uses_current_model(tmp_path):
    log = _log(tmp_path, ["old whisper text 哈什"], model="mlx-community/whisper-large-v3-mlx")
    called = []
    res = vm.run(lambda *a, **k: called.append(1) or "[]", _vocab(tmp_path), log,
                 str(tmp_path / "s.json"), model=MODEL, min_new=1)
    assert res.examined == 0 and not called


def test_run_needs_minimum_new_dictations(tmp_path):
    log = _log(tmp_path, ["a 哈什", "b"])
    called = []
    res = vm.run(lambda *a, **k: called.append(1) or "[]", _vocab(tmp_path), log,
                 str(tmp_path / "s.json"), model=MODEL, min_new=5)
    assert res.skipped and not called


def test_run_pauses_while_busy_then_finishes_with_progress(tmp_path):
    log = _log(tmp_path, ["x 哈什 " * 400] * 10)          # several chunks
    busy = iter([True, True, False] + [False] * 50)
    progress, called, slept = [], [], []
    res = vm.run(lambda *a, **k: called.append(1) or "[]", _vocab(tmp_path), log,
                 str(tmp_path / "s.json"), model=MODEL, min_new=1,
                 should_pause=lambda: next(busy), on_progress=lambda d, t, p: progress.append((d, t, p)),
                 sleep=slept.append)
    assert not res.stopped and called
    assert slept                                          # waited while busy
    assert progress[0][2] is True                          # reported "paused"
    assert progress[-1][0] == progress[-1][1]              # finished all chunks


def test_run_gives_up_after_long_pause(tmp_path):
    log = _log(tmp_path, ["x 哈什"] * 3)
    res = vm.run(lambda *a, **k: "[]", _vocab(tmp_path), log, str(tmp_path / "s.json"), model=MODEL,
                 min_new=1, should_pause=lambda: True, sleep=lambda s: None, max_pause_s=5)
    assert res.stopped


def test_chunks_split_long_history():
    chunks = vm.chunk_texts(["a" * 3000, "b" * 3000, "c" * 100], limit=4000)
    assert len(chunks) == 2


def test_due_and_mark_run(tmp_path):
    s = str(tmp_path / "s.json")
    assert vm.due(s, interval_s=3600, now=1000.0) is True
    vm.mark_run(s, now=1000.0)
    assert vm.due(s, interval_s=3600, now=2000.0) is False
    assert vm.due(s, interval_s=3600, now=5000.0) is True


import pytest


@pytest.mark.parametrize("heard,correct", [
    ("fold的这个东西", "fold's thing"),                 # mixed-script phrase / translation
    ("MCP不包括这个功能", "MCP does not include this feature"),
    ("它它那个session", "it that session"),
    ("At Atlassian MCP", "Atlassian MCP"),             # stutter removal
    ("flip the those two parameters", "flip the two parameters"),
    ("back up", "backup"),                             # spacing only
    ("provisioning on the Thailand", "provisioning in Thailand"),
])
def test_validate_rejects_rewrites(heard, correct):
    corpus = f"xx {heard} yy"
    assert vm.validate([{"heard": heard, "correct": correct, "confidence": "high"}], corpus,
                       _vocab_entries=[]) == []


def test_activation_policy():
    term = vm.Candidate("哈什", "hash", "high", count=6)
    once = vm.Candidate("Gira", "Jira", "high", count=1)
    phrase = vm.Candidate("Snow House Query", "Snowflake Query", "high", count=3)
    medium = vm.Candidate("MCB", "MCP", "medium", count=5)
    assert vm.auto_activate(term) is True
    assert vm.auto_activate(once) is False
    assert vm.auto_activate(phrase) is False
    assert vm.auto_activate(medium) is False


@pytest.mark.parametrize("heard,correct", [
    ("decommissioning this", "decommissioning the"),
    ("Snowhouse Query", "Snowsight Query"),
    ("One SH", "One App"),
    ("Snowflakes", "Snowflake"),
])
def test_validate_rejects_phrase_and_inflection_edits(heard, correct):
    assert vm.validate([{"heard": heard, "correct": correct, "confidence": "high"}], f"x {heard} y",
                       _vocab_entries=[]) == []


def test_validate_rejects_reversing_a_known_term():
    out = vm.validate([{"heard": "MCP", "correct": "MCB", "confidence": "high"}], "MCP MCP MCP",
                      _vocab_entries=[Entry("MCP", aliases=("MCB",))])
    assert out == []


def test_ai_pending_never_self_activates(tmp_path):
    v = _vocab(tmp_path)
    for _ in range(3):
        v.add_correction("Gira", "Jira", source="suggested")
    assert v.find("Jira").status == "pending"
    v.add_correction("Gira", "Jira", source="learned")      # real retype evidence
    v.add_correction("Gira", "Jira", source="learned")
    assert v.find("Jira").status == "active"


def test_review_approves_kept_and_remembers_rejected(tmp_path):
    v = _vocab(tmp_path)
    v.add_correction("Gira", "Jira", source="suggested")
    v.add_correction("One SH", "One App", source="suggested")
    state = str(tmp_path / "s.json")
    lines = vm.pending_lines(v)
    assert lines == ["Gira = Jira", "One SH = One App"]
    approved, rejected = vm.apply_review(v, state, kept_text="Gira = Jira\n")
    assert approved == [("Gira", "Jira")] and rejected == [("One SH", "One App")]
    assert v.find("Jira").status == "active"
    assert v.find("One App") is None
    # rejected pair is filtered out of future suggestions
    out = vm.validate([{"heard": "One SH", "correct": "One App", "confidence": "high"}], "One SH",
                      _vocab_entries=[], rejected=vm.rejected_pairs(state))
    assert out == []


def test_auto_activate_requires_english_target():
    assert vm.auto_activate(vm.Candidate("memory", "内存", "high", count=5)) is False
    assert vm.auto_activate(vm.Candidate("哈什", "hash", "high", count=2)) is True


def test_latin_count_uses_word_boundaries():
    out = vm.validate([{"heard": "CI", "correct": "CSE", "confidence": "high"}], "CICD CICD CI",
                      _vocab_entries=[])
    assert out[0].count == 1


def test_partial_review_keeps_only_kept_aliases(tmp_path):
    v = _vocab(tmp_path)
    v.add_correction("Gira", "Jira", source="suggested")
    v.add_correction("Java Management", "Jira", source="suggested")
    vm.apply_review(v, str(tmp_path / "s.json"), kept_text="Gira = Jira")
    assert v.find("Jira").aliases == ("Gira",) and v.find("Jira").status == "active"


def test_rejections_survive_concurrent_run(tmp_path):
    state = str(tmp_path / "s.json")
    vm._save_state(state, {"last_timestamp": "", "rejected": []})
    v = _vocab(tmp_path)
    v.add_correction("One SH", "One App", source="suggested")
    vm.apply_review(v, state, kept_text="")
    vm._update_state(state, last_timestamp="2026-10-21T10:00:00")
    assert ("one sh", "one app") in vm.rejected_pairs(state)
