"""correction_flow: per-dictation correction + learning glue used by the app."""

import correction_flow as cf
from vocabulary import Entry, Vocabulary


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _flow(tmp_path, entries=None):
    v = Vocabulary(path=str(tmp_path / "v.json"), entries=list(entries or []))
    changed = []
    clock = Clock()
    flow = cf.CorrectionFlow(v, on_change=lambda: changed.append(1), now=clock)
    return flow, v, changed, clock


def test_process_applies_rules_and_reports(tmp_path):
    flow, v, _, _ = _flow(tmp_path, [Entry("CSE", aliases=("CIC",))])
    out = flow.process("我是C I C")
    assert out.text == "我是CSE"
    assert out.raw_text == "我是C I C"
    assert out.corrections
    assert v.find("CSE").hits == 1


def test_teach_command_adds_active_entry_and_saves(tmp_path):
    flow, v, changed, _ = _flow(tmp_path)
    assert flow.teach("漫威应该是memory") == ("漫威", "memory")
    assert v.find("memory").status == "active"
    assert (tmp_path / "v.json").exists()
    assert changed
    assert flow.process("这个agent的漫威").text == "这个agent的memory"


def test_teach_unparsable(tmp_path):
    flow, _, changed, _ = _flow(tmp_path)
    assert flow.teach("今天天气不错") is None
    assert not changed


def test_retype_learns_after_two_sightings(tmp_path):
    flow, v, changed, clock = _flow(tmp_path)
    flow.process("这个agent的漫威不够")
    clock.t += 5
    out = flow.process("这个agent的memory不够")
    assert out.learned == [("漫威", "memory")]
    assert v.find("memory").status == "pending"
    clock.t += 300
    flow.process("我的漫威满了 agent")
    clock.t += 5
    flow.process("我的memory满了 agent")
    assert v.find("memory").status == "active"
    assert changed


def test_retype_outside_window_ignored(tmp_path):
    flow, v, _, clock = _flow(tmp_path)
    flow.process("这个agent的漫威不够")
    clock.t += cf.RETYPE_WINDOW_S + 1
    assert flow.process("这个agent的memory不够").learned == []
    assert v.find("memory") is None


def test_empty_text(tmp_path):
    flow, _, _, _ = _flow(tmp_path)
    out = flow.process("")
    assert out.text == "" and out.learned == []


def test_hotwords(tmp_path):
    flow, _, _, _ = _flow(tmp_path, [Entry("CSE"), Entry("X", status="pending")])
    assert flow.hotwords() == ["CSE"]


def test_retype_that_undoes_a_rule_is_not_learned(tmp_path):
    flow, v, _, clock = _flow(tmp_path, [Entry("CSE", aliases=("CIC",))])
    flow.process("我在 CIC team")          # becomes "我在 CSE team"
    clock.t += 5
    out = flow.process("我在 CIC team")     # rules again -> same text, nothing learned
    assert out.learned == []
    assert v.find("CIC") is None


def test_hotword_list_echo_dropped(tmp_path):
    flow, _, _, _ = _flow(tmp_path, [Entry("CSE"), Entry("Snowflake"), Entry("Cortex")])
    assert flow.process("CSE, Snowflake, Cortex").text == ""
    assert flow.process("CSE and Snowflake").text == "CSE and Snowflake"


def test_add_manual_pairs(tmp_path):
    flow, v, changed, _ = _flow(tmp_path)
    assert flow.add_manual([("哈什", "hash"), ("哈时", "hash")]) == [("哈什", "hash"), ("哈时", "hash")]
    assert v.find("hash").aliases == ("哈什", "哈时")
    assert changed
    assert flow.process("这个哈什会变吗 version").text == "这个hash会变吗 version"


def test_external_edit_is_picked_up(tmp_path):
    import json, os, time
    flow, v, _, _ = _flow(tmp_path)
    p = tmp_path / "v.json"
    p.write_text(json.dumps({"version": 1, "entries": [{"term": "hash", "aliases": ["哈什"]}]}))
    os.utime(p, (time.time() + 5, time.time() + 5))
    assert flow.process("这个哈什").text == "这个hash"
