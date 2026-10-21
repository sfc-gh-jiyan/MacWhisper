"""correction_report: what was corrected, what was learned, the vocabulary."""

import json

import correction_report as cr
from vocabulary import Entry, Vocabulary


def test_report_lists_vocab_corrections_and_teaching(tmp_path):
    log = tmp_path / "t.jsonl"
    rows = [
        {"timestamp": "2026-10-21T10:07:59", "text": "and GRA", "raw_text": "and G R A",
         "corrections": [["G R A", "GRA"]]},
        {"timestamp": "2026-10-21T10:08:00", "text": "no change"},
        {"timestamp": "2026-10-21T10:09:00", "type": "teach", "raw_text": "哈什应该是hash",
         "learned": ["哈什", "hash"]},
        {"timestamp": "2026-10-21T10:10:00", "type": "teach", "raw_text": "哈什应该是哈什", "learned": None},
    ]
    log.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    v = Vocabulary(path=str(tmp_path / "v.json"), entries=[
        Entry("CSE", aliases=("CIC",), hits=3), Entry("Snow", aliases=("斯诺",), status="pending", seen=1)])
    md = cr.render(v, str(log), vocab_path="~/v.json")
    assert "# MacWhisper corrections" in md
    assert "| CSE | CIC | 3 |" in md
    assert "Snow" in md and "pending" in md.lower()
    assert "G R A" in md and "GRA" in md
    assert "no change" not in md
    assert "哈什 → hash" in md
    assert "not understood" in md


def test_report_without_log(tmp_path):
    v = Vocabulary(path=str(tmp_path / "v.json"), entries=[])
    md = cr.render(v, str(tmp_path / "missing.jsonl"))
    assert "No corrections yet" in md
