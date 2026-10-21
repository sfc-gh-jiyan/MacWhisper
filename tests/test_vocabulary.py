"""vocabulary: user term list with aliases, learning thresholds, persistence."""

import json

import vocabulary as vb


def _vocab(tmp_path, entries=None):
    p = tmp_path / "vocabulary.json"
    if entries is not None:
        p.write_text(json.dumps({"version": 1, "entries": entries}))
    return vb.Vocabulary.load(str(p)), p


def test_missing_file_loads_seed(tmp_path):
    v, _ = _vocab(tmp_path)
    terms = {e.term for e in v.active()}
    assert {"CSE", "Snowflake", "memory"} <= terms


def test_roundtrip(tmp_path):
    v, p = _vocab(tmp_path, [])
    v.add_correction("漫威", "memory", source="voice")
    v.save()
    v2 = vb.Vocabulary.load(str(p))
    e = v2.find("memory")
    assert e is not None and "漫威" in e.aliases and e.status == "active"


def test_corrupt_file_falls_back_to_seed(tmp_path):
    p = tmp_path / "vocabulary.json"
    p.write_text("{not json")
    v = vb.Vocabulary.load(str(p))
    assert v.find("CSE") is not None


def test_voice_correction_is_active_immediately(tmp_path):
    v, _ = _vocab(tmp_path, [])
    e = v.add_correction("CIC", "CSE", source="voice")
    assert e.status == "active"


def test_learned_correction_needs_two_sightings(tmp_path):
    v, _ = _vocab(tmp_path, [])
    e = v.add_correction("斯诺", "Snow", source="learned")
    assert e.status == "pending"
    assert v.find("Snow") not in v.active()
    e = v.add_correction("斯诺", "Snow", source="learned")
    assert e.status == "active"


def test_add_alias_to_existing_term(tmp_path):
    v, _ = _vocab(tmp_path, [{"term": "CSE", "aliases": ["CIC"]}])
    v.add_correction("西艾斯", "CSE", source="voice")
    assert v.find("cse").aliases == ("CIC", "西艾斯")
    assert len(v.entries) == 1


def test_identity_correction_ignored(tmp_path):
    v, _ = _vocab(tmp_path, [])
    assert v.add_correction("CSE", "cse", source="voice") is None
    assert v.add_correction("", "x", source="voice") is None


def test_hotwords_ranked_by_hits(tmp_path):
    v, _ = _vocab(tmp_path, [
        {"term": "A1", "hits": 1}, {"term": "B2", "hits": 9}, {"term": "C3", "status": "pending"}])
    assert v.hotwords(limit=2) == ["B2", "A1"]


def test_record_hits_and_remove(tmp_path):
    v, _ = _vocab(tmp_path, [{"term": "CSE"}])
    v.record_hits(["CSE", "CSE"])
    assert v.find("CSE").hits == 2
    assert v.remove("CSE") is True
    assert v.find("CSE") is None


def test_approve_pending(tmp_path):
    v, _ = _vocab(tmp_path, [{"term": "X1", "status": "pending"}])
    assert v.approve("X1") is True
    assert v.find("X1").status == "active"


def test_corrupt_file_is_preserved_not_overwritten(tmp_path):
    p = tmp_path / "vocabulary.json"
    p.write_text("{not json")
    v = vb.Vocabulary.load(str(p))
    v.save()
    assert (tmp_path / "vocabulary.json.bad").read_text() == "{not json"


def test_reverse_correction_replaces_old_rule(tmp_path):
    v, _ = _vocab(tmp_path, [{"term": "CSE", "aliases": ["CIC"]}])
    v.add_correction("CSE", "CIC", source="voice")
    assert v.find("CSE") is None
    assert v.find("CIC").aliases == ("CSE",)


def test_save_cleans_temp_on_error(tmp_path, monkeypatch):
    v, p = _vocab(tmp_path, [])
    monkeypatch.setattr(vb.json, "dump", lambda *a, **k: (_ for _ in ()).throw(TypeError("x")))
    import pytest
    with pytest.raises(TypeError):
        v.save()
    assert not [f for f in tmp_path.iterdir() if f.suffix == ".tmp"]
