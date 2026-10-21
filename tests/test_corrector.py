"""corrector: deterministic vocabulary corrections."""

import corrector
from vocabulary import Entry, Vocabulary, SCOPE_MIXED


def _v(*entries):
    return Vocabulary(path="/nonexistent", entries=list(entries))


def test_joins_spelled_acronyms():
    r = corrector.correct("我们的A I agent 和 C S E team", _v())
    assert r.text == "我们的AI agent 和 CSE team"


def test_does_not_join_single_letter_or_words():
    assert corrector.correct("Plan A is fine", _v()).text == "Plan A is fine"
    assert corrector.correct("I am here", _v()).text == "I am here"


def test_alias_replaced_with_word_boundaries():
    v = _v(Entry("CSE", aliases=("CIC",)))
    r = corrector.correct("我是CIC，不是CICD", v)
    assert r.text == "我是CSE，不是CICD"
    assert ("CIC", "CSE") in r.changes


def test_spelled_alias_after_join():
    v = _v(Entry("CSE", aliases=("CIC",)))
    assert corrector.correct("我出现很多C I C", v).text == "我出现很多CSE"


def test_alias_case_insensitive_for_latin():
    v = _v(Entry("Snowflake", aliases=("snow flake",)))
    assert corrector.correct("用 Snow Flake 查一下", v).text == "用 Snowflake 查一下"


def test_cjk_alias_mixed_scope_requires_english_context():
    v = _v(Entry("memory", aliases=("漫威",), scope=SCOPE_MIXED))
    assert corrector.correct("这个agent的漫威不够", v).text == "这个agent的memory不够"
    assert corrector.correct("我喜欢看漫威电影", v).text == "我喜欢看漫威电影"


def test_canonical_casing():
    v = _v(Entry("Snowflake"), Entry("CSE"))
    assert corrector.correct("connect to snowflake as a cse", v).text == "connect to Snowflake as a CSE"


def test_lowercase_term_not_forced():
    v = _v(Entry("memory"))
    assert corrector.correct("Memory is low", v).text == "Memory is low"


def test_pending_entries_ignored():
    v = _v(Entry("CSE", aliases=("CIC",), status="pending"))
    assert corrector.correct("CIC", v).text == "CIC"


def test_longest_alias_first():
    v = _v(Entry("Snowpark", aliases=("snow park",)), Entry("park", aliases=("帕克",)))
    assert corrector.correct("用snow park", v).text == "用Snowpark"


def test_used_terms_reported():
    v = _v(Entry("CSE", aliases=("CIC",)))
    assert corrector.correct("CIC and CSE", v).terms == ["CSE"]


def test_empty_text():
    r = corrector.correct("", _v())
    assert r.text == "" and r.changes == []


def test_bilingual_prompt_with_terms():
    from text_utils import BILINGUAL_PROMPT, bilingual_prompt
    assert bilingual_prompt([]) == BILINGUAL_PROMPT
    p = bilingual_prompt(["CSE", "Snowflake"])
    assert p.startswith(BILINGUAL_PROMPT) and "CSE, Snowflake" in p
    assert bilingual_prompt([str(i) for i in range(50)], limit=3).endswith("0, 1, 2.")


def test_stutter_not_joined():
    v = _v()
    assert corrector.correct("Okay, I I saw it", v).text == "Okay, I I saw it"
    assert corrector.correct("A A A battery", v).text == "A A A battery"
    assert corrector.correct("W C S window", v).text == "WCS window"


def test_mixed_scope_needs_english_nearby():
    v = _v(Entry("memory", aliases=("漫威",), scope=SCOPE_MIXED))
    far = "我今天发了一个PR，然后晚上和朋友一起去电影院看了很久的漫威电影"
    assert corrector.correct(far, v).text == far


def test_single_char_cjk_alias_ignored():
    v = _v(Entry("Four", aliases=("四",)))
    assert corrector.correct("四点开会", v).text == "四点开会"


def test_short_terms_not_recased():
    v = _v(Entry("US"), Entry("Go"), Entry("Snowflake"))
    assert corrector.correct("tell us to go to snowflake", v).text == "tell us to go to Snowflake"
