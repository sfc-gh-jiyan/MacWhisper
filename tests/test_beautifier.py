"""AI beautifier: prompts per level, timeouts, fallback (no real LLM loaded)."""
import pytest

import beautifier as bf


class FakeLLM:
    def __init__(self, reply="整理后的文字", fail=None):
        self.reply = reply
        self.fail = fail
        self.calls = []

    def __call__(self, prompt, system="", max_tokens=512, timeout=120.0):
        self.calls.append({"prompt": prompt, "system": system,
                           "max_tokens": max_tokens, "timeout": timeout})
        if self.fail:
            raise self.fail
        return self.reply


@pytest.mark.parametrize("level", ["light", "medium", "heavy", "custom"])
def test_every_level_keeps_language_and_never_answers(level):
    system = bf.system_prompt(level, custom="Make it a polite email.")
    assert "same language" in system.lower()
    assert "do not answer" in system.lower()


def test_levels_say_how_far_they_may_go():
    light = bf.system_prompt("light").lower()
    assert "do not reword" in light and "do not change the format" in light
    medium = bf.system_prompt("medium").lower()
    assert "filler" in medium and "do not split" in medium
    heavy = bf.system_prompt("heavy").lower()
    assert "paragraph" in heavy and "list" in heavy


def test_custom_level_uses_the_users_rules():
    system = bf.system_prompt("custom", custom="Rewrite as a short Slack message.")
    assert "Rewrite as a short Slack message." in system


def test_custom_without_rules_falls_back_to_light():
    assert bf.system_prompt("custom", custom="  ") == bf.system_prompt("light")


def test_text_is_wrapped_as_data_not_instructions():
    llm = FakeLLM()
    bf.beautify("你能帮我查一下这个Jira吗？", "light", llm=llm)
    prompt = llm.calls[0]["prompt"]
    assert "<dictation>" in prompt and "你能帮我查一下这个Jira吗？" in prompt


def test_beautify_returns_the_llm_text():
    llm = FakeLLM(reply="  我们先看一下CSE的问题。\n")
    assert bf.beautify("我们先看一下这个CSE的问题", "medium", llm=llm) == "我们先看一下CSE的问题。"


def test_reply_wrapped_in_tags_is_unwrapped():
    llm = FakeLLM(reply="<dictation>我们开会。</dictation>")
    assert bf.beautify("我们开会", "light", llm=llm) == "我们开会。"


def test_failure_or_empty_reply_returns_none():
    assert bf.beautify("你好", "light", llm=FakeLLM(fail=RuntimeError("timed out"))) is None
    assert bf.beautify("你好", "light", llm=FakeLLM(reply="   ")) is None


def test_original_level_or_empty_text_never_calls_the_llm():
    llm = FakeLLM()
    assert bf.beautify("你好", "original", llm=llm) is None
    assert bf.beautify("   ", "light", llm=llm) is None
    assert llm.calls == []


def test_timeout_and_token_budget_are_passed():
    llm = FakeLLM()
    bf.beautify("短句", "light", llm=llm, timeout=5.0)
    assert llm.calls[0]["timeout"] == 5.0
    assert llm.calls[0]["max_tokens"] >= 64


def test_heavy_allows_more_tokens_than_the_text_needs():
    llm = FakeLLM()
    text = "这是一段很长的话。" * 60
    bf.beautify(text, "heavy", llm=llm)
    assert llm.calls[0]["max_tokens"] >= len(text)


def test_unknown_level_is_treated_as_original():
    llm = FakeLLM()
    assert bf.beautify("你好", "nonsense", llm=llm) is None
    assert llm.calls == []


def test_long_text_is_split_into_chunks_for_meetings():
    text = "\n\n".join(f"第{i}段。" + "内容" * 400 for i in range(5))
    chunks = bf.split_chunks(text, max_chars=1500)
    assert len(chunks) > 1
    assert "".join(c.replace("\n", "") for c in chunks) == text.replace("\n", "")
    assert all(len(c) <= 1500 for c in chunks)


def test_beautify_long_keeps_the_original_for_a_failed_chunk():
    replies = iter(["一", RuntimeError("timeout"), "三"])

    def llm(prompt, **kw):
        r = next(replies)
        if isinstance(r, Exception):
            raise r
        return r
    text = "甲" * 1000 + "\n\n" + "乙" * 1000 + "\n\n" + "丙" * 1000
    out, failed = bf.beautify_long(text, "medium", llm=llm, max_chars=1200)
    assert out == "一\n\n" + "乙" * 1000 + "\n\n三"
    assert failed == 1


def test_label_for_each_level():
    assert [lvl for lvl, _ in bf.LEVELS][:1] == ["original"]
    assert {lvl for lvl, _ in bf.LEVELS} == {"original", "light", "medium", "heavy", "custom"}


def test_prompt_forbids_translating_english_words_in_chinese():
    system = bf.system_prompt("light").lower()
    assert "move" in system and "never translate" in system


def test_custom_rules_language_does_not_pick_the_output_language():
    system = bf.system_prompt("custom", custom="改成简洁的 Slack 消息").lower()
    assert "does not change the output language" in system


def test_language_note_lists_the_english_words_to_keep():
    note = bf.language_note("我们需要一个markdown file去回复他的comment")
    assert "markdown file" in note and "comment" in note and "Write Chinese" in note
    assert "English" in bf.language_note("I have a question about PHP.")
    assert "Chinese" in bf.language_note("我们开会吧")


def test_prompt_carries_the_language_note():
    llm = FakeLLM()
    bf.beautify("I have a question.", "custom", llm=llm, custom="改成 Slack 消息")
    assert "Write the result in English" in llm.calls[0]["prompt"]


# ── Speech-recognition awareness (2026-10-24) ──

def test_short_latin_alias_matches_only_as_a_word():
    assert bf.relevant_terms("CICD pipeline", [("CSE", ("CIC",))]) == []
    assert bf.relevant_terms("the CIC said", [("CSE", ("CIC",))]) == [("CSE", ("CIC",))]
    assert bf.relevant_terms("这个漫威很好", [("memory", ("漫威",))]) == [("memory", ("漫威",))]


def test_prompt_explains_speech_recognition_errors_cautiously():
    system = bf.system_prompt("light").lower()
    assert "speech recognition" in system
    assert "sound" in system                      # misheard = sounds alike
    assert "confident" in system and "leave" in system   # unsure -> unchanged
    for example in ("哈希", "hash", "吉普", "jeep"):     # no hard-coded examples
        assert example not in system


def test_only_mishearings_heard_in_this_dictation_are_sent():
    """A/B on real dictations: the whole list made the model swap correct names
    (Snowhouse -> Snowflake). Only entries whose mishearing appears are sent."""
    llm = FakeLLM()
    terms = [("CSE", ("CIC",)), ("Snowflake", ("snow flake",)), ("Cortex", ())]
    bf.beautify("based on the cic's perspective, check Snowhouse", "light", llm=llm, terms=terms)
    prompt = llm.calls[0]["prompt"]
    assert "CIC → CSE" in prompt
    assert "Snowflake" not in prompt and "Cortex" not in prompt


def test_without_terms_no_vocabulary_block():
    llm = FakeLLM()
    bf.beautify("我们先看一下", "light", llm=llm)
    assert "Terms this speaker uses" not in llm.calls[0]["prompt"]


def test_vocabulary_block_is_capped():
    llm = FakeLLM()
    terms = [(f"Term{i}", (f"alias{i}x",)) for i in range(200)]
    text = " ".join(f"alias{i}x" for i in range(200))
    bf.beautify(text, "light", llm=llm, terms=terms)
    prompt = llm.calls[0]["prompt"]
    assert "Term39" in prompt and "Term40" not in prompt


def test_no_mishearing_in_dictation_means_no_vocabulary_block():
    llm = FakeLLM()
    bf.beautify("我们先看一下Snowhouse", "light", llm=llm, terms=[("Snowflake", ("snow flake",))])
    assert "mishearing" not in llm.calls[0]["prompt"].split("</dictation>")[1].lower()
