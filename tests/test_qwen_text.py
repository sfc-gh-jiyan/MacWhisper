"""Tests for Qwen3-ASR output post-processing (examples from real outputs)."""

import qwen_text as qt


def test_tight_spacing_between_cjk_and_latin():
    assert qt.tighten_spacing("你帮我查下现在的这个 commit 的情况") == "你帮我查下现在的这个commit的情况"
    assert qt.tighten_spacing("这个 trust center 有没有") == "这个trust center有没有"
    assert qt.tighten_spacing("It's beautiful. 我很喜欢") == "It's beautiful.我很喜欢"


def test_spacing_keeps_english_word_spaces():
    s = "Wait for a second, this does not make any sense."
    assert qt.tighten_spacing(s) == s


def test_strip_standalone_fillers():
    assert qt.strip_fillers("OK，我们再来录一段。嗯，This is the first day.") == \
        "OK，我们再来录一段。This is the first day."
    assert qt.strip_fillers("我打算用，嗯，这个这个太多了") == "我打算用，这个这个太多了"
    assert qt.strip_fillers("Like um, we not only search") == "Like we not only search"
    assert qt.strip_fillers("呃，你好") == "你好"


def test_fillers_inside_words_are_kept():
    assert qt.strip_fillers("这是额外的工作") == "这是额外的工作"       # 额 in 额外
    assert qt.strip_fillers("好啊，我同意") == "好啊，我同意"           # sentence particle
    assert qt.strip_fillers("嗯哼") == "嗯哼"                            # not a bare filler


def test_filler_only_output_becomes_empty():
    assert qt.postprocess("嗯。") == ""
    assert qt.postprocess("  呃，嗯。 ") == ""


def test_long_repeated_english_is_not_discarded():
    """Real 67 s clip: Whisper's word_repeat rule wiped a correct transcript."""
    text = ("For the Jira search, we just stay focused on the Jira search. We want to "
            "improve the Jira search results to get exactly what what they want.")
    assert qt.postprocess(text) == text


def test_known_hallucination_phrases_still_filtered():
    assert qt.postprocess("谢谢大家") == ""
    assert qt.postprocess("Thank you for watching.") == ""


def test_traditional_to_simplified():
    assert qt.postprocess("這個問題") == "这个问题"


def test_options_can_be_disabled():
    raw = "嗯，这个 commit 的情况"
    assert qt.postprocess(raw, strip_filler_words=False, tight=False) == raw
