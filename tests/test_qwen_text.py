"""Tests for Qwen3-ASR output post-processing (examples modelled on real outputs)."""

import qwen_text as qt


def test_tight_spacing_between_cjk_and_latin():
    assert qt.tighten_spacing("你帮我看下今天的这个 menu 的价格") == "你帮我看下今天的这个menu的价格"
    assert qt.tighten_spacing("这个 coffee shop 有没有") == "这个coffee shop有没有"
    assert qt.tighten_spacing("It's sunny. 我很开心") == "It's sunny.我很开心"


def test_spacing_keeps_english_word_spaces():
    s = "Hold on a moment, this recipe needs more salt."
    assert qt.tighten_spacing(s) == s


def test_strip_standalone_fillers():
    assert qt.strip_fillers("OK，我们再来唱一遍。嗯，This is the second song.") == \
        "OK，我们再来唱一遍。This is the second song."
    assert qt.strip_fillers("我打算买，嗯，这个这个太贵了") == "我打算买，这个这个太贵了"
    assert qt.strip_fillers("Like um, we not only cook") == "Like we not only cook"
    assert qt.strip_fillers("呃，你好") == "你好"


def test_fillers_inside_words_are_kept():
    assert qt.strip_fillers("这是额外的工作") == "这是额外的工作"       # 额 in 额外
    assert qt.strip_fillers("好啊，我同意") == "好啊，我同意"           # sentence particle
    assert qt.strip_fillers("嗯哼") == "嗯哼"                            # not a bare filler


def test_filler_only_output_becomes_empty():
    assert qt.postprocess("嗯。") == ""
    assert qt.postprocess("  呃，嗯。 ") == ""


def test_long_repeated_english_is_not_discarded():
    """Long clip (modelled on a real one): Whisper's word_repeat rule wiped a correct transcript."""
    text = ("For the garden plan, we just stay focused on the garden plan. We want to "
            "improve the garden plan so the kids get exactly what what they like.")
    assert qt.postprocess(text) == text
    # Kept whole; only the adjacent stutter "what what" is collapsed (roadmap D)
    assert qt.postprocess(text, collapse=True) == text.replace("what what", "what")


def test_known_hallucination_phrases_still_filtered():
    assert qt.postprocess("谢谢大家") == ""
    assert qt.postprocess("Thank you for watching.") == ""


def test_traditional_to_simplified():
    assert qt.postprocess("這個問題") == "这个问题"


def test_options_can_be_disabled():
    raw = "嗯，这个 menu 的价格"
    assert qt.postprocess(raw, strip_filler_words=False, tight=False) == raw


# ── Stutter repeats (roadmap D, modelled on real dictations) ──

import pytest  # noqa: E402

MUST_CHANGE = [
    ("你先别急着改，我我我们先讨论一下。", "你先别急着改，我们先讨论一下。"),
    ("周末的报告还是交给老师会会会好。", "周末的报告还是交给老师会好。"),
    ("然后认真的看一下它它的color", "然后认真的看一下它的color"),
    ("但是我们的我们的plan还是go to the city park。",
     "但是我们的plan还是go to the city park。"),
    ("我平时基本上都喝的是这个这个茶。", "我平时基本上都喝的是这个茶。"),
    ("我的自行车经常会被借走，被借走，我也不知道", "我的自行车经常会被借走，我也不知道"),
    ("这个蛋糕你可以你可以尝一下", "这个蛋糕你可以尝一下"),
    ("Okay, next, I I read the shopping list.", "Okay, next, I read the shopping list."),
    ("I fed the the cat.", "I fed the cat."),
    ("We we tried it. You you see.", "We tried it. You see."),
    ("Let let me think", "Let me think"),
    ("然后就是就是那个想法", "然后就是那个想法"),
    ("这这这个还是那那个", "这个还是那个"),
    ("write a a poem", "write a poem"),
]

MUST_KEEP = [
    "你能找找钥匙在哪里吗？",
    "看看到底是什么原因导致的这个变化？",
    "把它怎么分类和整整改。",
    "慢慢来，好好休息，谢谢。",
    "然后大段大段的这个example",
    "水果什么什么的都可以",
    "你觉得好不好？对对，就是这样。",
    "She had had lunch. He knew that that was true.",
    "the weather report said rain, and the weather report is often wrong",
    "三三两两的鸟",
    "我们每天天天都在跑步",
    "的的确确是这样，是是非非很难讲，和和气气",
    "你再想想看",
    # false-positive patterns found replaying real dictations (2026-10-24)
    "我想打开这个A P P。",
    "这个五月十九、五月二十、五月二十一",
    "一个学生一定对应一个座位，一个座位对应一个学生",
    "this is very very simple, really really nice",
    "我的想法，我的想法是，不着急",   # comma + clause goes on: too risky (see 一个座位)
]


@pytest.mark.parametrize("before,after", MUST_CHANGE)
def test_stutter_repeats_are_collapsed(before, after):
    assert qt.collapse_repeats(before) == after


@pytest.mark.parametrize("text", MUST_KEEP)
def test_normal_reduplication_is_kept(text):
    assert qt.collapse_repeats(text) == text


def test_postprocess_collapses_only_when_asked():
    """The app collapses the final text itself so it can log the original."""
    assert qt.postprocess("我我我们走吧") == "我我我们走吧"
    assert qt.postprocess("我我我们走吧", collapse=True) == "我们走吧"


# ── Intentional repeats vs. runaway loops (2026-11-07) ──
# Whisper's strip_trailing_repetition deleted any tail repeated 3+ times, so
# "Silly, silly, silly." was pasted as "" and "The puppy is silly,
# silly, silly." as "The puppy is". People repeat 2-4 times on purpose;
# a looping model repeats dozens of times.

@pytest.mark.parametrize("text", [
    "Silly, silly, silly.",
    "The puppy is silly, silly, silly.",
    "Bye bye bye.",
    "No, no, no.",
    "我们走吧走吧走吧。",
    "对对对对。",
    "哈哈哈哈哈哈哈哈",
    "对对对对对对，就是这样。",
    "No no no no no no.",
])
def test_intentional_repeats_are_kept(text):
    assert qt.postprocess(text, strip_filler_words=False) == text


def test_runaway_loop_keeps_one_copy():
    """2026-10-29 meeting: the live decoder looped on 我们 ~225 times."""
    text = "what time is it now? " + "我们" * 120
    out = qt.postprocess(text, strip_filler_words=False, tight=False)
    assert out.startswith("what time is it now?")
    assert out.count("我们") == 1


def test_runaway_english_loop_keeps_one_copy():
    out = qt.postprocess("Okay. " + "Thank you. " * 9, strip_filler_words=False)
    assert out.lower().count("thank you") == 1 and out.startswith("Okay.")


def test_six_repeats_of_a_phrase_count_as_a_loop():
    out = qt.trim_runaway_loop("so " + "let me check " * 6)
    assert out.count("let me check") == 1


def test_five_repeats_are_still_speech():
    text = "go go go go go"
    assert qt.trim_runaway_loop(text) == text
