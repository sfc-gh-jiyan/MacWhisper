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
    # Kept whole; only the adjacent stutter "what what" is collapsed (roadmap D)
    assert qt.postprocess(text, collapse=True) == text.replace("what what", "what")


def test_known_hallucination_phrases_still_filtered():
    assert qt.postprocess("谢谢大家") == ""
    assert qt.postprocess("Thank you for watching.") == ""


def test_traditional_to_simplified():
    assert qt.postprocess("這個問題") == "这个问题"


def test_options_can_be_disabled():
    raw = "嗯，这个 commit 的情况"
    assert qt.postprocess(raw, strip_filler_words=False, tight=False) == raw


# ── Stutter repeats (roadmap D, from the user's real dictations 2026-10) ──

import pytest  # noqa: E402

MUST_CHANGE = [
    ("你先别着急做，我我我们再思考一下。", "你先别着急做，我们再思考一下。"),
    ("没有主任的Jira还是发给Review会会会花。", "没有主任的Jira还是发给Review会花。"),
    ("然后仔细的研究一下它它的history", "然后仔细的研究一下它的history"),
    ("但是我们的我们的priority还是work on the customer case。",
     "但是我们的priority还是work on the customer case。"),
    ("我现在基本上都用的是这个这个app。", "我现在基本上都用的是这个app。"),
    ("我的八四零零端口经常会被占用，被占用，我也不知道", "我的八四零零端口经常会被占用，我也不知道"),
    ("这个东西你可以你可以试一下", "这个东西你可以试一下"),
    ("Okay, next, I I saw the customer list.", "Okay, next, I saw the customer list."),
    ("I felt the the PR.", "I felt the PR."),
    ("We we need it. You you know.", "We need it. You know."),
    ("Let let me check", "Let me check"),
    ("然后就是就是这个问题", "然后就是这个问题"),
    ("这这这个或者那那个", "这个或者那个"),
    ("start a a design", "start a design"),
]

MUST_KEEP = [
    "你能找找是什么问题吗？",
    "看看到底是什么问题导致的这个结果？",
    "把它怎么合并和整整改。",
    "慢慢来，好好想想，谢谢。",
    "然后大段大段的这个detail",
    "什么什么的都可以",
    "你说得对不对？对对，就是这样。",
    "He had had enough. I said that that was fine.",
    "the Jira search returned it, and the Jira search is slow",
    "三三两两的人",
    "我们每天天天都在写",
    "的的确确是这样，是是非非说不清，和和气气",
    "你想想看",
    # false positives found replaying 1288 real dictations (2026-10-24)
    "我想动这个A P P。",
    "这个十月十九、十月二十、十月二十一",
    "一个email一定对应一个账户，一个账户对应一个email",
    "this is very very important, really really good",
    "我的意思，我的意思是，不需要",   # comma + clause goes on: too risky (see 一个账户)
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
