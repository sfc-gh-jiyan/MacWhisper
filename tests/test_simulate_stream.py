"""Pure-logic tests for simulate_stream (no models)."""

import numpy as np

import simulate_stream as ss


def test_tokenize_mixed_keeps_display_and_normalizes_keys():
    toks = ss.tokenize("我们 commit 了，OK!")
    assert [k for _, k in toks] == ["我", "们", "commit", "了", "ok"]
    assert "".join(d for d, _ in toks) == "我们 commit 了，OK!"


def test_stable_text_confirms_agreeing_prefix_only():
    st = ss.StableText()
    st.update("你好 world")
    assert st.confirmed_text == ""            # nothing to agree with yet
    st.update("你好 world again")
    assert st.confirmed_text == "你好 world "
    assert st.tail_text == "again"


def test_stable_text_ignores_punctuation_changes():
    st = ss.StableText()
    st.update("今天，天气")
    st.update("今天。天气好")
    assert st.confirmed_text.startswith("今天")
    assert "天气" in st.confirmed_text


def test_confirmed_never_shrinks_and_counts_contradictions():
    st = ss.StableText()
    st.update("a b c")
    st.update("a b c d")
    before = st.confirmed_text
    st.update("x y")                          # model changed its mind
    assert st.confirmed_text == before
    assert st.contradictions == 1


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_simulate_charges_decode_time_to_virtual_clock():
    clock = FakeClock()
    words = "one two three four five six".split()

    def decode(chunk):
        clock.now += 0.2                      # every decode costs 0.2 s
        n = max(1, int(len(chunk) / ss.SAMPLE_RATE))
        return " ".join(words[:n])

    audio = np.zeros(int(4.5 * ss.SAMPLE_RATE), dtype=np.float32)
    r = ss.simulate(audio, decode, step=1.0, clock=clock)
    assert r["decodes"] == 4                  # ticks at 1, 2, 3, 4 s
    assert r["mean_decode_s"] == 0.2
    assert r["first_text_s"] == 1.2           # first decode starts at 1.0 s
    assert r["final"] == "one two three four"
    assert r["contradictions"] == 0
    assert r["wait_after_release_s"] == 0.2   # only the final decode


def test_simulate_slow_decode_delays_next():
    clock = FakeClock()

    def decode(chunk):
        clock.now += 1.5                      # slower than the 1 s step
        return "x"

    audio = np.zeros(int(5 * ss.SAMPLE_RATE), dtype=np.float32)
    r = ss.simulate(audio, decode, step=1.0, clock=clock)
    assert r["decodes"] == 2                  # starts at 1.0 and 3.5 (2.5 + 1)


def test_leading_filler_does_not_freeze_confirmation():
    """Real trace: silence before speech decoded as '嗯。' twice, got
    confirmed, then every later decode (which drops it) contradicted it and
    nothing was ever confirmed again."""
    st = ss.StableText()
    for h in ("嗯。", "嗯。", "我上周。", "我上周去了三藩。",
              "我上周去了 San Francisco 参加。", "我上周去了 San Francisco，参加了一个 conference。"):
        st.update(h)
    assert "我上周去了" in st.confirmed_text
    assert "嗯" not in st.confirmed_text


def test_fillers_ignored_in_agreement():
    st = ss.StableText()
    st.update("我们 um 今天")
    st.update("我们今天 去")
    assert st.confirmed_text.replace(" ", "").startswith("我们")
    assert "今天" in st.confirmed_text


def test_rebase_when_two_decodes_agree_against_confirmed():
    st = ss.StableText()
    st.update("三藩 很好")
    st.update("三藩 很好 啊")                   # confirms "三藩 很好"
    st.update("San Francisco 很好")            # first disagreement: wait
    assert "三" in st.confirmed_text
    st.update("San Francisco 很好 天气")        # agrees with previous → rebase
    assert st.confirmed_text.startswith("San Francisco")
    assert st.rewrites == 1
