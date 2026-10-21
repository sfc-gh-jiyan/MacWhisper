"""learn: voice correction commands and retype-diff learning."""

import pytest

import learn


@pytest.mark.parametrize("text,expected", [
    ("CIC应该是CSE", ("CIC", "CSE")),
    ("CIC 应该是 C S E。", ("CIC", "CSE")),
    ("把漫威改成memory", ("漫威", "memory")),
    ("漫威换成 memory", ("漫威", "memory")),
    ("不是漫威，是memory", ("漫威", "memory")),
    ("“斯诺”应该写成“Snow”", ("斯诺", "Snow")),
    ("CIC should be CSE.", ("CIC", "CSE")),
    ("change CIC to CSE", ("CIC", "CSE")),
    ("replace snow flake with Snowflake", ("snow flake", "Snowflake")),
])
def test_parse_command(text, expected):
    assert learn.parse_command(text) == expected


@pytest.mark.parametrize("text", [
    "", "今天天气不错", "应该是CSE", "CIC应该是", "change it", "X应该是X",
    "这一整段话非常非常长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长长应该是短的",
])
def test_parse_command_rejects(text):
    assert learn.parse_command(text) is None


def test_retype_single_term_fix():
    prev = "这个agent的漫威不够用"
    cur = "这个agent的memory不够用"
    assert learn.retype_corrections(prev, cur) == [("漫威", "memory")]


def test_retype_latin_word_expanded():
    prev = "我是一个CIC engineer"
    cur = "我是一个CSE engineer"
    assert learn.retype_corrections(prev, cur) == [("CIC", "CSE")]


def test_retype_ignores_punctuation_only():
    assert learn.retype_corrections("好的，我们开始", "好的我们开始。") == []


def test_retype_ignores_different_sentence():
    assert learn.retype_corrections("我们明天开会讨论预算", "帮我查一下这个错误日志") == []


def test_retype_ignores_large_rewrite():
    prev = "我觉得这个方案可以先上线然后再看看"
    cur = "我觉得那个计划应该晚点上线再说吧看看"
    assert learn.retype_corrections(prev, cur) == []


def test_retype_identical():
    assert learn.retype_corrections("一样的", "一样的") == []


@pytest.mark.parametrize("prev,cur", [
    ("明天下午三点开会", "明天下午四点开会"),      # rewording a number
    ("我们明天开会吧", "我们后天开会吧"),          # rewording Chinese
    ("version 3 is out", "version 4 is out"),     # digits
])
def test_retype_only_learns_term_fixes(prev, cur):
    assert learn.retype_corrections(prev, cur) == []


@pytest.mark.parametrize("text,expected", [
    ("哈什 = hash", [("哈什", "hash")]),
    ("哈时=hash; 哈什 -> hash", [("哈时", "hash"), ("哈什", "hash")]),
    ("漫威 → memory\nCIC=>CSE", [("漫威", "memory"), ("CIC", "CSE")]),
    ("哈什应该是hash", [("哈什", "hash")]),
    ("", []), ("nonsense", []), ("a = a", []),
])
def test_parse_manual(text, expected):
    assert learn.parse_manual(text) == expected


def test_same_sides_detected():
    assert learn.is_same_sides("哈什应该是哈什")
    assert not learn.is_same_sides("哈什应该是hash")
    assert not learn.is_same_sides("今天天气不错")
