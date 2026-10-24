"""Find misheard terms in recent dictations with the local LLM and teach them.

    res = vocab_miner.run(llm.complete, vocab, TRANSCRIPT_LOG, STATE_PATH, model=current_asr_model)

Only dictations made with the *current* ASR model since the last run are read
(different engines mishear differently). Every LLM suggestion is checked
against the real text: the "heard" form must occur verbatim, and numbers, single
characters and identities are rejected. High-confidence fixes become active at
once (source "ai"); medium ones stay pending until seen again.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field, replace

from corrector import _latin_pattern

from vocabulary import Vocabulary

logger = logging.getLogger(__name__)

CHUNK_CHARS = 6000          # dictation text per LLM request
MIN_NEW_DICTATIONS = 20     # don't bother the GPU for a handful of sentences
MAX_PER_RUN = 25

SYSTEM = (
    "You review speech-to-text output and find words the recognizer MISHEARD. "
    "Answer with a JSON array only."
)

PROMPT = """The speaker is a Snowflake Cloud Support Engineer (CSE). He mixes Chinese and English and talks about
Snowflake, support cases, AI agents, Cortex / Cortex Code (CoCo), MCP, GitHub PRs, Jira, Confluence, Slack, Zoom,
Python, macOS apps (including his own dictation app MacWhisper).

Typical recognition errors:
- English technical words written as Chinese homophones (e.g. 哈什 = hash, 漫威 = memory)
- English terms split into wrong English words (e.g. "Clouds Polynomial" = Cloud Support Engineer)
- product names / acronyms with wrong letters (e.g. MCB = MCP, Gira = Jira)

Already known (do NOT repeat): {known}

Dictations:
<<<
{texts}
>>>

List up to {limit} misheard words. Each item:
{{"heard": "<copy EXACTLY as it appears above>", "correct": "<intended word>", "confidence": "high|medium|low", "reason": "<few words>"}}
Rules:
- Only single misheard TERMS (a word, name, acronym or short fixed phrase), never whole sentences.
- Do NOT fix grammar, word order, repetitions/stutters or spacing, and do NOT translate Chinese to English.
- Do NOT "correct" real names you are unsure about (e.g. Snowhouse is a real internal Snowflake system).
If none, answer []."""


@dataclass
class Candidate:
    heard: str
    correct: str
    confidence: str
    reason: str = ""
    count: int = 0
    example: str = ""


@dataclass
class MiningResult:
    examined: int = 0
    active: list[Candidate] = field(default_factory=list)
    pending: list[Candidate] = field(default_factory=list)
    skipped: bool = False
    stopped: bool = False
    error: str = ""


# ── parsing / validation ─────────────────────────────────────

def parse_response(text: str) -> list[dict]:
    """Extract the JSON array from an LLM reply (fences and prose tolerated)."""
    m = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return []
    out = []
    for d in data if isinstance(data, list) else []:
        if isinstance(d, dict) and isinstance(d.get("heard"), str) and isinstance(d.get("correct"), str):
            out.append(d)
    return out


_NUMERIC = re.compile(r"^[\d\s零一二三四五六七八九十百千万两.,%]+$")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def _example(corpus: str, heard: str, width: int = 24) -> str:
    i = corpus.find(heard)
    return corpus[max(0, i - width): i + len(heard) + width].replace("\n", " ") if i >= 0 else ""


_LATIN_WORDS = re.compile(r"[A-Za-z][A-Za-z0-9'._-]*")
MAX_LATIN_WORDS = 3
MAX_CJK_CHARS = 6


def _words(s: str) -> list[str]:
    return [w.lower() for w in _LATIN_WORDS.findall(s)]


def _is_rewrite(heard: str, correct: str) -> bool:
    """Grammar fixes, translations, stutter removal, spacing — not vocabulary."""
    if _CJK.search(heard) and re.search(r"[A-Za-z]", heard):
        return True                                   # mixed-script phrase ("fold的这个东西")
    if len(_words(heard)) > MAX_LATIN_WORDS or len(_words(correct)) > MAX_LATIN_WORDS + 1:
        return True
    if len(_CJK.findall(heard)) > MAX_CJK_CHARS:
        return True
    norm = lambda s: re.sub(r"[\s_-]+", "", s.lower())
    if norm(heard) == norm(correct):
        return True                                   # "back up" -> "backup"
    hw, cw = _words(heard), _words(correct)
    if hw and cw and (set(cw) < set(hw) or set(hw) < set(cw)):
        return True                                   # only words added/removed (stutters)
    if len(hw) > 1 and len(cw) > 1 and set(hw) & set(cw):
        return True                                   # phrase edit ("decommissioning this" -> "the")
    a, b = heard.lower(), correct.lower()
    if (a.startswith(b) or b.startswith(a)) and abs(len(a) - len(b)) <= 2:
        return True                                   # inflection ("Snowflakes" -> "Snowflake")
    return False


# Everyday words make dangerous "mishearings": an active 既然 → Jira rewrote
# every 既然 the user said (4 dictations, found 2026-10-24). They may still be
# real mishearings (绘画 → 会话 is), so they are never activated without review
# and the review window warns about them.
_COMMON_ZH = frozenset("""
既然 但是 因为 所以 如果 虽然 然后 而且 或者 还是 就是 不过 可是 只是 于是 因此 并且 而是 不是 只要
只有 除非 无论 不管 即使 哪怕 尽管 假如 要是 比如 例如 其实 当然 确实 真的 已经 正在 马上 立刻 一直
总是 经常 有时 刚才 现在 以前 以后 之前 之后 今天 明天 昨天 时候 时间 问题 事情 东西 地方 情况 方面
方法 方式 结果 原因 意思 意见 想法 看法 办法 部分 内容 过程 目的 目标 计划 系统 工作 公司 客户 用户
会议 会话 会谈 话题 讨论 文章 论文 报告 文件 文档 资料 数据 信息 消息 邮件 电话 网络 程序 软件 硬件
电脑 手机 设置 配置 功能 能力 问题 答案 回答 回复 解释 说明 描述 介绍 总结 分析 研究 测试 检查 确认
赛事 比赛 赛程 绘画 画画 学习 学生 老师 朋友 大家 我们 你们 他们 她们 它们 自己 别人 人们 这个 那个
这些 那些 这样 那样 这里 那里 哪里 什么 怎么 为什么 多少 几个 一个 一些 一下 一点 一样 一起 一定 一般
可能 可以 应该 需要 必须 能够 愿意 希望 觉得 认为 知道 了解 理解 明白 记得 忘记 发现 看到 听到 说到
提到 想到 开始 结束 继续 停止 完成 准备 打算 决定 选择 改变 变成 成为 出现 发生 存在 包括 包含 使用
利用 通过 根据 关于 对于 按照 为了 除了 以及 还有 甚至 尤其 特别 非常 比较 更加 最后 首先 其次 另外
同时 一边 还要 不要 不用 没有 不能 不会 不行 可惜 幸好 好像 似乎 大概 也许 或许 肯定 一共 全部 所有
每个 任何 其他 别的 主要 重要 简单 复杂 容易 困难 清楚 明确 正确 错误 安全 方便 负责 负担 压力 风险
""".split())
_EN_WORDS: frozenset | None = None
_EN_DICT_PATH = "/usr/share/dict/words"


def _english_words() -> frozenset:
    global _EN_WORDS
    if _EN_WORDS is None:
        try:
            with open(_EN_DICT_PATH, encoding="utf-8", errors="ignore") as f:
                _EN_WORDS = frozenset(w.strip().lower() for w in f if w.strip())
        except OSError:
            _EN_WORDS = frozenset()
    return _EN_WORDS


def _is_english_word(w: str) -> bool:
    w = w.lower()
    words = _english_words() | {"the", "a", "an", "and", "or", "of", "to"}
    return w in words or (w.endswith("s") and w[:-1] in words) or \
        (w.endswith("es") and w[:-2] in words)


def is_everyday_word(heard: str) -> bool:
    """A common Chinese word, or English made only of dictionary words."""
    heard = heard.strip()
    if _CJK.search(heard):
        return heard in _COMMON_ZH
    # One English word only: a phrase ("conference page") must match whole, so it
    # rarely fires by accident; CIC, co-citation, R4P are not dictionary words
    if not re.fullmatch(r"[A-Za-z]{2,}", heard) or heard.isupper():
        return False
    return _is_english_word(heard)


def review_warning(vocab: Vocabulary) -> str:
    """Warning line for the review window: suggestions that are everyday words."""
    risky = [line for line in pending_lines(vocab)
             if is_everyday_word(line.split(" = ", 1)[0])]
    if not risky:
        return ""
    return ("⚠ Everyday words — keep only if you are sure, every time you say them they "
            "will be changed: " + "; ".join(risky))


def reject_pairs(state_path: str, pairs) -> None:
    """Never suggest these (heard, correct) pairs again."""
    pairs = {(a, b) for a, b in pairs}
    _update_state(state_path, rejected=lambda s: sorted(
        {tuple(p) for p in s.get("rejected", [])} | pairs))


def auto_activate(c: "Candidate") -> bool:
    """Apply without asking only for clear, repeated, single-term fixes."""
    if is_everyday_word(c.heard):
        return False
    single = len(_words(c.heard)) <= 1 and len(_CJK.findall(c.heard)) <= 4
    english_target = bool(re.search(r"[A-Za-z]{2,}", c.correct)) and not _CJK.search(c.correct)
    return c.confidence == "high" and single and english_target and c.count >= 2


def validate(cands: list[dict], corpus: str, _vocab_entries, rejected=frozenset()) -> list[Candidate]:
    known = {a.lower() for e in _vocab_entries for a in e.aliases}
    terms = {e.term.lower() for e in _vocab_entries}
    seen, out = set(), []
    for d in cands:
        heard, correct = d["heard"].strip(), d["correct"].strip()
        conf = str(d.get("confidence", "low")).lower()
        if conf not in ("high", "medium") or not heard or not correct:
            continue
        if heard.lower() == correct.lower() or heard.lower() in known or (heard, correct) in seen:
            continue
        if (heard.lower(), correct.lower()) in rejected:
            continue
        if correct.lower() in known:                  # a misheard form is never the right answer
            continue
        if heard.lower() in terms:                    # would reverse an existing correction
            continue
        if _NUMERIC.match(heard) or _NUMERIC.match(correct):
            continue
        if len(heard) > 30 or len(correct) > 40:
            continue
        if (_CJK.search(heard) and len(heard) < 2) or (not _CJK.search(heard) and len(heard) < 2):
            continue
        if _is_rewrite(heard, correct):
            continue
        count = (len(_latin_pattern(heard).findall(corpus)) if re.search(r"[A-Za-z]", heard)
                 else corpus.count(heard))
        if count == 0:
            logger.debug("Miner: dropped %r (not in transcripts)", heard)
            continue
        seen.add((heard, correct))
        out.append(Candidate(heard, correct, conf, str(d.get("reason", ""))[:80], count,
                             _example(corpus, heard)))
    return out[:MAX_PER_RUN]


# ── state / input ────────────────────────────────────────────

def _load_state(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


_STATE_LOCK = threading.Lock()


def _update_state(path: str, **fields) -> dict:
    """Read-modify-write under a lock so concurrent writers never lose each other's fields."""
    with _STATE_LOCK:
        state = _load_state(path)
        for k, v in fields.items():
            state[k] = v(state) if callable(v) else v
        _save_state(path, state)
        return state


def _save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def new_dictations(log_path: str, since: str, model: str | None) -> list[tuple[str, str]]:
    """(timestamp, text) of dictations newer than `since` made with `model`."""
    out = []
    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("type") == "teach" or not d.get("text"):
                    continue
                ts = d.get("timestamp", "")
                if ts <= since or (model and d.get("model") != model):
                    continue
                out.append((ts, d.get("raw_text") or d["text"]))
    except OSError:
        pass
    return out


def chunk_texts(texts: list[str], limit: int = CHUNK_CHARS) -> list[list[str]]:
    chunks, cur, size = [], [], 0
    for t in texts:
        if cur and size + len(t) > limit:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(t)
        size += len(t)
    if cur:
        chunks.append(cur)
    return chunks


# ── main entry ───────────────────────────────────────────────

def run(complete, vocab: Vocabulary, log_path: str, state_path: str, model: str | None = None,
        min_new: int = MIN_NEW_DICTATIONS, should_pause=lambda: False, on_progress=None,
        sleep=None, max_pause_s: float = 1800.0) -> MiningResult:
    """complete(prompt, system=..., max_tokens=..., timeout=...) -> str (e.g. LLMSidecarBackend.complete)."""
    state = _load_state(state_path)
    since = state.get("last_timestamp", "")
    rows = new_dictations(log_path, since, model)
    res = MiningResult(examined=len(rows))
    if not rows:
        return res
    if len(rows) < min_new:
        res.skipped = True
        logger.info("Miner: %d new dictations (< %d) — skipped", len(rows), min_new)
        return res

    texts = [t for _, t in rows]
    corpus = "\n".join(texts)
    known = ", ".join(f"{a}={e.term}" for e in vocab.active() for a in e.aliases)[:1500] or "none"
    found: list[Candidate] = []
    import time as _time
    sleep = sleep or _time.sleep
    chunks = chunk_texts(texts)
    report = on_progress or (lambda done, total, paused: None)
    for i, chunk in enumerate(chunks):
        waited = 0.0
        while should_pause():          # never compete with dictation for the GPU
            if waited == 0.0:
                report(i, len(chunks), True)
            if waited >= max_pause_s:
                res.stopped = True
                logger.warning("Miner: gave up after waiting %.0fs for the user to finish", waited)
                return res
            sleep(1.0)
            waited += 1.0
        report(i, len(chunks), False)
        prompt = PROMPT.format(known=known, texts="\n".join(chunk), limit=10)
        reply = complete(prompt, system=SYSTEM, max_tokens=700, timeout=180.0)
        found += validate(parse_response(reply), corpus, vocab.entries, rejected=rejected_pairs(state_path))
    report(len(chunks), len(chunks), False)

    unique, seen = [], set()
    for c in found:
        if (c.heard, c.correct) not in seen:
            seen.add((c.heard, c.correct))
            unique.append(c)
    for c in unique[:MAX_PER_RUN]:
        entry = vocab.add_correction(c.heard, c.correct, source="ai" if auto_activate(c) else "suggested")
        if entry is None:
            continue
        (res.active if entry.status == "active" else res.pending).append(c)
    vocab.save()
    run_info = {"at": rows[-1][0], "examined": len(rows),
                "active": len(res.active), "pending": len(res.pending)}
    _update_state(state_path, last_timestamp=rows[-1][0],
                  runs=lambda s: (s.get("runs", []) + [run_info])[-30:])
    logger.warning("Vocabulary miner: %d dictations, %d active, %d pending: %s",
                   len(rows), len(res.active), len(res.pending),
                   [(c.heard, c.correct) for c in res.active + res.pending])
    return res


def due(state_path: str, interval_s: float, now: float) -> bool:
    """True when the last automatic run was at least interval_s ago (or never)."""
    last = _load_state(state_path).get("last_run_epoch")
    return last is None or now - float(last) >= interval_s


def mark_run(state_path: str, now: float) -> None:
    _update_state(state_path, last_run_epoch=now)


# ── user review of pending suggestions ───────────────────────

def rejected_pairs(state_path: str) -> set[tuple[str, str]]:
    return {(a.lower(), b.lower()) for a, b in _load_state(state_path).get("rejected", [])}


def pending_lines(vocab: Vocabulary) -> list[str]:
    """One "heard = correct" line per pending suggestion (what the review window shows)."""
    out = []
    for e in vocab.pending():
        forms = e.aliases if e.status == "pending" else e.pending_aliases
        out += [f"{a} = {e.term}" for a in forms]
    return out


def typed_risks(vocab: Vocabulary, heard: str, correct: str) -> list[str]:
    """Why a line the user typed in the review window may do harm (empty = safe)."""
    reasons = []
    owner = next((e for e in vocab.entries
                  if any(a.lower() == correct.lower() for a in e.aliases)), None)
    if owner is not None:
        reasons.append(f"“{correct}” is how “{owner.term}” gets misheard — this would undo "
                       f"“{correct} → {owner.term}”.")
    known = vocab.find(heard)
    if known is not None and known.status == "active":
        reasons.append(f"“{heard}” is a word in your vocabulary — every time you say it, "
                       f"it would become “{correct}”.")
    return reasons


def apply_review(vocab: Vocabulary, state_path: str, kept_text: str, confirm=None):
    """Lines left in the review window are approved; deleted ones are rejected for good.

    Lines the user typed or edited are added too, unless they look harmful
    (reversed, or breaking an existing rule): then `confirm(typed, alternative,
    reasons)` decides — "alternative", "typed" or "skip". Without `confirm`
    a risky line is skipped.
    """
    from learn import parse_manual

    kept = {(a.lower(), b.lower()) for a, b in parse_manual(kept_text)}
    approved, rejected = [], []
    for e in list(vocab.pending()):
        forms = e.aliases if e.status == "pending" else e.pending_aliases
        keep = tuple(a for a in forms if (a.lower(), e.term.lower()) in kept)
        approved += [(a, e.term) for a in keep]
        rejected += [(a, e.term) for a in forms if a not in keep]
        with vocab._lock:
            if e.status == "pending":
                vocab.remove(e.term)
                if keep:
                    vocab._put(replace(e, aliases=keep, status="active", pending_aliases=()))
            else:   # active term: promote kept forms, drop the rest
                vocab._put(replace(e, aliases=e.aliases + keep, pending_aliases=()))
    # Lines the user typed themselves (not from the list) are taught as manual corrections
    listed = {(a.lower(), b.lower()) for a, b in approved + rejected}
    for a, b in parse_manual(kept_text):
        if (a.lower(), b.lower()) in listed:
            continue
        reasons = typed_risks(vocab, a, b)
        # Left side edited on a suggestion: the user most likely typed the right word on the left
        orig = next(((h, c) for h, c in rejected if c.lower() == b.lower() and h.lower() != a.lower()), None)
        alternative = (orig[0], a) if orig else None
        if orig:
            reasons.insert(0, f"You changed the left side of “{orig[0]} = {orig[1]}”. Left is what "
                              f"MacWhisper heard, right is what it should become.")
        choice = "typed" if not reasons else (confirm((a, b), alternative, reasons) if confirm else "skip")
        pair = alternative if choice == "alternative" and alternative else (a, b) if choice == "typed" else None
        if pair is None:
            logger.warning("Vocabulary review: skipped risky line %r: %s", (a, b), reasons)
            continue
        vocab.add_correction(pair[0], pair[1], source="manual")
        approved.append(pair)
    vocab.save()
    _update_state(state_path, rejected=lambda s: sorted(
        {tuple(p) for p in s.get("rejected", [])} | set(rejected)))
    logger.warning("Vocabulary review: approved %s, rejected %s", approved, rejected)
    return approved, rejected
