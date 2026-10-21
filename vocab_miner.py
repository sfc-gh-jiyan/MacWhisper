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


def auto_activate(c: "Candidate") -> bool:
    """Apply without asking only for clear, repeated, single-term fixes."""
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


def apply_review(vocab: Vocabulary, state_path: str, kept_text: str):
    """Lines left in the review window are approved; deleted ones are rejected for good."""
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
        if (a.lower(), b.lower()) not in listed:
            vocab.add_correction(a, b, source="manual")
            approved.append((a, b))
    vocab.save()
    _update_state(state_path, rejected=lambda s: sorted(
        {tuple(p) for p in s.get("rejected", [])} | set(rejected)))
    logger.warning("Vocabulary review: approved %s, rejected %s", approved, rejected)
    return approved, rejected
