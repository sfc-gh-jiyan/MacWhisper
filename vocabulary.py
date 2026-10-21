"""User vocabulary: canonical terms with the ways ASR mishears them.

Stored as JSON in <data dir>/vocabulary.json:
    {"version": 1, "entries": [{"term": "CSE", "aliases": ["CIC"], ...}]}

Entries taught explicitly (voice command, manual edit) are active at once.
Corrections inferred automatically ("learned") stay pending until the same
correction has been seen LEARN_THRESHOLD times, so one odd diff never rewrites
your text.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field, replace

logger = logging.getLogger(__name__)

LEARN_THRESHOLD = 2
ACTIVE, PENDING = "active", "pending"
SCOPE_ANY, SCOPE_MIXED = "any", "mixed"  # mixed: only in text that also has English words

SEED = (
    {"term": "CSE", "aliases": ["CIC", "C I C"], "note": "Cloud Support Engineer"},
    {"term": "Snowflake", "aliases": ["snow flake", "斯诺弗雷克"], "note": ""},
    {"term": "Cortex", "aliases": [], "note": "Snowflake Cortex"},
    {"term": "memory", "aliases": ["漫威"], "note": "", "scope": SCOPE_MIXED},
)


@dataclass(frozen=True)
class Entry:
    term: str
    aliases: tuple[str, ...] = ()
    note: str = ""
    scope: str = SCOPE_ANY
    source: str = "manual"   # manual | voice | learned | seed
    status: str = ACTIVE
    hits: int = 0
    seen: int = 1            # sightings of an inferred correction

    @classmethod
    def from_dict(cls, d: dict) -> "Entry":
        return cls(
            term=str(d["term"]).strip(),
            aliases=tuple(str(a) for a in d.get("aliases", ()) if str(a).strip()),
            note=str(d.get("note", "")),
            scope=d.get("scope", SCOPE_ANY) if d.get("scope") in (SCOPE_ANY, SCOPE_MIXED) else SCOPE_ANY,
            source=str(d.get("source", "manual")),
            status=d.get("status", ACTIVE) if d.get("status") in (ACTIVE, PENDING) else ACTIVE,
            hits=int(d.get("hits", 0)),
            seen=int(d.get("seen", 1)),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["aliases"] = list(self.aliases)
        return d


@dataclass
class Vocabulary:
    path: str
    entries: list[Entry] = field(default_factory=list)

    def __post_init__(self):
        self._lock = threading.RLock()
        self._mtime = self._file_mtime()

    def _file_mtime(self) -> float | None:
        try:
            return os.stat(self.path).st_mtime
        except OSError:
            return None

    def reload_if_changed(self) -> bool:
        """Pick up edits made to the file outside the app (e.g. in a text editor)."""
        mtime = self._file_mtime()
        if mtime is None or mtime == self._mtime:
            return False
        fresh = Vocabulary.load(self.path)
        with self._lock:
            self.entries = fresh.entries
            self._mtime = self._file_mtime()
        logger.info("Vocabulary reloaded from %s (%d entries)", self.path, len(self.entries))
        return True

    # ── persistence ──────────────────────────────────────────
    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            entries = [Entry.from_dict(d) for d in raw.get("entries", []) if d.get("term")]
            logger.debug("Loaded %d vocabulary entries from %s", len(entries), path)
        except FileNotFoundError:
            entries = [replace(Entry.from_dict(d), source="seed") for d in SEED]
            logger.info("No vocabulary at %s — starting with %d seed terms", path, len(entries))
        except (OSError, ValueError, KeyError, TypeError) as e:
            entries = [replace(Entry.from_dict(d), source="seed") for d in SEED]
            try:  # keep the user's file for recovery; never overwrite it with the seed
                os.replace(path, path + ".bad")
            except OSError:
                pass
            logger.warning("Unreadable vocabulary %s (%s) — kept as .bad, using seed terms", path, e)
        return cls(path=path, entries=entries)

    def save(self) -> None:
        with self._lock:
            data = {"version": 1, "entries": [e.to_dict() for e in self.entries]}
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path) or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
            self._mtime = self._file_mtime()
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ── queries ──────────────────────────────────────────────
    def find(self, term: str) -> Entry | None:
        key = term.strip().lower()
        with self._lock:
            return next((e for e in self.entries if e.term.lower() == key), None)

    def active(self) -> list[Entry]:
        with self._lock:
            return [e for e in self.entries if e.status == ACTIVE]

    def pending(self) -> list[Entry]:
        with self._lock:
            return [e for e in self.entries if e.status == PENDING]

    def hotwords(self, limit: int = 50) -> list[str]:
        ranked = sorted(self.active(), key=lambda e: -e.hits)
        return [e.term for e in ranked[:limit]]

    # ── updates ──────────────────────────────────────────────
    def _put(self, new: Entry) -> Entry:
        for i, e in enumerate(self.entries):
            if e.term.lower() == new.term.lower():
                self.entries[i] = new
                return new
        self.entries.append(new)
        return new

    def add_correction(self, wrong: str, right: str, source: str = "voice") -> Entry | None:
        """Teach that `wrong` (as heard) should be `right`. Returns the entry or None."""
        wrong, right = wrong.strip(), right.strip()
        if not wrong or not right or wrong.lower() == right.lower():
            return None
        with self._lock:
            # A reverse correction replaces the old rule instead of fighting it
            for i, e in enumerate(self.entries):
                if right in e.aliases:
                    self.entries[i] = replace(e, aliases=tuple(a for a in e.aliases if a != right))
            old = self.find(wrong)
            if old is not None and old.term.lower() != right.lower():
                self.entries.remove(old)
            cur = self.find(right)
            inferred = source in ("learned", "suggested")
            if cur is None:
                cur = Entry(term=right, source=source,
                            status=PENDING if inferred else ACTIVE, seen=0)
            aliases = cur.aliases if wrong in cur.aliases else cur.aliases + (wrong,)
            seen = cur.seen + 1 if source == "learned" else cur.seen
            status = cur.status
            # "suggested" (AI, not sure) only becomes active through approval or retype evidence
            if not inferred or (source == "learned" and seen >= LEARN_THRESHOLD):
                status = ACTIVE
            entry = self._put(replace(cur, aliases=aliases, seen=seen, status=status))
        logger.info("Vocabulary %s: %r -> %r (%s, seen=%d)", entry.status, wrong, right, source, entry.seen)
        return entry

    def record_hits(self, terms) -> None:
        with self._lock:
            for t in terms:
                e = self.find(t)
                if e is not None:
                    self._put(replace(e, hits=e.hits + 1))

    def approve(self, term: str) -> bool:
        with self._lock:
            e = self.find(term)
            if e is None:
                return False
            self._put(replace(e, status=ACTIVE))
            return True

    def remove(self, term: str) -> bool:
        with self._lock:
            e = self.find(term)
            if e is None:
                return False
            self.entries.remove(e)
            return True
