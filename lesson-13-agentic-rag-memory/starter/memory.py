"""Long-term memory (TODO 1, TODO 5) of the support agent: facts about the user, rules from feedback, past sessions.

One JSON file for all users (output/memory.json), or nothing on disk when path is None (tests, eval).
    {"u1": {"facts": {"bike_model": {"value": "V2", "updated": "2026-11-20T19:31:02"}},
            "feedback": ["Відповідай до трьох речень."],
            "episodes": [{"session": 1, "summary": "питання: ...; документи: v2-errors"}]}}
"""
import json
import re
from datetime import datetime
from pathlib import Path

FACT_KEYS = ("bike_model", "city", "preference")
MAX_VALUE = 80          # a fact is a short value, not a paragraph of instructions
MAX_FEEDBACK = 5
MAX_EPISODES = 20


class MemoryStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path and self.path.exists() else {}

    def user(self, user_id: str) -> dict:
        return self.data.setdefault(user_id, {"facts": {}, "feedback": [], "episodes": []})

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")

    # TODO 1 · Goal: long-term facts about the user. remember(user_id, key, value) saves one fact and returns a short
    #   text for the model ("saved city=Одеса (was Київ)"); profile(user_id) returns {key: value}.
    #   Why: the user said "у мене V2 Trek" last week; today they ask "що означає E3?" and the answer depends on the model.
    #   The model writes memory through a tool, so memory must reject what does not belong there.
    #   Done when: key not in FACT_KEYS -> ValueError; empty value or longer than MAX_VALUE -> ValueError;
    #   bike_model is normalised to "V1" | "V2" | "V3" ("V2 Trek" -> "V2", "V7" -> ValueError); the newest value replaces
    #   the old one and is stored as {"value": ..., "updated": ISO time}; call self.save() (checks 1 and 6).
    def remember(self, user_id: str, key: str, value: str) -> str:
        raise NotImplementedError("TODO 1")

    def profile(self, user_id: str) -> dict:
        raise NotImplementedError("TODO 1")

    # TODO 5 · Goal: record_feedback(user_id, rating, comment) turns a thumbs-down with a comment into a rule for future
    #   answers (self.user(user_id)["feedback"], shown in the context by memory_block). Returns True if a rule was stored.
    #   Why: "занадто довго" after one answer should change the next answers, not only a dashboard.
    #   Done when: rating not "up"/"down" -> ValueError; "up" or an empty comment stores nothing (False);
    #   the same comment twice is stored once; only the newest MAX_FEEDBACK rules are kept (check 5).
    def record_feedback(self, user_id: str, rating: str, comment: str = "") -> bool:
        raise NotImplementedError("TODO 5")

    # ------------------------------------------------------------ episodes (ready)
    def add_episode(self, user_id: str, session: int, summary: str) -> None:
        eps = self.user(user_id)["episodes"]
        eps.append({"session": session, "summary": summary})
        del eps[:-MAX_EPISODES]
        self.save()

    def recent_episodes(self, user_id: str, n: int = 3) -> list[dict]:
        return self.user(user_id)["episodes"][-n:]

    def feedback_rules(self, user_id: str) -> list[str]:
        return list(self.user(user_id)["feedback"])
