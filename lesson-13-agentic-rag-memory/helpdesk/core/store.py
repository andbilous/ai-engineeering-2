"""Long-term memory and the helpdesk records in one SQLite file. Ready, do not change.

What makes it «enterprise»:
- isolation: every method takes the employee id and touches only that employee's rows; the API takes the id
  from the login header, never from the request body;
- audit: every write to memory is logged with who wrote it, from where, the old and the new value;
- forget: an employee can delete all their memory; the audit keeps the fact of deletion but not the values;
- tickets and approvals: access to systems is never granted by the agent, it creates a request for the manager.

Tables: facts(user, key, value, updated, source) · episodes · rules · audit · tickets · approvals.
Dates are ISO strings in local time, e.g. "2026-11-23T19:31:02".
"""
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMPLOYEES = {e["id"]: e for e in json.loads((ROOT / "data" / "employees.json").read_text(encoding="utf-8"))}
MAX_RULES = 5
MAX_EPISODES = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (user TEXT, key TEXT, value TEXT, updated TEXT, source TEXT, PRIMARY KEY (user, key));
CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY, user TEXT, session TEXT, summary TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS rules (id INTEGER PRIMARY KEY, user TEXT, rule TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY, ts TEXT, user TEXT, actor TEXT, action TEXT,
                                  key TEXT, old TEXT, new TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS tickets (id INTEGER PRIMARY KEY, user TEXT, summary TEXT, priority TEXT, details TEXT,
                                    status TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS approvals (id INTEGER PRIMARY KEY, user TEXT, manager TEXT, system TEXT, reason TEXT,
                                      status TEXT, created TEXT, decided TEXT);
"""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _q(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return [dict(r) for r in cur.fetchall()]

    def _audit(self, user, actor, action, key="", old=None, new=None, source=""):
        self._q("INSERT INTO audit (ts, user, actor, action, key, old, new, source) VALUES (?,?,?,?,?,?,?,?)",
                (now(), user, actor, action, key, old, new, source))

    # ------------------------------------------------------------ directory (read-only, from HR)
    @staticmethod
    def employee(user: str) -> dict:
        if user not in EMPLOYEES:
            raise KeyError(f"unknown employee {user!r}")
        return EMPLOYEES[user]

    # ------------------------------------------------------------ semantic memory
    def set_fact(self, user: str, key: str, value: str, actor: str = "agent", source: str = "") -> str:
        """The new value replaces the old one; the write goes to the audit with the old value."""
        old = self._q("SELECT value FROM facts WHERE user=? AND key=?", (user, key))
        old = old[0]["value"] if old else None
        self._q("INSERT INTO facts (user, key, value, updated, source) VALUES (?,?,?,?,?) "
                "ON CONFLICT(user, key) DO UPDATE SET value=excluded.value, updated=excluded.updated, source=excluded.source",
                (user, key, value, now(), source))
        self._audit(user, actor, "set_fact", key, old, value, source)
        return f"saved {key}={value}" + (f" (was {old})" if old and old != value else "")

    def profile(self, user: str) -> dict:
        return {r["key"]: r["value"] for r in self._q("SELECT key, value FROM facts WHERE user=? ORDER BY key", (user,))}

    def facts(self, user: str) -> list[dict]:
        return self._q("SELECT key, value, updated, source FROM facts WHERE user=? ORDER BY key", (user,))

    # ------------------------------------------------------------ episodic memory
    def add_episode(self, user: str, session: str, summary: str) -> None:
        self._q("INSERT INTO episodes (user, session, summary, created) VALUES (?,?,?,?)", (user, session, summary, now()))
        self._q("DELETE FROM episodes WHERE user=? AND id NOT IN (SELECT id FROM episodes WHERE user=? ORDER BY id DESC LIMIT ?)",
                (user, user, MAX_EPISODES))
        self._audit(user, "system", "add_episode", "", None, summary, session)

    def episodes(self, user: str, n: int = 3) -> list[dict]:
        rows = self._q("SELECT session, summary, created FROM episodes WHERE user=? ORDER BY id DESC LIMIT ?", (user, n))
        return rows[::-1]

    # ------------------------------------------------------------ procedural memory
    def add_rule(self, user: str, rule: str, source: str = "feedback") -> bool:
        if rule in self.rules(user):
            return False
        self._q("INSERT INTO rules (user, rule, created) VALUES (?,?,?)", (user, rule, now()))
        self._q("DELETE FROM rules WHERE user=? AND id NOT IN (SELECT id FROM rules WHERE user=? ORDER BY id DESC LIMIT ?)",
                (user, user, MAX_RULES))
        self._audit(user, "user", "add_rule", "", None, rule, source)
        return True

    def rules(self, user: str) -> list[str]:
        return [r["rule"] for r in self._q("SELECT rule FROM rules WHERE user=? ORDER BY id", (user,))]

    # ------------------------------------------------------------ audit and forget
    def audit(self, user: str, n: int = 50) -> list[dict]:
        return self._q("SELECT ts, actor, action, key, old, new, source FROM audit WHERE user=? ORDER BY id DESC LIMIT ?",
                       (user, n))

    def forget(self, user: str) -> dict:
        """Delete all memory of the employee. The audit keeps that it happened, the old values are wiped."""
        counts = {t: self._q(f"SELECT COUNT(*) AS n FROM {t} WHERE user=?", (user,))[0]["n"]
                  for t in ("facts", "episodes", "rules")}
        for t in ("facts", "episodes", "rules"):
            self._q(f"DELETE FROM {t} WHERE user=?", (user,))
        self._q("UPDATE audit SET old=NULL, new=NULL WHERE user=?", (user,))
        self._audit(user, "user", "forget", "", None, json.dumps(counts), "")
        return counts

    # ------------------------------------------------------------ tickets
    def create_ticket(self, user: str, summary: str, priority: str, details: str = "") -> dict:
        if priority not in ("P1", "P2", "P3", "P4"):
            raise ValueError("priority must be P1, P2, P3 or P4")
        self._q("INSERT INTO tickets (user, summary, priority, details, status, created) VALUES (?,?,?,?,?,?)",
                (user, summary[:200], priority, details[:1000], "open", now()))
        t = self._q("SELECT * FROM tickets WHERE user=? ORDER BY id DESC LIMIT 1", (user,))[0]
        return {**t, "number": f"IT-{1000 + t['id']}"}

    def tickets(self, user: str) -> list[dict]:
        return [{**t, "number": f"IT-{1000 + t['id']}"}
                for t in self._q("SELECT * FROM tickets WHERE user=? ORDER BY id DESC", (user,))]

    # ------------------------------------------------------------ approvals (human-in-the-loop)
    def request_access(self, user: str, system: str, reason: str) -> dict:
        manager = self.employee(user)["manager"]
        if not manager:
            raise ValueError("this employee has no manager in the directory: ask IT directly")
        self._q("INSERT INTO approvals (user, manager, system, reason, status, created) VALUES (?,?,?,?,?,?)",
                (user, manager, system[:60], reason[:300], "pending", now()))
        a = self._q("SELECT * FROM approvals WHERE user=? ORDER BY id DESC LIMIT 1", (user,))[0]
        self._audit(user, "agent", "request_access", system, None, "pending", f"approval {a['id']}")
        return a

    def my_requests(self, user: str) -> list[dict]:
        return self._q("SELECT * FROM approvals WHERE user=? ORDER BY id DESC", (user,))

    def to_approve(self, manager: str) -> list[dict]:
        return self._q("SELECT * FROM approvals WHERE manager=? ORDER BY status='pending' DESC, id DESC", (manager,))

    def decide(self, manager: str, approval_id: int, approve: bool) -> dict:
        rows = self._q("SELECT * FROM approvals WHERE id=? AND manager=?", (approval_id, manager))
        if not rows:
            raise PermissionError("not your approval")
        if rows[0]["status"] != "pending":
            raise ValueError("already decided")
        status = "approved" if approve else "rejected"
        self._q("UPDATE approvals SET status=?, decided=? WHERE id=?", (status, now(), approval_id))
        self._audit(rows[0]["user"], manager, "decide_access", rows[0]["system"], "pending", status, f"approval {approval_id}")
        return {**rows[0], "status": status}
