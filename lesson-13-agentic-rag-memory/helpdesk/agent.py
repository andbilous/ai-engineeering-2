"""IT-асистент компанії Northwind. Цей файл ви пишете на занятті: усе, що тут позначено TODO, — AI-частина.

Готове поруч і не змінюється:
    core/kb.py      пошук по 24 інструкціях IT з фільтром за ОС, kb.search(...) і kb.get(id)
    core/store.py   пам'ять у SQLite з ізоляцією, журналом і «забути мене», заявки, запити на доступ
    server.py       API на FastAPI, web/ — інтерфейс; сервер викликає лише три методи Agent нижче

Поки метод не написано, сервер показує в чаті «TODO …», а решта інтерфейсу працює.
Допоміжні функції (валідація фактів, бюджет контексту, критик тощо) — у todo.py.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
import anthropic

from core.kb import KB
from core.store import Store

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
MODEL = os.getenv("HELPDESK_MODEL", "claude-sonnet-5-5")


@dataclass
class Session:
    """Готово. Одна розмова одного співробітника: short-term пам'ять живе тут."""
    user: str
    id: str = field(default_factory=lambda: uuid4().hex[:8])
    history: list = field(default_factory=list)      # [{"role": "user"|"assistant", "content": str}, ...]
    summary: str = ""                                # переказ старішої частини розмови (compaction)
    turns: list = field(default_factory=list)        # трейси ходів, з них end_session пише епізод


def make_llm() -> anthropic.Anthropic | None:
    """Готово. Клієнт Claude API з ANTHROPIC_API_KEY з .env або None, якщо ключа немає."""
    return anthropic.Anthropic() if os.getenv("ANTHROPIC_API_KEY") else None


# ------------------------------------------------------------ TODO A: system prompt і описи інструментів
SYSTEM = """TODO A: роль асистента і правила (див. README, крок 1)."""

# TODO A: search_docs, get_doc, remember, create_ticket, request_access
# формат Claude API: {"name": ..., "description": ..., "input_schema": {"type": "object", "properties": {...}, "required": [...]}}
TOOLS: list[dict] = []


class Agent:
    def __init__(self, llm, store: Store, kb: KB, *, budget: int | None = 1500):
        self.llm, self.store, self.kb, self.budget = llm, store, kb, budget

    # ------------------------------------------------------------ TODO B: один хід агента
    def turn(self, s: Session, message: str) -> dict:
        """Один хід: повідомлення співробітника → відповідь. Сервер віддає результат у браузер як є.

        Що має бути всередині (кроки README 1–6): журнал → субагент, пам'ять, контекст у межах бюджету
        з compaction, цикл моделі з інструментами, критик, запис ходу в s.history і s.turns.

        Інтерфейс читає з результату такі ключі (будь-який можна пропустити, тоді блок не показується):
            answer: str                         текст відповіді
            calls: [{"step", "name", "args", "result"}]
            critic: [str]                       зауваження критика
            context: {"parts": {"system", "memory", "summary", "history", "question", "tools"}, "dropped": int}
            compacted: int, summary: str        скільки повідомлень переказано і сам переказ
            log: {"lines", "tokens_raw", "tokens_trimmed", "trimmed", "analysis"}
            tokens_in: int, cached_tokens: int, usd: float
                                                (з response.usage: input_tokens, cache_read_input_tokens)
            retrieved: [id], tickets: ["IT-1003"], access: ["CRM: pending"]   (для епізоду)
        """
        raise NotImplementedError("TODO B: Agent.turn")

    # ------------------------------------------------------------ TODO C: кінець сесії → епізод
    def end_session(self, s: Session) -> str | None:
        """Сервер викликає це на «Нова сесія». Запишіть епізод через self.store.add_episode і поверніть його текст."""
        raise NotImplementedError("TODO C: Agent.end_session")

    # ------------------------------------------------------------ TODO D: відгук → правило
    def feedback(self, user: str, rating: str, comment: str = "") -> dict:
        """👍 / 👎 з коментарем. Поверніть {"accepted": bool, "rule": str | None, "reason": str}."""
        raise NotImplementedError("TODO D: Agent.feedback")
