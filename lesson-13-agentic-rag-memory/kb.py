"""Knowledge base of the e-bike maker «Вольт»: 20 documents and a small BM25 search. You do not need to change it.

The search is deliberately simple (words cut to their first 5 letters instead of a real Ukrainian stemmer), so that
the lesson is about who decides what to search, not about the quality of embeddings.

    from kb import KB
    kb = KB()
    kb.search("що означає E3", model="V2", k=3)   # V2 documents and the common ones ("all"), best first
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

DOCS = Path(__file__).resolve().parent / "data" / "docs.jsonl"
MODELS = ("V1", "V2", "V3")
STOP = set("що як це на у в і й та з із до від для по за чи не а але мій моя моє мої мені мене я ви ваш "
           "можна треба є був буде якщо коли де який яка яке які".split())


def tokenize(text: str) -> list[str]:
    text = re.sub(r"\b[еЕ](\d)\b", r"e\1", text)          # Cyrillic «Е3» typed by a user = Latin «E3» in the docs
    return [w[:5] for w in re.findall(r"\w+", text.lower()) if w not in STOP]


class KB:
    def __init__(self, path: Path = DOCS, k1: float = 1.5, b: float = 0.75):
        self.docs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.by_id = {d["id"]: d for d in self.docs}
        self.toks = [tokenize(f'{d["title"]} {d["title"]} {d["text"]}') for d in self.docs]
        self.avg = sum(map(len, self.toks)) / len(self.toks)
        df = Counter(t for toks in self.toks for t in set(toks))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def score(self, q: list[str], i: int) -> float:
        tf, dl, s = Counter(self.toks[i]), len(self.toks[i]), 0.0
        for t in q:
            if t in tf:
                s += self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avg))
        return s

    def search(self, query: str, model: str | None = None, k: int = 4) -> list[dict]:
        """Top-k documents for the query. model="V2" keeps V2 documents and the common ones; None searches everything."""
        if model is not None and model not in MODELS:
            raise ValueError(f"model must be one of {MODELS} or None, got {model!r}")
        q = tokenize(query)
        hits = []
        for i, d in enumerate(self.docs):
            if model and d["model"] not in (model, "all"):
                continue
            s = self.score(q, i)
            if s > 0:
                hits.append({**d, "score": round(s, 3)})
        return sorted(hits, key=lambda h: -h["score"])[:k]
