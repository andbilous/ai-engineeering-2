"""Knowledge base of the IT department of Northwind: 24 internal instructions and a small BM25 search. Ready, do not change.

Instructions differ by operating system (VPN, software), so search takes an `os` filter:
    kb.search("сертифікат vpn", os="mac")   # mac documents and the common ones ("all"), best first
Search returns short snippets on purpose (just-in-time context): the full text comes from kb.get(doc_id).
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "data" / "docs.jsonl"
OS_LIST = ("mac", "windows", "linux")
SNIPPET = 160
STOP = set("що як це на у в і й та з із до від для по за чи не а але мій моя моє мої мені мене я ви ваш "
           "можна треба є був буде якщо коли де який яка яке які".split())


def tokenize(text: str) -> list[str]:
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

    def _score(self, q: list[str], i: int) -> float:
        tf, dl, s = Counter(self.toks[i]), len(self.toks[i]), 0.0
        for t in q:
            if t in tf:
                s += self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avg))
        return s

    def search(self, query: str, os: str | None = None, k: int = 4) -> list[dict]:
        """Top-k {id, title, os, snippet, score}. os="mac" keeps mac documents and the common ones."""
        if os is not None and os not in OS_LIST:
            raise ValueError(f"os must be one of {OS_LIST} or empty, got {os!r}")
        q = tokenize(query)
        hits = []
        for i, d in enumerate(self.docs):
            if os and d["os"] not in (os, "all"):
                continue
            s = self._score(q, i)
            if s > 0:
                snip = d["text"] if len(d["text"]) <= SNIPPET else d["text"][:SNIPPET].rsplit(" ", 1)[0] + " …"
                hits.append({"id": d["id"], "title": d["title"], "os": d["os"], "snippet": snip, "score": round(s, 3)})
        return sorted(hits, key=lambda h: -h["score"])[:k]

    def get(self, doc_id: str) -> dict:
        if doc_id not in self.by_id:
            raise ValueError(f"no document {doc_id!r}")
        return self.by_id[doc_id]
