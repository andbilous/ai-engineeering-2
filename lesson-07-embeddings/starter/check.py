"""Checks the lesson 7 practice. Usage: python starter/check.py [path/to/search.py]

Each check names the TODO it depends on. A check that crashes counts as failed.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "starter" / "search.py"

spec = importlib.util.spec_from_file_location("search", PATH)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

CHUNKS = s.load_chunks()
QUESTIONS = json.loads(s.QUESTIONS.read_text(encoding="utf-8"))
state = {}


def index():
    if "client" not in state:
        state["client"] = s.build_index(CHUNKS)
    return state["client"]


def bm25():
    if "bm25" not in state:
        state["bm25"] = s.BM25Index(CHUNKS)
    return state["bm25"]


def ids(hits):
    return [h["chunk_id"] for h in hits]


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5))


def check_embed():
    a, b, c = s.embed(["Can I work from home?", "Remote work is allowed three days per week.",
                       "Batteries are charged in the charging room."])
    return len(a) == 384 and cos(a, b) > cos(a, c)


def check_cache():
    s.embed(["Can I work from home?"])
    return s.embed.last_missing == 0 and s.CACHE.exists()


def check_index():
    client = index()
    info = client.count(s.COLLECTION)
    point = client.retrieve(s.COLLECTION, [0], with_payload=True)[0]
    return info.count == len(CHUNKS) and "chunk_id" in point.payload


def check_vector():
    hits = s.vector_search(index(), "How many days a week can I work from home?", ["all_employees"])
    return "policies/remote_work_policy_v2.docx#2" in ids(hits) and all("score" in h for h in hits)


def check_eval():
    fake = {"a": ["x", "r1", "y"], "b": ["r2", "x"], "c": ["x", "y", "z"]}
    qs = [{"id": "a", "question": "a", "groups": [], "relevant": ["r1"]},
          {"id": "b", "question": "b", "groups": [], "relevant": ["r2"]},
          {"id": "c", "question": "c", "groups": [], "relevant": ["r3"]}]
    r = s.evaluate(lambda q, g: [{"chunk_id": c} for c in fake[q]], qs, k=5)
    return abs(r["recall"] - 2 / 3) < 1e-9 and abs(r["mrr"] - 0.5) < 1e-9 and r["misses"] == ["c"]


def check_bm25():
    hits = bm25().search("agreement BV-2013-02", ["all_employees", "warehouse"])
    return bool(hits) and hits[0]["chunk_id"] == "operations/legacy_warehouse_manual_rev7.pdf#16"


def check_rrf():
    r = s.rrf([[{"chunk_id": "A"}, {"chunk_id": "B"}, {"chunk_id": "C"}], [{"chunk_id": "C"}, {"chunk_id": "D"}]])
    return ids(r)[:2] == ["C", "A"] and abs(r[0]["score"] - (1 / 63 + 1 / 61)) < 1e-9


def check_access():
    q = "What were the total operating costs in Q3?"
    hidden = all("finance" not in h["access"] for h in s.vector_search(index(), q, ["all_employees"]) +
                 bm25().search(q, ["all_employees"]))
    visible = "finance/q3_budget_summary.pdf#0" in ids(s.vector_search(index(), q, ["finance"]))
    return hidden and visible


def check_rerank():
    target = "operations/legacy_warehouse_manual_rev7.pdf#16"
    cands = [c for c in CHUNKS if c["chunk_id"] != target][:9] + [c for c in CHUNKS if c["chunk_id"] == target]
    return ids(s.rerank("agreement BV-2013-02 works council", cands))[0] == target


def check_hybrid():
    configs = s.make_configs(index(), bm25())
    vector = s.evaluate(configs["vector"], QUESTIONS)
    hybrid = s.evaluate(configs["hybrid"], QUESTIONS)
    state["line"] = f"vector {vector['recall']:.2f} | hybrid {hybrid['recall']:.2f}"
    return hybrid["recall"] >= 0.9 and hybrid["recall"] > vector["recall"]


def check_pairs():
    pairs = s.similar_pairs(CHUNKS)
    src = {c["chunk_id"]: c["source"] for c in CHUNKS}
    return len(pairs) == 5 and all(-1 <= p[0] <= 1 and src[p[1]] != src[p[2]] for p in pairs)


CHECKS = [
    ("TODO 1: embeddings put similar texts closer", check_embed),
    ("TODO 2: second run takes vectors from the cache", check_cache),
    ("TODO 3: one Qdrant point per chunk, chunk as payload", check_index),
    ("TODO 4: vector search finds the remote work rule", check_vector),
    ("TODO 5: recall@k and MRR computed correctly", check_eval),
    ("TODO 6: BM25 finds an exact document code", check_bm25),
    ("TODO 7: RRF fuses rankings by rank", check_rrf),
    ("TODO 8: finance chunk hidden from other groups", check_access),
    ("TODO 9: reranker moves the right chunk to the top", check_rerank),
    ("TODO 6-7: hybrid recall@5 above vector-only", check_hybrid),
    ("TODO 10 (optional): semantic near-duplicate pairs", check_pairs),
]


def main():
    passed = 0
    for name, fn in CHECKS:
        try:
            ok = bool(fn())
        except Exception as e:  # a TODO that is not done yet usually raises
            ok, name = False, f"{name}  [{type(e).__name__}: {str(e)[:60]}]"
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{passed}/{len(CHECKS)} checks passed" + (f"  ({state['line']})" if "line" in state else ""))


if __name__ == "__main__":
    main()
