"""Lesson 7 practice (starter): vector, BM25, hybrid and reranked search over the lesson 6 chunks.

Usage:
    python starter/search.py                  # evaluate all configurations on data/questions.json
    python starter/search.py "vpn backup"     # show the top results of every configuration for one query
    python starter/search.py --groups finance "operating costs"
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "chunks.jsonl"
QUESTIONS = ROOT / "data" / "questions.json"
OUT = ROOT / "output"
CACHE = OUT / "embeddings_cache.json"
MODELS = ROOT / ".models"   # downloaded once, ~300 MB

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"   # 384 dims, ~220 MB, uk + en
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"                                # ~80 MB cross-encoder
COLLECTION = "northwind"
K = 5                # how many results we evaluate (recall@K)
CANDIDATES = 20      # how many hybrid candidates go to the reranker
RRF_K = 60

_embedder = None
_reranker = None


def load_chunks() -> list[dict]:
    return [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines()]


def text_for_embedding(chunk: dict) -> str:
    """Contextual header: the section path in front of the text (lesson 6, chunking)."""
    return f"{chunk['section']}\n{chunk['text']}" if chunk["section"] else chunk["text"]


# ---------------------------------------------------------------- embeddings

# TODO 1-2 · Goal: turn every chunk into a vector, and do it only once.
#   Why: search compares vectors, not words; the model is slow, so unchanged chunks must come from the cache.
#   Done when: similar texts get closer vectors (check 1) and a second run prints "embedded now 0" (check 2).
def embed(texts: list[str]) -> list[list[float]]:
    """TODO 1: embed texts with EMBED_MODEL.
    TODO 2: cache vectors in output/embeddings_cache.json by sha256(model + text): a re-run embeds nothing."""
    # Hint: TextEmbedding(EMBED_MODEL, cache_dir=str(MODELS)).embed(texts) yields numpy arrays; store them as lists of floats.
    # Set embed.last_missing = number of texts that were not in the cache.
    raise NotImplementedError("TODO 1-2: embed()")


embed.last_missing = 0


# ---------------------------------------------------------------- index

# TODO 3 · Goal: put all chunk vectors into Qdrant together with their metadata.
#   Why: the vector database finds the nearest vectors fast and returns the chunk (source, section, access) with them.
#   Done when: the collection has 69 points and each point carries its chunk as payload (check 3).
def build_index(chunks: list[dict]) -> QdrantClient:
    """TODO 3: create a Qdrant collection (cosine) and upsert one point per chunk with the chunk as payload."""
    # Hint: QdrantClient(":memory:"), create_collection(..., VectorParams(size, Distance.COSINE)), upsert(PointStruct(...))
    raise NotImplementedError("TODO 3: build_index()")


# TODO 8 · Goal: a user only gets chunks of the groups they belong to.
#   Why: access rules from lesson 6 must work inside the search; a prompt "do not show finance data" is not protection.
#   Done when: the Q3 budget is hidden from all_employees but found for finance (check 8, "access filter: OK").
def access_filter(groups: list[str]) -> models.Filter:
    """TODO 8: only chunks whose `access` shares at least one group with the user."""
    return None   # no filter yet: every user sees every chunk


def allowed(chunk: dict, groups: list[str]) -> bool:
    return bool(set(chunk["access"]) & set(groups))


# ---------------------------------------------------------------- search

# TODO 4 · Goal: semantic search: the query becomes a vector, Qdrant returns the k nearest chunks.
#   Why: it finds the answer even when the user's words differ from the document (another language included).
#   Done when: "How many days a week can I work from home?" finds the remote work rule (check 4).
def vector_search(client: QdrantClient, query: str, groups: list[str], k: int = K) -> list[dict]:
    """TODO 4: embed the query with the same model and return the top-k chunks (with "score")."""
    # Hint: client.query_points(COLLECTION, query=<vector>, limit=k, query_filter=access_filter(groups)).points
    raise NotImplementedError("TODO 4: vector_search()")


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# TODO 6 · Goal: classic keyword search (BM25) over the same chunks.
#   Why: embeddings miss exact codes and names like "BV-2013-02"; BM25 finds them at once.
#   Done when: "agreement BV-2013-02" returns the manual chunk 6.3 first (check 6).
class BM25Index:
    def __init__(self, chunks: list[dict]):
        """TODO 6: build BM25 over the tokenized texts (section path included, like for embeddings)."""
        self.chunks = chunks
        self.bm25 = None

    def search(self, query: str, groups: list[str], k: int = K) -> list[dict]:
        # Hint: self.bm25.get_scores(tokenize(query)); skip zero scores and chunks the user may not read (allowed())
        return []


# TODO 7 · Goal: merge the vector and BM25 rankings into one list.
#   Why: their scores are on different scales (0.36 vs 13.0), so we combine ranks, not scores.
#   Done when: a chunk found by both searches beats one found by a single search (check 7),
#   and hybrid recall@5 is higher than vector-only (check "TODO 6-7").
def rrf(rankings: list[list[dict]], k: int = K) -> list[dict]:
    """TODO 7: Reciprocal Rank Fusion: score = sum of 1 / (RRF_K + rank), rank starts at 1."""
    return rankings[0][:k] if rankings else []   # for now: just the first ranking


# TODO 9 · Goal: re-order the top-20 hybrid candidates with a cross-encoder.
#   Why: it reads the query and the chunk together, so the right chunk moves to the top (higher MRR).
#   Done when: the right chunk goes from last to first place (check 9). Then compare MRR and ms/query in the table.
def rerank(query: str, candidates: list[dict], k: int = K) -> list[dict]:
    """TODO 9: score every (query, chunk) pair with the cross-encoder and keep the top-k."""
    # Hint: TextCrossEncoder(RERANK_MODEL, cache_dir=str(MODELS)).rerank(query, texts) yields one score per text
    return candidates[:k]   # for now: no reranking


def make_configs(client: QdrantClient, bm25: BM25Index) -> dict:
    return {
        "vector": lambda q, g: vector_search(client, q, g),
        "bm25": lambda q, g: bm25.search(q, g),
        "hybrid": lambda q, g: rrf([vector_search(client, q, g, CANDIDATES), bm25.search(q, g, CANDIDATES)]),
        "hybrid+rerank": lambda q, g: rerank(q, rrf([vector_search(client, q, g, CANDIDATES),
                                                     bm25.search(q, g, CANDIDATES)], CANDIDATES)),
    }


# ---------------------------------------------------------------- evaluation

# TODO 5 · Goal: measure the search with numbers instead of "looks fine".
#   Why: without recall@k and MRR we cannot tell whether hybrid or the reranker actually helps.
#   Done when: the fake test gives recall 0.67 and MRR 0.50 (check 5) and the table shows real numbers per config.
def evaluate(search, questions: list[dict], k: int = K) -> dict:
    """TODO 5: recall@k (a relevant chunk is in the top-k) and MRR (1 / rank of the first relevant chunk)."""
    return {"recall": 0.0, "mrr": 0.0, "misses": [q["id"] for q in questions], "ms": 0.0}


def check_access(client: QdrantClient, bm25: BM25Index) -> bool:
    """A user without the finance group must never get the Q3 budget, whatever they ask."""
    q, groups = "What were the total operating costs in Q3?", ["all_employees"]
    hits = vector_search(client, q, groups) + bm25.search(q, groups)
    return all("finance" not in h["access"] for h in hits)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5))


# TODO 10 (optional) · Goal: find chunks from different documents that say almost the same.
#   Why: embeddings catch duplicates in other words, which the shingles of lesson 6 miss.
#   Done when: --pairs prints 5 pairs with cosine between -1 and 1 (check 10).
def similar_pairs(chunks: list[dict], top: int = 5) -> list[tuple]:
    """TODO 10 (optional): the most similar chunk pairs from different documents (semantic near-duplicates).
    This model does not normalize its vectors, so a plain dot product is not a cosine."""
    return []


# ---------------------------------------------------------------- main

def show(configs: dict, query: str, groups: list[str]):
    for name, search in configs.items():
        print(f"\n== {name}")
        for h in search(query, groups):
            print(f"  {h['score']:8.4f}  {h['chunk_id']:<50} {h['section'][:45]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?")
    parser.add_argument("--groups", default="all_employees,warehouse")
    parser.add_argument("--pairs", action="store_true", help="show semantic near-duplicates (TODO 10)")
    args = parser.parse_args()

    chunks = load_chunks()
    client = build_index(chunks)
    bm25 = BM25Index(chunks)
    configs = make_configs(client, bm25)
    print(f"chunks {len(chunks)} | embedded now {embed.last_missing} (rest from cache) | model {EMBED_MODEL}")

    if args.pairs:
        for score, a, b in similar_pairs(chunks):
            print(f"  {score:.3f}  {a}  ~  {b}")
        return
    if args.query:
        show(configs, args.query, args.groups.split(","))
        return

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    print(f"\n{'config':<15}{'recall@' + str(K):>10}{'MRR':>8}{'ms/query':>10}   misses")
    results = {}
    for name, search in configs.items():
        r = evaluate(search, questions)
        results[name] = r
        print(f"{name:<15}{r['recall']:>10.2f}{r['mrr']:>8.2f}{r['ms']:>10.0f}   {' '.join(r['misses']) or '-'}")
    access_ok = check_access(client, bm25)
    print("\naccess filter: " + ("OK, finance chunk hidden from all_employees" if access_ok
                                   else "LEAK, all_employees can see the finance chunk (TODO 8)"))
    line = " | ".join(f"{n} {r['recall']:.2f}/{r['mrr']:.2f}" for n, r in results.items())
    print(f"\nFOR CHAT: recall@{K}/MRR on {len(questions)} questions | {line}")
    (OUT / "eval.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except NotImplementedError as e:
        sys.exit(f"Not implemented yet: {e}")
