"""Hybrid search from lesson 7, ready to use: embeddings + Qdrant + BM25 + RRF + access filter.

You do not need to change this file. It is the lesson 7 solution packed into one class,
so that lesson 8 works even if your lesson 7 practice is not finished.

    r = Retriever()
    hits = r.search("How many days a week can I work from home?", ["all_employees"], k=5)
    hits[0]["chunk_id"], hits[0]["score"], hits[0]["cosine"]
"""
import hashlib
import json
import re
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent
CHUNKS = ROOT / "data" / "chunks.jsonl"
OUT = ROOT / "output"
CACHE = OUT / "embeddings_cache.json"
MODELS = ROOT / ".models"

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"   # 384 dims, uk + en
COLLECTION = "northwind"
CANDIDATES = 20
RRF_K = 60


def load_chunks() -> list[dict]:
    return [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines()]


def text_for_embedding(chunk: dict) -> str:
    return f"{chunk['section']}\n{chunk['text']}" if chunk["section"] else chunk["text"]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self, chunks: list[dict] | None = None):
        self.chunks = chunks or load_chunks()
        self._embedder = None
        vectors = self.embed([text_for_embedding(c) for c in self.chunks])
        self.client = QdrantClient(":memory:")
        self.client.create_collection(COLLECTION, vectors_config=models.VectorParams(
            size=len(vectors[0]), distance=models.Distance.COSINE))
        self.client.upsert(COLLECTION, points=[models.PointStruct(id=i, vector=v, payload=c)
                                               for i, (c, v) in enumerate(zip(self.chunks, vectors))])
        self.bm25 = BM25Okapi([tokenize(text_for_embedding(c)) for c in self.chunks])

    def embed(self, texts: list[str]) -> list[list[float]]:
        cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        keys = [hashlib.sha256(f"{EMBED_MODEL}\n{t}".encode()).hexdigest() for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in cache]
        if missing:
            if self._embedder is None:
                self._embedder = TextEmbedding(EMBED_MODEL, cache_dir=str(MODELS))
            for (k, _), vec in zip(missing, self._embedder.embed([t for _, t in missing])):
                cache[k] = [round(float(x), 6) for x in vec]
            OUT.mkdir(exist_ok=True)
            CACHE.write_text(json.dumps(cache))
        return [cache[k] for k in keys]

    def vector_search(self, query: str, groups: list[str], k: int) -> list[dict]:
        flt = models.Filter(must=[models.FieldCondition(key="access", match=models.MatchAny(any=groups))])
        hits = self.client.query_points(COLLECTION, query=self.embed([query])[0], limit=k, query_filter=flt).points
        return [{**h.payload, "cosine": h.score} for h in hits]

    def bm25_search(self, query: str, groups: list[str], k: int) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])
        return [self.chunks[i] for i in order
                if scores[i] > 0 and set(self.chunks[i]["access"]) & set(groups)][:k]

    def search(self, query: str, groups: list[str], k: int = 5) -> list[dict]:
        """Hybrid search: vector + BM25 merged with RRF. Every hit has "score" (RRF) and "cosine" (vector
        similarity to the query, 0.0 when only BM25 found it)."""
        vec = self.vector_search(query, groups, CANDIDATES)
        cos = {h["chunk_id"]: h["cosine"] for h in vec}
        scores, by_id = {}, {}
        for ranking in (vec, self.bm25_search(query, groups, CANDIDATES)):
            for rank, hit in enumerate(ranking, 1):
                cid = hit["chunk_id"]
                scores[cid] = scores.get(cid, 0.0) + 1 / (RRF_K + rank)
                by_id.setdefault(cid, hit)
        best = sorted(scores, key=lambda c: -scores[c])[:k]
        return [{**by_id[c], "score": scores[c], "cosine": cos.get(c, 0.0)} for c in best]
