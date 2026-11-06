"""The RAG pipeline from lesson 8, ready to use. Lesson 9 does not change it, it measures it.

    llm, retriever = LLM(), Retriever()
    r = rag(llm, retriever, "How much can I spend on a hotel?", ["all_employees"], prompt="v2")
    r["found"], r["answer"], r["cited"]          # cited = chunk ids the answer relies on
    r["context"]                                  # {chunk_id: text} the model saw (groundedness needs it)
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from retrieval import Retriever, load_chunks  # noqa: E402

load_dotenv(ROOT / ".env")

PROMPTS = ROOT / "prompts"
OUT = ROOT / "output"

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-4o-mini"
PRICE_IN, PRICE_OUT = 0.15, 0.60          # USD per 1M tokens, check openrouter.ai/models before you rely on it
# Local model instead (lesson 3): ollama pull llama3.2:3b
# BASE_URL, API_KEY, MODEL, PRICE_IN, PRICE_OUT = "http://localhost:11434/v1", "ollama", "llama3.2:3b", 0, 0

K = 5                      # how many chunks retrieval returns
CONTEXT_BUDGET = 1500      # max tokens of sources we put into the prompt
PROMPT_VERSION = "v2"      # v1 = lesson 8 starter, v2 = after lesson 8, v3 = "more helpful" (lesson 9)
HELPERS = "v2"             # the condense prompt lives in prompts/v2
NOT_FOUND = "I could not find this in the documents available to you."

ENC = tiktoken.get_encoding("o200k_base")   # the tokenizer of gpt-4o / gpt-4o-mini


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    found: bool             # false when the sources do not contain the answer
    answer: str
    citations: list[str]    # source labels used in the answer, e.g. ["S1", "S3"]


ANSWER_FORMAT = {"type": "json_schema",
                 "json_schema": {"name": "answer", "strict": True, "schema": Answer.model_json_schema()}}


class LLM:
    """A thin wrapper over the chat API: one call, its tokens, cost and time. Ready to use."""

    def __init__(self):
        if not API_KEY:
            raise SystemExit("OPENROUTER_API_KEY is missing: copy .env.example to .env and paste your key")
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=60, max_retries=2)
        self.calls, self.tokens_in, self.tokens_out = 0, 0, 0

    def chat(self, messages: list[dict], schema: dict | None = None, max_tokens: int = 500) -> str:
        r = self.client.chat.completions.create(model=MODEL, messages=messages, temperature=0,
                                                max_tokens=max_tokens, **({"response_format": schema} if schema else {}))
        self.calls += 1
        if r.usage:
            self.tokens_in += r.usage.prompt_tokens
            self.tokens_out += r.usage.completion_tokens
        return r.choices[0].message.content or ""

    def cost(self) -> float:
        return (self.tokens_in * PRICE_IN + self.tokens_out * PRICE_OUT) / 1_000_000


def load_prompt(version: str, name: str = "system") -> str:
    for base in (PROMPTS, Path(__file__).resolve().parent / "prompts"):
        path = base / version / f"{name}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    raise SystemExit(f"Prompt not found: prompts/{version}/{name}.txt")


# ---------------------------------------------------------------- context

def build_context(hits: list[dict], budget: int = CONTEXT_BUDGET) -> tuple[str, dict]:
    """Return the context text and {label: chunk} for the chunks that fit into `budget` tokens."""
    parts, sources, used = [], {}, 0
    for hit in hits:
        label = f"S{len(sources) + 1}"
        date = f" · effective {hit['effective']}" if hit.get("effective") else ""
        block = f"[{label}] {hit['source']} · {hit['section'] or 'no section'}{date}\n{hit['text']}"
        size = count_tokens(block)
        if used + size > budget:
            continue
        parts.append(block)
        sources[label] = hit
        used += size
    return "\n\n".join(parts), sources


# ---------------------------------------------------------------- answer

def generate(llm: LLM, question: str, context: str, system: str) -> Answer:
    """Build messages (system prompt + sources + question), call llm.chat with ANSWER_FORMAT, parse."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"<sources>\n{context}\n</sources>\n\nQuestion: {question}"}]
    return Answer.model_validate_json(llm.chat(messages, schema=ANSWER_FORMAT))


def validate_citations(ans: Answer, sources: dict) -> tuple[Answer, list[str]]:
    """Normalize labels ("[S1]" -> "S1"), return the cleaned answer and the dropped (invalid) labels."""
    labels = [c.strip().strip("[]").strip() for c in ans.citations]
    valid = [c for c in dict.fromkeys(labels) if c in sources]
    invalid = [c for c in labels if c not in sources]
    found = ans.found and bool(valid)
    return Answer(found=found, answer=ans.answer, citations=valid if found else []), invalid


def condense(llm: LLM, history: list[list[str]], question: str) -> str:
    """One LLM call that returns the standalone question (plain text)."""
    if not history:
        return question
    dialog = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history)
    messages = [{"role": "system", "content": load_prompt(HELPERS, "condense")},
                {"role": "user", "content": f"<dialog>\n{dialog}\n</dialog>\n\nFollow-up: {question}"}]
    return llm.chat(messages, max_tokens=100).strip()


def rag(llm: LLM, retriever, question: str, groups: list[str], history: list | None = None,
        k: int = K, budget: int = CONTEXT_BUDGET, prompt: str = PROMPT_VERSION, use_condense: bool = True) -> dict:
    """Condense -> retrieve -> build_context -> generate -> validate_citations -> NOT_FOUND if needed."""
    query = condense(llm, history or [], question) if use_condense else question
    hits = retriever.search(query, groups, k)
    result = {"question": question, "query": query, "retrieved": [h["chunk_id"] for h in hits],
              "found": False, "answer": NOT_FOUND, "cited": [], "invalid": []}
    if not hits:
        return result
    context, sources = build_context(hits, budget)
    ans, invalid = validate_citations(generate(llm, query, context, load_prompt(prompt)), sources)
    result.update(found=ans.found, invalid=invalid, context_tokens=count_tokens(context),
                  in_context=[s["chunk_id"] for s in sources.values()],
                  context={s["chunk_id"]: s["text"] for s in sources.values()})
    if ans.found:
        result.update(answer=ans.answer, cited=[sources[c]["chunk_id"] for c in ans.citations])
    return result
