"""Lesson 8 practice: RAG over the Northwind corpus with sources, "not found" and a context budget.

Usage:
    python starter/rag.py                         # answer all questions in data/questions.json and score them
    python starter/rag.py "How much can I spend on a hotel?"     # one question, answer with sources
    python starter/rag.py --groups finance "operating costs in Q3"
    python starter/rag.py --prompt v1             # run with another prompt version (TODO 6)
    python starter/rag.py --sweep                 # k = 3 / 5 / 10: quality, tokens, cost, latency (TODO 7)
    python starter/rag.py --summary               # top-5 vs whole document vs map-reduce (TODO 8)
    python starter/rag.py --no-condense           # follow-up questions without rewriting (TODO 9)
"""
import argparse
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from retrieval import Retriever, load_chunks  # noqa: E402

load_dotenv(ROOT / ".env")

QUESTIONS = ROOT / "data" / "questions.json"
SUMMARY_TASK = ROOT / "data" / "summary_task.json"
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
MAP_BUDGET = 1500          # tokens of document text per map call (TODO 8)
PROMPT_VERSION = "v1"      # TODO 6: write prompts/v2/system.txt, then switch to "v2"
HELPERS = "v2"             # condense / map / reduce prompts are ready in prompts/v2
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

# TODO 1 · Goal: turn the retrieved chunks into numbered sources that fit a token budget.
#   Why: the model can only cite what has a label, and every token of context costs money and latency.
#   Done when: sources are labelled S1, S2, ... with document, section and date, and the total stays
#   within the budget: a chunk that does not fit is skipped, a smaller one after it may still go in (check 1).
def build_context(hits: list[dict], budget: int = CONTEXT_BUDGET) -> tuple[str, dict]:
    """TODO 1: return the context text and {label: chunk} for the chunks that fit into `budget` tokens."""
    raise NotImplementedError("TODO 1: build_context")


# ---------------------------------------------------------------- answer

# TODO 2 (ready: read it, no code to write) · Goal: ask the model to answer only from the sources and return JSON {found, answer, citations}.
#   Why: structured output (lesson 4) lets the code check citations and "not found" instead of parsing prose.
#   Done when: the user message holds the sources and the question, the call uses ANSWER_FORMAT,
#   and the reply is parsed into Answer (check 2).
def generate(llm: LLM, question: str, context: str, system: str) -> Answer:
    """TODO 2: build messages (system prompt + sources + question), call llm.chat with ANSWER_FORMAT, parse."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"<sources>\n{context}\n</sources>\n\nQuestion: {question}"}]
    return Answer.model_validate_json(llm.chat(messages, schema=ANSWER_FORMAT))


# TODO 3 · Goal: keep only citations that point to sources we actually gave the model.
#   Why: a model can invent "S7" when there were five sources, write "[S3]" instead of "S3",
#   or answer without citing anything. An answer that cites nothing real is not grounded, so it becomes "not found".
#   Done when: "[S1]" counts as S1, unknown labels are dropped,
#   and found=True with no valid citation becomes found=False (check 3).
def validate_citations(ans: Answer, sources: dict) -> tuple[Answer, list[str]]:
    """TODO 3: normalize labels ("[S1]" -> "S1"), return the cleaned answer and the dropped (invalid) labels."""
    raise NotImplementedError("TODO 3: validate_citations")


# TODO 9 · Goal: rewrite a follow-up ("what is their e-mail?") into a question that stands on its own.
#   Why: retrieval sees only the current message; "their" finds nothing without the previous turn.
#   Done when: with history the model rewrites the question before the search, without history
#   there is no extra call (check 9). Compare follow-ups with and without --no-condense.
def condense(llm: LLM, history: list[list[str]], question: str) -> str:
    """TODO 9: one LLM call that returns the standalone question (plain text)."""
    return question   # TODO 9: replace with the rewriting call; until then follow-ups go to search as they are


# TODO 4 · Goal: put the steps together and say "not found" instead of guessing.
#   Why: when the documents do not contain the answer, a made-up answer is worse than no answer.
#   Done when: no hits -> NOT_FOUND without calling the model; found=False -> NOT_FOUND and no sources;
#   otherwise the answer with the cited chunks (check 4).
def rag(llm: LLM, retriever, question: str, groups: list[str], history: list | None = None,
        k: int = K, budget: int = CONTEXT_BUDGET, prompt: str = PROMPT_VERSION, use_condense: bool = True) -> dict:
    """TODO 4: condense -> retrieve -> build_context -> generate -> validate_citations -> NOT_FOUND if needed."""
    raise NotImplementedError("TODO 4: rag")


# ---------------------------------------------------------------- evaluation

def has(text: str, options: list[str]) -> bool:
    return any(o.lower() in text.lower() for o in options)


# TODO 5 · Goal: score the answers with simple deterministic checks.
#   Why: "the answer looks right" is not a metric; lesson 9 builds full evals on top of these numbers.
#   retrieved and in_context split the blame: was the right chunk found, did it reach the model?
#   Done when: the fake test in check 5 gives retrieved 0.50, in_context 0.50, correct 0.50, cited 0.50,
#   abstain 1.00, false_abstain 0.50.
def evaluate(results: list[dict], questions: list[dict]) -> dict:
    """TODO 5: for answerable questions: retrieved (a relevant chunk among the retrieved ones), in_context (a relevant
    chunk made it into the context), correct (found and all `must` facts in the answer), cited (a relevant chunk among
    the cited ones), false_abstain (said NOT_FOUND). For no_answer questions: abstain (said NOT_FOUND)."""
    raise NotImplementedError("TODO 5: evaluate")


def run_all(llm: LLM, retriever, questions: list[dict], verbose: bool = True, **kw) -> tuple[list, dict]:
    results, calls0, tin0, tout0 = [], llm.calls, llm.tokens_in, llm.tokens_out
    start = time.perf_counter()
    for q in questions:
        r = rag(llm, retriever, q["question"], q["groups"], q.get("history"), **kw)
        results.append(r)
        if verbose:
            mark = "+" if r["found"] else "-"
            print(f"  {q['id']} {q['kind']:<12} {mark} {r['answer'][:70]!r:<74} {', '.join(r['cited'])[:60]}")
    n = len(questions)
    m = evaluate(results, questions)
    m.update(tokens_in=(llm.tokens_in - tin0) / n, calls=(llm.calls - calls0) / n,
             usd_per_1000=((llm.tokens_in - tin0) * PRICE_IN + (llm.tokens_out - tout0) * PRICE_OUT) / n * 1000 / 1e6,
             ms=(time.perf_counter() - start) * 1000 / n)
    return results, m


# ---------------------------------------------------------------- budget sweep

# TODO 7 · Goal: measure what more context buys you: k = 3, 5, 10 on the same questions.
#   Why: more chunks raise the chance the answer is in context, but cost tokens, money and latency,
#   and can add noise. The right k is the one your numbers support.
#   Done when: --sweep prints one row per k with correct, cited, abstain, tokens, $/1000 and ms (check 7).
def sweep(llm: LLM, retriever, questions: list[dict], ks=(3, 5, 10)) -> list[dict]:
    """TODO 7: run_all for every k with a budget big enough for k chunks; return one row per k."""
    raise NotImplementedError("TODO 7: sweep")


# ---------------------------------------------------------------- more data than the window

def windows(chunks: list[dict], budget: int) -> list[list[dict]]:
    out, cur, used = [], [], 0
    for c in chunks:
        size = count_tokens(c["text"])
        if cur and used + size > budget:
            out.append(cur)
            cur, used = [], 0
        cur.append(c)
        used += size
    return out + ([cur] if cur else [])


# TODO 8 · Goal: answer a question that needs a whole document, not 5 chunks: map-reduce.
#   Why: "list every temperature rule in the manual" is spread over many sections; top-k sees a few of them.
#   Map: each window of the document -> the relevant facts with [chunk_id], or NONE.
#   Reduce: one call merges the facts into the final answer.
#   Done when: one map call per window, NONE results are dropped, one reduce call (check 8);
#   --summary: map-reduce finds more of the 8 facts than top-5 AND than the whole document in one call.
def map_reduce(llm: LLM, question: str, chunks: list[dict], budget: int = MAP_BUDGET) -> str:
    """TODO 8: map over windows(chunks, budget) with prompts map.txt, then reduce with reduce.txt."""
    raise NotImplementedError("TODO 8: map_reduce")


def whole_document(llm: LLM, question: str, chunks: list[dict]) -> str:
    """Ready, not a TODO: the whole document in ONE call with the same map prompt. The baseline map-reduce must beat:
    the manual is ~4.5k tokens and fits the window, so why split it at all? Run --summary and compare."""
    text = "\n\n".join(f"[{c['chunk_id']}] {c['section']}\n{c['text']}" for c in chunks)
    return llm.chat([{"role": "system", "content": load_prompt(HELPERS, "map")},
                     {"role": "user", "content": f"<document>\n{text}\n</document>\n\nQuestion: {question}"}],
                    max_tokens=1500).strip()


SUMMARY_RUNS = 3   # the model is not stable on long lists even at temperature 0, so each method runs 3 times


def summary_demo(llm: LLM, retriever):
    task = json.loads(SUMMARY_TASK.read_text(encoding="utf-8"))
    doc = [c for c in load_chunks() if c["source"] == task["source"]]
    q, facts = task["question"], task["facts"]
    print(f"Question: {q}\nDocument: {task['source']}, {len(doc)} chunks, "
          f"{sum(count_tokens(c['text']) for c in doc)} tokens, {len(windows(doc, MAP_BUDGET))} map windows\n")
    methods = [("top-5", lambda: rag(llm, retriever, q, task["groups"])["answer"]),
               ("whole doc", lambda: whole_document(llm, q, doc)),
               ("map-reduce", lambda: map_reduce(llm, q, doc))]
    rows = {}
    for name, fn in methods:
        runs = []
        for _ in range(SUMMARY_RUNS):
            c0, t0, o0, s = llm.calls, llm.tokens_in, llm.tokens_out, time.perf_counter()
            try:
                text = fn()
            except NotImplementedError as e:
                print(f"== {name}: {e}\n")
                break
            runs.append({"hit": [i for i, f in enumerate(facts, 1) if has(text, f)], "calls": llm.calls - c0,
                         "tokens_in": llm.tokens_in - t0, "s": time.perf_counter() - s, "answer": text,
                         "usd": ((llm.tokens_in - t0) * PRICE_IN + (llm.tokens_out - o0) * PRICE_OUT) / 1e6})
        if runs:
            rows[name] = runs
            print(f"== {name} (run 1 of {len(runs)})\n{runs[0]['answer']}\n")
    if not rows:
        return
    names = list(rows)
    print(f"{'fact (source chunk)':<44}" + "".join(f"{n:>12}" for n in names))
    for i, label in enumerate(task["fact_sources"], 1):
        print(f"{label[:43]:<44}" + "".join(f"{sum(i in r['hit'] for r in rows[n])}/{len(rows[n])}".rjust(12)
                                            for n in names))
    avg = lambda n, key: sum(r[key] for r in rows[n]) / len(rows[n])
    print(f"{'facts found, per run':<44}" + "".join(
        (" ".join(str(len(r["hit"])) for r in rows[n]) + f" of {len(facts)}").rjust(12) for n in names))
    for label, key, fmt in [("calls", "calls", "{:.0f}"), ("tokens in", "tokens_in", "{:.0f}"),
                            ("$ per question", "usd", "{:.5f}"), ("seconds", "s", "{:.1f}")]:
        print(f"{label + ', average':<44}" + "".join(fmt.format(avg(n, key)).rjust(12) for n in names))
    OUT.mkdir(exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------- main

def print_metrics(name: str, m: dict):
    print(f"{name:<10}{m['retrieved']:>7.2f}{m['in_context']:>7.2f}{m['correct']:>9.2f}{m['cited']:>8.2f}"
          f"{m['abstain']:>9.2f}{m['false_abstain']:>8.2f}{m['tokens_in']:>11.0f}{m['usd_per_1000']:>10.3f}{m['ms']:>8.0f}"
          f"   {' '.join(m['failed']) or '-'}")


HEADER = (f"{'':<10}{'retr':>7}{'ctx':>7}{'correct':>9}{'cited':>8}{'abstain':>9}{'f.abst':>8}{'tokens in':>11}"
          f"{'$/1000':>10}{'ms':>8}   failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--groups", default="all_employees,warehouse")
    parser.add_argument("--prompt", default=PROMPT_VERSION)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--no-condense", action="store_true")
    args = parser.parse_args()

    llm, retriever = LLM(), Retriever()
    groups = args.groups.split(",")

    if args.summary:
        return summary_demo(llm, retriever)
    if args.question:
        r = rag(llm, retriever, args.question, groups, k=args.k, prompt=args.prompt)
        print(r["answer"])
        for cid in r["cited"]:
            print(f"  source: {cid}")
        print(f"\nretrieved: {', '.join(r['retrieved'])}")
        return

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    if args.sweep:
        rows = sweep(llm, retriever, questions)
        print(HEADER)
        for row in rows:
            print_metrics(f"k={row['k']}", row)
        (OUT / "sweep.json").write_text(json.dumps(rows, indent=2))
        return

    print(f"model {MODEL} | prompt {args.prompt} | k {args.k} | budget {CONTEXT_BUDGET} tokens\n")
    results, m = run_all(llm, retriever, questions, k=args.k, prompt=args.prompt, use_condense=not args.no_condense)
    print("\n" + HEADER)
    print_metrics(args.prompt, m)
    print(f"\nFOR CHAT: {MODEL} {args.prompt} k={args.k} | correct {m['correct']:.2f} | cited {m['cited']:.2f} | "
          f"abstain {m['abstain']:.2f} | ${m['usd_per_1000']:.2f}/1000 q | {m['ms']:.0f} ms")
    tag = args.prompt + ("-nocondense" if args.no_condense else "")
    (OUT / f"eval-{tag}.json").write_text(json.dumps(m, indent=2))
    with (OUT / f"answers-{tag}.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except NotImplementedError as e:
        sys.exit(f"Not implemented yet: {e}")
