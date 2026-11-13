"""Lesson 11 practice: an agent that answers expense and remote-work questions with 4 tools.

Usage:
    python starter/agent.py --lint                        # check data/tasks.json and the tool descriptions, no model calls
    python starter/agent.py --ask "Скільки компенсують за таксі 35 EUR?"   # one question, you confirm side effects
    python starter/agent.py                               # all tasks once: system v4, tools v2
    python starter/agent.py --repeat 3                    # every task 3 times: pass@1, passed every run, flaky
    python starter/agent.py --system v1 --tools v1        # the first versions of the prompt and the descriptions
    python starter/agent.py --disable calculate,days_between   # ablation: the model counts in its head
    python starter/agent.py --max-steps 2                 # ablation: a tight step limit
    python starter/agent.py --autonomy auto --only side_effect # ablation: side effects without confirmation
    python starter/agent.py --min-pass 0.9                # CI: exit code 1 if pass < 0.9 or a claim went out unconfirmed

Writes output/eval-<tag>.json (metrics and every run) and output/traces/<tag>/<task>-r<run>.json (every step).
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tools  # noqa: E402

load_dotenv(ROOT / ".env")

PROMPTS = ROOT / "prompts"
TASKS = ROOT / "data" / "tasks.json"
OUT = ROOT / "output"

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-4o-mini"
PRICE_IN, PRICE_OUT = 0.15, 0.60          # USD per 1M tokens, check openrouter.ai/models before you rely on it
# Local model instead (lesson 3): ollama pull qwen2.5:7b (llama3.2:3b calls tools poorly)
# BASE_URL, API_KEY, MODEL, PRICE_IN, PRICE_OUT = "http://localhost:11434/v1", "ollama", "qwen2.5:7b", 0, 0

SYSTEM_VERSION = "v4"      # v1 = "be helpful, use tools"; v2 = rules; v3 = search first; v4 = search budget (prompts/CHANGELOG.md)
TOOLS_VERSION = "v2"       # prompts/tools-v1.json = one-word descriptions; v2 = when to use, formats, examples
TODAY = "2026-11-09"       # frozen, so that "is it too late?" has one right answer in evals; None = the real date
AUTONOMY = "confirm"       # read_only: no side effects at all; confirm: a person says yes first; auto: no questions
LIMITS = {"max_steps": 6,          # model calls per task
          "max_tokens": 20_000,    # input + output tokens per task
          "max_usd": 0.01,         # cost per task
          "max_seconds": 60,       # wall time per task
          "max_repeats": 2}        # the same tool with the same arguments
TOOL_TIMEOUT = 10          # seconds for one tool call
SMALL = 10                 # unverified_numbers ignores numbers below this: list items, "3 nights" in words
NOT_FOUND = ("не знайш", "не знайд", "немає інформац", "немає даних", "не містять", "не міститься",
             "не вказан", "could not find", "couldn't find", "not find", "no information")


def today() -> str:
    return TODAY or date.today().isoformat()


def usd(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1_000_000


def norm(text: str) -> str:
    """Lower case, one kind of space, decimal comma -> dot: '43,50 EUR' and '43.50 eur' compare equal. Ready to use."""
    text = text.lower().replace("\xa0", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", re.sub(r"(\d),(\d)", r"\1.\2", text))


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str             # JSON text, exactly as the model wrote it: it may be broken


@dataclass
class Reply:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0


class LLM:
    """A thin wrapper over the chat API with tools: one call -> Reply. Ready to use."""

    def __init__(self):
        if not API_KEY:
            raise SystemExit("OPENROUTER_API_KEY is missing: copy .env.example to .env and paste your key")
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=60, max_retries=2)

    def chat(self, messages: list[dict], specs: list[dict] | None = None, max_tokens: int = 600) -> Reply:
        r = self.client.chat.completions.create(model=MODEL, messages=messages, temperature=0, max_tokens=max_tokens,
                                                **({"tools": specs} if specs else {}))
        m = r.choices[0].message
        calls = [ToolCall(c.id, c.function.name, c.function.arguments) for c in (m.tool_calls or [])]
        return Reply(m.content, calls, r.usage.prompt_tokens if r.usage else 0, r.usage.completion_tokens if r.usage else 0)


def assistant_message(reply: Reply) -> dict:
    """The model's reply in the format the API expects back in `messages`. Ready to use."""
    msg = {"role": "assistant", "content": reply.content or ""}
    if reply.tool_calls:
        msg["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                             for c in reply.tool_calls]
    return msg


def load_prompt(version: str) -> str:
    path = PROMPTS / version / "system.txt"
    if not path.exists():
        raise SystemExit(f"Prompt not found: prompts/{version}/system.txt")
    return path.read_text(encoding="utf-8").strip().replace("{today}", today())


def load_descriptions(version: str) -> dict:
    path = PROMPTS / f"tools-{version}.json"
    if not path.exists():
        raise SystemExit(f"Tool descriptions not found: prompts/tools-{version}.json")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- tool schemas

# TODO 1 · Goal: describe the tools so that the model knows when to call each one and what to pass.
#   Why: the model sees only the name, the description and the JSON schema of every tool. It never sees your code.
#   "Calculator." does not say that min() is allowed or that the dot is the decimal separator, and
#   "from" does not say YYYY-MM-DD. Keep the descriptions in a file, so that v1 and v2 can be compared on the eval.
#   Done when: tool_specs returns one {"type": "function", "function": {name, description, parameters}} per name;
#   parameters = the JSON schema of tools.ARGS[name] without "title" keys, every property with its description from
#   the file, "additionalProperties": false; a missing tool or parameter description raises ValueError (check 1).
def tool_specs(descriptions: dict, names: list[str]) -> list[dict]:
    """TODO 1: OpenAI tool specs for `names` from tools.ARGS and the descriptions file."""
    raise NotImplementedError("TODO 1: tool_specs")


# ---------------------------------------------------------------- the executor

def call_with_timeout(fn, args, timeout: float):
    """Run fn(args) in a thread; raise TimeoutError if it takes longer than `timeout` seconds. Ready to use."""
    box = {}

    def target():
        try:
            box["result"] = fn(args)
        except Exception as e:          # noqa: BLE001 - handed to the caller
            box["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"no result after {timeout} s")
    if "error" in box:
        raise box["error"]
    return box["result"]


# TODO 2 · Goal: run one tool call and turn every failure into a message the model can act on.
#   Why: the model writes the arguments, so they will sometimes be wrong: "20.09.2026" instead of 2026-09-20,
#   a tool that does not exist, broken JSON. If the agent crashes, the user gets nothing. If the model reads
#   "ERROR: date_from: Input should be a valid date", it usually fixes the call on the next step.
#   Done when: execute returns (text, status) and never raises: unknown tool, invalid JSON, arguments that fail
#   tools.ARGS[name] validation, an exception in the tool and a timeout give status "error" and a text starting with
#   "ERROR:"; after validation it calls guard(name, args_dict): a returned message means status "blocked" and
#   the tool does not run; otherwise status "ok" and the tool result as JSON (check 2).
def execute(name: str, raw_args: str, available: dict, guard=lambda name, args: None,
            timeout: float = TOOL_TIMEOUT) -> tuple[str, str]:
    """TODO 2: available = {tool name: function}. Returns (text for the model, "ok" | "error" | "blocked")."""
    raise NotImplementedError("TODO 2: execute")


# ---------------------------------------------------------------- the agent loop

# TODO 3 · Goal: the loop itself: model -> tool calls -> results -> model ... -> final answer, with a trace.
#   Why: this is all an agent is. Frameworks hide this loop, and when the agent "falls apart on step two"
#   you need to see every step: what the model asked for, what it got back, how many tokens it cost.
#   Done when: run_agent starts with [system, user], calls llm.chat(messages, specs), appends
#   assistant_message(reply), runs every tool call with run_tool(name, arguments) and appends
#   {"role": "tool", "tool_call_id": ..., "content": record["result"]}; it stops with stop="answer" when the reply
#   has no tool calls, or with the reason from stop_reason(trace, limits, seconds) checked before every model call.
#   The trace has question, answer (None if stopped), stop, steps, tool_calls (the run_tool records with "step"),
#   tokens_in, tokens_out, usd, seconds, messages (check 3).
def run_agent(llm, question: str, specs: list[dict], run_tool, system: str, limits: dict = LIMITS) -> dict:
    """TODO 3: run_tool(name, raw_args) -> {"name", "args", "status", "result", "ms"}."""
    raise NotImplementedError("TODO 3: run_agent")


# ---------------------------------------------------------------- execution control

# TODO 4 · Goal: stop the agent before it burns money or time, and say why.
#   Why: a loop with a model inside has no natural end. The model can search for the same thing again and again,
#   or keep calling tools on a question it cannot answer. Each step costs tokens, and the context grows with it.
#   Done when: check_limits returns None or a reason that starts with the limit's name: "max_steps" (len(steps)
#   reached it), "max_tokens" (tokens_in + tokens_out), "max_usd" (usd), "max_seconds" (seconds), "max_repeats"
#   (the same tool with the same arguments, compared as parsed JSON, called more than max_repeats times) (check 4).
def check_limits(trace: dict, limits: dict, seconds: float) -> str | None:
    """TODO 4: one reason to stop, or None to continue."""
    raise NotImplementedError("TODO 4: check_limits")


def stop_reason(trace: dict, limits: dict, seconds: float) -> str | None:
    """check_limits() once TODO 4 is done; until then only the step limit, so that TODO 3 can run first."""
    try:
        return check_limits(trace, limits, seconds)
    except NotImplementedError:
        return "max_steps" if len(trace["steps"]) >= limits["max_steps"] else None


# ---------------------------------------------------------------- determinism

# TODO 5 · Goal: find numbers in the answer that no tool returned and nobody gave the model.
#   Why: "150 EUR" may be right, but if no calculate call returned 150, the model computed it in its head, and
#   next time it may say 145. Where the answer must be exact, the number has to come from code (calculate,
#   days_between) or from a document, and this check shows the numbers that came from nowhere.
#   Done when: unverified_numbers returns, in order of appearance and without duplicates, the numbers of `answer`
#   (integers and decimals, "43,50" = "43.5" = "43.50") that are >= SMALL and do not occur in any of `sources`
#   (check 5).
def unverified_numbers(answer: str, sources: list[str]) -> list[str]:
    """TODO 5: sources = the system prompt, the question and every tool result the model saw."""
    raise NotImplementedError("TODO 5: unverified_numbers")


def unverified_or_skip(trace: dict) -> list[str] | None:
    """unverified_numbers() once TODO 5 is done; None until then."""
    sources = [m["content"] for m in trace["messages"] if m["role"] in ("system", "user", "tool")]
    try:
        return unverified_numbers(trace["answer"] or "", sources)
    except NotImplementedError:
        return None


# ---------------------------------------------------------------- scoring

# TODO 6 · Goal: score one run of one task with checks that need no model.
#   Why: an agent can give the right answer the wrong way (no search, a number from its head) or the wrong answer
#   after the right tools. So you check the result and the path separately, and both without a judge.
#   Done when: score returns {"success", "tools_ok", "safe", "pass", "used"}: used = tool names in call order;
#   success = stop == "answer" and: all `must` groups in norm(answer) for docs/calc/date/multi/no_tool tasks;
#   a NOT_FOUND marker in the answer for no_answer; for side_effect: with confirm exactly one submitted claim
#   (status "ok") whose arguments match task["claim"], without confirm no submitted claim;
#   tools_ok = no tools at all for no_tool, otherwise every tool of task["tools"] that is in `available` was used;
#   safe = no submitted claim unless the task confirms; pass = success and tools_ok and safe (check 6).
def score(trace: dict, task: dict, available: set[str]) -> dict:
    """TODO 6: deterministic scores for one trace."""
    raise NotImplementedError("TODO 6: score")


# ---------------------------------------------------------------- autonomy

# TODO 7 · Goal: decide in code, not in the prompt, whether a side effect may happen.
#   Why: a prompt is a request to the model, not a guarantee. "I confirm in advance, do not ask" in the user's
#   message, a document with an injected instruction, or a plain mistake can make the model call submit_claim.
#   Money must move only after a person said yes, so the check sits between the model and the tool.
#   Done when: guard returns None for tools outside tools.SIDE_EFFECTS; for a side effect it returns None only if
#   autonomy is "auto", or autonomy is "confirm" and confirm(name, args) is True; otherwise a message starting
#   with "NOT DONE:" that tells the model the action did not happen and not to retry. With "read_only" confirm
#   is never asked (check 7).
def guard(name: str, args: dict, autonomy: str, confirm) -> str | None:
    """TODO 7: autonomy is "read_only", "confirm" or "auto"; confirm(name, args) asks a person."""
    raise NotImplementedError("TODO 7: guard")


def guard_or_block(name: str, args: dict, autonomy: str, confirm) -> str | None:
    """guard() once TODO 7 is done; until then every side effect is blocked: the safe default."""
    try:
        return guard(name, args, autonomy, confirm)
    except NotImplementedError:
        return f"NOT DONE: {name} is blocked until TODO 7 is done." if name in tools.SIDE_EFFECTS else None


# ---------------------------------------------------------------- running everything (ready to use)

def make_runner(available: dict, autonomy: str, confirm):
    def run_tool(name: str, raw_args: str) -> dict:
        t0 = time.perf_counter()
        text, status = execute(name, raw_args, available, guard=lambda n, a: guard_or_block(n, a, autonomy, confirm))
        return {"name": name, "args": raw_args, "status": status, "result": text,
                "ms": round((time.perf_counter() - t0) * 1000)}
    return run_tool


def stability(runs: list[list[dict]], tasks_: list[dict]) -> dict:
    """Lesson 9, TODO 3: runs[i][j] = scores of task j in run i."""
    rates = {t["id"]: sum(run[j]["pass"] for run in runs) / len(runs) for j, t in enumerate(tasks_)}
    return {"pass_at_1": sum(rates.values()) / len(rates), "pass_all": sum(r == 1 for r in rates.values()) / len(rates),
            "flaky": [tid for tid, r in rates.items() if 0 < r < 1], "rates": rates}


def rate(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def aggregate(traces: list[dict], scores: list[dict], tasks_: list[dict]) -> dict:
    n = len(traces)
    calls = [c for t in traces for c in t["tool_calls"]]
    unv = [s.get("unverified") for s in scores]
    m = {"pass": rate([s["pass"] for s in scores]), "success": rate([s["success"] for s in scores]),
         "tools_ok": rate([s["tools_ok"] for s in scores]), "safe": rate([s["safe"] for s in scores]),
         "invented_numbers": rate([bool(u) if u is not None else None for u in unv]),
         "steps": sum(len(t["steps"]) for t in traces) / n, "tool_calls": len(calls) / n,
         "tool_errors": sum(c["status"] == "error" for c in calls), "blocked": sum(c["status"] == "blocked" for c in calls),
         "tokens": sum(t["tokens_in"] + t["tokens_out"] for t in traces) / n,
         "usd_per_task": sum(t["usd"] for t in traces) / n, "usd": sum(t["usd"] for t in traces),
         "seconds": sum(t["seconds"] for t in traces) / n, "seconds_max": max(t["seconds"] for t in traces),
         "stops": dict(Counter(t["stop"].split(":")[0] for t in traces))}
    kinds = sorted({t["kind"] for t in tasks_})
    m["by_kind"] = {k: rate([s["pass"] for s, t in zip(scores, tasks_) if t["kind"] == k]) for k in kinds}
    m["failed"] = [t["id"] for s, t in zip(scores, tasks_) if not s["pass"]]
    return m


def lint(tasks_: list[dict], descriptions: dict) -> list[str]:
    """Problems in the task set and the tool descriptions (lesson 9, TODO 1 for an agent)."""
    chunks = {c["chunk_id"]: c["text"] for c in (json.loads(l) for l in (ROOT / "data" / "chunks.jsonl")
                                                 .read_text(encoding="utf-8").splitlines())}
    problems, kinds = [], Counter(t["kind"] for t in tasks_)
    for t in tasks_:
        problems += [f"{t['id']}: unknown chunk id {r}" for r in t["relevant"] if r not in chunks]
        text = " ".join(chunks.get(r, "") for r in t["relevant"])
        problems += [f"{t['id']}: policy fact {p!r} is not in the relevant chunks" for p in t["policy"] if p not in text]
        problems += [f"{t['id']}: `must` {g} is already in the question" for g in t["must"]
                     if any(norm(o) in norm(t["question"]) for o in g)]
        problems += [f"{t['id']}: unknown tool {x}" for x in t["tools"] if x not in tools.FUNCS]
    problems += [f"kind {k}: only {n} task(s)" for k, n in kinds.items() if n < 2]
    try:
        tool_specs(descriptions, list(tools.FUNCS))
    except ValueError as e:
        problems.append(f"tool descriptions: {e}")
    except NotImplementedError:
        print("(tool schemas are not checked until TODO 1 is done)")
    for name, d in descriptions.items():   # a rule of thumb: one line cannot say when to use the tool and how
        if len(d.get("description", "").split()) < 10:
            problems.append(f"tool descriptions: {name}: {len(d.get('description', '').split())} word(s), "
                            "say when to use it, what to pass and what it returns")
    return problems


def run_task(llm, task: dict, specs: list[dict], available: dict, system: str, autonomy: str, limits: dict) -> dict:
    asked = []

    def confirm(name, args):           # a scripted person: says what the task says, and we record that we asked
        asked.append({"tool": name, "args": args})
        return bool(task.get("confirm", False))

    trace = run_agent(llm, task["question"], specs, make_runner(available, autonomy, confirm), system, limits)
    trace["confirm_asked"] = asked
    return trace


def ask_person(name: str, args: dict) -> bool:
    print(f"\n  The agent wants to run {name}({json.dumps(args, ensure_ascii=False)})")
    return input("  Allow? [y/N] ").strip().lower() in ("y", "yes", "т", "так")


def show(trace: dict):
    for i, step in enumerate(trace["steps"], 1):
        for c in step["tool_calls"]:
            rec = next((r for r in trace["tool_calls"] if r["step"] == i and r["name"] == c["name"] and r["args"] == c["args"]), {})
            print(f"  step {i}: {c['name']}({c['args']}) -> {rec.get('status')}: {rec.get('result', '')[:160]}")
    print(f"\n{trace['answer'] if trace['answer'] is not None else 'STOPPED: ' + trace['stop']}")
    print(f"\n  stop {trace['stop']} | {len(trace['steps'])} model calls | {trace['tokens_in'] + trace['tokens_out']} tokens"
          f" | ${trace['usd']:.5f} | {trace['seconds']} s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ask")
    p.add_argument("--lint", action="store_true")
    p.add_argument("--system", default=SYSTEM_VERSION)
    p.add_argument("--tools", default=TOOLS_VERSION)
    p.add_argument("--disable", default="", help="comma-separated tool names to take away from the agent")
    p.add_argument("--autonomy", default=AUTONOMY, choices=["read_only", "confirm", "auto"])
    p.add_argument("--max-steps", type=int, default=LIMITS["max_steps"])
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--only", default="", help="comma-separated task kinds or ids")
    p.add_argument("--min-pass", type=float)
    args = p.parse_args()

    descriptions, all_tasks = load_descriptions(args.tools), json.loads(TASKS.read_text(encoding="utf-8"))
    if args.lint:
        problems = lint(all_tasks, descriptions)
        print("\n".join(problems) or "tasks and tool descriptions OK")
        print(f"\n{len(all_tasks)} tasks, {len(problems)} problem(s)")
        return

    disabled = {x for x in args.disable.split(",") if x}
    names = [n for n in tools.FUNCS if n not in disabled and not (args.autonomy == "read_only" and n in tools.SIDE_EFFECTS)]
    available = {n: tools.FUNCS[n] for n in names}
    specs, system, limits = tool_specs(descriptions, names), load_prompt(args.system), {**LIMITS, "max_steps": args.max_steps}
    llm = LLM()

    if args.ask:
        return show(run_agent(llm, args.ask, specs, make_runner(available, args.autonomy, ask_person), system, limits))

    only = {x for x in args.only.split(",") if x}
    tasks_ = [t for t in all_tasks if not only or t["kind"] in only or t["id"] in only]
    tag = f"s{args.system}-t{args.tools}" + "".join(f"-no-{d}" for d in sorted(disabled)) + \
        (f"-steps{args.max_steps}" if args.max_steps != LIMITS["max_steps"] else "") + \
        (f"-{args.autonomy}" if args.autonomy != AUTONOMY else "") + (f"-{'+'.join(sorted(only))}" if only else "") + \
        (f"-x{args.repeat}" if args.repeat > 1 else "")
    trace_dir = OUT / "traces" / tag
    trace_dir.mkdir(parents=True, exist_ok=True)
    print(f"model {MODEL} | system {args.system} | tools {args.tools}: {', '.join(names)} | autonomy {args.autonomy}"
          f" | max_steps {args.max_steps} | {len(tasks_)} tasks x {args.repeat}\n")

    runs_traces, runs_scores = [], []
    for r in range(args.repeat):
        traces, scores = [], []
        for t in tasks_:
            trace = run_task(llm, t, specs, available, system, args.autonomy, limits)
            s = score(trace, t, set(names))
            s["unverified"] = unverified_or_skip(trace)
            traces.append(trace)
            scores.append(s)
            (trace_dir / f"{t['id']}-r{r + 1}.json").write_text(
                json.dumps({"task": t, "scores": s, **trace}, ensure_ascii=False, indent=1), encoding="utf-8")
        runs_traces.append(traces)
        runs_scores.append(scores)
        print(f"run {r + 1}/{args.repeat}: pass {aggregate(traces, scores, tasks_)['pass']:.2f}")

    print(f"\n{'task':<5}{'kind':<13}{'pass':<6}{'tools used (run 1)':<44}{'steps':>6}{'tokens':>8}{'$':>9}{'s':>6}  stop / invented")
    for j, t in enumerate(tasks_):
        tr, sc = runs_traces[0][j], runs_scores[0][j]
        marks = "".join("+" if run[j]["pass"] else "-" for run in runs_scores)
        used = ",".join(x.replace("search_docs", "search").replace("days_between", "days").replace("calculate", "calc")
                        .replace("submit_claim", "submit") for x in sc["used"]) or "-"
        note = (tr["stop"] if tr["stop"] != "answer" else "") + (f" invented {sc['unverified']}" if sc["unverified"] else "")
        print(f"{t['id']:<5}{t['kind']:<13}{marks:<6}{used[:43]:<44}{len(tr['steps']):>6}"
              f"{tr['tokens_in'] + tr['tokens_out']:>8}{tr['usd']:>9.5f}{tr['seconds']:>6.1f}  {note.strip()}")

    flat_t, flat_s = [x for run in runs_traces for x in run], [x for run in runs_scores for x in run]
    m = aggregate(flat_t, flat_s, tasks_ * args.repeat)
    m["runs"] = [aggregate(tr, sc, tasks_) for tr, sc in zip(runs_traces, runs_scores)]
    m["failed"] = sorted({tid for run in m["runs"] for tid in run["failed"]})
    if args.repeat > 1:
        m.update(stability(runs_scores, tasks_))
    print(f"\nall runs: pass {m['pass']:.2f} | success {m['success']:.2f} | tools_ok {m['tools_ok']:.2f} | safe {m['safe']:.2f}"
          + (f" | invented numbers {m['invented_numbers']:.2f}" if m["invented_numbers"] is not None else ""))
    print(f"per task: {m['steps']:.1f} model calls, {m['tool_calls']:.1f} tool calls, {m['tokens']:.0f} tokens,"
          f" ${m['usd_per_task']:.5f}, {m['seconds']:.1f} s (max {m['seconds_max']:.1f} s)")
    print(f"tool errors {m['tool_errors']} | blocked side effects {m['blocked']} | stops {m['stops']} | total ${m['usd']:.4f}")
    print("by kind  " + "  ".join(f"{k} {v:.2f}" for k, v in m["by_kind"].items()))
    print(f"failed   {' '.join(m['failed']) or '-'}")
    if args.repeat > 1:
        print(f"pass@1 {m['pass_at_1']:.2f} | passed every run {m['pass_all']:.2f} | flaky {' '.join(m['flaky']) or '-'}")

    (OUT / f"eval-{tag}.json").write_text(json.dumps(
        {"model": MODEL, "system": args.system, "tools": args.tools, "names": names, "autonomy": args.autonomy,
         "limits": limits, "today": today(), **m,
         "rows": [{"run": r + 1, "id": t["id"], "kind": t["kind"], **{k: v for k, v in sc.items()},
                   "stop": tr["stop"], "steps": len(tr["steps"]), "tokens": tr["tokens_in"] + tr["tokens_out"],
                   "usd": tr["usd"], "seconds": tr["seconds"], "answer": tr["answer"]}
                  for r, (trs, scs) in enumerate(zip(runs_traces, runs_scores))
                  for t, tr, sc in zip(tasks_, trs, scs)]}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwritten: output/eval-{tag}.json, output/traces/{tag}/")

    if args.min_pass is not None:
        unsafe = m["safe"] < 1
        if m["pass"] < args.min_pass or unsafe:
            print(f"GATE FAILED: pass {m['pass']:.2f} (need {args.min_pass})" + (", a claim went out unconfirmed" if unsafe else ""))
            sys.exit(1)
        print("GATE PASSED")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        raise SystemExit(f"{e} is not done yet: read the comment above the function")
