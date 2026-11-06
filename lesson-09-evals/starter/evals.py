"""Lesson 9 practice: evals for the RAG bot from lesson 8.

Usage:
    python starter/evals.py --lint                # TODO 1: check the testset itself, no model calls
    python starter/evals.py                       # run the bot (prompt v2) on the testset, score it, write the report
    python starter/evals.py --prompt v3           # the same for another prompt version
    python starter/evals.py --repeat 3            # TODO 3: every question 3 times: pass rate, flaky cases
    python starter/evals.py --judge               # + TODO 4 LLM judge and TODO 6 groundedness
    python starter/evals.py --calibrate           # TODO 5: the judge against human labels (data/labels.json)
    python starter/evals.py --save-baseline       # remember these metrics as the baseline (baseline.json)
    python starter/evals.py --prompt v3 --gate    # TODO 7: exit code 1 if the metrics fell below the baseline
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import rag  # noqa: E402

TESTSET = ROOT / "data" / "testset.json"
LABELS = ROOT / "data" / "labels.json"
BASELINE = ROOT / "baseline.json"
OUT = ROOT / "output"
KINDS_MIN = 2          # every kind of question needs at least this many cases
TOLERANCE = 0.05       # the gate allows a metric to drop this much (one case out of 37 is 0.03)
LOWER_IS_BETTER = {"false_abstain"}   # for these metrics a rise is the regression
JUDGE = "judge-v3"     # judge prompts: v1 before calibration, v2 and v3 after it (prompts/CHANGELOG.md)


def load_testset() -> list[dict]:
    return json.loads(TESTSET.read_text(encoding="utf-8"))


def chunk_texts() -> dict:
    return {c["chunk_id"]: c["text"] for c in rag.load_chunks()}


def norm(text: str) -> str:
    """Lower case, one kind of minus and space: '−15 °C' and '-15  °c' compare equal. Ready to use."""
    text = text.lower().replace("−", "-").replace("–", "-").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text)


# ---------------------------------------------------------------- the testset

# TODO 1 · Goal: test the tests before you trust their numbers.
#   Why: a typo in a chunk id or a `must` fact that the document never says makes a case fail forever,
#   and you will "fix" the bot for a bug that lives in the testset. A kind with one case says nothing.
#   Done when: lint returns a list of problems: unknown chunk ids, `must` facts not found in the relevant
#   chunks, no_answer cases with `must` or `relevant`, duplicate ids, kinds with fewer than KINDS_MIN cases (check 1).
def lint(cases: list[dict], chunks: dict) -> list[str]:
    """TODO 1: return human-readable problems, one string per problem, e.g. "q07: unknown chunk id x#9"."""
    raise NotImplementedError("TODO 1: lint")


# ---------------------------------------------------------------- deterministic scores

# TODO 2 · Goal: score one answer with checks that need no model: facts, citation, abstain.
#   Why: deterministic checks are free, fast and never disagree with themselves. Use them wherever they work
#   and keep the expensive judge for what they cannot see.
#   Done when: score returns {"facts", "cited", "abstain", "pass"}; facts = found and every `must` group
#   has one option in the answer (after norm), cited = a relevant chunk among the cited ones,
#   abstain = for no_answer cases the bot said not found; scores that do not apply are None;
#   pass = facts and cited for answerable cases, abstain for no_answer ones (check 2).
def score(result: dict, case: dict) -> dict:
    """TODO 2: deterministic scores for one answer."""
    raise NotImplementedError("TODO 2: score")


def rate(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def aggregate(scores: list[dict], cases: list[dict], results: list[dict]) -> dict:
    """Metrics over one run. Ready to use."""
    answerable = [r for r, c in zip(results, cases) if c["kind"] != "no_answer"]
    m = {"pass": rate([s["pass"] for s in scores]), "facts": rate([s["facts"] for s in scores]),
         "cited": rate([s["cited"] for s in scores]), "abstain": rate([s["abstain"] for s in scores]),
         "false_abstain": rate([not r["found"] for r in answerable])}
    for key in ("judge", "grounded"):
        if any(key in s for s in scores):
            m[key] = rate([s.get(key) for s in scores])
    kinds = sorted({c["kind"] for c in cases})
    m["by_kind"] = {k: rate([s["pass"] for s, c in zip(scores, cases) if c["kind"] == k]) for k in kinds}
    m["failed"] = [c["id"] for s, c in zip(scores, cases) if not s["pass"]]
    return m


# ---------------------------------------------------------------- repeated runs

# TODO 3 · Goal: run every question several times and see which cases are flaky.
#   Why: an LLM is not a function. Even at temperature 0 the same question can pass once and fail the next time,
#   so one green run proves little. pass@1 is the average chance to pass; "passed every run" is what users feel.
#   Done when: stability returns {"pass_at_1", "pass_all", "flaky", "rates"}: rates = {case id: share of runs
#   it passed}, pass_at_1 = their mean, pass_all = share of cases that passed in every run,
#   flaky = ids that passed in some runs but not in all (check 3).
def stability(runs: list[list[dict]], cases: list[dict]) -> dict:
    """TODO 3: runs[i][j] = scores of case j in run i."""
    raise NotImplementedError("TODO 3: stability")


# ---------------------------------------------------------------- LLM as a judge

class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str                                   # reason first: the model explains before it decides
    verdict: Literal["correct", "incorrect"]


VERDICT_FORMAT = {"type": "json_schema",
                  "json_schema": {"name": "verdict", "strict": True, "schema": Verdict.model_json_schema()}}


# TODO 4 · Goal: let a model grade what string matching cannot: "No, upgrades are not reimbursed" vs "No".
#   Why: `must` lists break on yes/no questions, paraphrases and other languages. A judge compares meaning.
#   Done when: judge sends prompts/<JUDGE>/correct.txt as the system prompt, the question, the reference and the
#   answer in the user message, calls llm.chat with schema=VERDICT_FORMAT and returns a Verdict (check 4).
def judge(llm, case: dict, answer: str) -> Verdict:
    """TODO 4: one judge call for one answer."""
    raise NotImplementedError("TODO 4: judge")


# TODO 5 · Goal: measure the judge before you trust it.
#   Why: the judge is a model too. If it calls wrong answers correct, every number built on it is inflated.
#   Compare it with labels people set by hand and read every disagreement.
#   Done when: calibrate returns {"agreement", "false_pass", "false_fail", "disagreements"}: agreement = share of
#   items where judge == person, false_pass = share of items the judge passed but a person failed,
#   false_fail = the opposite, disagreements = ids of the items where they differ (check 5).
def calibrate(judged: list[bool], human: list[bool], ids: list[str]) -> dict:
    """TODO 5: judged[i] and human[i] are True when answer i was marked correct."""
    raise NotImplementedError("TODO 5: calibrate")


# ---------------------------------------------------------------- groundedness

class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str
    supported: bool


class Claims(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[Claim]


CLAIMS_FORMAT = {"type": "json_schema",
                 "json_schema": {"name": "claims", "strict": True, "schema": Claims.model_json_schema()}}


# TODO 6 · Goal: catch answers that say more than their sources.
#   Why: an answer with a valid citation can still add a number or a rule the source never had.
#   Correctness compares with the reference; groundedness compares with the sources, so it also works
#   where nobody wrote a reference, for example on production logs.
#   Done when: grounded returns (None, []) for a not-found answer without a model call; otherwise it sends
#   prompts/<JUDGE>/grounded.txt as the system prompt, the sources and the answer, parses Claims and returns
#   (all claims supported, [texts of the unsupported claims]) (check 6).
def grounded(llm, answer: str, found: bool, sources: list[str]) -> tuple[bool | None, list[str]]:
    """TODO 6: sources = texts of the chunks the answer cites."""
    raise NotImplementedError("TODO 6: grounded")


# ---------------------------------------------------------------- the gate

# TODO 7 · Goal: turn the eval into a check that can block a release.
#   Why: evals nobody runs catch nothing. In CI the job runs on every change of a prompt or a model
#   and fails the build when quality drops, before users see it.
#   Done when: gate returns a list of failures: every number in the baseline (pass, facts, cited, abstain,
#   grounded ...) that is now lower by more than TOLERANCE (for LOWER_IS_BETTER metrics: higher), plus every case that passed in the baseline and fails
#   now, written as "regression: q16". An empty list means the gate is green (check 7).
def gate(metrics: dict, baseline: dict) -> list[str]:
    """TODO 7: compare the current metrics with the baseline."""
    raise NotImplementedError("TODO 7: gate")


# ---------------------------------------------------------------- running everything (ready to use)

def run_bot(llm, retriever, cases: list[dict], prompt: str) -> list[dict]:
    return [{"id": c["id"], **rag.rag(llm, retriever, c["question"], c["groups"], c.get("history"), prompt=prompt)}
            for c in cases]


def grounded_or_skip(llm, answer, found, sources):
    """grounded() once TODO 6 is done; until then (None, []) so that TODO 4 and 5 can run first."""
    try:
        return grounded(llm, answer, found, sources)
    except NotImplementedError:
        return None, []


def add_judges(judge_llm, results: list[dict], scores: list[dict], cases: list[dict], chunks: dict):
    for r, s, c in zip(results, scores, cases):
        v = judge(judge_llm, c, r["answer"])
        s["judge"], s["judge_reason"] = v.verdict == "correct", v.reason
        s["grounded"], s["unsupported"] = grounded_or_skip(judge_llm, r["answer"], r["found"],
                                                           [chunks[i] for i in r["cited"]])


def write_report(prompt: str, m: dict, results, scores, cases, bot_usd, eval_usd) -> Path:
    """A markdown report for people: the numbers, then every case worth reading."""
    f = lambda v: "-" if v is None else f"{v:.2f}"
    lines = [f"# Eval report: prompt {prompt}, model {rag.MODEL}", "",
             f"{len(cases)} cases. Bot ${bot_usd:.4f}, judge ${eval_usd:.4f}.", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {f(m[k])} |" for k in ("pass", "facts", "cited", "abstain", "false_abstain", "judge",
                                                 "grounded", "pass_at_1", "pass_all") if m.get(k) is not None]
    if m.get("flaky") is not None:
        lines += ["", f"Flaky cases: {', '.join(m['flaky']) or 'none'}"]
    lines += ["", "## By kind", "", "| kind | pass |", "|---|---|"] + [f"| {k} | {f(v)} |" for k, v in m["by_kind"].items()]
    lines += ["", "## Cases to read", ""]
    for r, s, c in zip(results, scores, cases):
        if s["pass"] and s.get("judge") is not False and s.get("grounded") is not False:
            continue
        lines += [f"### {c['id']} ({c['kind']}): {c['question']}", "", f"- answer: {r['answer']}",
                  f"- reference: {c['reference']}", f"- deterministic: facts {s['facts']}, cited {s['cited']}, "
                  f"abstain {s['abstain']}"]
        if "judge" in s:
            lines.append(f"- judge: {'correct' if s['judge'] else 'incorrect'}. {s['judge_reason']}")
        if s.get("unsupported"):
            lines.append(f"- not in the cited sources: {'; '.join(s['unsupported'])}")
        lines.append("")
    path = OUT / f"report-{prompt}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_calibration(judge_llm, chunks: dict):
    items = json.loads(LABELS.read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in load_testset()}
    judged, ground = [], []
    for it in items:
        v = judge(judge_llm, cases[it["case"]], it["answer"])
        judged.append(v.verdict == "correct")
        g, bad = grounded_or_skip(judge_llm, it["answer"], it["found"], [chunks[i] for i in it["cited"]])
        ground.append(g)
        it.update(judge_reason=v.reason, judge_ok=judged[-1], grounded_model=g, unsupported=bad)
    ids, human = [it["id"] for it in items], [it["correct"] for it in items]
    det = [score(it, cases[it["case"]])["pass"] for it in items]
    res = {"deterministic": calibrate(det, human, ids), "judge": calibrate(judged, human, ids)}
    g_items = [(g, it) for g, it in zip(ground, items) if g is not None]
    if g_items:   # empty until TODO 6 is done
        res["grounded"] = calibrate([g for g, _ in g_items], [it["grounded"] for _, it in g_items],
                                    [it["id"] for _, it in g_items])
    print(f"{len(items)} answers labelled by people ({len(g_items)} with a found answer), judge {rag.MODEL}, prompts {JUDGE}\n")
    print(f"{'':<24}{'agreement':>10}{'false pass':>12}{'false fail':>12}")
    for name, c in res.items():
        print(f"{name:<24}{c['agreement']:>10.2f}{c['false_pass']:>12.2f}{c['false_fail']:>12.2f}")
    by_id = {it["id"]: it for it in items}
    for name, c in res.items():
        print(f"\n{name} disagrees with people on: {', '.join(c['disagreements']) or 'nothing'}")
        for i in c["disagreements"]:
            it = by_id[i]
            why = {"judge": it["judge_reason"], "grounded": "; ".join(it["unsupported"]) or "all claims supported"}.get(name)
            print(f"  {i} [{it['case']}] {it['answer'][:100]!r}\n      person: {it['note']}"
                  + (f"\n      model:  {why}" if why else ""))
    OUT.mkdir(exist_ok=True)
    (OUT / f"calibration-{JUDGE}.json").write_text(json.dumps({"model": rag.MODEL, "prompts": JUDGE, **res, "items": items},
                                                     indent=2, ensure_ascii=False))


def main():
    global JUDGE
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default=rag.PROMPT_VERSION)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--judge", action="store_true")
    p.add_argument("--lint", action="store_true")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--save-baseline", action="store_true")
    p.add_argument("--gate", action="store_true")
    p.add_argument("--judge-prompt", default=JUDGE, help="judge-v1, judge-v2 or judge-v3")
    args = p.parse_args()

    JUDGE = args.judge_prompt
    cases, chunks = load_testset(), chunk_texts()
    if args.lint:
        problems = lint(cases, chunks)
        print("\n".join(problems) or "testset OK")
        print(f"\n{len(cases)} cases, {len(problems)} problem(s)")
        return
    if args.calibrate:
        return run_calibration(rag.LLM(), chunks)

    bot, judge_llm, retriever = rag.LLM(), rag.LLM(), rag.Retriever()
    runs_results, runs_scores = [], []
    for i in range(args.repeat):
        results = run_bot(bot, retriever, cases, args.prompt)
        runs_results.append(results)
        runs_scores.append([score(r, c) for r, c in zip(results, cases)])
        print(f"run {i + 1}/{args.repeat}: pass {aggregate(runs_scores[-1], cases, results)['pass']:.2f}")
    results, scores = runs_results[0], runs_scores[0]
    if args.judge:
        add_judges(judge_llm, results, scores, cases, chunks)
    m = aggregate(scores, cases, results)
    if args.repeat > 1:
        st = stability(runs_scores, cases)
        m.update(pass_at_1=st["pass_at_1"], pass_all=st["pass_all"], flaky=st["flaky"], rates=st["rates"])

    print(f"\nprompt {args.prompt} | {len(cases)} cases | bot ${bot.cost():.4f} | judge ${judge_llm.cost():.4f}\n")
    for key in ("pass", "facts", "cited", "abstain", "false_abstain", "judge", "grounded", "pass_at_1", "pass_all"):
        if m.get(key) is not None:
            print(f"  {key:<14}{m[key]:.2f}")
    print("  by kind       " + "  ".join(f"{k} {v:.2f}" for k, v in m["by_kind"].items()))
    print(f"  failed        {' '.join(m['failed']) or '-'}")
    if "flaky" in m:
        print(f"  flaky         {' '.join(m['flaky']) or '-'}")

    OUT.mkdir(exist_ok=True)
    tag = args.prompt + (f"-x{args.repeat}" if args.repeat > 1 else "") + ("-judge" if args.judge else "")
    (OUT / f"eval-{tag}.json").write_text(json.dumps({**m, "bot_usd": bot.cost(), "judge_usd": judge_llm.cost()},
                                                     indent=2))
    with (OUT / f"answers-{tag}.jsonl").open("w", encoding="utf-8") as fh:
        for run, (res, sc) in enumerate(zip(runs_results, runs_scores)):
            for r, s in zip(res, sc):
                fh.write(json.dumps({"run": run, **{k: v for k, v in r.items() if k != "context"}, "scores": s},
                                    ensure_ascii=False) + "\n")
    print(f"\nreport: {write_report(args.prompt, m, results, scores, cases, bot.cost(), judge_llm.cost()).relative_to(ROOT)}")

    if args.save_baseline:
        BASELINE.write_text(json.dumps({k: v for k, v in m.items() if k not in ("by_kind", "flaky", "rates")}, indent=2))
        print(f"baseline saved: {BASELINE.name}")
    if args.gate:
        if not BASELINE.exists():
            raise SystemExit("no baseline.json: run --save-baseline on the version you trust first")
        failures = gate(m, json.loads(BASELINE.read_text()))
        print("\nGATE " + ("FAILED" if failures else "PASSED"))
        for line in failures:
            print(f"  {line}")
        if failures:
            sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        raise SystemExit(f"{e} is not done yet: read the comment above the function")
