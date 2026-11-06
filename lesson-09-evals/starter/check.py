"""Checks for the lesson 9 practice. Checks 1-7 run offline with a fake model (no key, no cost);
check 8 asks the real judge two questions if OPENROUTER_API_KEY is set.

    python starter/check.py              # checks starter/evals.py
    python starter/check.py --solution   # checks solution/evals.py (after the lesson)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ("solution" if "--solution" in sys.argv else "starter")))
import evals  # noqa: E402
import rag  # noqa: E402


class FakeLLM:
    """Same interface as rag.LLM, but returns prepared replies and records what it was asked."""

    def __init__(self, replies=()):
        self.replies, self.asked, self.calls = list(replies), [], 0

    def chat(self, messages, schema=None, max_tokens=500):
        self.asked.append({"messages": messages, "schema": schema})
        self.calls += 1
        return self.replies.pop(0)


def case(cid="a", kind="exact", must=(("120",),), relevant=("x#1",), **kw):
    c = {"id": cid, "kind": kind, "groups": ["all_employees"], "question": "How much for a hotel?",
         "must": [list(m) for m in must], "relevant": list(relevant), "reference": "120 EUR per night."}
    return {**c, **kw}


def result(found=True, answer="120 EUR per night.", cited=("x#1",)):
    return {"found": found, "answer": answer, "cited": list(cited)}


def check_1():
    chunks = {"x#1": "Hotel, per night: 120 EUR.", "x#2": "Taxi 30 EUR per trip."}
    good = [case("a"), case("b", must=(("30",),), relevant=("x#2",)),
            case("c", kind="no_answer", must=(), relevant=()), case("d", kind="no_answer", must=(), relevant=())]
    assert evals.lint(good, chunks) == [], f"a clean testset must give no problems, got {evals.lint(good, chunks)}"
    bad = good + [case("e", relevant=("x#9",)), case("f", must=(("45",),)), case("a"),
                  case("g", kind="no_answer"), case("h", kind="yes_no")]
    text = "\n".join(evals.lint(bad, chunks))
    for what, needle in [("unknown chunk id x#9", "x#9"), ("`must` fact 45 not in the chunks", "f:"),
                         ("duplicate id a", "duplicate"), ("no_answer with must/relevant", "g:"),
                         ("kind yes_no with one case", "yes_no")]:
        assert needle in text, f"lint must report: {what}\n  got:\n{text}"


def check_2():
    s = evals.score(result(), case())
    assert s == {"facts": True, "cited": True, "abstain": None, "pass": True}, f"a good answer, got {s}"
    s = evals.score(result(answer="Reject above −15 °C."), case(must=(("-15 °c",),)))
    assert s["facts"], 'use norm(): "−15 °C" in the answer must match "-15 °c" in `must`'
    s = evals.score(result(cited=("x#5",)), case())
    assert s["facts"] and not s["cited"] and not s["pass"], f"right fact, wrong source: cited False, pass False, got {s}"
    s = evals.score(result(found=False, answer=rag.NOT_FOUND, cited=()), case())
    assert not s["facts"] and not s["pass"], "not found on an answerable case is a fail"
    s = evals.score(result(found=False, answer=rag.NOT_FOUND, cited=()), case(kind="no_answer", must=(), relevant=()))
    assert s == {"facts": None, "cited": None, "abstain": True, "pass": True}, f"abstain on no_answer, got {s}"
    s = evals.score(result(), case(kind="no_answer", must=(), relevant=()))
    assert s["abstain"] is False and not s["pass"], "an answer to a no_answer case is a fail"


def check_3():
    cases = [case("a"), case("b"), case("c")]
    p, f = {"pass": True}, {"pass": False}
    st = evals.stability([[p, p, f], [p, f, f], [p, p, f]], cases)
    assert st["rates"] == {"a": 1.0, "b": 2 / 3, "c": 0.0}, f"rates per case, got {st['rates']}"
    assert abs(st["pass_at_1"] - 5 / 9) < 1e-9, f"pass_at_1 = mean of rates = 0.56, got {st['pass_at_1']}"
    assert abs(st["pass_all"] - 1 / 3) < 1e-9, f"pass_all: only a passed every run = 0.33, got {st['pass_all']}"
    assert st["flaky"] == ["b"], f"flaky: passed sometimes = ['b'], got {st['flaky']}"


def check_4():
    llm = FakeLLM([json.dumps({"reason": "Same value: 120 EUR.", "verdict": "correct"})])
    v = evals.judge(llm, case(), "You can spend 120 euros a night.")
    assert isinstance(v, evals.Verdict) and v.verdict == "correct", "parse the reply into Verdict"
    msgs = llm.asked[0]["messages"]
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == rag.load_prompt(evals.JUDGE, "correct"), \
        "system message: prompts/<JUDGE>/correct.txt"
    user = msgs[-1]["content"]
    for part in ("How much for a hotel?", "120 EUR per night.", "120 euros a night"):
        assert part in user, f"the user message needs the question, the reference and the answer; missing {part!r}"
    assert llm.asked[0]["schema"] == evals.VERDICT_FORMAT, "call llm.chat with schema=VERDICT_FORMAT"


def check_5():
    c = evals.calibrate([True, True, False, True, False], [True, False, False, True, True], list("abcde"))
    got = (round(c["agreement"], 2), round(c["false_pass"], 2), round(c["false_fail"], 2), c["disagreements"])
    assert got == (0.6, 0.2, 0.2, ["b", "e"]), \
        f"expected agreement 0.6, false_pass 0.2, false_fail 0.2, ['b', 'e'], got {got}"


def check_6():
    llm = FakeLLM()
    assert evals.grounded(llm, rag.NOT_FOUND, False, []) == (None, []) and llm.calls == 0, \
        "not found: (None, []) without a model call"
    reply = {"claims": [{"claim": "Hotel limit is 120 EUR", "supported": True},
                        {"claim": "Hostels are not allowed", "supported": False}]}
    llm = FakeLLM([json.dumps(reply)])
    got = evals.grounded(llm, "120 EUR; hostels are not allowed.", True, ["Hotel, per night: 120 EUR."])
    assert got == (False, ["Hostels are not allowed"]), f"expected (False, ['Hostels are not allowed']), got {got}"
    msgs = llm.asked[0]["messages"]
    assert msgs[0]["content"] == rag.load_prompt(evals.JUDGE, "grounded"), "system message: prompts/<JUDGE>/grounded.txt"
    assert "Hotel, per night: 120 EUR." in msgs[-1]["content"] and "hostels" in msgs[-1]["content"], \
        "the user message needs the sources and the answer"
    assert llm.asked[0]["schema"] == evals.CLAIMS_FORMAT, "call llm.chat with schema=CLAIMS_FORMAT"
    llm = FakeLLM([json.dumps({"claims": [{"claim": "120 EUR", "supported": True}]})])
    assert evals.grounded(llm, "120 EUR.", True, ["120 EUR"]) == (True, []), "all supported: (True, [])"


def check_7():
    base = {"pass": 0.90, "facts": 0.90, "abstain": 1.0, "false_abstain": 0.03, "grounded": 0.95, "failed": ["q09"]}
    same = {"pass": 0.87, "facts": 0.92, "abstain": 1.0, "false_abstain": 0.06, "grounded": 0.95, "failed": ["q09"]}
    assert evals.gate(same, base) == [], f"a drop of 0.03 is within TOLERANCE 0.05, got {evals.gate(same, base)}"
    worse = {"pass": 0.80, "facts": 0.90, "abstain": 0.71, "false_abstain": 0.10, "grounded": 0.95,
             "failed": ["q09", "q16", "q20"]}
    got = evals.gate(worse, base)
    text = "\n".join(got)
    for needle in ("pass", "abstain:", "false_abstain", "regression: q16", "regression: q20"):
        assert needle in text, f"gate must report {needle!r}, got {got}"
    assert not any(line.startswith("facts") or "q09" in line for line in got), \
        f"facts did not drop and q09 failed already in the baseline, got {got}"


def check_8():
    if not rag.API_KEY:
        raise SkipCheck("no OPENROUTER_API_KEY in .env")
    llm = rag.LLM()
    c = case(question="How much can I spend on a hotel during a business trip?",
             reference="120 EUR per night, approved by the team lead.")
    good = evals.judge(llm, c, "You can spend up to 120 EUR per night on a hotel during a business trip.")
    bad = evals.judge(llm, c, "You can spend up to 150 EUR per night on a hotel during a business trip.")
    assert good.verdict == "correct", f"120 EUR must be correct, judge said: {good.reason}"
    assert bad.verdict == "incorrect", f"150 EUR must be incorrect, judge said: {bad.reason}"


class SkipCheck(Exception):
    pass


CHECKS = [("1", "TODO 1  lint: test the testset", check_1),
          ("2", "TODO 2  score: facts, citation, abstain", check_2),
          ("3", "TODO 3  stability: pass@1, passed every run, flaky", check_3),
          ("4", "TODO 4  judge: LLM-as-a-judge call", check_4),
          ("5", "TODO 5  calibrate: judge vs people", check_5),
          ("6", "TODO 6  grounded: claims vs sources", check_6),
          ("7", "TODO 7  gate: metrics and regressions vs baseline", check_7),
          ("8", "live    the real judge on 120 vs 150 EUR", check_8)]


def main():
    passed = skipped = 0
    for num, name, fn in CHECKS:
        try:
            fn()
            print(f"PASS  {num:>2}  {name}")
            passed += 1
        except NotImplementedError as e:
            print(f"TODO  {num:>2}  {name}  ({e})")
        except SkipCheck as e:
            print(f"SKIP  {num:>2}  {name}  ({e})")
            skipped += 1
        except Exception as e:
            print(f"FAIL  {num:>2}  {name}\n          {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(CHECKS) - skipped} checks passed" + (f", {skipped} skipped" if skipped else ""))


if __name__ == "__main__":
    main()
