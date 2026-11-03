"""Checks for the lesson 8 practice. Checks 1-9 run offline with a fake model (no key, no cost);
check 10 asks the real model one question if OPENROUTER_API_KEY is set.

    python starter/check.py              # checks starter/rag.py
    python starter/check.py --solution   # checks solution/rag.py (after the lesson)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ("solution" if "--solution" in sys.argv else "starter")))
import rag  # noqa: E402


class FakeLLM:
    """Same interface as rag.LLM, but returns prepared replies and records what it was asked."""

    def __init__(self, replies=()):
        self.replies, self.asked = list(replies), []
        self.calls, self.tokens_in, self.tokens_out = 0, 0, 0

    def chat(self, messages, schema=None, max_tokens=500):
        self.asked.append({"messages": messages, "schema": schema})
        self.calls += 1
        self.tokens_in += sum(len(m["content"]) for m in messages) // 4
        return self.replies.pop(0) if self.replies else '{"found": false, "answer": "-", "citations": []}'


class FakeRetriever:
    def __init__(self, hits):
        self.hits, self.queries = hits, []

    def search(self, query, groups, k=5):
        self.queries.append(query)
        return self.hits[:k]


def chunk(i, text="Hotel limit is 120 EUR per night.", source="handbook/expense_policy.md"):
    return {"chunk_id": f"{source}#{i}", "source": source, "section": f"Expense Policy > Part {i}",
            "access": ["all_employees"], "effective": "2026-03-01", "text": text}


def answer(found=True, text="120 EUR per night.", cites=("S1",)):
    return json.dumps({"found": found, "answer": text, "citations": list(cites)})


def check_1():
    hits = [chunk(0), chunk(1, "word " * 2000), chunk(2, "Taxi limit is 30 EUR.")]
    context, sources = rag.build_context(hits, budget=200)
    assert list(sources) == ["S1", "S2"], f"expected S1, S2 (the 2000-word chunk skipped), got {list(sources)}"
    assert sources["S2"]["chunk_id"].endswith("#2"), "S2 must be the small chunk after the skipped one"
    assert "[S1]" in context and "handbook/expense_policy.md" in context and "Part 0" in context, \
        "each source needs its label, document and section"
    assert "2026-03-01" in context, "put the effective date into the source header"
    assert rag.count_tokens(context) <= 200, "context is over the budget"


def check_2():
    llm = FakeLLM([answer()])
    ans = rag.generate(llm, "How much for a hotel?", "[S1] doc\nHotel 120 EUR", "SYSTEM PROMPT")
    assert isinstance(ans, rag.Answer) and ans.found and ans.citations == ["S1"], "parse the reply into Answer"
    msgs = llm.asked[0]["messages"]
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == "SYSTEM PROMPT", "first message: the system prompt"
    user = msgs[-1]["content"]
    assert "Hotel 120 EUR" in user and "How much for a hotel?" in user, "user message needs sources and question"
    assert llm.asked[0]["schema"] == rag.ANSWER_FORMAT, "call llm.chat with schema=ANSWER_FORMAT"


def check_3():
    sources = {"S1": chunk(0), "S2": chunk(1)}
    ans, invalid = rag.validate_citations(rag.Answer(found=True, answer="x", citations=["[S1]", "S7"]), sources)
    assert ans.citations == ["S1"] and ans.found, f'"[S1]" must count as S1 and S7 be dropped, got {ans.citations}'
    assert invalid == ["S7"], f"invalid labels: expected ['S7'], got {invalid}"
    ans, _ = rag.validate_citations(rag.Answer(found=True, answer="x", citations=["S9"]), sources)
    assert not ans.found and ans.citations == [], "an answer with no valid citation is not grounded: found=False"


def check_4():
    llm = FakeLLM()
    r = rag.rag(llm, FakeRetriever([]), "q", ["all_employees"])
    assert r["answer"] == rag.NOT_FOUND and llm.calls == 0, "no hits: NOT_FOUND without calling the model"
    llm = FakeLLM([answer(found=False, text="The documents do not say.", cites=())])
    r = rag.rag(llm, FakeRetriever([chunk(0), chunk(1)]), "q", ["all_employees"])
    assert r["answer"] == rag.NOT_FOUND and r["cited"] == [] and not r["found"], "found=False: NOT_FOUND, no sources"
    llm = FakeLLM([answer(cites=("S2",))])
    r = rag.rag(llm, FakeRetriever([chunk(0), chunk(1)]), "q", ["all_employees"])
    assert r["found"] and r["cited"] == ["handbook/expense_policy.md#1"], f"S2 must map to chunk #1, got {r['cited']}"


def check_5():
    qs = [{"id": "a", "kind": "paraphrase", "must": [["40"]], "relevant": ["x#1"]},
          {"id": "b", "kind": "paraphrase", "must": [["120"]], "relevant": ["x#2"]},
          {"id": "c", "kind": "no_answer", "relevant": []}]
    rs = [{"found": True, "answer": "Up to 40 EUR.", "cited": ["x#1"], "retrieved": ["x#1", "x#5"], "in_context": ["x#1"]},
          {"found": False, "answer": rag.NOT_FOUND, "cited": [], "retrieved": ["x#7"], "in_context": ["x#7"]},
          {"found": False, "answer": rag.NOT_FOUND, "cited": [], "retrieved": [], "in_context": []}]
    m = rag.evaluate(rs, qs)
    keys = ("retrieved", "in_context", "correct", "cited", "abstain", "false_abstain")
    got = tuple(round(m.get(k, -1), 2) for k in keys)
    assert got == (0.5, 0.5, 0.5, 0.5, 1.0, 0.5), f"expected {dict(zip(keys, (0.5, 0.5, 0.5, 0.5, 1.0, 0.5)))}, got {dict(zip(keys, got))}"
    assert m["failed"] == ["b"], f"failed: expected ['b'], got {m['failed']}"


def check_6():
    v2 = next((p for p in (ROOT / "prompts" / "v2" / "system.txt",
                           Path(rag.__file__).parent / "prompts" / "v2" / "system.txt") if p.exists()), None)
    if not v2:
        raise NotImplementedError("TODO 6: write prompts/v2/system.txt")
    assert v2.read_text() != (ROOT / "prompts" / "v1" / "system.txt").read_text(), "v2 must differ from v1"
    assert rag.PROMPT_VERSION == "v2", 'switch PROMPT_VERSION to "v2" after the comparison'


def check_7():
    rows = rag.sweep(FakeLLM(), FakeRetriever([chunk(i) for i in range(10)]), [
        {"id": "a", "kind": "paraphrase", "question": "q", "groups": ["all_employees"], "must": [["x"]], "relevant": []}])
    assert [r["k"] for r in rows] == [3, 5, 10], "one row per k = 3, 5, 10"
    t = [r["tokens_in"] for r in rows]
    assert t[0] < t[1] < t[2], f"more chunks must mean more input tokens, got {t}"


def check_8():
    chunks = [chunk(i, "temperature rule " * 60) for i in range(6)]
    n = len(rag.windows(chunks, 300))
    llm = FakeLLM(["- fact A [x#0]", "NONE"] + ["- fact B [x#4]"] * (n - 2) + ["FINAL ANSWER"])
    out = rag.map_reduce(llm, "List all temperature rules", chunks, budget=300)
    assert n >= 3, "test setup: expected at least 3 windows"
    assert llm.calls == n + 1, f"expected {n} map calls + 1 reduce, got {llm.calls} calls"
    reduce_in = llm.asked[-1]["messages"][-1]["content"]
    assert "fact A" in reduce_in and "fact B" in reduce_in and "NONE" not in reduce_in, "drop NONE, pass facts to reduce"
    assert out == "FINAL ANSWER", "return what the reduce call returns"


def check_9():
    probe = FakeLLM(["rewritten"])
    if rag.condense(probe, [["a", "b"]], "q") == "q" and probe.calls == 0:
        raise NotImplementedError("TODO 9: condense")
    llm = FakeLLM()
    assert rag.condense(llm, [], "Same question?") == "Same question?" and llm.calls == 0, "no history: no call"
    llm, ret = FakeLLM(["What is the yearly limit for home office equipment?", answer()]), FakeRetriever([chunk(0)])
    rag.rag(llm, ret, "What is the limit for that?", ["all_employees"],
            history=[["Can I claim home office equipment?", "Yes, a keyboard or a mouse."]])
    assert ret.queries == ["What is the yearly limit for home office equipment?"], \
        f"search must get the rewritten question, got {ret.queries}"
    assert "keyboard" in llm.asked[0]["messages"][-1]["content"], "pass the dialog to the rewriting call"


def check_10():
    if not rag.API_KEY:
        raise SkipCheck("no OPENROUTER_API_KEY in .env")
    from retrieval import Retriever
    r = rag.rag(rag.LLM(), Retriever(), "How much can I spend on a hotel during a business trip?", ["all_employees"])
    assert r["found"] and "120" in r["answer"], f"expected 120 EUR, got {r['answer']!r}"
    assert "handbook/expense_policy.md#2" in r["cited"], f"expected the limits table among sources, got {r['cited']}"


class SkipCheck(Exception):
    pass


CHECKS = [("1", "TODO 1  build_context: labels, headers, budget", check_1),
          ("2", "TODO 2  generate (ready): messages + structured output", check_2),
          ("3", "TODO 3  validate_citations", check_3),
          ("4", "TODO 4  rag: not found instead of guessing", check_4),
          ("5", "TODO 5  evaluate on a fake run", check_5),
          ("6", "TODO 6  prompt v2 written and switched on", check_6),
          ("7", "TODO 7  sweep k = 3 / 5 / 10", check_7),
          ("8", "TODO 8  map_reduce", check_8),
          ("9", "TODO 9  condense follow-ups", check_9),
          ("10", "live    one real question through the whole pipeline", check_10)]


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
