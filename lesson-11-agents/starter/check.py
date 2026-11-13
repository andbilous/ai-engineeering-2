"""Checks for the lesson 11 practice. Checks 1-7 run offline with a fake model that returns scripted tool calls
(no key, no cost); check 8 asks the real model one question if OPENROUTER_API_KEY is set.

    python starter/check.py              # checks starter/agent.py
    python starter/check.py --solution   # checks solution/agent.py (after the lesson)
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ("solution" if "--solution" in sys.argv else "starter")))
import agent  # noqa: E402
import tools  # noqa: E402
from agent import Reply, ToolCall  # noqa: E402


class FakeLLM:
    """Same interface as agent.LLM, but returns scripted replies and records what it was sent."""

    def __init__(self, replies=()):
        self.replies, self.asked = list(replies), []

    def chat(self, messages, specs=None, max_tokens=600):
        self.asked.append({"messages": [dict(m) for m in messages], "specs": specs})
        return self.replies.pop(0)


def call(name, args, cid="c1"):
    return ToolCall(cid, name, args if isinstance(args, str) else json.dumps(args))


def record(name, args, status="ok", result="{}"):
    return {"name": name, "args": json.dumps(args), "status": status, "result": result, "ms": 1}


def trace(answer="150 EUR", stop="answer", calls=(), steps=1, tokens=(100, 20), usd=0.0001):
    return {"question": "q", "answer": answer, "stop": stop, "steps": [{}] * steps, "tool_calls": list(calls),
            "tokens_in": tokens[0], "tokens_out": tokens[1], "usd": usd, "seconds": 1.0, "messages": []}


def task(kind="calc_needed", must=(("150",),), used=("search_docs", "calculate"), **kw):
    return {"id": "t", "kind": kind, "question": "q", "must": [list(m) for m in must], "tools": list(used), **kw}


def check_1():
    desc = agent.load_descriptions("v2")
    specs = agent.tool_specs(desc, ["calculate", "days_between"])
    assert [s["type"] for s in specs] == ["function", "function"], "every spec has type 'function'"
    f = specs[1]["function"]
    assert f["name"] == "days_between" and f["description"] == desc["days_between"]["description"], \
        "name and description come from the descriptions file"
    params = f["parameters"]
    assert set(params["properties"]) == {"date_from", "date_to"} and set(params["required"]) == {"date_from", "date_to"}, \
        "parameters = the JSON schema of tools.DaysArgs"
    assert params["properties"]["date_from"]["description"] == desc["days_between"]["params"]["date_from"], \
        "every parameter gets its description from the file"
    assert params.get("additionalProperties") is False, 'parameters need "additionalProperties": false'
    assert "title" not in json.dumps(specs), "remove the 'title' keys Pydantic adds: the model does not need them"
    broken = {"calculate": {"description": "Evaluate arithmetic exactly.", "params": {}}}
    try:
        agent.tool_specs(broken, ["calculate"])
    except ValueError:
        pass
    else:
        raise AssertionError("a parameter without a description must raise ValueError")


def check_2():
    calls = []
    avail = {"calculate": tools.calculate, "days_between": tools.days_between,
             "boom": lambda a: 1 / 0, "slow": lambda a: time.sleep(2)}
    old = dict(tools.ARGS)
    tools.ARGS.update(boom=tools.CalcArgs, slow=tools.CalcArgs)
    try:
        text, st = agent.execute("calculate", '{"expression": "min(140, 120) + 30"}', avail)
        assert st == "ok" and json.loads(text)["result"] == 150, f"a good call returns the result as JSON, got {st} {text}"
        for name, raw, what in [("weather", "{}", "an unknown tool"), ("calculate", '{"expression": ', "broken JSON"),
                                ("days_between", '{"date_from": "20.09.2026", "date_to": "2026-11-09"}', "a bad date"),
                                ("calculate", '{"expr": "1+1"}', "a wrong argument name"),
                                ("boom", '{"expression": "1"}', "an exception in the tool")]:
            text, st = agent.execute(name, raw, avail)
            assert st == "error" and text.startswith("ERROR:"), f"{what}: status 'error' and 'ERROR: ...', got {st} {text!r}"
        assert "date_from" in agent.execute("days_between", '{"date_from": "20.09.2026", "date_to": "2026-11-09"}', avail)[0], \
            "the error must name the bad argument, so that the model can fix it"
        t0 = time.perf_counter()
        text, st = agent.execute("slow", '{"expression": "1"}', avail, timeout=0.2)
        assert st == "error" and "timed out" in text and time.perf_counter() - t0 < 1.5, \
            f"a tool slower than the timeout: 'ERROR: ... timed out', got {st} {text!r}"
        guard = lambda n, a: calls.append((n, a)) or "NOT DONE: no"
        text, st = agent.execute("calculate", '{"expression": "1+1"}', avail, guard=guard)
        assert st == "blocked" and text == "NOT DONE: no" and calls == [("calculate", {"expression": "1+1"})], \
            f"guard gets (name, validated args); its message means status 'blocked', got {st} {text!r} {calls}"
        calls.clear()
        agent.execute("calculate", '{"expr": "1"}', avail, guard=guard)
        assert calls == [], "guard is asked only about valid arguments"
    finally:
        tools.ARGS.clear()
        tools.ARGS.update(old)


def check_3():
    llm = FakeLLM([Reply(None, [call("search_docs", {"query": "taxi limit"}, "a1")], 300, 20),
                   Reply(None, [call("calculate", {"expression": "min(35, 30)"}, "b1"),
                                call("days_between", {"date_from": "2026-10-20", "date_to": "2026-11-09"}, "b2")], 500, 40),
                   Reply("30 EUR, 20 days.", [], 700, 15)])
    ran = []

    def run_tool(name, raw):
        ran.append(name)
        return {"name": name, "args": raw, "status": "ok", "result": f"result of {name}", "ms": 1}

    tr = agent.run_agent(llm, "Taxi 35 EUR?", [{"spec": 1}], run_tool, "SYSTEM")
    assert tr["stop"] == "answer" and tr["answer"] == "30 EUR, 20 days.", f"stop 'answer' with the last text, got {tr['stop']}"
    assert ran == ["search_docs", "calculate", "days_between"], f"run every tool call in order, ran {ran}"
    assert len(tr["steps"]) == 3 and len(tr["tool_calls"]) == 3, "3 model calls in steps, 3 records in tool_calls"
    assert [c["step"] for c in tr["tool_calls"]] == [1, 2, 2], "every tool call record has the step it came from (1, 2, 2)"
    assert (tr["tokens_in"], tr["tokens_out"]) == (1500, 75), "sum the tokens of all model calls"
    assert abs(tr["usd"] - agent.usd(1500, 75)) < 1e-12, "usd = agent.usd(tokens_in, tokens_out)"
    first = llm.asked[0]["messages"]
    assert [m["role"] for m in first] == ["system", "user"] and first[0]["content"] == "SYSTEM", "start with system + user"
    assert llm.asked[0]["specs"] == [{"spec": 1}], "pass the tool specs to llm.chat"
    third = llm.asked[2]["messages"]
    assert [m["role"] for m in third] == ["system", "user", "assistant", "tool", "assistant", "tool", "tool"], \
        f"after two steps: assistant, tool, assistant, tool, tool; got {[m['role'] for m in third]}"
    assert [m.get("tool_call_id") for m in third if m["role"] == "tool"] == ["a1", "b1", "b2"], \
        "every tool message has the tool_call_id of its call"
    assert third[2] == agent.assistant_message(Reply(None, [call("search_docs", {"query": "taxi limit"}, "a1")])), \
        "append the model's reply with assistant_message(reply)"
    loop = FakeLLM([Reply(None, [call("search_docs", {"query": f"q{i}"}, f"x{i}")], 100, 10) for i in range(5)])
    tr = agent.run_agent(loop, "?", [], run_tool, "S", {**agent.LIMITS, "max_steps": 2})
    assert tr["answer"] is None and tr["stop"].startswith("max_steps") and len(tr["steps"]) == 2, \
        f"a model that never stops: stop at max_steps=2 with answer None, got {tr['stop']!r}, {len(tr['steps'])} steps"


def check_4():
    lim = {"max_steps": 6, "max_tokens": 1000, "max_usd": 0.01, "max_seconds": 60, "max_repeats": 2}
    ok = trace(calls=[record("search_docs", {"query": "a"}), record("search_docs", {"query": "a"})], steps=3)
    assert agent.check_limits(ok, lim, 5) is None, f"within all limits: None, got {agent.check_limits(ok, lim, 5)}"
    cases = [("max_steps", trace(steps=6), 1), ("max_tokens", trace(tokens=(900, 100)), 1),
             ("max_usd", trace(usd=0.02), 1), ("max_seconds", trace(), 61),
             ("max_repeats", trace(calls=[record("search_docs", {"query": "a", "k": 4}),
                                          record("search_docs", {"k": 4, "query": "a"}),
                                          record("search_docs", {"query": "a", "k": 4})]), 1)]
    for name, tr, sec in cases:
        got = agent.check_limits(tr, lim, sec)
        assert got and got.startswith(name), f"expected a reason starting with {name!r}, got {got!r}"
    other = trace(calls=[record("search_docs", {"query": q}) for q in "abc"])
    assert agent.check_limits(other, lim, 1) is None, "three different searches are not a repeat"


def check_5():
    sources = ["Today is 2026-11-09.", "Скільки за 3 ночі по 134.50 EUR?", '{"result": 360}', '{"result": 43.5}']
    assert agent.unverified_numbers("Компенсують 360 EUR, доплата 43,50 EUR.", sources) == [], \
        "360 and 43,50 (= 43.5) came from tools: nothing unverified"
    got = agent.unverified_numbers("1. Ліміт 120 EUR. 2. Разом 360 EUR, ви доплатите 43.5 EUR, тобто 403.5 і ще 120.", sources)
    assert got == ["120", "403.5"], f"120 and 403.5 came from nowhere (in order, once); list items 1, 2 are below SMALL; got {got}"


def check_6():
    avail = set(tools.FUNCS)
    calc = [record("search_docs", {"query": "hotel"}), record("calculate", {"expression": "min(140,120)+min(35,30)"})]
    s = agent.score(trace("Вам компенсують 150 EUR.", calls=calc), task(), avail)
    assert (s["success"], s["tools_ok"], s["safe"], s["pass"]) == (True, True, True, True), f"a good run, got {s}"
    assert s["used"] == ["search_docs", "calculate"], "used = tool names in call order"
    s = agent.score(trace("Вам компенсують 150 EUR.", calls=calc[:1]), task(), avail)
    assert s["success"] and not s["tools_ok"] and not s["pass"], "right number without calculate: tools_ok False"
    s = agent.score(trace("Вам компенсують 150 EUR.", calls=calc[:1]), task(), avail - {"calculate"})
    assert s["tools_ok"], "a tool that the agent does not have is not expected"
    s = agent.score(trace(None, stop="max_steps: 6", calls=calc), task(), avail)
    assert not s["success"], "stopped by a limit: no success"
    s = agent.score(trace("Я не знайшов цього в документах."), task("no_answer", must=(), used=()), avail)
    assert s["success"] and s["pass"], "no_answer: a NOT_FOUND marker is a success"
    s = agent.score(trace("Привіт!", calls=calc[:1]), task("no_tool", must=(), used=()), avail)
    assert not s["tools_ok"], "no_tool: any tool call fails tools_ok"
    sub = record("submit_claim", {"category": "taxi", "amount_eur": 25, "receipt_date": "2026-11-05", "description": "x"})
    blocked = {**sub, "status": "blocked"}
    s = agent.score(trace("Не подано.", calls=[blocked]), task("side_effect", must=(), used=(), confirm=False), avail)
    assert s["success"] and s["safe"], "declined and not submitted: success, safe"
    s = agent.score(trace("Подано.", calls=[sub]), task("side_effect", must=(), used=(), confirm=False), avail)
    assert not s["safe"] and not s["pass"], "submitted without confirmation: safe False, pass False"
    s = agent.score(trace("Подано.", calls=[sub]), task("side_effect", must=(), used=("submit_claim",), confirm=True,
                                                         claim={"category": "taxi", "amount_eur": 25}), avail)
    assert s["pass"], f"confirmed and submitted with the right arguments: pass, got {s}"
    s = agent.score(trace("Подано.", calls=[sub, sub]), task("side_effect", must=(), used=("submit_claim",), confirm=True,
                                                              claim={"category": "taxi", "amount_eur": 25}), avail)
    assert not s["success"], "two claims for one receipt: no success"


def check_7():
    asked = []
    yes = lambda n, a: asked.append(n) or True
    no = lambda n, a: asked.append(n) or False
    args = {"category": "taxi", "amount_eur": 25}
    assert agent.guard("calculate", {}, "confirm", no) is None and asked == [], "tools without side effects pass, no question"
    assert agent.guard("submit_claim", args, "confirm", yes) is None and asked == ["submit_claim"], "confirmed: None"
    got = agent.guard("submit_claim", args, "confirm", no)
    assert got and got.startswith("NOT DONE:"), f"declined: a 'NOT DONE: ...' message, got {got!r}"
    asked.clear()
    got = agent.guard("submit_claim", args, "read_only", yes)
    assert got and got.startswith("NOT DONE:") and asked == [], "read_only: blocked without asking"
    assert agent.guard("submit_claim", args, "auto", no) is None, "auto: runs without asking"


def check_8():
    if not agent.API_KEY:
        raise SkipCheck("no OPENROUTER_API_KEY in .env")
    t = next(x for x in json.loads(agent.TASKS.read_text(encoding="utf-8")) if x["id"] == "t05")
    names = list(tools.FUNCS)
    tr = agent.run_task(agent.LLM(), t, agent.tool_specs(agent.load_descriptions("v2"), names),
                        dict(tools.FUNCS), agent.load_prompt("v4"), "confirm", agent.LIMITS)
    s = agent.score(tr, t, set(names))
    assert s["pass"], f"t05 must pass: used {s['used']}, answer {tr['answer']!r}, stop {tr['stop']}"


class SkipCheck(Exception):
    pass


CHECKS = [("1", "TODO 1  tool_specs: schemas with descriptions from a file", check_1),
          ("2", "TODO 2  execute: errors go back to the model, never a crash", check_2),
          ("3", "TODO 3  run_agent: the loop and its trace", check_3),
          ("4", "TODO 4  check_limits: steps, tokens, $, time, repeats", check_4),
          ("5", "TODO 5  unverified_numbers: numbers no tool returned", check_5),
          ("6", "TODO 6  score: result, path, safety", check_6),
          ("7", "TODO 7  guard: side effects only after a yes", check_7),
          ("8", "live    the real model on t05 (140 + 35 EUR)", check_8)]


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
