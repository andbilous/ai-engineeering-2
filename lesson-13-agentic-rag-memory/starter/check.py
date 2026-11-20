"""Checks 1-6 run without an API key; check 7 runs one scenario through the model.

    python starter/check.py              # your code in starter/
    python starter/check.py --solution   # the reference solution (after the lesson)
Each line is PASS, FAIL (with the reason), TODO (the function is not written yet) or SKIP.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / ("solution" if "--solution" in sys.argv else "starter")
sys.path[:0] = [str(DIR), str(ROOT)]

import agent  # noqa: E402
from context import build_messages, estimate_tokens, memory_block  # noqa: E402
from kb import KB  # noqa: E402
from memory import MemoryStore  # noqa: E402

results = []


def report(n, title, fn):
    try:
        msg = fn()
        results.append((n, "PASS", title, msg or ""))
    except NotImplementedError as e:
        results.append((n, "TODO", title, str(e)))
    except AssertionError as e:
        results.append((n, "FAIL", title, str(e)))
    except Exception as e:  # noqa: BLE001
        results.append((n, "FAIL", title, f"{type(e).__name__}: {e}"))


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


# 1 facts
def c1():
    m = MemoryStore()
    m.remember("u", "city", "Київ")
    out = m.remember("u", "city", "  Одеса ")
    assert m.profile("u") == {"city": "Одеса"}, f"the newest value must win: {m.profile('u')}"
    assert "Київ" in out, "tell the model what the old value was: 'saved city=Одеса (was Київ)'"
    m.remember("u", "bike_model", "V2 Trek")
    assert m.profile("u")["bike_model"] == "V2", "bike_model 'V2 Trek' -> 'V2'"
    assert raises(lambda: m.remember("u", "bike_model", "V7")), "bike_model V7 must be rejected"
    assert raises(lambda: m.remember("u", "warranty", "10 років")), "unknown key must be rejected"
    assert raises(lambda: m.remember("u", "preference", "x" * 500)), "a 500-character value must be rejected"
    assert "updated" in m.data["u"]["facts"]["city"], "store {'value': ..., 'updated': ISO time}"
report(1, "remember/profile: newest wins, keys and values validated", c1)


# 2 context
def c2():
    hist = []
    for i in range(10):
        hist += [{"role": "user", "content": f"питання {i} " + "а" * 300},
                 {"role": "assistant", "content": f"відповідь {i} " + "б" * 300}]
    msgs, dropped = build_messages("SYS", "MEM", hist, "Q?", None)
    assert dropped == 0 and len(msgs) == 23, "budget=None keeps everything: system + memory + 20 + question"
    assert [m["content"] for m in msgs[:2]] == ["SYS", "MEM"] and msgs[-1] == {"role": "user", "content": "Q?"}, \
        "order: system, memory, history, question"
    msgs, dropped = build_messages("SYS", "", hist, "Q?", None)
    assert msgs[1]["content"].startswith("питання 0"), "no memory -> no empty system message"
    msgs, dropped = build_messages("SYS", "MEM", hist, "Q?", 1000)
    total = sum(estimate_tokens(m["content"]) for m in msgs)
    assert dropped and dropped % 2 == 0, f"drop whole pairs, dropped={dropped}"
    assert total <= 1000, f"{total} estimated tokens > budget 1000"
    assert "питання 9" in msgs[-3]["content"] and "відповідь 9" in msgs[-2]["content"], "keep the newest pair"
    assert not any("питання 0" in m["content"] for m in msgs), "the oldest pair goes first"
    assert any(f"{dropped} старіших" in m["content"] for m in msgs if m["role"] == "system"), "add the note about dropped"
    msgs, dropped = build_messages("S" * 9000, "MEM", hist, "Q?", 1000)
    assert msgs[0]["content"].startswith("S") and msgs[-1]["content"] == "Q?", "system and question always stay"
    return f"budget 1000: dropped {build_messages('SYS', 'MEM', hist, 'Q?', 1000)[1]} of 20 messages"
report(2, "build_messages: order, budget, whole pairs, note", c2)


# 3 search tool
def c3():
    kb = KB()
    text, ids = agent.run_search({"query": "помилка E3"}, {"bike_model": "V2"}, kb)
    assert ids and ids[0] == "v2-errors", f"profile V2 -> filter by V2: {ids}"
    assert not any(i.startswith(("v1-", "v3-")) for i in ids), f"other models leaked in: {ids}"
    assert "[v2-errors]" in text, "each hit as '[id] title: text'"
    _, ids = agent.run_search({"query": "помилка E3", "model": "V3"}, {"bike_model": "V2"}, kb)
    assert ids[0] == "v3-errors", "an explicit model argument wins over the profile"
    _, ids = agent.run_search({"query": "помилка E3"}, {}, kb)
    assert {"v1-errors", "v2-errors", "v3-errors"} <= set(ids), "no model known -> search all models"
    text, ids = agent.run_search({"query": "квантовий телепорт"}, {}, kb)
    assert ids == [] and text, "nothing found -> say so, empty id list"
report(3, "run_search: model from args or memory, empty result", c3)


# 4 critic
def c4():
    c = agent.critic
    assert c("E3 — перегрів мотора [v2-errors].", {"v2-errors"}, True) is None, "a correct answer must pass"
    bad = c("Використовуйте C1 [v1-charging].", {"all-chargers"}, True)
    assert bad and "v1-charging" in bad, "name the id that was not shown"
    assert c("E3 — перегрів мотора.", {"v2-errors"}, True), "after a search an answer without citations fails"
    assert c("Привіт! Чим допомогти?", set(), False) is None, "no search, no citation -> fine"
    assert c("Гарантія 2 роки [all-warranty].", set(), False), "a citation without a search this turn fails"
report(4, "critic: unknown ids, missing citations", c4)


# 5 feedback
def c5():
    m = MemoryStore()
    assert m.record_feedback("u", "up", "супер") is False, "'up' stores no rule"
    assert m.record_feedback("u", "down", "") is False, "no comment, no rule"
    assert m.record_feedback("u", "down", "Відповідай коротко.") is True
    assert m.record_feedback("u", "down", "Відповідай коротко.") is False, "the same rule once"
    for i in range(7):
        m.record_feedback("u", "down", f"правило {i}")
    rules = m.feedback_rules("u")
    assert len(rules) == 5 and rules[-1] == "правило 6", f"keep the newest 5: {rules}"
    assert raises(lambda: m.record_feedback("u", "meh", "x")), "rating must be up or down"
    assert "правило 6" in memory_block({}, rules, []), "rules reach the context through memory_block"
report(5, "record_feedback: thumbs-down with a comment becomes a rule", c5)


# 6 memory survives a restart
def c6():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "memory.json"
        m = MemoryStore(p)
        m.remember("u", "bike_model", "V3")
        m.add_episode("u", 1, "питання: E3")
        again = MemoryStore(p)
        assert again.profile("u") == {"bike_model": "V3"}, "facts must be saved to the file"
        assert json.loads(p.read_text(encoding="utf-8"))["u"]["episodes"], "episodes too"
report(6, "memory in a file: a new process sees the facts", c6)


# 7 live: scenario s01 through the model
def live():
    import os
    if not os.getenv("OPENROUTER_API_KEY"):
        results.append((7, "SKIP", "live: scenario s01 (two sessions)", "no OPENROUTER_API_KEY in .env"))
        return
    out = subprocess.run([sys.executable, str(DIR / "agent.py"), "--eval", "--only", "s01", "--tag", "check"],
                         capture_output=True, text=True, cwd=ROOT, timeout=180)
    if "NotImplementedError" in out.stderr:
        results.append((7, "TODO", "live: scenario s01 (two sessions)", "TODO 1-4"))
    elif "ok=True" in out.stdout:
        results.append((7, "PASS", "live: scenario s01 (two sessions)", out.stdout.splitlines()[0][:100]))
    else:
        results.append((7, "FAIL", "live: scenario s01 (two sessions)", (out.stdout + out.stderr).strip()[-200:]))


live()
for n, st, title, msg in sorted(results):
    print(f"{n}. {st:<4} {title}" + (f"  · {msg}" if msg else ""))
