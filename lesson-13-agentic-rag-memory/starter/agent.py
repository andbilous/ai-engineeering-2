"""Support agent of the e-bike maker «Вольт» (TODO 3, TODO 4): agentic RAG, long-term memory, a context budget and a critic.

    python starter/agent.py --chat                    # talk to it; /new starts a new session, /down <comment> is feedback
    python starter/agent.py --eval                    # all scenarios from data/scenarios.json
    python starter/agent.py --eval --no-memory        # every session starts from scratch
    python starter/agent.py --eval --fixed-rag        # always search with the raw question, no search tool
    python starter/agent.py --eval --no-critic        # answers go out without the citation check
    python starter/agent.py --eval --budget 0         # no context budget: the whole history goes in
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

from context import build_messages, memory_block  # noqa: E402
from kb import KB, MODELS  # noqa: E402
from memory import FACT_KEYS, MemoryStore  # noqa: E402

load_dotenv(ROOT / ".env")
MODEL = "openai/gpt-4o-mini"
PRICE_IN, PRICE_OUT = 0.15, 0.60          # USD per 1M tokens, check openrouter.ai/models
MAX_STEPS = 5
CRITIC_RETRIES = 1
K = 3

SYSTEM = """Ти асистент підтримки виробника електровелосипедів «Вольт». Моделі: V1 City, V2 Trek, V3 Cargo.
Відповідай українською, по суті.
Факти про велосипеди, помилки, гарантію, сервіс і зарядку бери лише з search_docs, не з власних знань.
Коди помилок і характеристики різні для різних моделей: шукай з фільтром model, якщо модель користувача відома.
Якщо модель невідома, а відповідь від неї залежить, спитай, яка в користувача модель.
Кожен факт з документації підкріплюй id документа у квадратних дужках, наприклад [v2-errors].
Коли користувач повідомляє модель велосипеда, місто чи побажання щодо відповідей, збережи це через remember.
Для привітань і подяк пошук не потрібен."""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "Пошук у документації «Вольт»: коди помилок, характеристики, догляд, гарантія, сервісні центри, "
                       "батареї й зарядки. Пиши короткий запит ключовими словами, а не все питання. "
                       "Якщо результат не відповідає на питання, переформулюй запит і шукай ще раз.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "ключові слова, напр. «помилка E3» або «тиск шин»"},
            "model": {"type": "string", "enum": list(MODELS),
                      "description": "модель велосипеда; без неї пошук іде по всіх моделях"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "remember",
        "description": "Зберегти факт про користувача на майбутні розмови: модель велосипеда, місто або побажання "
                       "щодо форми відповідей. Не зберігай сюди правила гарантії, ціни чи інші факти про продукт.",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string", "enum": list(FACT_KEYS)},
            "value": {"type": "string", "description": "коротко: «V2», «Львів», «відповідати коротко»"}},
            "required": ["key", "value"]}}},
]


# TODO 3 · Goal: run_search(args, profile, kb) executes the search_docs tool and returns (text for the model, ids shown).
#   Why: this is the "agentic" part of RAG: the model decides whether to search and with what query; your code decides
#   how. E3 means three different things on V1, V2 and V3, so a search without the bike model mixes them up.
#   Done when: model = args["model"] if given, else profile["bike_model"], else None (all models); kb.search(query,
#   model, k=K); each hit becomes a line "[id] title: text"; nothing found -> a text that says so and suggests other
#   keywords, with an empty id list (check 3).
def run_search(args: dict, profile: dict, kb: KB) -> tuple[str, list[str]]:
    raise NotImplementedError("TODO 3")


# TODO 4 · Goal: critic(answer, retrieved, searched) checks the answer before it goes to the user: None if it is fine,
#   otherwise a short instruction that the agent sends back to the model for one more try.
#   Why: the model writes [v1-charging] when no such document was shown, or answers from memory without searching.
#   A cheap code check catches both without a second model.
#   Done when: ids are found with the CITE regex; any cited id not in retrieved -> a message that names it;
#   searched=True and no citation -> a message asking for citations; otherwise None (check 4).
CITE = re.compile(r"\[([a-z0-9]+(?:-[a-z0-9]+)+)\]")     # [v2-errors] -> v2-errors


def critic(answer: str, retrieved: set[str], searched: bool) -> str | None:
    raise NotImplementedError("TODO 4")


# ------------------------------------------------------------ the agent (ready)
class Agent:
    def __init__(self, llm, store: MemoryStore, kb: KB, *, memory=True, fixed_rag=False, use_critic=True,
                 budget: int | None = 1500):
        self.llm, self.store, self.kb = llm, store, kb
        self.memory, self.fixed_rag, self.use_critic, self.budget = memory, fixed_rag, use_critic, budget

    def tools(self):
        return [t for t in TOOLS if not (self.fixed_rag and t["function"]["name"] == "search_docs")]

    def turn(self, user: str, question: str, history: list[dict]) -> dict:
        profile = self.store.profile(user)
        mem = memory_block(profile, self.store.feedback_rules(user), self.store.recent_episodes(user)) if self.memory else ""
        retrieved: list[str] = []
        q = question
        if self.fixed_rag:                                   # the classic pipeline: one search with the raw question
            text, ids = run_search({"query": question}, {}, self.kb)
            retrieved += ids
            q = f"{question}\n\nДокументи для відповіді:\n{text}"
        messages, dropped = build_messages(SYSTEM, mem, history, q, self.budget)
        tr = {"question": question, "answer": "", "calls": [], "critic": [], "dropped": dropped,
              "tokens_in": 0, "tokens_out": 0, "first_prompt_tokens": None, "stop": "max_steps"}
        retries = 0
        for step in range(1, MAX_STEPS + 1):
            r = self.llm.chat.completions.create(model=MODEL, messages=messages, tools=self.tools(),
                                                 temperature=0, max_tokens=500)
            m = r.choices[0].message
            tr["tokens_in"] += r.usage.prompt_tokens
            tr["tokens_out"] += r.usage.completion_tokens
            tr["first_prompt_tokens"] = tr["first_prompt_tokens"] or r.usage.prompt_tokens
            messages.append(m.model_dump(exclude_none=True))
            if not m.tool_calls:
                answer = m.content or ""
                searched = self.fixed_rag or any(c["name"] == "search_docs" for c in tr["calls"])
                verdict = critic(answer, set(retrieved), searched) if self.use_critic else None
                if verdict and retries < CRITIC_RETRIES:
                    retries += 1
                    tr["critic"].append(verdict)
                    messages.append({"role": "system", "content": "Перевірка відповіді: " + verdict})
                    continue
                tr["answer"], tr["stop"] = answer, "answer"
                break
            for c in m.tool_calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                    if c.function.name == "search_docs":
                        text, ids = run_search(args, profile, self.kb)
                        retrieved += ids
                    elif c.function.name == "remember":
                        text = self.store.remember(user, args.get("key", ""), args.get("value", ""))
                        profile = self.store.profile(user)
                    else:
                        text, ids = f"ERROR: unknown tool {c.function.name}", []
                except (ValueError, json.JSONDecodeError) as e:
                    text, ids = f"ERROR: {e}", []
                tr["calls"].append({"step": step, "name": c.function.name, "args": c.function.arguments,
                                    "result": text[:200], "ids": ids if c.function.name == "search_docs" else []})
                messages.append({"role": "tool", "tool_call_id": c.id, "content": text})
        tr["retrieved"] = retrieved
        tr["usd"] = (tr["tokens_in"] * PRICE_IN + tr["tokens_out"] * PRICE_OUT) / 1e6
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": tr["answer"]}]
        return tr

    def end_session(self, user: str, session: int, traces: list[dict]) -> None:
        """Episodic memory: a short, deterministic summary of the session (no extra model call)."""
        qs = " | ".join(t["question"][:70] for t in traces)
        ids = sorted({i for t in traces for i in t["retrieved"]})
        self.store.add_episode(user, session, f"питання: {qs}" + (f"; документи: {', '.join(ids)}" if ids else ""))


# ------------------------------------------------------------ eval (ready)
def check(tr: dict, c: dict) -> dict:
    a = tr["answer"].lower()
    searches = [x for x in tr["calls"] if x["name"] == "search_docs"]
    res = {}
    if "must_contain" in c:
        res["contains"] = all(s.lower() in a for s in c["must_contain"])
    if "must_contain_any" in c:
        res["contains_any"] = any(s.lower() in a for s in c["must_contain_any"])
    if "must_not_contain" in c:
        res["not_contains"] = not any(s.lower() in a for s in c["must_not_contain"])
    if "max_chars" in c:
        res["max_chars"] = len(tr["answer"]) <= c["max_chars"]
    if "expect_search" in c:
        res["search"] = bool(searches) == c["expect_search"]
    if "search_args_contain" in c:
        res["search_args"] = any(c["search_args_contain"].lower() in x["args"].lower() for x in searches)
    if "expect_fact" in c:
        res["fact"] = True                                  # filled in by run_scenario from the memory store
    return res


def run_scenario(llm, kb, sc: dict, cfg: dict) -> dict:
    store = MemoryStore()
    agent = Agent(llm, store, kb, **cfg)
    user, turns, checks = sc["id"], [], []
    for n, session in enumerate(sc["sessions"], 1):
        if not cfg["memory"]:
            agent.store = store = MemoryStore()               # nothing survives between sessions
        history, traces = [], []
        for item in session:
            if "feedback" in item:
                store.record_feedback(user, item["feedback"], item.get("comment", ""))
                continue
            if "padding" in item:
                history += json.loads(json.dumps(sc["padding"]))  # earlier turns of this session, without model calls
                continue
            tr = agent.turn(user, item["user"], history)
            tr["session"] = n
            traces.append(tr)
            if "check" in item:
                res = check(tr, item["check"])
                if "expect_fact" in item["check"]:
                    k, v = item["check"]["expect_fact"]
                    res["fact"] = store.profile(user).get(k) == v
                cited = set(CITE.findall(tr["answer"]))
                checks.append({"session": n, "question": item["user"], "ok": all(res.values()), "res": res,
                               "cited_ok": cited <= set(tr["retrieved"]), "cited": sorted(cited)})
        turns += traces
        agent.end_session(user, n, traces)
    return {"id": sc["id"], "kind": sc["kind"], "ok": all(c["ok"] for c in checks), "checks": checks,
            "usd": sum(t["usd"] for t in turns), "tokens_in": sum(t["tokens_in"] for t in turns),
            "calls": sum(len(t["calls"]) for t in turns), "critic_retries": sum(len(t["critic"]) for t in turns),
            "turns": turns, "memory": store.data.get(user, {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--only", help="comma-separated scenario ids")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--fixed-rag", action="store_true")
    ap.add_argument("--no-critic", action="store_true")
    ap.add_argument("--budget", type=int, default=1500, help="0 = no budget")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is missing: copy .env.example to .env and paste your key")
    llm = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=60)
    kb = KB()
    cfg = {"memory": not a.no_memory, "fixed_rag": a.fixed_rag, "use_critic": not a.no_critic,
           "budget": a.budget or None}
    if a.chat:
        store = MemoryStore(ROOT / "output" / "memory.json")
        agent, history, traces, session = Agent(llm, store, kb, **cfg), [], [], 1
        print("/new — нова сесія, /down <коментар> — поганий відгук, /quit — вихід")
        while (q := input("> ").strip()) != "/quit":
            if q == "/new":
                agent.end_session("me", session, traces)
                history, traces, session = [], [], session + 1
                continue
            if q.startswith("/down"):
                store.record_feedback("me", "down", q[5:].strip())
                continue
            tr = agent.turn("me", q, history)
            traces.append(tr)
            for c in tr["calls"]:
                print(f"  {c['name']}({c['args']}) -> {c['result'][:80]!r}")
            for v in tr["critic"]:
                print(f"  critic: {v[:90]}")
            print(tr["answer"])
        agent.end_session("me", session, traces)
        return
    scenarios = json.loads((ROOT / "data" / "scenarios.json").read_text(encoding="utf-8"))
    if a.only:
        scenarios = [s for s in scenarios if s["id"] in a.only.split(",")]
    rows = []
    for sc in scenarios:
        t0 = time.perf_counter()
        r = run_scenario(llm, kb, sc, cfg)
        rows.append(r)
        last = r["turns"][-1]["answer"] if r["turns"] else ""
        print(f"{sc['id']} {sc['kind']:<13} ok={r['ok']!s:<5} calls={r['calls']} critic={r['critic_retries']} "
              f"{time.perf_counter() - t0:4.1f}s | {last[:80]!r}")
    n = len(rows)
    checks = [c for r in rows for c in r["checks"]]
    summary = {"scenarios": n, "ok": round(sum(r["ok"] for r in rows) / n, 2),
               "checks_ok": round(sum(c["ok"] for c in checks) / len(checks), 2),
               "cited_ok": round(sum(c["cited_ok"] for c in checks) / len(checks), 2),
               "critic_retries": sum(r["critic_retries"] for r in rows),
               "tokens_in": sum(r["tokens_in"] for r in rows), "usd": round(sum(r["usd"] for r in rows), 4),
               "model": MODEL, **cfg}
    print(json.dumps(summary, ensure_ascii=False))
    tag = a.tag or "-".join(k for k, on in (("nomem", a.no_memory), ("fixed", a.fixed_rag), ("nocritic", a.no_critic),
                                              ("nobudget", not a.budget)) if on) or "default"
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    (out / f"eval-{tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print("->", out / f"eval-{tag}.json")


if __name__ == "__main__":
    main()
