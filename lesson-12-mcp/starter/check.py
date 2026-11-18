"""Checks 1-7 run without an API key; check 8 calls the model once.

    python starter/check.py              # your code in starter/
    python starter/check.py --solution   # the reference solution (after the lesson)
Each line is PASS, FAIL (with the reason) or TODO (the function is not written yet).
"""
import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / ("solution" if "--solution" in sys.argv else "starter")


def load(name):
    spec = importlib.util.spec_from_file_location(f"{DIR.name}_{name}", DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


os.environ["CUSTOMER_ID"] = "C-17"
srv = load("server")
host = load("host")
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


async def main():
    async with Client(srv.server) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}

        async def call(name, args):
            return await c.call_tool(name, args)

        # 1 schema and description of search_products
        def c1():
            if "search_products" not in tools:
                raise NotImplementedError("TODO 1")
            t = tools["search_products"]
            props = t.input_schema.get("properties", {})
            assert set(props) >= {"query", "category", "max_price_eur", "in_stock_only"}, f"arguments: {sorted(props)}"
            cat = json.dumps(props["category"])
            assert all(x in cat for x in ("family", "party", "strategy")), "category must be a Literal of 3 values"
            assert "exclusiveMinimum" in json.dumps(props["max_price_eur"]), "max_price_eur needs Field(gt=0)"
            assert t.annotations and t.annotations.read_only_hint, "add ToolAnnotations(read_only_hint=True)"
            assert t.description and len(t.description) > 60, "the docstring is the description: say when to use it"
        report(1, "search_products: schema, annotations, description", c1)

        # 2 behaviour of search_products
        async def c2():
            if "search_products" not in tools:
                raise NotImplementedError("TODO 1")
            r = json.loads((await call("search_products", {"query": "каркасон"})).content[0].text)
            assert [p["sku"] for p in r["results"]] == ["G-001", "G-006"], f"query 'каркасон': {r}"
            r = json.loads((await call("search_products", {"category": "strategy", "in_stock_only": True})).content[0].text)
            assert sorted(p["sku"] for p in r["results"]) == ["G-004", "G-008"], "strategy + in stock"
            r = json.loads((await call("search_products", {"max_price_eur": 20})).content[0].text)
            assert [p["sku"] for p in r["results"]] == ["G-012"], "max_price_eur 20"
            r = json.loads((await call("search_products", {})).content[0].text)
            assert len(r["results"]) == srv.MAX_RESULTS and r["total"] == 12, "at most MAX_RESULTS, total = all matches"
            bad = await call("search_products", {"category": "puzzle"})
            assert bad.is_error, "category 'puzzle' must be rejected by the schema"
        await run_async(2, "search_products: filters, limit, schema rejects bad input", c2)

        # 3 get_order and ownership
        async def c3():
            if "get_order" not in tools:
                raise NotImplementedError("TODO 2")
            r = await call("get_order", {"order_id": "ORD-1002"})
            assert not r.is_error and "NP-20488102" in r.content[0].text, "own order ORD-1002"
            assert "customer_id" not in r.content[0].text, "do not return customer_id"
            foreign = await call("get_order", {"order_id": "ORD-1003"})
            missing = await call("get_order", {"order_id": "ORD-9999"})
            assert foreign.is_error and missing.is_error, "ORD-1003 (other customer) and ORD-9999 must be errors"
            assert "processing" not in foreign.content[0].text and "97.9" not in foreign.content[0].text, "leak"
            assert foreign.content[0].text.replace("1003", "X") == missing.content[0].text.replace("9999", "X"), \
                "the same message for 'other customer' and 'does not exist'"
            assert "not found" in foreign.content[0].text, "use ToolError so the model sees the reason"
            assert (await call("get_order", {"order_id": "1002"})).is_error, "order_id needs pattern ^ORD-\\d{4}$"
        await run_async(3, "get_order: own order, foreign = missing, pattern", c3)

        # 4 resource
        async def c4():
            res = [str(r.uri) for r in (await c.list_resources()).resources]
            if "shop://policy" not in res:
                raise NotImplementedError("TODO 3")
            txt = (await c.read_resource("shop://policy")).contents[0].text
            assert "14 днів" in txt, "return POLICY"
        await run_async(4, "resource shop://policy", c4)

        # 5 mcp_spec
        def c5():
            t = tools.get("get_order") or next(iter(tools.values()), None)
            if t is None:
                raise NotImplementedError("write at least one tool first")
            s = host.mcp_spec(t)
            assert s["type"] == "function" and s["function"]["name"] == t.name
            assert s["function"]["parameters"] == t.input_schema, "use input_schema as is"
            assert s["function"]["description"] == (t.description or "")
        report(5, "mcp_spec: MCP tool -> chat API format", c5)

        # 6 call_mcp
        async def c6():
            if "get_order" not in tools:
                raise NotImplementedError("TODO 2")
            text, st = await host.call_mcp(c, "get_order", '{"order_id": "ORD-1002"}')
            assert st == "ok" and "NP-20488102" in text, (st, text[:80])
            text, st = await host.call_mcp(c, "get_order", '{"order_id": "ORD-1003"}')
            assert st == "error" and text.startswith("ERROR:"), (st, text[:80])
            text, st = await host.call_mcp(c, "get_order", '{"order_id": ')
            assert st == "error" and text.startswith("ERROR:"), "broken JSON"

            class Broken:
                async def call_tool(self, *a, **k):
                    raise ConnectionError("server died")
            text, st = await host.call_mcp(Broken(), "get_order", '{"order_id": "ORD-1002"}')
            assert st == "error" and text.startswith("ERROR:"), "a broken connection must not raise"
        await run_async(6, "call_mcp: ok, isError, broken JSON, broken connection", c6)
    await print_and_live()


async def run_async(n, title, coro_fn):
    try:
        msg = await coro_fn()
        results.append((n, "PASS", title, msg or ""))
    except NotImplementedError as e:
        results.append((n, "TODO", title, str(e)))
    except AssertionError as e:
        results.append((n, "FAIL", title, str(e)))
    except Exception as e:  # noqa: BLE001
        results.append((n, "FAIL", title, f"{type(e).__name__}: {e}"))


async def print_and_live():
    # 7 the real transport: the server as a subprocess over stdio; anything printed to stdout breaks it
    async def c7():
        p = StdioServerParameters(command=sys.executable, args=[str(DIR / "server.py")], env={**os.environ})
        async with Client(p) as c:
            names = [t.name for t in (await c.list_tools()).tools]
        if not names:
            raise NotImplementedError("the server starts, but has no tools yet (TODO 1-2)")
        return f"tools over stdio: {names}"
    await run_async(7, "stdio: the host starts the server as a subprocess", c7)

    # 8 live: one question through the model (needs OPENROUTER_API_KEY)
    if not os.getenv("OPENROUTER_API_KEY"):
        results.append((8, "SKIP", "live: --ask through the model", "no OPENROUTER_API_KEY in .env"))
    else:
        out = subprocess.run([sys.executable, str(DIR / "host.py"), "--server", f"{DIR.name}/server.py",
                              "--ask", "Де моє замовлення ORD-1002?"], capture_output=True, text=True, cwd=ROOT, timeout=120)
        if "NotImplementedError" in out.stderr or "Unknown resource" in out.stderr:
            results.append((8, "TODO", "live: --ask through the model", "TODO 1-5"))
        elif "NP-20488102" in out.stdout:
            results.append((8, "PASS", "live: --ask through the model", out.stdout.strip().splitlines()[-1][:90]))
        else:
            results.append((8, "FAIL", "live: --ask through the model", (out.stdout + out.stderr).strip()[-200:]))
    for n, st, title, msg in sorted(results):
        print(f"{n}. {st:<4} {title}" + (f"  · {msg}" if msg else ""))


asyncio.run(main())
