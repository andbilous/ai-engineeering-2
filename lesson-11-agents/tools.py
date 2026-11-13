"""The agent's tools, ready to use. Lesson 11 does not change them: it teaches the model to use them well.

    tools.search_docs(tools.SearchArgs(query="hotel limit"))      # chunks the employee may read
    tools.calculate(tools.CalcArgs(expression="min(140, 120) + min(35, 30)"))
    tools.days_between(tools.DaysArgs(date_from="2026-09-20", date_to="2026-11-09"))
    tools.submit_claim(tools.ClaimArgs(...))                     # a side effect: the agent must ask first

Every tool takes one Pydantic model with its arguments and returns a dict that can go to JSON.
The descriptions the model sees are NOT here: they live in prompts/tools-v1.json and prompts/tools-v2.json.
"""
import ast
import itertools
import operator
import sys
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

USER_GROUPS = ["all_employees"]   # the assistant works for an ordinary employee: no finance or warehouse documents


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchArgs(Args):
    query: str = Field(min_length=2, max_length=300)
    k: int = Field(default=4, ge=1, le=8)


class CalcArgs(Args):
    expression: str = Field(min_length=1, max_length=200)


class DaysArgs(Args):
    date_from: date
    date_to: date


class ClaimArgs(Args):
    category: Literal["hotel", "meals", "taxi", "training", "home_office"]
    amount_eur: float = Field(gt=0, le=5000)
    receipt_date: date
    description: str = Field(min_length=3, max_length=200)


# ---------------------------------------------------------------- search_docs

_retriever = None


def search_docs(args: SearchArgs) -> dict:
    """Hybrid search from lesson 7 over the Northwind documents the employee has access to."""
    global _retriever
    if _retriever is None:
        from retrieval import Retriever
        _retriever = Retriever()
    hits = _retriever.search(args.query, USER_GROUPS, args.k)
    return {"results": [{"chunk_id": h["chunk_id"], "section": h["section"], "text": h["text"]} for h in hits]}


# ---------------------------------------------------------------- calculate

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"min": min, "max": max, "round": round, "abs": abs}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("exponent is too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"not allowed in an expression: {ast.unparse(node)!r}")


def calculate(args: CalcArgs) -> dict:
    """Safe arithmetic: numbers, + - * / // % **, brackets, min, max, round, abs. No eval() of arbitrary code."""
    result = _eval(ast.parse(args.expression, mode="eval"))
    return {"expression": args.expression, "result": round(result, 6)}


# ---------------------------------------------------------------- days_between

def days_between(args: DaysArgs) -> dict:
    """Calendar days from date_from to date_to (negative if date_to is earlier)."""
    return {"date_from": args.date_from.isoformat(), "date_to": args.date_to.isoformat(),
            "days": (args.date_to - args.date_from).days}


# ---------------------------------------------------------------- submit_claim (side effect)

SIDE_EFFECTS = {"submit_claim"}    # tools that change something outside the conversation
_claim_ids = itertools.count(1)


def submit_claim(args: ClaimArgs) -> dict:
    """Submit an expense claim. A demo: nothing leaves your laptop, but treat it as if money moved."""
    return {"status": "submitted", "claim_id": f"CLM-{next(_claim_ids):04d}", **args.model_dump(mode="json")}


ARGS = {"search_docs": SearchArgs, "calculate": CalcArgs, "days_between": DaysArgs, "submit_claim": ClaimArgs}
FUNCS = {"search_docs": search_docs, "calculate": calculate, "days_between": days_between, "submit_claim": submit_claim}
