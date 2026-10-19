import json
import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

# TODO 1
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-4o-mini"

# MODEL = "meta-llama/llama-3.1-8b-instruct"

# BASE_URL = "http://localhost:11434/v1"
# API_KEY = "ollama"
# MODEL = "llama3.2:3b"

# TODO 2
PRICES = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "meta-llama/llama-3.1-8b-instruct": (0.05, 0.08),
}

DATA = Path(__file__).resolve().parent.parent / "data" / "tickets.json"


class Ticket(BaseModel):
    category: Literal["delivery", "payment", "refund", "technical", "other"]
    priority: Literal["low", "medium", "high"]
    order_id: str | None = Field(description="Order number, only if it is present in the text. Otherwise null.")
    summary: str = Field(description="Short summary of the request, up to 15 words.")


SYSTEM_PROMPT = (
    "You classify customer support requests for an online store. "
    "Respond only with JSON that matches the given schema. "
    "category: delivery — shipping and parcels; payment — payments and card charges; "
    "refund — the customer wants their money back or to return an item; "
    "technical — problems with the website, account or app; other — anything else. "
    "priority: high — the customer is angry, threatens, or money is at stake; "
    "medium — a regular problem; low — a simple question. "
    "order_id: only if the order number is explicitly present in the text, otherwise null. Never invent values."
)


def classify(client: OpenAI, text: str) -> dict:
    start = time.perf_counter()
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "ticket", "schema": Ticket.model_json_schema()},
            },
        )
    except OpenAIError as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "seconds": time.perf_counter() - start}

    seconds = time.perf_counter() - start
    content = r.choices[0].message.content or ""
    result = {
        "seconds": seconds,
        "input_tokens": r.usage.prompt_tokens if r.usage else 0,
        "output_tokens": r.usage.completion_tokens if r.usage else 0,
        "raw": content,
    }
    try:
        result["ticket"] = Ticket.model_validate_json(content)
        result["ok"] = True
    except ValidationError as e:
        result["ok"] = False
        result["error"] = f"invalid JSON: {e.errors()[0]['msg']}"
    return result


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES.get(MODEL, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def main() -> None:
    if not API_KEY:
        raise SystemExit("OPENROUTER_API_KEY is missing in .env")
    if "localhost" not in BASE_URL and MODEL not in PRICES:
        print(f"Warning: no price for {MODEL} in PRICES, cost will be 0 (TODO 2)\n")

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=30, max_retries=2)
    tickets = json.loads(DATA.read_text(encoding="utf-8"))

    print(f"Model: {MODEL}  ·  {BASE_URL}\n")
    total_s = total_cost = 0.0
    valid = correct = 0

    for t in tickets:
        res = classify(client, t["text"])
        total_s += res["seconds"]
        if "input_tokens" in res:
            total_cost += cost_usd(res["input_tokens"], res["output_tokens"])

        if res["ok"]:
            tk = res["ticket"]
            valid += 1
            correct += tk.category == t["expected_category"]
            mark = "✓" if tk.category == t["expected_category"] else f"✗ (expected {t['expected_category']})"
            print(f"#{t['id']}  {res['seconds']:.1f}s  {tk.category:<9} {mark}  {tk.priority:<6}  "
                  f"order_id={tk.order_id}  {tk.summary}")
        else:
            print(f"#{t['id']}  {res['seconds']:.1f}s  ERROR: {res['error']}")

    # TODO 3

    n = len(tickets)
    print(f"\nFOR CHAT: {MODEL} | {n} requests | {total_s:.1f}s | ${total_cost:.6f} | "
          f"valid JSON {valid}/{n} | category {correct}/{n}")


if __name__ == "__main__":
    main()
