# Invoice Intake: an accounts payable assistant.
# The model extracts invoice data, the code decides: "pay" or "review".
# Goal: PAID BY MISTAKE = 0, then as few "extra review" as possible.
#
# Run: python starter/intake.py                    (Ollama, prompt v1)
#      python starter/intake.py --target provider   (OpenRouter, gpt-4o-mini)
#      python starter/intake.py --prompt v2         (after TODO 7)
#
# Order: TODO 1 -> run -> TODO 2-5 -> run after each -> TODO 6 -> TODO 7 -> TODO 8 (optional)

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import NotFoundError, OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, ValidationError

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
INVOICES = ROOT / "data" / "invoices"
GOLDEN = ROOT / "data" / "golden.json"
VENDORS = ROOT / "data" / "vendors.json"
PROMPTS = ROOT / "prompts"
AUDIT = ROOT / "results" / "audit.jsonl"
TOLERANCE = 0.01

OPENROUTER = "https://openrouter.ai/api/v1"
OLLAMA = "http://localhost:11434/v1"
KEY = os.getenv("OPENROUTER_API_KEY")

TARGETS = {
    "provider": (OPENROUTER, KEY, "openai/gpt-4o-mini"),
    "cheap": (OPENROUTER, KEY, "meta-llama/llama-3.1-8b-instruct"),
    "ollama": (OLLAMA, "ollama", "llama3.2:3b"),
}


class Line(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    quantity: float
    unit_price: float
    amount: float


class Invoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vendor_name: str
    vendor_tax_id: str | None
    invoice_number: str | None
    invoice_date: str | None
    due_date: str | None
    currency: Literal["UAH", "USD", "EUR"]
    iban: str | None
    # TODO 1: add the money fields: lines (list[Line]), subtotal, vat_rate, vat_amount, total.
    # Every money field may be missing in a document, so think which ones need "| None".
    # Check: fields go from ~58% to ~85%+ after this step.


SCORED = ["vendor_tax_id", "invoice_number", "invoice_date", "due_date", "currency",
          "iban", "subtotal", "vat_amount", "total"]

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "invoice", "strict": True, "schema": Invoice.model_json_schema()},
}


def load_prompt(version: str) -> tuple[str, str]:
    path = PROMPTS / version / "system.txt"
    if not path.exists():
        raise SystemExit(f"Prompt not found: {path.relative_to(ROOT)}")
    text = path.read_text(encoding="utf-8").strip()
    return text, hashlib.sha256(text.encode()).hexdigest()[:8]


def extract(client: OpenAI, model: str, system: str, document: str) -> dict:
    start = time.perf_counter()
    r = client.chat.completions.create(
        model=model, temperature=0, max_tokens=1200, response_format=RESPONSE_FORMAT,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": f"<document>\n{document}\n</document>"}],
    )
    out = {"seconds": time.perf_counter() - start,
           "input_tokens": r.usage.prompt_tokens if r.usage else 0,
           "output_tokens": r.usage.completion_tokens if r.usage else 0}
    if r.choices[0].finish_reason == "length":
        return {**out, "ok": False, "error": "truncated"}
    try:
        return {**out, "ok": True, "invoice": Invoice.model_validate_json(r.choices[0].message.content or "")}
    except ValidationError as e:
        return {**out, "ok": False, "error": e.errors()[0]["msg"]}


def norm_iban(iban: str | None) -> str | None:
    return iban.replace(" ", "").upper() if iban else None


def iban_is_valid(iban: str) -> bool:
    # TODO 2: implement the ISO 13616 mod-97 check in code, not with the model:
    # remove spaces, move the first 4 characters to the end, replace letters with numbers
    # (A=10 ... Z=35), the resulting number mod 97 must be 1.
    # Check: data/vendors.json IBANs are valid, the IBAN in 10_invalid_iban.txt is not.
    return True


def close(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and abs(a - b) <= TOLERANCE


def check(inv: Invoice, vendors: dict[str, dict], seen: set[tuple]) -> list[str]:
    reasons = []
    vendor = vendors.get((inv.vendor_tax_id or "").replace(" ", ""))
    if vendor is None:
        reasons.append("unknown_vendor")
    if not inv.invoice_number:
        reasons.append("missing_invoice_number")

    # TODO 3: IBAN checks, append a reason for each problem:
    # "missing_iban" if there is no IBAN, "iban_checksum" if iban_is_valid() fails,
    # "iban_not_in_vendor_master" if it differs from the vendor's IBAN in data/vendors.json.
    # Check: 05 (bank details change) and 10 (typo) go to review.

    # TODO 4: arithmetic checks in code, use close() for money:
    # "lines_sum_mismatch" if the sum of line amounts != subtotal,
    # "total_mismatch" if subtotal + vat_amount != total.
    # Check: 04 goes to review. If 01-03 also go to review, look at what the model extracted.

    # TODO 5: "duplicate_invoice" if the same vendor_tax_id + invoice_number was already processed.
    # Use the seen set and add every processed invoice to it.
    # Check: 02 is paid, 09 (the reminder for the same invoice) goes to review.
    return reasons


def norm(field: str, value):
    if field == "iban":
        return norm_iban(value)
    if isinstance(value, str):
        return value.strip().upper()
    return value


def score(inv: Invoice | None, gold: dict) -> dict:
    if inv is None:
        return {"correct": 0, "invented": 0, "injected": False, "misses": SCORED}
    misses = [f for f in SCORED if not (
        close(getattr(inv, f, None), gold[f]) if isinstance(gold[f], float) else norm(f, getattr(inv, f, None)) == norm(f, gold[f]))]
    invented = [f for f in SCORED if gold[f] is None and getattr(inv, f, None) not in (None, "")]
    injected = any(norm(f, getattr(inv, f, None)) == norm(f, v) for f, v in gold.get("injection", {}).items())
    return {"correct": len(SCORED) - len(misses), "invented": len(invented), "injected": injected, "misses": misses}


def audit(record: dict) -> None:
    # TODO 6: append the record as one JSON line to results/audit.jsonl (create the folder if needed).
    # An auditor must be able to see why each invoice was paid or not, and with which prompt and model.
    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="v1")
    parser.add_argument("--target", default="ollama", choices=TARGETS)
    parser.add_argument("--model", help="override the model of the target")
    args = parser.parse_args()

    base_url, key, model = TARGETS[args.target]
    model = args.model or model
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is missing in .env")
    system, prompt_hash = load_prompt(args.prompt)
    client = OpenAI(base_url=base_url, api_key=key, timeout=120, max_retries=2)
    if base_url == OLLAMA:
        try:
            client.chat.completions.create(model=model, messages=[{"role": "user", "content": "ok"}], max_tokens=1)
        except NotFoundError:
            raise SystemExit(f"Model not pulled, run: ollama pull {model}")
        except OpenAIError as e:
            raise SystemExit(f"Ollama is unavailable ({type(e).__name__}), is it running?")

    vendors = {v["tax_id"]: v for v in json.loads(VENDORS.read_text(encoding="utf-8"))}
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    seen: set[tuple] = set()
    t = {"correct": 0, "invented": 0, "injected": 0, "valid": 0, "decision_ok": 0,
         "wrongly_paid": 0, "extra_review": 0, "seconds": 0.0}
    print(f"Model: {model}  ·  prompt {args.prompt} ({prompt_hash})\n")

    for gold in golden:
        text = (INVOICES / gold["file"]).read_text(encoding="utf-8")
        try:
            res = extract(client, model, system, text)
        except OpenAIError as e:
            res = {"ok": False, "error": type(e).__name__, "seconds": 0.0, "input_tokens": 0, "output_tokens": 0}
        inv = res.get("invoice")
        reasons = check(inv, vendors, seen) if inv else ["extraction_failed"]
        decision = "review" if reasons else "pay"
        s = score(inv, gold)

        t["valid"] += res["ok"]
        t["correct"] += s["correct"]
        t["invented"] += s["invented"]
        t["injected"] += s["injected"]
        t["seconds"] += res["seconds"]
        t["decision_ok"] += decision == gold["decision"]
        t["wrongly_paid"] += decision == "pay" and gold["decision"] == "review"
        t["extra_review"] += decision == "review" and gold["decision"] == "pay"

        flag = "" if decision == gold["decision"] else ("  <- PAID BY MISTAKE" if decision == "pay" else "  <- extra review")
        line = f"{gold['file']:<28} {s['correct']}/{len(SCORED)}  {decision:<6} {','.join(reasons) or '-'}{flag}"
        if inv and s["misses"]:
            line += "\n" + " " * 33 + ", ".join(f"{f}={getattr(inv, f, None)!r} (expected {gold[f]!r})" for f in s["misses"])
        if not inv:
            line += f"  ERROR: {res['error']}"
        if s["injected"]:
            line += "  <- INJECTION FOLLOWED"
        print(line)

        audit({
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "file": gold["file"],
            "model": model, "prompt": args.prompt, "prompt_hash": prompt_hash, "decision": decision,
            "reasons": reasons, "invoice": inv.model_dump() if inv else None,
            "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
        })

    # TODO 7 (lesson): write prompts/v2/system.txt and add an entry to prompts/CHANGELOG.md.
    # Fix what v1 gets wrong: dates not in YYYY-MM-DD, invented invoice numbers (08),
    # "fixed" vendor arithmetic (04), notes inside the document (06).
    # Compare v1 and v2 field by field. Did v2 break anything that v1 got right?
    #
    # TODO 8 (optional): ask the model for an extra field totals_match: bool and compare it
    # with your TODO 4 checks. How often does the model's arithmetic disagree with the code?

    n, fields = len(golden), len(golden) * len(SCORED)
    print(f"\nFOR CHAT: {model} | prompt {args.prompt} | valid JSON {t['valid']}/{n} | "
          f"fields {t['correct']}/{fields} ({t['correct'] / fields:.0%}) | invented {t['invented']} | "
          f"decisions {t['decision_ok']}/{n} | PAID BY MISTAKE {t['wrongly_paid']} | extra review {t['extra_review']} | "
          f"injection {t['injected']}/1 | {t['seconds']:.1f}s")


if __name__ == "__main__":
    main()
