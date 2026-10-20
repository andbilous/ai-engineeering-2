import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, ValidationError

load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-4o-mini"

# BASE_URL = "http://localhost:11434/v1"
# API_KEY = "ollama"
# MODEL = "llama3.2:3b"

ROOT = Path(__file__).resolve().parent.parent
POSTINGS = ROOT / "data" / "postings"
GOLDEN = ROOT / "data" / "golden.json"
PROMPTS = ROOT / "prompts"
LOG = ROOT / "results" / "runs.jsonl"


class JobPosting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    company: str | None
    seniority: Literal["junior", "middle", "senior", "lead", "unknown"]
    work_format: Literal["remote", "hybrid", "office", "unknown"]
    salary_min: int | None
    salary_max: int | None
    # TODO 1: add the remaining fields from data/golden.json:
    # currency (USD / EUR / UAH or null), salary_period (hour / month / year or null),
    # years_experience (int or null), contact_email (str or null)

    # TODO 8 (lesson): add validators for facts the schema cannot check,
    # e.g. salary_max is between salary_min and 3 x salary_min, contact_email is written in the posting.
    # Hint: model_validator(mode="after") and ValidationInfo.context


SCORED = ["company", "seniority", "work_format", "salary_min", "salary_max",
          "currency", "salary_period", "years_experience", "contact_email"]
EMPTY = (None, "unknown")

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "job_posting", "strict": True, "schema": JobPosting.model_json_schema()},
}


def load_prompt(version: str) -> tuple[str, str]:
    # TODO 2: read prompts/<version>/system.txt and return (text, short sha256 hash of the text)
    return "You extract structured data from job postings.", "no-hash"


def build_messages(system: str, posting: str) -> list[dict]:
    # TODO 3: wrap the posting in <posting>...</posting> tags so the model can tell data from instructions
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": posting},
    ]


def extract(client: OpenAI, messages: list[dict], text: str, model: str = MODEL) -> dict:
    start = time.perf_counter()
    r = client.chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=400,
        response_format=RESPONSE_FORMAT,
    )
    result = {
        "seconds": time.perf_counter() - start,
        "input_tokens": r.usage.prompt_tokens if r.usage else 0,
        "output_tokens": r.usage.completion_tokens if r.usage else 0,
    }
    content = r.choices[0].message.content or ""
    # TODO 4: if finish_reason is "length", return ok=False with an error;
    # otherwise validate content with JobPosting: on success set result["posting"] and ok=True,
    # on ValidationError set ok=False and result["error"]
    # TODO 8 (lesson): pass context={"text": text} to validation; on ValidationError send the error
    # back to the model once and ask for corrected JSON (repair)
    return {**result, "ok": False, "error": "not validated yet (TODO 4)"}


def score(pred: JobPosting | None, gold: dict) -> dict:
    # TODO 5: compare pred with gold for every field in SCORED (strings case-insensitive).
    # correct: how many fields match; misses: names of fields that do not match;
    # invented: fields where gold is empty (EMPTY) but pred is not;
    # injected: True if pred contains a value from gold["injection"]
    return {"correct": 0, "invented": 0, "injected": False, "misses": []}


def log_run(record: dict) -> None:
    # TODO 6: append record as one JSON line to results/runs.jsonl (create the folder if needed)
    pass


def main() -> None:
    version = sys.argv[1] if len(sys.argv) > 1 else "v1"
    if not API_KEY:
        raise SystemExit("OPENROUTER_API_KEY is missing in .env")
    system, prompt_hash = load_prompt(version)
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=60, max_retries=2)
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    print(f"Model: {MODEL}  ·  prompt {version} ({prompt_hash})\n")
    totals = {"correct": 0, "invented": 0, "injected": 0, "valid": 0, "seconds": 0.0}

    for gold in golden:
        text = (POSTINGS / gold["file"]).read_text(encoding="utf-8")
        try:
            res = extract(client, build_messages(system, text), text)
        except OpenAIError as e:
            res = {"ok": False, "error": type(e).__name__, "seconds": 0.0, "input_tokens": 0, "output_tokens": 0}
        s = score(res.get("posting"), gold)

        totals["valid"] += res["ok"]
        totals["correct"] += s["correct"]
        totals["invented"] += s["invented"]
        totals["injected"] += s["injected"]
        totals["seconds"] += res["seconds"]

        line = f"{gold['file']:<26} {res['seconds']:>5.1f}s  {s['correct']}/{len(SCORED)}"
        if res["ok"]:
            p = res["posting"]
            line += "  " + ", ".join(f"{f}={getattr(p, f, None)!r} (expected {gold[f]!r})" for f in s["misses"])
        else:
            line += f"  ERROR: {res['error']}"
        if s["injected"]:
            line += "  <- INJECTION FOLLOWED"
        print(line)

        log_run({
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "file": gold["file"], "model": MODEL, "prompt": version, "prompt_hash": prompt_hash,
            "temperature": 0, "ok": res["ok"], "seconds": round(res["seconds"], 2),
            "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
            "output": res["posting"].model_dump() if res["ok"] else None, "misses": s["misses"],
        })

    # TODO 7 (lesson): write prompts/v2/system.txt, add an entry to prompts/CHANGELOG.md,
    # run "python starter/extract.py v2" and compare with v1 field by field

    n, fields = len(golden), len(golden) * len(SCORED)
    print(f"\nFOR CHAT: {MODEL} | prompt {version} | valid JSON {totals['valid']}/{n} | "
          f"fields {totals['correct']}/{fields} ({totals['correct'] / fields:.0%}) | "
          f"invented {totals['invented']} | injection {totals['injected']}/1 | {totals['seconds']:.1f}s")


if __name__ == "__main__":
    main()
