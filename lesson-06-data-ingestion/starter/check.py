"""Check the ingest output (output/) against data/expected.json.

    python starter/check.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"


def main() -> None:
    expected = json.loads((ROOT / "data" / "expected.json").read_text(encoding="utf-8"))
    if not (OUT / "report.json").exists():
        sys.exit("no output/report.json: run the ingest script first")
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    chunks = [json.loads(line) for line in (OUT / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
    docs = report["docs"]
    by_source: dict[str, str] = {}
    for c in chunks:
        by_source[c["source"]] = by_source.get(c["source"], "") + "\n" + c["text"]
    all_text = "\n".join(by_source.values())

    checks: list[tuple[str, bool, str]] = []

    def check(todo: str, name: str, ok: bool, detail: str = "") -> None:
        checks.append((f"TODO {todo}: {name}", ok, detail))

    check("1", "legacy cp1251 file decoded",
          "захисне взуття" in by_source.get("legacy/warehouse_safety_ua.txt", ""))
    check("2", "HTML boilerplate removed",
          not any(s in all_text for s in ("Accept all cookies", "dataLayer", "All rights reserved")),
          "cookie banner, scripts or footer still in chunks")
    check("3", "scanned PDF flagged for OCR",
          all(docs.get(s, {}).get("status") == "ocr_required" for s in expected["ocr_required"]))
    check("4", "DOCX table kept as rows",
          all(t in by_source.get("it/vpn_setup.docx", "") for t in expected["must_contain"]["it/vpn_setup.docx"]))
    bad = [s for s in expected["must_not_contain"] if s in all_text]
    check("5", "headers, page numbers, zero-width chars removed", not bad, f"found: {[repr(b) for b in bad]}")
    check("5", "hyphenation across lines fixed",
          "lease renegotiation" in by_source.get("finance/q3_budget_summary.pdf", ""))

    dup, original = expected["exact_duplicates"][0]
    check("6", "exact duplicate dropped", docs.get(dup, {}).get("status") == "exact_duplicate"
          and docs.get(original, {}).get("status") == "ok")
    older, newer = expected["near_duplicates"][0]
    check("7", "near duplicate: older version dropped, newer kept",
          docs.get(older, {}).get("status") == "near_duplicate" and docs.get(newer, {}).get("status") == "ok")

    mid = [c["chunk_id"] for c in chunks if c["text"][:1].islower()]
    cut = [c["chunk_id"] for c in chunks
           if any("|" in line and not (line.startswith("|") and line.endswith("|"))
                  for line in c["text"].strip().split("\n"))]
    check("8", "chunks keep sentences and table rows whole", bool(chunks) and not mid and not cut,
          f"starts mid-sentence: {mid[:3]}, cut table rows: {cut[:3]}")
    check("8", "chunks know their section",
          any(c.get("section") for c in chunks) and
          "| Hotel, per night | 120 EUR | Team lead |" in by_source.get("handbook/expense_policy.md", ""))

    missing = sorted({f for c in chunks for f in expected["chunk_fields"] if f not in c})
    check("9", "chunk metadata fields present", bool(chunks) and not missing, f"missing: {missing}")
    wrong = [s for s, groups in expected["access"].items()
             if not any(c["source"] == s and c.get("access") == groups for c in chunks)]
    check("9", "access groups from data/access.json", not wrong, f"wrong for: {wrong}")

    manifest = OUT / "manifest.json"
    ok = manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")).get("dataset_version")
    check("10", "manifest with dataset_version written", bool(ok))
    check("10", "audit log written", (OUT / "audit.jsonl").exists())

    passed = 0
    for name, ok, detail in checks:
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok or not detail else f"  ({detail})"))
    print(f"\n{passed}/{len(checks)} checks passed")


if __name__ == "__main__":
    main()
