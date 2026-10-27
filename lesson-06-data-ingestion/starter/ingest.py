"""Lesson 6: ingest a folder of heterogeneous documents into clean, deduplicated chunks.

    python starter/ingest.py             # ingest data/raw -> output/
    python starter/ingest.py --full      # ignore the manifest and re-ingest everything
    python starter/check.py              # check the output against data/expected.json

Pipeline: extract text -> normalize -> deduplicate -> chunk -> attach metadata -> manifest.
TODO 1-5 at home (extract, normalize), TODO 6-10 in the lesson.
"""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup
from charset_normalizer import from_bytes
from docx import Document
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
ACCESS = ROOT / "data" / "access.json"
OUT = ROOT / "output"

PIPELINE_VERSION = "ingest-v1"
CHUNK_TOKENS = 120          # max tokens per chunk (small, because the demo documents are short)
FIXED_OVERLAP = 20          # overlap for the fixed-size baseline
NEAR_DUP_JACCARD = 0.8      # shingle overlap above which two documents are near-duplicates
MIN_CHARS_PER_PAGE = 50     # a PDF page with less text than this has no usable text layer

ENC = tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


@dataclass
class Doc:
    source: str                       # path relative to data/raw, e.g. "it/vpn_setup.docx"
    raw_hash: str                     # sha256 of the file bytes: did the file change?
    text: str = ""
    title: str = ""
    effective: str | None = None      # "Effective YYYY-MM-DD" found in the document
    status: str = "ok"                # ok | ocr_required | exact_duplicate | near_duplicate | error
    note: str = ""
    doc_hash: str = ""                # sha256 of the normalized text: is the content the same?
    unchanged: bool = False           # same file as in the last run: reuse its chunks
    chunks: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------- 1. extract

def read_text_file(path: Path) -> str:
    """TODO 1: .txt / .md. Detect the encoding instead of assuming UTF-8."""
    # Hint: charset_normalizer.from_bytes(data).best(); strip the BOM ("\ufeff").
    return path.read_text(encoding="utf-8", errors="replace")


def extract_html(path: Path) -> str:
    """TODO 2: keep the article, drop navigation, cookie banners, scripts and the footer.
    Headings become markdown headings so the chunker can see the structure."""
    # Hint: BeautifulSoup; decompose script/style/nav/footer and the cookie banner;
    # take <main>; turn <h1>-<h3> into "# ", "## ", "### " lines and <li> into "- " lines.
    return read_text_file(path)


def extract_pdf(path: Path) -> tuple[str, bool]:
    """TODO 3: text per page, pages separated by \\f. Returns (text, needs_ocr).
    A scan has pages but (almost) no text layer: we flag it instead of indexing nothing."""
    # Hint: compare the number of characters with MIN_CHARS_PER_PAGE * number of pages.
    reader = PdfReader(path)
    return "\f".join(page.extract_text() or "" for page in reader.pages), False


def extract_docx(path: Path) -> str:
    """TODO 4: keep headings (as markdown), numbered lists and tables (as markdown rows).
    python-docx lists paragraphs and tables separately, so walk the body in document order."""
    # Hint: Title -> "# ", "Heading 1" -> "## ", "List Number" -> "1. ", a table -> "| a | b |" rows.
    doc = Document(path)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract(path: Path) -> tuple[str, bool]:
    """Dispatch by file type. Returns (text, needs_ocr)."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return read_text_file(path), False
    if suffix in (".html", ".htm"):
        return extract_html(path), False
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix == ".docx":
        return extract_docx(path), False
    raise ValueError(f"unsupported file type: {suffix}")


# ---------------------------------------------------------------- 2. normalize

PAGE_NUMBER = re.compile(r"^\s*(page\s+)?\d+\s*(of|/)\s*\d+\s*$", re.IGNORECASE)
HEADING = re.compile(r"^(#{1,6} |\d+\.\s+[A-ZА-ЯІЇЄҐ])")


def is_heading(line: str) -> bool:
    """A markdown heading, or a short numbered title like "2. Eligibility" (a PDF has no styles).
    "1. Open the Software Center." is a list item, not a heading: it is a sentence."""
    line = line.strip()
    if line.startswith("#"):
        return bool(HEADING.match(line))
    return bool(HEADING.match(line)) and len(line) < 60 and not line.endswith((".", ":", ";"))




def unwrap(text: str) -> str:
    """Join lines that a PDF broke inside a sentence; keep headings, list items and table rows."""
    out: list[str] = []
    for line in text.split("\n"):
        prev = out[-1] if out else ""
        joinable = (prev and line and not HEADING.match(line) and not HEADING.match(prev)
                    and not line.lstrip().startswith(("- ", "|")) and not prev.lstrip().startswith("|")
                    and not re.match(r"^\d+\.\s", line) and not prev.rstrip().endswith((".", ":", "!", "?")))
        if joinable:
            out[-1] = f"{prev.rstrip()} {line.lstrip()}"
        else:
            out.append(line)
    return "\n".join(out)


def normalize(text: str) -> str:
    """TODO 5: one canonical text form, so that hashing, dedup and chunking see the same thing."""
    # Steps, in this order:
    # 1. unicodedata.normalize("NFKC", ...), CRLF -> LF, remove zero-width characters and the BOM
    # 2. split pages on "\f", remove lines that repeat on most pages (headers, footers)
    # 3. remove page numbers ("Page 2 of 3") with PAGE_NUMBER
    # 4. join words hyphenated across lines: "secur-\nity" -> "security"
    # 5. strip trailing spaces, collapse runs of spaces, unwrap(text), max one blank line
    return text


# ---------------------------------------------------------------- 3. deduplicate

def content_hash(text: str) -> str:
    """TODO 6: the same content gives the same hash, whatever the line endings or spacing."""
    # Hint: hash a canonical form, not the raw string.
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.casefold())
    return {tuple(words[i:i + k]) for i in range(max(1, len(words) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def near_duplicates(docs: list[Doc]) -> list[tuple[Doc, Doc, float]]:
    """TODO 7: pairs of documents whose word 5-shingles overlap above NEAR_DUP_JACCARD.
    O(n^2) is fine for a folder; for millions of documents use MinHash + LSH."""
    # Hint: shingles() and jaccard() are ready; compare every pair once.
    return []


# ---------------------------------------------------------------- 4. chunk

def chunk_fixed(text: str, size: int = CHUNK_TOKENS, overlap: int = FIXED_OVERLAP) -> list[dict]:
    """Baseline: a sliding window of tokens. Simple, but cuts sentences, tables and sections."""
    tokens = ENC.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        piece = ENC.decode(tokens[start:start + size])
        chunks.append({"text": piece, "section": ""})
        if start + size >= len(tokens):
            break
        start += size - overlap
    return chunks


def chunk_by_headings(text: str, max_tokens: int = CHUNK_TOKENS) -> list[dict]:
    """TODO 8: split on headings, then pack whole paragraphs up to max_tokens.
    A paragraph or a table is never cut in the middle; each chunk knows its section path."""
    # Hint: split the text into blocks on blank lines; is_heading(first line) starts a new section;
    # keep the heading path (["Expense Policy", "Limits"]) and pack paragraphs while n_tokens() fits.
    return chunk_fixed(text, max_tokens, FIXED_OVERLAP)


def chunk_stats(chunks: list[dict]) -> dict:
    """How often a chunk starts mid-sentence or cuts a table: the cost of naive chunking."""
    mid = sum(1 for c in chunks if c["text"][:1].islower())
    cut_tables = sum(1 for c in chunks if any("|" in line and not (line.startswith("|") and line.endswith("|"))
                                              for line in c["text"].strip().split("\n")))
    sizes = [n_tokens(c["text"]) for c in chunks] or [0]
    return {"chunks": len(chunks), "starts_mid_sentence": mid, "cut_tables": cut_tables,
            "avg_tokens": round(sum(sizes) / len(sizes)), "max_tokens": max(sizes)}


# ---------------------------------------------------------------- 5. metadata

def load_access() -> dict[str, list[str]]:
    rules = json.loads(ACCESS.read_text(encoding="utf-8"))
    return {k: v for k, v in rules.items() if not k.startswith("_")}


def access_for(source: str, rules: dict[str, list[str]]) -> list[str]:
    prefix = max((p for p in rules if source.startswith(p)), key=len)
    return rules[prefix]


def title_of(doc: Doc) -> str:
    for line in doc.text.splitlines():
        if line.strip():
            return line.lstrip("#").strip()[:120]
    return Path(doc.source).stem


def chunk_records(doc: Doc, chunks: list[dict], rules: dict[str, list[str]], now: str) -> list[dict]:
    """TODO 9: everything retrieval will need later: where it came from, who may see it,
    which version of the document and of the pipeline produced it."""
    # Fields: chunk_id, source, title, section, access (access_for), effective, doc_hash,
    # chunk_index, n_tokens, text, pipeline_version, ingested_at
    return [{"chunk_id": f"{doc.source}#{i}", "source": doc.source, "chunk_index": i, "text": c["text"]}
            for i, c in enumerate(chunks)]


# ---------------------------------------------------------------- 6. manifest, incremental runs, audit

def load_manifest() -> dict:
    path = OUT / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"files": {}}


def dataset_version(files: dict) -> str:
    """TODO 10: one id for the exact set of inputs + pipeline version: same inputs, same id."""
    # Hint: sha256 of json.dumps(..., sort_keys=True) over PIPELINE_VERSION and each file's raw_hash.
    return ""


def audit(event: str, source: str, **extra) -> None:
    """TODO 10: record what happened to each file, so any chunk can be traced back."""
    # One JSON line per event in OUT / "audit.jsonl": ts, event, source and the extra fields.
    pass


# ---------------------------------------------------------------- run

def run(full: bool) -> dict:
    OUT.mkdir(exist_ok=True)
    rules = load_access()
    manifest = {"files": {}} if full else load_manifest()
    previous = manifest["files"]
    old_chunks = {}
    if not full and (OUT / "chunks.jsonl").exists():
        for line in (OUT / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            old_chunks.setdefault(c["source"], []).append(c)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    docs: list[Doc] = []
    skipped = 0
    for path in sorted(p for p in RAW.rglob("*") if p.is_file() and not p.name.startswith(".")):
        source = path.relative_to(RAW).as_posix()
        doc = Doc(source=source, raw_hash=hashlib.sha256(path.read_bytes()).hexdigest())
        prev = previous.get(source)
        if prev and prev["raw_hash"] == doc.raw_hash and prev.get("pipeline_version") == PIPELINE_VERSION:
            doc.status, doc.note, doc.doc_hash = prev["status"], prev.get("note", ""), prev.get("doc_hash", "")
            doc.title, doc.effective = prev.get("title", ""), prev.get("effective")
            doc.chunks = old_chunks.get(source, [])
            doc.text = "\n".join(c["text"] for c in doc.chunks)
            doc.unchanged = True
            skipped += 1
            docs.append(doc)
            continue
        try:
            raw_text, needs_ocr = extract(path)
        except Exception as e:  # one broken file must not stop the batch
            doc.status, doc.note = "error", f"{type(e).__name__}: {e}"
            audit("error", source, error=doc.note)
            docs.append(doc)
            continue
        if needs_ocr:
            doc.status, doc.note = "ocr_required", "no text layer: send to OCR, not to the index"
            audit("ocr_required", source)
            docs.append(doc)
            continue
        m = re.search(r"Effective (\d{4}-\d{2}-\d{2})", raw_text)
        doc.effective = m.group(1) if m else None
        doc.text = normalize(raw_text)
        doc.doc_hash = content_hash(doc.text)
        doc.title = title_of(doc)
        docs.append(doc)

    # exact duplicates: same normalized content -> keep the first path, drop the rest
    seen: dict[str, Doc] = {}
    for doc in docs:
        if doc.status != "ok" or doc.unchanged:
            if doc.status == "ok":
                seen.setdefault(doc.doc_hash, doc)
            continue
        if doc.doc_hash in seen:
            doc.status, doc.note = "exact_duplicate", f"same content as {seen[doc.doc_hash].source}"
            audit("exact_duplicate", doc.source, of=seen[doc.doc_hash].source)
        else:
            seen[doc.doc_hash] = doc

    # near duplicates: keep the newer version (by effective date), drop the older one
    active = [d for d in docs if d.status == "ok"]
    for a, b, score in near_duplicates(active):
        older, newer = sorted([a, b], key=lambda d: d.effective or "")
        older.status, older.note = "near_duplicate", f"older version of {newer.source} (jaccard {score:.2f})"
        older.chunks = []
        audit("near_duplicate", older.source, kept=newer.source, jaccard=round(score, 2))

    fixed_all, heading_all = [], []
    for doc in docs:
        if doc.status != "ok" or doc.unchanged:
            continue
        fixed_all += chunk_fixed(doc.text)
        chunks = chunk_by_headings(doc.text)
        heading_all += chunks
        doc.chunks = chunk_records(doc, chunks, rules, now)
        audit("ingested", doc.source, chunks=len(doc.chunks), doc_hash=doc.doc_hash[:12])

    files = {d.source: {"raw_hash": d.raw_hash, "doc_hash": d.doc_hash, "status": d.status, "note": d.note,
                        "title": d.title, "effective": d.effective, "chunks": len(d.chunks),
                        "pipeline_version": PIPELINE_VERSION} for d in docs}
    version = dataset_version(files)
    (OUT / "manifest.json").write_text(json.dumps({"dataset_version": version, "pipeline_version": PIPELINE_VERSION,
                                                   "created_at": now, "files": files}, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    all_chunks = [c for d in docs if d.status == "ok" for c in d.chunks]
    with (OUT / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    report = {"dataset_version": version, "files": len(docs), "skipped_unchanged": skipped,
              "docs": {d.source: {"status": d.status, "note": d.note, "chunks": len(d.chunks)} for d in docs},
              "chunking": {"fixed": chunk_stats(fixed_all), "by_headings": chunk_stats(heading_all)}}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def print_report(report: dict) -> None:
    print(f"{'file':<40} {'status':<16} chunks  note")
    for source, d in report["docs"].items():
        print(f"{source:<40} {d['status']:<16} {d['chunks']:>6}  {d['note']}")
    fixed, heading = report["chunking"]["fixed"], report["chunking"]["by_headings"]
    if fixed["chunks"]:
        print("\nchunking            chunks  mid-sentence  cut tables  avg tok  max tok")
        for name, s in (("fixed 120/20", fixed), ("by headings", heading)):
            print(f"{name:<18} {s['chunks']:>7} {s['starts_mid_sentence']:>13} {s['cut_tables']:>11} "
                  f"{s['avg_tokens']:>8} {s['max_tokens']:>8}")
    ok = sum(1 for d in report["docs"].values() if d["status"] == "ok")
    n_chunks = sum(d["chunks"] for d in report["docs"].values())
    print(f"\nFOR CHAT: files {report['files']} | indexed {ok} | chunks {n_chunks} | "
          f"skipped unchanged {report['skipped_unchanged']} | dataset {report['dataset_version']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="ignore the manifest and re-ingest every file")
    args = ap.parse_args()
    if not RAW.exists():
        sys.exit(f"no input folder {RAW}")
    print_report(run(args.full))
