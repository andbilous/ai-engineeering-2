# Invoice Intake

An accounts payable assistant. Supplier invoices arrive as emails, PDF text and OCR scans.
The model extracts the invoice data, and the code decides what happens next:

- **pay** — every check passed, the invoice can be paid automatically;
- **review** — something is wrong, a person must look at it.

A mistake here costs real money: a wrong IBAN sends the payment to the wrong account.
That is why the model only reads the document, and every decision is made by checks in code.

## Project goal

Build an invoice intake pipeline that a finance team could trust:
**no invoice is ever paid by mistake**, and as few correct invoices as possible are sent to manual review.

In numbers: `PAID BY MISTAKE = 0`, then the lowest possible `extra review`, with fields extracted at 95%+ on the golden set.

By the end you will have practiced the core of prompt engineering as an engineering discipline:

- **Structured output** — a nested schema with invoice lines, money fields and nulls where data is missing.
- **Reducing hallucinations** — the model must not invent an invoice number, a date or an IBAN; the `invented` metric shows it.
- **Model vs code** — arithmetic and IBAN checksums are done in code, not by the model.
- **Prompt injection** — instructions hidden in a document and a fake "bank details change" email.
- **Prompt versioning and testing** — v1 → v2 with a changelog, compared field by field on a golden set to catch regressions.
- **Production practices** — a vendor master, an audit log and human review for anything uncertain.

## Data

| Path | What it is |
|---|---|
| `data/invoices/` | 10 invoices in different formats, see the table below |
| `data/vendors.json` | vendor master: approved vendors and their bank accounts |
| `data/golden.json` | expected fields and the expected decision with reasons for each invoice |
| `prompts/v1/system.txt` | baseline prompt |
| `prompts/CHANGELOG.md` | log of prompt changes and why they were made |
| `starter/intake.py` | the script you complete |

| Invoice | What it tests |
|---|---|
| 01–03 | a clean email, PDF text and an OCR scan with noise: should be paid |
| 04 | the vendor's arithmetic is wrong: the model must not fix it |
| 05 | "our bank details have changed": a classic payment fraud scenario |
| 06 | a hidden note for "automated systems" inside the document |
| 07 | EUR invoice with a German IBAN and reverse charge VAT |
| 08 | no invoice number and no IBAN |
| 09 | a reminder for an invoice that was already processed |
| 10 | an IBAN with a typo in one digit |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env              # paste your OpenRouter key
ollama pull llama3.2:3b           # for the local run
```

## Run

```bash
python starter/intake.py                     # Ollama, prompt v1
python starter/intake.py --target provider   # OpenRouter, gpt-4o-mini
python starter/intake.py --prompt v2         # after task 7
```

Each line shows the decision and its reasons. The last line is the summary:

```
FOR CHAT: llama3.2:3b | prompt v1 | valid JSON 10/10 | fields 52/90 (58%) | invented 4 |
decisions 5/10 | PAID BY MISTAKE 5 | extra review 0 | injection 0/1 | 11.2s
```

This is the starter as is: 5 of 10 invoices are paid by mistake, including the bank details change in 05.

## Review reasons

| Reason | When |
|---|---|
| `unknown_vendor` | the tax ID is not in `vendors.json` |
| `missing_invoice_number` | the invoice has no number |
| `missing_iban` | the invoice has no IBAN |
| `iban_checksum` | the IBAN fails the mod-97 check |
| `iban_not_in_vendor_master` | the IBAN is valid but differs from the vendor's account |
| `lines_sum_mismatch` | line amounts do not add up to the subtotal |
| `total_mismatch` | subtotal + VAT is not equal to the total |
| `duplicate_invoice` | the same vendor and invoice number were already processed |

## Tasks

All tasks are marked `TODO` in `starter/intake.py`. Run the script after each one.

| # | Task | Check |
|---|---|---|
| 1 | Add the money fields to the `Invoice` schema: `lines`, `subtotal`, `vat_rate`, `vat_amount`, `total` | fields go from ~58% to ~85%+ |
| 2 | Implement the IBAN mod-97 check in code | IBANs in `vendors.json` are valid, the one in 10 is not |
| 3 | IBAN checks: missing, wrong checksum, not in the vendor master | 05 and 10 go to review |
| 4 | Arithmetic checks in code: lines vs subtotal, subtotal + VAT vs total | 04 goes to review |
| 5 | Duplicate invoices by vendor and invoice number | 02 is paid, 09 goes to review |
| 6 | Audit log: write every decision to `results/audit.jsonl` | each line shows model, prompt, decision and reasons |
| 7 | Write `prompts/v2/system.txt`, add an entry to `prompts/CHANGELOG.md`, compare v1 and v2 field by field | v2 improves fields and breaks nothing v1 got right |
| 8 | *Optional.* Ask the model for `totals_match: bool` and compare it with your task 4 checks | how often the model's arithmetic disagrees with the code |

Hints for task 7: look at what v1 gets wrong — dates not in `YYYY-MM-DD`, invented invoice numbers, "fixed" arithmetic, notes inside the document.
Be careful with every new rule and example: check that it does not make other fields worse.

## Rules

- The model extracts, the code decides. Never ask the model whether to pay.
- Anything the code can check (IBAN, sums, duplicates) is checked by code.
- Keep keys in `.env` only. Do not commit `.env` or `results/`.
