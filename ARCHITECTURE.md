# Invoice → Decision — Multi-Agent Architecture (PS-1)

> Takes an invoice (clean PDF, scanned image, or messy mixed format) and produces a
> **routed, reasoned AP decision** with a full audit trail. Every number is traceable
> to where it came from on the page.

## The one question the system answers
**Should we pay this invoice — in full, now — and can we prove why?**

Three failure modes it exists to prevent:
1. Paying for what we didn't order/receive → **PO match + tolerances**
2. Paying twice → **duplicate detection**
3. Paying a fraudster / wrong bank account → **vendor master + bank-change check**

## Design principles
- **One shared state object** (`InvoiceCase`). Agents don't message each other; each
  reads the case, does its job, writes its slice back. The case *is* the audit trail
  and the thing the UI renders.
- **Provenance over trust.** Every extracted value carries confidence + source
  (page/bbox/raw text). A reviewer can click any number and see the crop it came from.
- **Decisions are routed, never bare yes/no.** Output is auto-approve / review / hold /
  reject, each with specific `IssueCode`s and required actions.
- **Deterministic rules decide; the LLM only extracts/reads.** Money decisions run
  through a transparent rule engine, not a model's vibe. This is what makes it defensible.

## Agents (pipeline, orchestrated)

| # | Agent | Job | Key output on the case |
|---|-------|-----|------------------------|
| 0 | **Orchestrator** | Runs the pipeline, handles failures, decides early-exit (e.g. confirmed duplicate → stop) | `trace[]` |
| 1 | **Ingestion / Classifier** | Detect format (native vs scanned), classify doc type, route scans to OCR | `source_format`, `document_type`, `raw_text` |
| 2 | **Extraction** | LLM structured extraction into `InvoiceData` with per-field confidence | `extracted` |
| 3 | **Vendor Resolution** | Match vendor to master (fuzzy name/alias), check status, compare bank details | `vendor` |
| 4 | **Validation** | Arithmetic integrity (lines sum, subtotal+tax−disc=total), completeness, tax plausibility, date sanity | `validations[]` |
| 5 | **PO Matching** | Find candidate PO(s), score match, apply tolerances, handle split/partial + over-billing | `match` |
| 6 | **Duplicate Detection** | Exact (vendor+invoice#) and fuzzy (amount+date+vendor) against processed ledger | `duplicate` |
| 7 | **Decision / Policy** | Run rule engine over all evidence → `Decision` with reasons + routing | `decision` |

Agents 3–6 are independent and can run **in parallel** after extraction. The
Decision agent is the only one that reads everything.

```
                 ┌─────────────┐
  file ─────────▶│  Ingestion  │──▶ raw_text, format, type
                 └─────────────┘
                        │
                 ┌─────────────┐
                 │ Extraction  │──▶ InvoiceData (+confidence)
                 └─────────────┘
                        │
        ┌───────────┬───┴────────┬──────────────┐   (parallel)
        ▼           ▼            ▼               ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
  │  Vendor  │ │Validation│ │ PO Match │ │  Duplicate   │
  └──────────┘ └──────────┘ └──────────┘ └──────────────┘
        └───────────┴───┬────────┴───────────────┘
                        ▼
                 ┌─────────────┐
                 │  Decision   │──▶ auto / review / hold / reject + reasons
                 └─────────────┘
```

## Decision rule engine (sketch — deterministic, ordered)
Hard stops first, then graded routing:
1. `VENDOR_BLOCKED` or `DUPLICATE_CONFIRMED` → **REJECT**
2. `PO_NOT_FOUND` (and PO required) or any **CRITICAL** validation fail → **HOLD** / **ROUTE_FOR_REVIEW** with required actions
3. `BANK_DETAILS_CHANGED` → **ROUTE_FOR_REVIEW** (manager) — fraud guard, always human
4. Within all tolerances, vendor approved, no warnings, extraction confidence high →
   **AUTO_APPROVE**
5. Within tolerances but minor warnings (rounding, low-confidence field) →
   **APPROVE_WITH_EXCEPTION** or **ROUTE_FOR_REVIEW** by amount threshold

## Edge cases (the 2–4 to actually build — these reveal judgment)
1. **Scanned/photographed invoice** → OCR path, lower field confidence, still extracts and decides.
2. **Split PO** — vendor bills one PO across two invoices. Second invoice must match on
   *remaining balance*, not full PO amount; flag `OVER_BILLING` if cumulative > PO.
3. **Near-duplicate** — same vendor + amount, different invoice number, days apart →
   `DUPLICATE_SUSPECTED` (fuzzy), route for review rather than auto-reject.
4. **Bank-detail change** — invoice bank account ≠ vendor master → hold for manager
   (classic AP fraud / business-email-compromise signal).
   *(Bonus: price within tolerance but quantity off → `QTY_VARIANCE_EXCEEDED`.)*

## Stack recommendation
- **Pydantic v2** for the schema/contract (done — `schemas.py`).
- **Orchestration:** start with a plain Python orchestrator (a function calling each
  agent and writing to the case). Move to **LangGraph** only if you want the graph/parallel
  + retry semantics for free. Don't reach for CrewAI/AutoGen — too much magic for a demo
  you must explain line-by-line.
- **Extraction LLM:** Claude (Opus/Sonnet 4.x) with structured outputs / tool use — strong
  on messy document layout. OCR for scans via Tesseract or a vision model directly.
- **Stores:** PO master + processed-invoice ledger as CSV/JSON to start (the FAQ says make
  your own test data); swap to SQLite when you add the dashboard.
- **UI (later):** live-run view = stream the `trace[]`; dashboard = list of `InvoiceCase`s
  with outcome + reasons. The schema already carries everything both views need.
