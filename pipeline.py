"""
pipeline — run the AP agents end-to-end on one document.

    python pipeline.py data/invoice_01_happy.pdf

Stages:
  1. extract            detect format → text + confidence (extractors package)
  2. ExtractionAgent    LLM reads text → structured invoice fields (OpenAI)
  3. VerificationAgent  gate: mandatory fields present & confident → proceed?
  4. POMatchingAgent    fuzzy vendor + strict PO match + billing history

If the verification gate blocks, the pipeline stops there (no PO matching).
"""

from __future__ import annotations

import sys

# Windows consoles default to cp1252; LLM output (and our em-dashes) can contain
# characters it can't encode. Force UTF-8 so printing never crashes.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from extractors import extract
from extraction_agent import ExtractionAgent
from verification_agent import VerificationAgent
from po_matching_agent import POMatchingAgent
from decision_agent import DecisionAgent
from summary_agent import SummaryAgent
from po_store import POStore




def run(path: str, commit: bool = True) -> None:
    print(f"\n=== 1. EXTRACT === {path}")
    doc = extract(path)
    print(f"format={doc.fmt.value}  confidence={doc.confidence:.0%}  "
          f"pages={len(doc.pages)}  chars={len(doc.text)}")

    print("\n=== 2. EXTRACTION AGENT (LLM) ===")
    result = ExtractionAgent().run(doc)
    f = result.fields
    for label, ef in [
        ("Vendor", f.vendor_name), ("Invoice #", f.invoice_number),
        ("Invoice date", f.invoice_date), ("Due date", f.due_date),
        ("Period start", f.billing_period_start), ("Period end", f.billing_period_end),
        ("Subtotal", f.subtotal), ("Tax", f.tax_total), ("Total", f.total_amount),
    ]:
        print(f"  {label:14s} {str(ef.value):<26} [{ef.confidence:4.0%}]")
    for i, po in enumerate(f.po_numbers):
        print(f"  {'PO #' + str(i + 1):14s} {str(po.value):<26} [{po.confidence:4.0%}]")

    print("\n=== 3. VERIFICATION GATE ===")
    verdict = VerificationAgent().run(result)
    print(verdict.summary())
    if not verdict.proceed:
        print("\nStopped at verification gate.")
        return

    store = POStore()                       # shared store (so commit hits the same table)

    print("\n=== 4. PO MATCHING ===")
    match = POMatchingAgent(store=store).run(result)
    print(match.summary())

    print("\n=== 5. DECISION  ===")
    decision = DecisionAgent(store=store).decide(result, verdict, match)
    print(f"duplicate: {decision.duplicate.status}")
    print(decision.summary())

    print("\n=== 6. SUMMARY (LLM) ===")
    summary = SummaryAgent().run(result, verdict, match, decision)
    print(summary)

    # 7. Commit: only approved outcomes write the invoice to the ledger.
    if commit and decision.approved and match.po_found:
        inv = result.fields.invoice_number.value
        applied = store.apply_invoice(
            invoice_number=inv,
            po_number=match.po_number_input,
            vendor=match.vendor_matched,
            amount=match.invoice_amount,
            status="Approved",
        )
        print(f"\n=== 7. COMMIT === {applied.condition}: {applied.message}")
    elif commit:
        print("\n=== 7. COMMIT === skipped (not approved)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--commit"]
    target = args[0] if args else "data/invoice_01_happy.pdf"
    run(target, commit="--commit" in sys.argv)
