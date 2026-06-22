"""Validate the demo story: run each invoice in order, commit approvals, print outcome.

Mirrors the UI: a fresh POStore per invoice (so it sees prior commits), commit only
on approval. Skips the summary LLM call to save tokens.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
from extractors import extract
from extraction_agent import ExtractionAgent
from verification_agent import VerificationAgent
from po_matching_agent import POMatchingAgent
from decision_agent import DecisionAgent
from po_store import POStore

ORDER = [
    "01_abc_happy.pdf", "02_abc_near_duplicate.pdf", "03_abc_split_po.pdf",
    "04_abc_overbilled.pdf", "05_abc_exact_duplicate.pdf", "06_mega_within_tolerance.pdf",
    "07_global_overbilled.pdf", "08_vendor_mismatch.pdf", "09_po_not_found.pdf",
    "10_metro_happy_overdue.pdf", "11_invoice_number_collision.pdf", "12_missing_po.pdf",
]

EXPECTED = {
    "01_abc_happy.pdf": "AUTO_APPROVE",
    "02_abc_near_duplicate.pdf": "ROUTE_FOR_REVIEW",
    "03_abc_split_po.pdf": "AUTO_APPROVE",
    "04_abc_overbilled.pdf": "ROUTE_FOR_REVIEW",
    "05_abc_exact_duplicate.pdf": "REJECT",
    "06_mega_within_tolerance.pdf": "APPROVE_WITH_EXCEPTION",
    "07_global_overbilled.pdf": "ROUTE_FOR_REVIEW",
    "08_vendor_mismatch.pdf": "ROUTE_FOR_REVIEW",
    "09_po_not_found.pdf": "HOLD",
    "10_metro_happy_overdue.pdf": "AUTO_APPROVE",
    "11_invoice_number_collision.pdf": "ROUTE_FOR_REVIEW",
    "12_missing_po.pdf": "ROUTE_FOR_REVIEW",
}

agent = ExtractionAgent()
ok = 0
for fname in ORDER:
    path = Path("data") / fname
    doc = extract(str(path))
    result = agent.run(doc)
    verdict = VerificationAgent(threshold=0.90).run(result)
    store = POStore()  # fresh, sees prior commits
    match = POMatchingAgent(store=store).run(result)
    decision = DecisionAgent(store=store).decide(result, verdict, match)

    if decision.approved and match.po_found:
        store.apply_invoice(
            invoice_number=result.fields.invoice_number.value,
            po_number=match.po_number_input, vendor=match.vendor_matched,
            amount=match.invoice_amount, status="Approved")

    exp = EXPECTED[fname]
    flag = "OK " if decision.outcome == exp else "XX "
    ok += decision.outcome == exp
    extra = []
    if not decision.duplicate.is_clear:
        extra.append(decision.duplicate.status)
    if decision.past_due:
        extra.append(f"past_due {decision.days_overdue}d")
    note = (" | " + ", ".join(extra)) if extra else ""
    print(f"{flag}{fname:34s} -> {decision.outcome:22s} (want {exp}){note}")

print(f"\n{ok}/{len(ORDER)} matched expected outcomes")
