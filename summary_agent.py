"""
SummaryAgent — the closing LLM call.

Gathers every signal from the run (extracted fields, verification, DUPLICATE
status, PO/vendor match, billing, and the final routed decision) into a compact
fact sheet, then asks the LLM to write a short plain-English summary for the
approver. The duplicate determination is always included in the fact sheet — so
the summary states whether the invoice was a duplicate even when some other rule
drove the decision.
"""

from __future__ import annotations

from extraction_agent import ExtractionResult
from verification_agent import VerificationResult
from po_matching_agent import POMatchResult
from decision_agent import Decision
from llm_client import DEFAULT_MODEL, get_client

_SYSTEM = (
    "You are an accounts-payable assistant. You are given a fact sheet about an "
    "invoice and the automated decision the system made. Write a concise summary "
    "(2-4 sentences) for the human approver. You MUST state: the final decision, "
    "whether the invoice is a duplicate, the PO/vendor match result, the key "
    "amounts (invoice vs pending/approved), and the action required. Use plain, "
    "professional English. Only use facts from the sheet — do not invent anything."
)


def build_fact_sheet(
    extraction: ExtractionResult,
    verification: VerificationResult,
    match: POMatchResult,
    decision: Decision,
) -> str:
    duplicate = decision.duplicate
    f = extraction.fields
    lines = [
        "INVOICE",
        f"  number: {f.invoice_number.value}",
        f"  vendor (on invoice): {f.vendor_name.value}",
        f"  currency/total: {f.currency.value} {f.total_amount.value}",
        f"  PO referenced: {[p.value for p in f.po_numbers]}",
        f"  source: {extraction.source_format} | OCR/text confidence: {extraction.ocr_confidence:.0%}",
        "",
        "DUPLICATE CHECK",
        f"  status: {duplicate.status}",
    ]
    for r in duplicate.reasons:
        lines.append(f"  - {r}")

    lines += [
        "",
        "VERIFICATION",
        f"  passed mandatory-field gate: {verification.proceed}",
    ]
    for i in verification.issues:
        lines.append(f"  - {i}")

    lines += [
        "",
        "PO MATCH",
        f"  PO found: {match.po_found}",
        f"  PO vendor of record: {match.po_vendor}",
        f"  vendor mismatch: {match.vendor_mismatch} (similarity {match.vendor_score:.0f})",
        f"  PO approval status: {match.po_status}",
        f"  approved: {match.approved_amount} | already billed: {match.total_billed} | pending: {match.pending_amount}",
        f"  invoices already raised on PO: {match.invoices_raised}",
        f"  over-billing (beyond tolerance): {match.over_billing} | within tolerance: {match.within_tolerance}",
        "",
        "DECISION",
        f"  outcome: {decision.outcome}",
        f"  past due: {decision.past_due}"
        + (f" (by {decision.days_overdue} days)" if decision.past_due else ""),
    ]
    for r in decision.reasons:
        lines.append(f"  reason: {r}")
    for a in decision.required_actions:
        lines.append(f"  action: {a}")
    return "\n".join(lines)


class SummaryAgent:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None, max_tokens: int = 400):
        self.model = model
        self.max_tokens = max_tokens
        self._client = get_client(api_key)

    def run(
        self,
        extraction: ExtractionResult,
        verification: VerificationResult,
        match: POMatchResult,
        decision: Decision,
    ) -> str:
        facts = build_fact_sheet(extraction, verification, match, decision)
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": facts},
            ],
        )
        return resp.choices[0].message.content.strip()
