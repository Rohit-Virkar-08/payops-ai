"""
DecisionAgent — the routing brain (now including duplicate detection).

Deterministic rule engine (NOT an LLM). It owns the double-payment guard too:
duplicate detection is folded in here, because a confirmed duplicate is simply a
hard-stop decision — if it's a duplicate we don't move forward, so there's no
reason for it to be a separate agent.

Inputs:
  ExtractionResult     (fields + per-field & OCR confidence)
  VerificationResult   (mandatory-field gate)
  POMatchResult        (vendor match, PO match, billing, tolerance flags)
The duplicate check runs internally against the store's ledger.

Outcomes (first matching rule wins — hard stops first):
  REJECT                 confirmed duplicate
  ROUTE_FOR_REVIEW       failed verification / vendor mismatch / suspected dup / over-billing
  HOLD                   PO missing / not approved
  APPROVE_WITH_EXCEPTION clean but a benign within-tolerance overage
  AUTO_APPROVE           everything clean

The Decision carries the DuplicateResult so the summary/UI can always show
whether the invoice was a duplicate, regardless of what drove the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from extraction_agent import ExtractionResult
from po_matching_agent import POMatchResult
from verification_agent import VerificationResult
from po_store import LedgerEntry, POStore

# --------------------------------------------------------------------------- #
# Duplicate detection (folded into the decision agent)
# --------------------------------------------------------------------------- #
_VENDOR_SIM = 85.0    # fuzzy vendor similarity threshold to count as same vendor


class DupStatus:
    NONE = "NONE"
    SUSPECTED = "DUPLICATE_SUSPECTED"
    CONFIRMED = "DUPLICATE_CONFIRMED"


@dataclass
class DuplicateResult:
    status: str = DupStatus.NONE
    exact_matches: list[LedgerEntry] = field(default_factory=list)
    near_matches: list[LedgerEntry] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def is_confirmed(self) -> bool:
        return self.status == DupStatus.CONFIRMED

    @property
    def is_clear(self) -> bool:
        return self.status == DupStatus.NONE


def _parse_date(s: str | None):
    if not s:
        return None
    try:
        return date.fromisoformat(s.strip()[:10])
    except (ValueError, AttributeError):
        return None


def detect_duplicate(result: ExtractionResult, store: POStore) -> DuplicateResult:
    """Duplicate identity is (vendor + invoice number).

    Invoice numbers are vendor-assigned and unique only within that vendor's own
    sequence, so the authoritative duplicate key is the combination of both:

      same vendor + same invoice #          -> CONFIRMED  (true re-billing, hard stop)
      same invoice # + DIFFERENT vendor     -> SUSPECTED  (number collision, needs a
                                                           human glance to confirm the
                                                           vendor identity is correct)

    Same vendor + same amount with a different invoice number is NOT flagged:
    vendors legitimately raise multiple invoices for the same amount (monthly
    retainers, repeat orders, split deliveries).
    """
    f = result.fields
    inv_no = (f.invoice_number.value or "").strip()
    vendor = f.vendor_name.value

    out = DuplicateResult()
    collisions: list[LedgerEntry] = []   # same invoice #, different vendor
    for e in store.ledger:
        if not (inv_no and e.invoice_number.strip().upper() == inv_no.upper()):
            continue
        same_vendor = store.vendor_similarity(vendor, e.vendor) >= _VENDOR_SIM
        (out.exact_matches if same_vendor else collisions).append(e)

    if out.exact_matches:
        out.status = DupStatus.CONFIRMED
        nums = ", ".join(sorted({e.invoice_number for e in out.exact_matches}))
        out.reasons.append(f"same vendor + invoice number {inv_no} already in ledger ({nums})")
    elif collisions:
        out.status = DupStatus.SUSPECTED
        for e in collisions:
            out.near_matches.append(e)
            out.reasons.append(
                f"invoice number {inv_no} also exists in ledger under a different vendor "
                f"('{e.vendor}') — likely a number coincidence, confirm vendor identity")
    return out


# --------------------------------------------------------------------------- #
# Decision
# --------------------------------------------------------------------------- #
class Outcome:
    AUTO_APPROVE = "AUTO_APPROVE"
    APPROVE_WITH_EXCEPTION = "APPROVE_WITH_EXCEPTION"
    REVIEW = "ROUTE_FOR_REVIEW"
    HOLD = "HOLD"
    REJECT = "REJECT"


@dataclass
class Decision:
    outcome: str
    reasons: list[str] = field(default_factory=list)
    required_actions: list[str] = field(default_factory=list)
    duplicate: DuplicateResult | None = None   # always attached for visibility
    past_due: bool = False                      # non-blocking flag
    days_overdue: int | None = None

    @property
    def approved(self) -> bool:
        return self.outcome in (Outcome.AUTO_APPROVE, Outcome.APPROVE_WITH_EXCEPTION)

    def summary(self) -> str:
        lines = [f"DECISION: {self.outcome}"]
        for r in self.reasons:
            lines.append(f"  reason: {r}")
        for a in self.required_actions:
            lines.append(f"  action: {a}")
        if self.past_due:
            lines.append(f"  flag: PAST DUE by {self.days_overdue} day(s)")
        return "\n".join(lines)


class DecisionAgent:
    def __init__(self, store: POStore | None = None):
        self.store = store or POStore()

    def decide(
        self,
        extraction: ExtractionResult,
        verification: VerificationResult,
        match: POMatchResult,
    ) -> Decision:
        dup = detect_duplicate(extraction, self.store)

        # Overdue check (post-match) — a non-blocking flag, not a halt: an overdue
        # invoice is urgent to pay, so it travels with the decision rather than
        # changing the outcome.
        past_due, days_overdue = False, None
        due = _parse_date(extraction.fields.due_date.value)
        if due is not None:
            delta = (date.today() - due).days
            if delta > 0:
                past_due, days_overdue = True, delta

        def mk(outcome, reasons, actions=None):
            return Decision(outcome, reasons, actions or [], duplicate=dup,
                            past_due=past_due, days_overdue=days_overdue)

        # 1. Hard stop — confirmed duplicate (double-payment guard).
        if dup.is_confirmed:
            return mk(Outcome.REJECT, ["confirmed duplicate"] + dup.reasons,
                      ["Do not pay — invoice already in the ledger"])

        # 2. Extraction wasn't trustworthy enough — a human must confirm fields.
        if not verification.proceed:
            return mk(Outcome.REVIEW, ["mandatory fields failed verification"] + verification.issues,
                      ["Confirm/correct the flagged fields"])

        # 3. PO not found — fixable, but blocks payment.
        if not match.po_found:
            return mk(Outcome.HOLD, [f"PO '{match.po_number_input}' not found in master"],
                      ["Verify the PO number on the invoice"])

        # 4. Vendor on the invoice doesn't match the PO's vendor — fraud / wrong PO.
        if match.vendor_mismatch:
            return mk(Outcome.REVIEW,
                      [f"vendor mismatch: invoice '{match.vendor_input}' vs PO vendor "
                       f"'{match.po_vendor}' (score {match.vendor_score:.0f})"],
                      ["Confirm the vendor identity / correct PO before payment"])

        # 5. PO exists but isn't approved.
        if (match.po_status or "").lower() != "approved":
            return mk(Outcome.HOLD,
                      [f"PO {match.po_number_input} status is '{match.po_status}', not Approved"],
                      ["Get the PO approved before payment"])

        # 6. Suspected (not confirmed) duplicate — route to a human, don't auto-reject.
        if dup.status == DupStatus.SUSPECTED:
            return mk(Outcome.REVIEW, ["suspected duplicate"] + dup.reasons,
                      ["Confirm this isn't a re-billing of an existing invoice"])

        # 7. Over-billing beyond tolerance — manager review.
        if match.over_billing:
            return mk(Outcome.REVIEW,
                      [f"over-billing beyond tolerance: invoice {match.invoice_amount}, "
                       f"pending {match.pending_amount}"],
                      ["Manager review for the over-billing exception"])

        # 8. Clean, but a small overage absorbed by tolerance — approve, but note it.
        if match.within_tolerance:
            return mk(Outcome.APPROVE_WITH_EXCEPTION,
                      [f"invoice {match.invoice_amount} exceeds pending {match.pending_amount} "
                       f"but within allowed tolerance"],
                      ["Logged as a tolerance exception"])

        # 9. All checks pass.
        return mk(Outcome.AUTO_APPROVE,
                  ["vendor matched, PO approved, amount within balance, mandatory fields confident"])


if __name__ == "__main__":
    # Self-test against the real ledger (INV-1001 exists; INV-9999 doesn't).
    from extraction_agent import ExtractedField, InvoiceFields

    def ef(v, c=0.97):
        return ExtractedField(value=v, confidence=c, evidence=None)

    def mk_result(inv, vendor="ABC Technologies", total="15000") -> ExtractionResult:
        flds = InvoiceFields(
            vendor_name=ef(vendor), invoice_number=ef(inv), po_numbers=[ef("PO-456")],
            invoice_date=ef("2024-03-12"), due_date=ef(None, 0.0),
            billing_period_start=ef(None, 0.0), billing_period_end=ef(None, 0.0),
            currency=ef("INR"), subtotal=ef(total), tax_total=ef("0"),
            total_amount=ef(total), line_items=[])
        return ExtractionResult(fields=flds, ocr_confidence=0.95,
                                source_format="pdf", source_path="mock", model="mock")

    def mk_match(**kw) -> POMatchResult:
        base = dict(vendor_input="ABC", vendor_matched="abc technologies", vendor_score=100.0,
                    po_number_input="PO-456", po_found=True, po_vendor="abc technologies",
                    vendor_mismatch=False, po_status="Approved", approved_amount=Decimal(50000),
                    invoices_raised=1, total_billed=Decimal(30000), pending_amount=Decimal(20000),
                    invoice_amount=Decimal(15000), over_billing=False, within_tolerance=False)
        base.update(kw)
        return POMatchResult(**base)

    ok = VerificationResult(proceed=True, threshold=0.90)
    agent = DecisionAgent()

    print("confirmed dup (INV-1001):  ", agent.decide(mk_result("INV-1001"), ok, mk_match()).outcome)
    print("clean (INV-9999):          ", agent.decide(mk_result("INV-9999"), ok, mk_match()).outcome)
    print("vendor mismatch:           ", agent.decide(mk_result("INV-9999"), ok, mk_match(vendor_mismatch=True)).outcome)
    print("over-billing:              ", agent.decide(mk_result("INV-9999"), ok, mk_match(over_billing=True)).outcome)