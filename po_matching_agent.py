"""
POMatchingAgent — runs after the verification gate gives a thumbs-up.

Pipeline within this agent:
  1. Resolve the vendor: the extracted vendor name rarely matches the PO master
     exactly (OCR noise, abbreviations), so we FUZZY-match it to a master vendor.
  2. Match the PO: STRICT exact match on the PO number, scoped to that vendor.
  3. Inspect the PO's billing history from the ledger: how many invoices were
     already raised, how much is billed, how much is still pending, and the PO's
     approval status.

Returns a POMatchResult the decision agent can act on. `over_billing` flags when
this invoice would push cumulative billing past the approved PO amount.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from extraction_agent import ExtractionResult
from po_store import POStore, _to_decimal


@dataclass
class POMatchResult:
    # vendor (cross-check against the PO's registered vendor)
    vendor_input: str | None
    vendor_matched: str | None        # the PO's vendor when consistent, else None
    vendor_score: float
    # PO match (primary key)
    po_number_input: str | None
    po_found: bool
    po_vendor: str | None = None      # vendor on record for this PO
    vendor_mismatch: bool = False     # invoice vendor != PO's vendor (red flag)
    po_status: str | None = None
    approved_amount: Decimal | None = None
    # billing history
    invoices_raised: int = 0
    total_billed: Decimal | None = None
    pending_amount: Decimal | None = None
    prior_invoices: list = field(default_factory=list)
    # this invoice vs the PO
    invoice_amount: Decimal | None = None
    over_billing: bool = False        # projected billing beyond approved + tolerance
    within_tolerance: bool = False    # over approved but inside the tolerance band
    issues: list[str] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        return self.po_found and not self.vendor_mismatch

    def summary(self) -> str:
        if not self.po_found:
            return f"PO '{self.po_number_input}' NOT FOUND in master"
        lines = [
            f"PO {self.po_number_input}: status={self.po_status}, approved={self.approved_amount}",
        ]
        if self.vendor_mismatch:
            lines.append(
                f"** VENDOR MISMATCH: invoice {self.vendor_input!r} vs PO vendor "
                f"{self.po_vendor!r} (score {self.vendor_score:.0f})"
            )
        else:
            lines.append(f"Vendor OK: {self.vendor_input!r} ~ {self.po_vendor!r} (score {self.vendor_score:.0f})")
        lines.append(
            f"Invoices already raised: {self.invoices_raised} | billed={self.total_billed} | pending={self.pending_amount}"
        )
        if self.over_billing:
            lines.append(f"** OVER-BILLING: this invoice ({self.invoice_amount}) exceeds pending ({self.pending_amount}) beyond tolerance")
        elif self.within_tolerance:
            lines.append(f"~ within tolerance: invoice ({self.invoice_amount}) exceeds pending ({self.pending_amount}) but inside the allowed band")
        return "\n".join(lines)


class POMatchingAgent:
    def __init__(self, store: POStore | None = None, vendor_threshold: float = 85.0):
        self.store = store or POStore()
        self.vendor_threshold = vendor_threshold

    def run(self, result: ExtractionResult) -> POMatchResult:
        fields = result.fields
        vendor_input = fields.vendor_name.value
        po_values = [ef.value for ef in fields.po_numbers if ef.value]
        po_input = po_values[0] if po_values else None
        invoice_amount = _to_decimal(fields.total_amount.value)

        out = POMatchResult(
            vendor_input=vendor_input,
            vendor_matched=None,
            vendor_score=0.0,
            po_number_input=po_input,
            po_found=False,
            invoice_amount=invoice_amount,
        )

        # 1. PO lookup by number (the authoritative, globally-unique key)
        po = self.store.get_po(po_input) if po_input else None
        if po is None:
            out.issues.append(f"PO '{po_input}' not found in master")
            return out
        out.po_found = True
        out.po_vendor = po.vendor
        out.po_status = po.approval_status
        out.approved_amount = po.approved_amount

        # 2. Vendor cross-check: does the invoice vendor match the PO's vendor?
        score = self.store.vendor_similarity(vendor_input, po.vendor)
        out.vendor_score = score
        if score >= self.vendor_threshold:
            out.vendor_matched = po.vendor
        else:
            out.vendor_mismatch = True
            out.issues.append(
                f"vendor mismatch: invoice '{vendor_input}' vs PO vendor "
                f"'{po.vendor}' (score {score:.0f})"
            )

        # 3. Billing history
        summary = self.store.billing_summary(po)
        out.invoices_raised = summary.invoices_raised
        out.total_billed = summary.total_billed
        out.pending_amount = summary.pending_amount
        out.prior_invoices = summary.invoices

        if po.approval_status.lower() != "approved":
            out.issues.append(f"PO status is '{po.approval_status}', not Approved")

        # Apply AP tolerance: billing is fine up to approved + allowed overage
        # (greater of the % tolerance or the flat floor).
        allowed = self.store.allowed_overage(po.approved_amount)
        projected = summary.total_billed + invoice_amount
        if projected > po.approved_amount + allowed:
            out.over_billing = True
            out.issues.append(
                f"over-billing: projected billed {projected} exceeds approved "
                f"{po.approved_amount} + allowed overage {allowed}"
            )
        elif projected > po.approved_amount:
            out.within_tolerance = True
            out.issues.append(
                f"within tolerance: projected billed {projected} over approved "
                f"{po.approved_amount} but inside allowed overage {allowed}"
            )
        return out


if __name__ == "__main__":
    # Self-test with mock extraction output (no API call needed).
    from extraction_agent import ExtractedField, InvoiceFields

    def ef(v, c=0.97):
        return ExtractedField(value=v, confidence=c, evidence=None)

    def make(vendor, po, total) -> ExtractionResult:
        flds = InvoiceFields(
            vendor_name=ef(vendor), invoice_number=ef("INV-9"),
            po_numbers=[ef(po)] if po else [],
            invoice_date=ef("2024-04-15"), due_date=ef(None, 0.0),
            billing_period_start=ef(None, 0.0), billing_period_end=ef(None, 0.0),
            currency=ef("INR"), subtotal=ef(total), tax_total=ef("0"),
            total_amount=ef(total), line_items=[],
        )
        return ExtractionResult(fields=flds, ocr_confidence=0.95,
                                source_format="pdf", source_path="mock", model="mock")

    agent = POMatchingAgent()

    print("A) PO-first, vendor cross-check OK ('ABC Tech.' ~ PO-456's 'abc technologies'):")
    print(agent.run(make("ABC Tech.", "PO-456", "15000")).summary())

    print("\nB) over-billing beyond tolerance (PO-456 pending 20000, billing 25000):")
    print(agent.run(make("ABC Technologies", "PO-456", "25000")).summary())

    print("\nC) PO not found:")
    print(agent.run(make("ABC Technologies", "PO-999", "1000")).summary())

    print("\nD) vendor mismatch (PO-456 belongs to ABC, invoice says Mega Industrial):")
    print(agent.run(make("Mega Industrial", "PO-456", "1000")).summary())
