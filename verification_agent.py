"""
VerificationAgent — the gate before downstream processing.

Takes the ExtractionAgent's output and decides whether the case is clean enough
to proceed. A case passes only when EVERY mandatory field is:
    1. present  (value is not null), and
    2. confident (per-field confidence >= threshold, default 0.90)

Optional fields are reported but never block. The result carries a single
`proceed` flag (the "heads up" to move to the next agent) plus per-field checks
and human-readable issues for anything that failed.

Mandatory by default: invoice_number, po_numbers, total_amount.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from extraction_agent import ExtractedField, ExtractionResult, InvoiceFields

DEFAULT_THRESHOLD = 0.90
DEFAULT_MANDATORY = ("invoice_number", "po_numbers", "total_amount")


@dataclass
class FieldCheck:
    name: str
    present: bool
    confidence: float
    value: str | None
    passed: bool
    reason: str | None = None


@dataclass
class VerificationResult:
    proceed: bool                 # the heads-up: True → move to the next agent
    threshold: float
    checks: list[FieldCheck] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def summary(self) -> str:
        verdict = "PROCEED" if self.proceed else "BLOCKED"
        head = f"{verdict}  (threshold {self.threshold:.0%})"
        lines = [head]
        for i in self.issues:
            lines.append(f"  - {i}")
        return "\n".join(lines)


def _resolve(fields: InvoiceFields, name: str) -> tuple[bool, float, str | None]:
    """Return (present, confidence, display_value) for a field by name.

    Scalar field  → its own value/confidence.
    List field (e.g. po_numbers) → present if any entry has a value; confidence
    is the best among entries that have a value.
    """
    attr = getattr(fields, name)
    if isinstance(attr, list):
        filled = [ef for ef in attr if ef.value not in (None, "")]
        if not filled:
            return False, 0.0, None
        best = max(filled, key=lambda ef: ef.confidence)
        return True, best.confidence, best.value
    ef: ExtractedField = attr
    present = ef.value not in (None, "")
    return present, (ef.confidence if present else 0.0), ef.value


class VerificationAgent:
    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        mandatory: tuple[str, ...] = DEFAULT_MANDATORY,
    ):
        self.threshold = threshold
        self.mandatory = mandatory

    def run(self, result: ExtractionResult) -> VerificationResult:
        fields = result.fields
        checks: list[FieldCheck] = []
        issues: list[str] = []

        for name in self.mandatory:
            present, conf, value = _resolve(fields, name)
            if not present:
                passed, reason = False, "missing"
                issues.append(f"{name}: missing (mandatory)")
            elif conf < self.threshold:
                passed, reason = False, f"low confidence {conf:.0%} < {self.threshold:.0%}"
                issues.append(f"{name}: low confidence ({conf:.0%})")
            else:
                passed, reason = True, None
            checks.append(FieldCheck(name, present, conf, value, passed, reason))

        return VerificationResult(
            proceed=not issues,
            threshold=self.threshold,
            checks=checks,
            issues=issues,
        )


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Self-test with mock extraction output (no API call needed).
    def ef(value, conf):
        return ExtractedField(value=value, confidence=conf, evidence=None)

    def make(inv, inv_c, po, po_c, total, total_c) -> ExtractionResult:
        flds = InvoiceFields(
            vendor_name=ef("ACME", 0.99),
            invoice_number=ef(inv, inv_c),
            po_numbers=[ef(po, po_c)] if po else [],
            invoice_date=ef("2024-04-15", 0.95),
            due_date=ef(None, 0.0),
            billing_period_start=ef(None, 0.0),
            billing_period_end=ef(None, 0.0),
            currency=ef("INR", 0.99),
            subtotal=ef("100", 0.9),
            tax_total=ef("18", 0.9),
            total_amount=ef(total, total_c),
            line_items=[],
        )
        return ExtractionResult(
            fields=flds, ocr_confidence=0.95,
            source_format="pdf", source_path="mock", model="mock",
        )

    agent = VerificationAgent()  # 90% threshold

    print("Case A — all good:")
    print(agent.run(make("INV-1001", 0.98, "PO-456", 0.95, "59000", 0.97)).summary())

    print("\nCase B — PO missing:")
    print(agent.run(make("INV-1001", 0.98, None, 0.0, "59000", 0.97)).summary())

    print("\nCase C — total low confidence:")
    print(agent.run(make("INV-1001", 0.98, "PO-456", 0.95, "59000", 0.72)).summary())
