"""
Invoice-to-Decision — core data schemas (Pydantic v2).

These models are the contract between agents. Everything an agent learns about an
invoice is written back onto a single `InvoiceCase` object, which is what the UI
renders and what becomes the audit trail. Design rule: nothing the system "knows"
should live only in a log line — it lives on the case, with provenance.

Financial-analyst lens baked in:
  - field-level confidence + provenance (so a human can trust/verify any number)
  - 2-way match with tolerances + partial/split-PO handling (over-billing guard)
  - duplicate detection (exact + fuzzy) — the #1 AP leakage source
  - vendor master + bank-detail-change check (payment fraud guard)
  - decisions are *routed* (auto / review / hold), never a bare yes/no
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Generic, Optional, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

# Money is Decimal everywhere — never float. Floats silently corrupt totals.
Money = Decimal


# --------------------------------------------------------------------------- #
# Enums                                                                        #
# --------------------------------------------------------------------------- #
class DocumentType(str, Enum):
    INVOICE = "invoice"
    PURCHASE_ORDER = "purchase_order"
    GOODS_RECEIPT = "goods_receipt"
    CREDIT_NOTE = "credit_note"
    VENDOR_STATEMENT = "vendor_statement"
    UNKNOWN = "unknown"


class SourceFormat(str, Enum):
    NATIVE_PDF = "native_pdf"      # machine-readable text layer present
    SCANNED_PDF = "scanned_pdf"    # image-only PDF -> needs OCR
    IMAGE = "image"               # jpg/png/tiff -> needs OCR


class VendorStatus(str, Enum):
    APPROVED = "approved"
    BLOCKED = "blocked"
    NEW = "new"                   # not in master / unverified


class POStatus(str, Enum):
    OPEN = "open"
    PARTIALLY_INVOICED = "partially_invoiced"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class MatchType(str, Enum):
    EXACT = "exact"               # PO ref explicit, lines + amounts line up
    WITHIN_TOLERANCE = "within_tolerance"
    PARTIAL_SPLIT = "partial_split"  # one PO billed across several invoices
    FUZZY = "fuzzy"               # PO inferred (no explicit ref)
    NONE = "none"


class CheckStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    SKIPPED = "skipped"           # couldn't run (e.g. data missing)


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class IssueCode(str, Enum):
    """Stable codes so rules + UI + tests can key off them."""
    # completeness / extraction
    MISSING_INVOICE_NUMBER = "missing_invoice_number"
    MISSING_INVOICE_DATE = "missing_invoice_date"
    MISSING_TOTAL = "missing_total"
    LOW_EXTRACTION_CONFIDENCE = "low_extraction_confidence"
    # arithmetic integrity
    LINE_ITEMS_DONT_SUM = "line_items_dont_sum"
    TOTAL_MISMATCH = "total_mismatch"          # subtotal + tax - disc != total
    TAX_IMPLAUSIBLE = "tax_implausible"
    # vendor / fraud
    VENDOR_NOT_APPROVED = "vendor_not_approved"
    VENDOR_BLOCKED = "vendor_blocked"
    BANK_DETAILS_CHANGED = "bank_details_changed"
    # PO matching
    PO_NOT_FOUND = "po_not_found"
    PO_CLOSED = "po_closed"
    CURRENCY_MISMATCH = "currency_mismatch"
    PRICE_VARIANCE_EXCEEDED = "price_variance_exceeded"
    QTY_VARIANCE_EXCEEDED = "qty_variance_exceeded"
    OVER_BILLING = "over_billing"              # cumulative billed > PO balance
    # duplicate / sanity
    DUPLICATE_SUSPECTED = "duplicate_suspected"
    DUPLICATE_CONFIRMED = "duplicate_confirmed"
    FUTURE_DATED = "future_dated"
    STALE_DATED = "stale_dated"


class DecisionOutcome(str, Enum):
    AUTO_APPROVE = "auto_approve"               # straight-through processing
    APPROVE_WITH_EXCEPTION = "approve_with_exception"
    ROUTE_FOR_REVIEW = "route_for_review"       # human needed, reasons attached
    HOLD = "hold"                               # block pending info
    REJECT = "reject"                           # duplicate / fraud / no PO


class ExtractionStatus(str, Enum):
    PENDING = "pending"                  # not yet run
    COMPLETE = "complete"                # all critical fields ≥ HIGH threshold
    PARTIAL = "partial"                  # some non-critical fields low-confidence; can proceed
    NEEDS_CLARIFICATION = "needs_clarification"  # critical field(s) below threshold; pipeline blocked
    ANALYST_REVIEW = "analyst_review"    # analyst opened it; awaiting their input
    CLARIFICATION_RESOLVED = "clarification_resolved"  # analyst filled gaps; ready to continue


class ClarificationStatus(str, Enum):
    OPEN = "open"
    ANALYST_PROVIDED = "analyst_provided"   # analyst entered a value
    CANNOT_DETERMINE = "cannot_determine"   # analyst confirms it's not on the doc
    RESOLVED = "resolved"                   # accepted and merged into extracted data


# --------------------------------------------------------------------------- #
# Provenance — every extracted value carries where it came from + confidence   #
# --------------------------------------------------------------------------- #
T = TypeVar("T")


class FieldSource(BaseModel):
    page: Optional[int] = None
    bbox: Optional[tuple[float, float, float, float]] = None  # x0,y0,x1,y1
    raw_text: Optional[str] = None        # exactly what was on the page
    method: Optional[str] = None          # "text_layer" | "ocr" | "llm"


class ExtractedField(BaseModel, Generic[T]):
    """A value plus how confident we are and where it came from.

    This is the unit that makes the whole thing auditable: the UI can show the
    number, the confidence, and let a reviewer click through to the source crop.
    """
    value: Optional[T] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: Optional[FieldSource] = None


# --------------------------------------------------------------------------- #
# Reference data (the "procurement system" side)                               #
# --------------------------------------------------------------------------- #
class BankAccount(BaseModel):
    account_name: Optional[str] = None
    account_number: Optional[str] = None  # store masked/hashed in real life
    iban: Optional[str] = None
    sort_code: Optional[str] = None
    swift_bic: Optional[str] = None


class VendorMaster(BaseModel):
    vendor_id: str
    legal_name: str
    aliases: list[str] = Field(default_factory=list)   # for fuzzy name resolution
    status: VendorStatus = VendorStatus.APPROVED
    tax_id: Optional[str] = None
    registered_country: Optional[str] = None           # ISO-3166 alpha-2
    default_currency: Optional[str] = None             # ISO-4217
    payment_terms_days: Optional[int] = None
    known_bank_accounts: list[BankAccount] = Field(default_factory=list)


class POLine(BaseModel):
    line_no: int
    sku: Optional[str] = None
    description: Optional[str] = None
    qty_ordered: Decimal
    qty_received: Decimal = Decimal(0)     # enables 3-way match if available
    qty_invoiced: Decimal = Decimal(0)     # cumulative, for split handling
    unit_price: Money
    line_amount: Money


class Tolerance(BaseModel):
    """How far an invoice may drift from the PO before a human looks.

    Defaults are deliberate finance choices, overridable per-PO or per-vendor.
    """
    price_pct: float = 0.05                # ±5% unit price
    price_abs: Money = Decimal("25")       # or ±$25, whichever is larger
    qty_pct: float = 0.0                   # qty usually must be exact
    total_abs: Money = Decimal("50")       # rounding slack on the grand total


class PurchaseOrder(BaseModel):
    po_number: str
    vendor_id: str
    status: POStatus = POStatus.OPEN
    currency: str = "USD"
    lines: list[POLine] = Field(default_factory=list)
    total_amount: Money = Decimal(0)
    amount_invoiced_to_date: Money = Decimal(0)   # for over-billing / split logic
    valid_from: Optional[date] = None
    valid_to: Optional[date] = None
    tolerance: Tolerance = Field(default_factory=Tolerance)

    @property
    def remaining_balance(self) -> Money:
        return self.total_amount - self.amount_invoiced_to_date


# --------------------------------------------------------------------------- #
# Extracted invoice                                                            #
# --------------------------------------------------------------------------- #
class InvoiceLineItem(BaseModel):
    description: ExtractedField[str] = Field(default_factory=ExtractedField)
    sku: ExtractedField[str] = Field(default_factory=ExtractedField)
    quantity: ExtractedField[Decimal] = Field(default_factory=ExtractedField)
    unit_price: ExtractedField[Money] = Field(default_factory=ExtractedField)
    line_total: ExtractedField[Money] = Field(default_factory=ExtractedField)
    tax_rate: ExtractedField[float] = Field(default_factory=ExtractedField)
    tax_amount: ExtractedField[Money] = Field(default_factory=ExtractedField)
    po_line_ref: ExtractedField[str] = Field(default_factory=ExtractedField)


class InvoiceData(BaseModel):
    vendor_name: ExtractedField[str] = Field(default_factory=ExtractedField)
    invoice_number: ExtractedField[str] = Field(default_factory=ExtractedField)
    invoice_date: ExtractedField[date] = Field(default_factory=ExtractedField)
    due_date: ExtractedField[date] = Field(default_factory=ExtractedField)
    po_references: list[ExtractedField[str]] = Field(default_factory=list)
    currency: ExtractedField[str] = Field(default_factory=ExtractedField)
    subtotal: ExtractedField[Money] = Field(default_factory=ExtractedField)
    tax_total: ExtractedField[Money] = Field(default_factory=ExtractedField)
    discount_total: ExtractedField[Money] = Field(default_factory=ExtractedField)
    total_amount: ExtractedField[Money] = Field(default_factory=ExtractedField)
    line_items: list[InvoiceLineItem] = Field(default_factory=list)
    bank_account: Optional[BankAccount] = None
    payment_terms: ExtractedField[str] = Field(default_factory=ExtractedField)


# --------------------------------------------------------------------------- #
# Agent outputs                                                                #
# --------------------------------------------------------------------------- #
class VendorResolution(BaseModel):
    matched_vendor_id: Optional[str] = None
    matched_name: Optional[str] = None
    status: VendorStatus = VendorStatus.NEW
    name_match_confidence: float = 0.0
    bank_details_match: Optional[bool] = None   # None = no master record to compare


class LineMatch(BaseModel):
    invoice_line_index: int
    po_line_no: Optional[int] = None
    price_variance_pct: Optional[float] = None
    qty_variance: Optional[Decimal] = None
    within_tolerance: bool = False


class MatchResult(BaseModel):
    po_number: Optional[str] = None
    match_type: MatchType = MatchType.NONE
    confidence: float = 0.0
    line_matches: list[LineMatch] = Field(default_factory=list)
    total_variance_abs: Optional[Money] = None
    cumulative_billed: Optional[Money] = None       # incl. this invoice
    remaining_po_balance: Optional[Money] = None
    is_over_billing: bool = False
    notes: Optional[str] = None


class DuplicateCheck(BaseModel):
    is_duplicate: bool = False
    confidence: float = 0.0
    matched_invoice_ids: list[str] = Field(default_factory=list)
    basis: Optional[str] = None   # "exact_invoice_no" | "fuzzy_amount_date_vendor"


class ValidationCheck(BaseModel):
    """One discrete check. The list of these IS the explanation of the decision."""
    code: IssueCode
    name: str
    status: CheckStatus
    severity: Severity
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None


class Decision(BaseModel):
    outcome: DecisionOutcome
    confidence: float = 0.0
    reasons: list[str] = Field(default_factory=list)        # human-readable
    triggered_issues: list[IssueCode] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)  # what unblocks it
    routed_to: Optional[str] = None     # "ap_clerk" | "ap_manager" | "auto"


# --------------------------------------------------------------------------- #
# Clarification — structured request for analyst input on low-confidence      #
# or missing fields                                                            #
# --------------------------------------------------------------------------- #

class FieldCriticality(str, Enum):
    """How badly the pipeline needs this field."""
    CRITICAL = "critical"     # blocks processing if missing/low-conf
    IMPORTANT = "important"   # degrades match quality; warn but continue
    OPTIONAL = "optional"     # nice to have


class FieldClarification(BaseModel):
    """One field that needs a human's eye.

    Rendered by the analyst UI as a form row: show the evidence excerpt, the
    auto-extracted value, and an input box. The analyst fills `analyst_value`
    and sets status. If they can't determine it, they set CANNOT_DETERMINE and
    optionally add a note — that is what flips the invoice to review-needed.
    """
    field_path: str                         # e.g. "invoice_number", "line_items[2].unit_price"
    field_label: str                        # human-readable label for the UI
    criticality: FieldCriticality = FieldCriticality.IMPORTANT
    reason: str                             # why we're asking (e.g. "No clear label found")
    evidence: Optional[str] = None          # the raw text snippet the model saw
    page: Optional[int] = None             # source page for UI to highlight
    auto_extracted_value: Optional[str] = None   # string repr of what we got
    auto_confidence: float = 0.0

    # filled by analyst
    analyst_value: Optional[str] = None
    analyst_note: Optional[str] = None
    status: ClarificationStatus = ClarificationStatus.OPEN
    resolved_at: Optional[datetime] = None


# --------------------------------------------------------------------------- #
# Confidence thresholds (finance-calibrated defaults)                          #
# --------------------------------------------------------------------------- #

class ConfidenceThresholds(BaseModel):
    """
    What counts as HIGH / MEDIUM / LOW confidence.

    - HIGH  (≥ auto_accept):  field accepted automatically
    - MEDIUM (review_flag ≤ x < auto_accept): proceed but flag in UI
    - LOW   (< review_flag):  request analyst input

    Critical-field LOW → NEEDS_CLARIFICATION (pipeline blocked)
    Non-critical LOW   → PARTIAL (pipeline continues with warning)
    """
    auto_accept: float = 0.85
    review_flag: float = 0.60

    def tier(self, confidence: float) -> str:
        if confidence >= self.auto_accept:
            return "high"
        if confidence >= self.review_flag:
            return "medium"
        return "low"


CRITICAL_FIELDS: list[tuple[str, str]] = [
    ("invoice_number",  "Invoice Number"),
    ("invoice_date",    "Invoice Date"),
    ("vendor_name",     "Vendor Name"),
    ("total_amount",    "Total Amount"),
    ("currency",        "Currency"),
]

IMPORTANT_FIELDS: list[tuple[str, str]] = [
    ("po_references",   "PO Reference(s)"),
    ("subtotal",        "Subtotal"),
    ("tax_total",       "Tax / VAT Total"),
    ("due_date",        "Due Date"),
    ("payment_terms",   "Payment Terms"),
]


# --------------------------------------------------------------------------- #
# Audit trace — one entry per agent step                                       #
# --------------------------------------------------------------------------- #
class AgentStep(BaseModel):
    agent: str
    status: str = "ok"                  # "ok" | "warning" | "error"
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    summary: Optional[str] = None       # one line for the live-run view
    error: Optional[str] = None
    token_cost: Optional[int] = None    # for cost visibility in the dashboard


# --------------------------------------------------------------------------- #
# The case — the single object that flows through every agent                  #
# --------------------------------------------------------------------------- #
class InvoiceCase(BaseModel):
    case_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    received_at: datetime = Field(default_factory=datetime.utcnow)

    # ingestion
    source_filename: Optional[str] = None
    source_format: Optional[SourceFormat] = None
    document_type: DocumentType = DocumentType.UNKNOWN
    raw_text: Optional[str] = None      # text layer or OCR output

    # accumulated evidence (each agent fills its slice)
    extracted: InvoiceData = Field(default_factory=InvoiceData)
    vendor: VendorResolution = Field(default_factory=VendorResolution)
    match: MatchResult = Field(default_factory=MatchResult)
    duplicate: DuplicateCheck = Field(default_factory=DuplicateCheck)
    validations: list[ValidationCheck] = Field(default_factory=list)
    decision: Optional[Decision] = None

    # audit
    trace: list[AgentStep] = Field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Extraction state — filled by IngestionAgent + ExtractionAgent     #
    # ------------------------------------------------------------------ #
    extraction_status: ExtractionStatus = ExtractionStatus.PENDING
    clarifications: list[FieldClarification] = Field(default_factory=list)

    def add_validation(self, check: ValidationCheck) -> None:
        self.validations.append(check)

    def add_clarification(self, req: "FieldClarification") -> None:
        # Deduplicate by field_path — update if already present
        for i, existing in enumerate(self.clarifications):
            if existing.field_path == req.field_path:
                self.clarifications[i] = req
                return
        self.clarifications.append(req)

    @property
    def has_critical_issue(self) -> bool:
        return any(v.severity == Severity.CRITICAL and v.status == CheckStatus.FAIL
                   for v in self.validations)

    @property
    def unresolved_clarifications(self) -> list["FieldClarification"]:
        return [c for c in self.clarifications
                if c.status != ClarificationStatus.RESOLVED]

    @property
    def is_extraction_blocked(self) -> bool:
        """True if critical fields are still unresolved — pipeline should pause."""
        return self.extraction_status in (
            ExtractionStatus.NEEDS_CLARIFICATION,
            ExtractionStatus.ANALYST_REVIEW,
        )
