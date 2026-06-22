"""
po_store — data access + update mechanism for the PO master and invoice ledger.

Design (agreed):
  - invoice_ledger.csv is the SOURCE OF TRUTH (append-only record of every
    invoice raised against a PO, with its own approval status).
  - po_dataset.csv is a PROJECTION: the master columns (vendor, approved amount,
    the PO's own approval status) are procurement-owned; the billing columns
    (billed, pending, billing status, invoice list) are RECOMPUTED from the
    ledger on every change. No drift, fully auditable.
  - Only APPROVED invoices reduce the pending balance. Pending/rejected invoices
    are recorded (and listed) but don't consume the PO budget.

Schema (po_dataset.csv):
  PO_Number, Vendor, Approved_Amount, Approval_Status,           <- master
  Billed_Amount, Pending_Amount, Billing_Status, Last_Updated    <- derived

(Invoice numbers are NOT duplicated here — the ledger is the source of truth for
which invoices were raised against each PO.)

apply_invoice() is the single robust entry point: it duplicate-guards, appends
to the ledger, recomputes the projection, and persists both files atomically.
"""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from rapidfuzz import fuzz, process, utils

_DATA = Path(__file__).parent / "data"
_PO_CSV = _DATA / "po_dataset.csv"
_LEDGER_CSV = _DATA / "invoice_ledger.csv"

# Rounding epsilon (currency units) for treating a balance as exactly "closed".
_TOL = Decimal("0.01")

# AP tolerance: cumulative billing may exceed the approved PO amount by up to the
# GREATER of a percentage (industry norm ~2%) or a flat floor amount — so small
# POs still get a minimum absolute slack instead of a tiny percentage.
DEFAULT_TOLERANCE = 0.02      # 2%
DEFAULT_TOLERANCE_ABS = 500   # flat floor (currency units)

INVOICE_APPROVED = "approved"   # ledger Status that counts toward billing


class BillingStatus:
    OPEN = "OPEN"                       # approved PO, nothing billed yet
    PARTIALLY_BILLED = "PARTIALLY_BILLED"
    FULLY_BILLED = "FULLY_BILLED"
    OVER_BILLED = "OVER_BILLED"        # approved billing exceeds approved amount
    ON_HOLD = "ON_HOLD"                # PO itself not approved


def _to_decimal(raw) -> Decimal:
    """Parse messy amount strings ('4,50,000.00', 'INR 59000') into Decimal."""
    if raw is None:
        return Decimal(0)
    s = "".join(ch for ch in str(raw) if ch.isdigit() or ch in ".-")
    try:
        return Decimal(s) if s else Decimal(0)
    except InvalidOperation:
        return Decimal(0)


@dataclass
class LedgerEntry:
    invoice_number: str
    po_number: str
    vendor: str
    amount: Decimal
    status: str
    date: str


@dataclass
class PORecord:
    # master
    po_number: str
    vendor: str
    approved_amount: Decimal
    approval_status: str
    # derived (recomputed from ledger)
    billed_amount: Decimal = Decimal(0)
    pending_amount: Decimal = Decimal(0)
    billing_status: str = BillingStatus.OPEN
    last_updated: str = ""


@dataclass
class BillingSummary:
    po_number: str
    approved_amount: Decimal
    invoices_raised: int
    total_billed: Decimal
    pending_amount: Decimal
    invoices: list[LedgerEntry] = field(default_factory=list)


@dataclass
class ApplyResult:
    invoice_number: str
    po_number: str
    condition: str           # DUPLICATE | PO_NOT_FOUND | ON_HOLD | RECORDED_PENDING
                             #  | PARTIAL | FULL | OVER_BILLED
    applied: bool            # did it change billed amount?
    message: str
    po: PORecord | None = None


class POStore:
    def __init__(
        self,
        po_csv: Path = _PO_CSV,
        ledger_csv: Path = _LEDGER_CSV,
        tolerance: float = DEFAULT_TOLERANCE,
        tolerance_abs: float = DEFAULT_TOLERANCE_ABS,
    ):
        self.po_csv = Path(po_csv)
        self.ledger_csv = Path(ledger_csv)
        self.tolerance = tolerance
        self.tolerance_abs = tolerance_abs
        self._tol_frac = Decimal(str(tolerance))
        self._tol_abs = Decimal(str(tolerance_abs))
        self.pos = self._load_master(self.po_csv)
        self.ledger = self._load_ledger(self.ledger_csv)
        self._recompute()

    def allowed_overage(self, approved_amount: Decimal) -> Decimal:
        """Max billing allowed above the approved amount: greater of % or flat floor."""
        return max(approved_amount * self._tol_frac, self._tol_abs)

    @staticmethod
    def vendor_similarity(a: str | None, b: str | None) -> float:
        """Pairwise fuzzy score (0-100) between two vendor names, case/punct-insensitive."""
        if not a or not b:
            return 0.0
        return fuzz.WRatio(a, b, processor=utils.default_process)

    # ----------------------------- loading ----------------------------- #
    @staticmethod
    def _load_master(path: Path) -> list[PORecord]:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        out = []
        for r in rows:
            # Accept either the old 'Status' header or the new 'Approval_Status'.
            approval = (r.get("Approval_Status") or r.get("Status") or "").strip()
            out.append(PORecord(
                po_number=r["PO_Number"].strip(),
                vendor=r["Vendor"].strip(),
                approved_amount=_to_decimal(r["Approved_Amount"]),
                approval_status=approval,
            ))
        return out

    @staticmethod
    def _load_ledger(path: Path) -> list[LedgerEntry]:
        if not path.exists():
            return []
        with open(path, newline="", encoding="utf-8") as f:
            return [
                LedgerEntry(
                    invoice_number=r["Invoice_Number"].strip(),
                    po_number=r["PO_Number"].strip(),
                    vendor=r["Vendor"].strip(),
                    amount=_to_decimal(r["Amount"]),
                    status=r["Status"].strip(),
                    date=r.get("Date", "").strip(),
                )
                for r in csv.DictReader(f)
            ]

    # --------------------------- projection ---------------------------- #
    def _recompute(self) -> None:
        """Rebuild every PO's derived columns from the ledger (source of truth)."""
        by_po: dict[str, list[LedgerEntry]] = {}
        for e in self.ledger:
            by_po.setdefault(e.po_number.upper(), []).append(e)

        for po in self.pos:
            entries = by_po.get(po.po_number.upper(), [])
            po.billed_amount = sum(
                (e.amount for e in entries if e.status.lower() == INVOICE_APPROVED),
                Decimal(0),
            )
            po.pending_amount = po.approved_amount - po.billed_amount
            po.last_updated = max((e.date for e in entries), default="") or ""
            po.billing_status = self._billing_status(po)

    def _billing_status(self, po: PORecord) -> str:
        if po.approval_status.lower() != "approved":
            return BillingStatus.ON_HOLD
        allowed = self.allowed_overage(po.approved_amount)   # max(2%, flat floor)
        if po.billed_amount <= _TOL:
            return BillingStatus.OPEN
        if po.billed_amount > po.approved_amount + allowed:   # beyond tolerance
            return BillingStatus.OVER_BILLED
        if po.billed_amount >= po.approved_amount - _TOL:      # reached approved (incl. within-tol overage)
            return BillingStatus.FULLY_BILLED
        return BillingStatus.PARTIALLY_BILLED

    # ----------------------------- queries ----------------------------- #
    def fuzzy_find_vendor(self, name: str | None, threshold: float = 85.0):
        if not name:
            return None, 0.0
        vendors = sorted({p.vendor for p in self.pos})
        # default_process lowercases, strips punctuation, and trims — so case and
        # punctuation differences ("ABC Technologies" vs "abc technologies.") don't
        # tank the score.
        match = process.extractOne(
            name, vendors, scorer=fuzz.WRatio, processor=utils.default_process
        )
        if not match:
            return None, 0.0
        vendor, score, _ = match
        return (vendor if score >= threshold else None), score

    def find_po(self, vendor: str, po_number: str | None) -> PORecord | None:
        """STRICT PO-number match, scoped to the (already-resolved) vendor."""
        if not po_number:
            return None
        target = po_number.strip().upper()
        for p in self.pos:
            if p.vendor == vendor and p.po_number.upper() == target:
                return p
        return None

    def get_po(self, po_number: str) -> PORecord | None:
        target = po_number.strip().upper()
        return next((p for p in self.pos if p.po_number.upper() == target), None)

    def billing_summary(self, po: PORecord) -> BillingSummary:
        entries = [e for e in self.ledger if e.po_number.upper() == po.po_number.upper()]
        return BillingSummary(
            po_number=po.po_number,
            approved_amount=po.approved_amount,
            invoices_raised=len(entries),
            total_billed=po.billed_amount,
            pending_amount=po.pending_amount,
            invoices=entries,
        )

    # -------------------------- the update path ------------------------ #
    def apply_invoice(
        self,
        invoice_number: str,
        po_number: str,
        vendor: str,
        amount,
        status: str = "Approved",
        when: str | None = None,
    ) -> ApplyResult:
        """Record an invoice against a PO and update the table.

        Robust to the conditions that matter:
          - duplicate invoice number      → rejected, no change
          - PO not found                  → rejected, no change
          - PO not approved               → recorded, ON_HOLD (no billing)
          - pending invoice               → recorded, balance unchanged
          - approved within balance       → PARTIAL or FULL
          - approved beyond balance       → OVER_BILLED (recorded + flagged)
        """
        amount = _to_decimal(amount)
        when = when or date.today().isoformat()

        # 1. duplicate guard (idempotent)
        if any(e.invoice_number.upper() == invoice_number.upper() for e in self.ledger):
            return ApplyResult(invoice_number, po_number, "DUPLICATE", False,
                               f"{invoice_number} already recorded — ignored",
                               self.get_po(po_number))

        # 2. PO must exist
        po = self.get_po(po_number)
        if po is None:
            return ApplyResult(invoice_number, po_number, "PO_NOT_FOUND", False,
                               f"PO {po_number} not in master — invoice not recorded")

        # 3. append to ledger (source of truth) + persist
        entry = LedgerEntry(invoice_number, po_number, vendor, amount, status, when)
        self.ledger.append(entry)
        self._append_ledger_row(entry)

        # 4. recompute projection + persist PO table
        self._recompute()
        self.save()

        # 5. classify what happened for the caller
        po = self.get_po(po_number)
        if status.lower() != INVOICE_APPROVED:
            cond, applied, msg = "RECORDED_PENDING", False, \
                f"{invoice_number} recorded as '{status}' — balance unchanged"
        elif po.billing_status == BillingStatus.ON_HOLD:
            cond, applied, msg = "ON_HOLD", False, \
                f"PO {po_number} is not approved — invoice held"
        elif po.billing_status == BillingStatus.OVER_BILLED:
            allowed = self.allowed_overage(po.approved_amount)
            cond, applied, msg = "OVER_BILLED", True, (
                f"OVER-BILLING: billed {po.billed_amount} > approved {po.approved_amount} "
                f"+ allowed overage {allowed}"
            )
        elif po.billing_status == BillingStatus.FULLY_BILLED:
            over = po.billed_amount - po.approved_amount
            if over > _TOL:
                cond, applied, msg = "FULL", True, (
                    f"PO {po_number} fully billed — {over} over approved, "
                    f"within allowed overage {self.allowed_overage(po.approved_amount)}"
                )
            else:
                cond, applied, msg = "FULL", True, f"PO {po_number} now fully billed"
        else:
            cond, applied, msg = "PARTIAL", True, \
                f"Billed {amount}; {po.pending_amount} still pending on {po_number}"

        return ApplyResult(invoice_number, po_number, cond, applied, msg, po)

    # ----------------------------- persistence ------------------------- #
    _PO_FIELDS = [
        "PO_Number", "Vendor", "Approved_Amount", "Approval_Status",
        "Billed_Amount", "Pending_Amount", "Billing_Status", "Last_Updated",
    ]

    def save(self) -> None:
        """Atomically write the projection to po_dataset.csv (temp + replace)."""
        fd, tmp = tempfile.mkstemp(dir=str(self.po_csv.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._PO_FIELDS)
                w.writeheader()
                for p in self.pos:
                    w.writerow({
                        "PO_Number": p.po_number,
                        "Vendor": p.vendor,
                        "Approved_Amount": p.approved_amount,
                        "Approval_Status": p.approval_status,
                        "Billed_Amount": p.billed_amount,
                        "Pending_Amount": p.pending_amount,
                        "Billing_Status": p.billing_status,
                        "Last_Updated": p.last_updated,
                    })
            os.replace(tmp, self.po_csv)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def _append_ledger_row(self, e: LedgerEntry) -> None:
        new_file = not self.ledger_csv.exists()
        with open(self.ledger_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["Invoice_Number", "PO_Number", "Vendor", "Amount", "Status", "Date"])
            w.writerow([e.invoice_number, e.po_number, e.vendor, e.amount, e.status, e.date])

    def rebuild(self) -> None:
        """Recompute the projection from the ledger and persist it."""
        self._recompute()
        self.save()


if __name__ == "__main__":
    # Rebuild the projection and print the current table.
    store = POStore()
    store.rebuild()
    print("PO table (projection from ledger):\n")
    for p in store.pos:
        print(f"  {p.po_number}  {p.vendor:18s} appr={p.approved_amount:>8} "
              f"billed={p.billed_amount:>8} pending={p.pending_amount:>8} "
              f"{p.billing_status}")
