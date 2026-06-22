"""
generate_demo_invoices.py — build the demo "story path" for AP Autopilot.

Run:  python generate_demo_invoices.py

It (1) resets the PO master + ledger to a clean starting state (the 5 vendors,
all OPEN, nothing billed), (2) saves that state as the canonical seed so the UI
"Reset demo data" button can restore it, and (3) generates 12 invoice PDFs that
walk the happy path and every edge case in narrative order.

Run the invoices in filename order in the UI with **Commit on approval = ON**
and the confidence gate at 0.90. See DEMO_STORY.md for the full script.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import fitz  # PyMuPDF

DATA = Path("data")
ACCENT = (0.42, 0.27, 0.76)   # purple
GREY = (0.45, 0.45, 0.45)
LINE = (0.75, 0.75, 0.75)
BUYER = "Northwind Retail Pvt Ltd"

# --------------------------------------------------------------------------- #
# 1. Starting state — the 5 vendors, all OPEN, nothing billed (ledger empty)
# --------------------------------------------------------------------------- #
PO_ROWS = [
    ("PO_Number", "Vendor", "Approved_Amount", "Approval_Status",
     "Billed_Amount", "Pending_Amount", "Billing_Status", "Last_Updated"),
    ("PO-456", "abc technologies",          "60000",  "Approved", "0", "60000",  "OPEN", ""),
    ("PO-777", "Mega Industrial",           "60000",  "Approved", "0", "60000",  "OPEN", ""),
    ("PO-888", "Global Parts",              "100000", "Approved", "0", "100000", "OPEN", ""),
    ("PO-900", "Scan Corp",                 "12000",  "Approved", "0", "12000",  "OPEN", ""),
    ("4587",   "Metro Components pvt ltd",  "45000",  "Approved", "0", "45000",  "OPEN", ""),
]
LEDGER_HEADER = [("Invoice_Number", "PO_Number", "Vendor", "Amount", "Status", "Date")]


def _write_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def reset_tables():
    _write_csv(DATA / "po_dataset.csv", PO_ROWS)
    _write_csv(DATA / "invoice_ledger.csv", LEDGER_HEADER)
    # canonical seed for the UI "Reset demo data" button
    _write_csv(DATA / "_seed_po.csv", PO_ROWS)
    _write_csv(DATA / "_seed_ledger.csv", LEDGER_HEADER)


# --------------------------------------------------------------------------- #
# 2. PDF invoice builder
# --------------------------------------------------------------------------- #
def _money(n: float) -> str:
    return f"{n:,.2f}"


def make_invoice(fname, vendor, invoice_no, invoice_date, due_date, po_number,
                 total, desc, vendor_gst="27ABCDE1234F1Z5", currency="INR"):
    tax = round(total * 0.18)
    subtotal = total - tax

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    x = 56

    page.insert_text((x, 72), "TAX INVOICE", fontsize=22, fontname="hebo", color=ACCENT)
    page.insert_text((x, 96), vendor, fontsize=13, fontname="hebo")
    page.insert_text((x, 112), f"GSTIN: {vendor_gst}", fontsize=9, fontname="helv", color=GREY)
    page.draw_line((x, 126), (539, 126), color=ACCENT, width=1.2)

    # Bill-to (left)
    page.insert_text((x, 156), "Bill To:", fontsize=10, fontname="hebo")
    page.insert_text((x, 172), BUYER, fontsize=10, fontname="helv")
    page.insert_text((x, 186), "Mumbai, India", fontsize=10, fontname="helv")

    # Meta (right)
    mx, my = 330, 156
    meta = [("Invoice Number", invoice_no), ("Invoice Date", invoice_date),
            ("Due Date", due_date)]
    if po_number:
        meta.append(("PO Number", po_number))
    meta.append(("Currency", currency))
    for k, v in meta:
        page.insert_text((mx, my), f"{k}:", fontsize=10, fontname="hebo")
        page.insert_text((mx + 112, my), str(v), fontsize=10, fontname="helv")
        my += 16

    # Line-item table
    ty = 270
    page.draw_line((x, ty - 14), (539, ty - 14), color=LINE, width=0.8)
    for label, lx in [("Description", x), ("Qty", 360), ("Unit Price", 415), ("Amount", 500)]:
        page.insert_text((lx, ty), label, fontsize=10, fontname="hebo")
    page.draw_line((x, ty + 6), (539, ty + 6), color=LINE, width=0.8)
    ly = ty + 28
    page.insert_text((x, ly), desc, fontsize=10, fontname="helv")
    page.insert_text((360, ly), "1", fontsize=10, fontname="helv")
    page.insert_text((415, ly), _money(subtotal), fontsize=10, fontname="helv")
    page.insert_text((500, ly), _money(subtotal), fontsize=10, fontname="helv")

    # Totals
    sy = ly + 44
    for k, v in [("Subtotal", subtotal), ("Tax (18%)", tax), ("Total", total)]:
        font = "hebo" if k == "Total" else "helv"
        page.insert_text((400, sy), f"{k}:", fontsize=11, fontname=font)
        page.insert_text((478, sy), f"{currency} {_money(v)}", fontsize=11, fontname=font)
        sy += 18

    page.insert_text((x, 806), "System-generated demo invoice — AP Autopilot.",
                     fontsize=8, fontname="helv", color=GREY)
    doc.save(str(DATA / fname))
    doc.close()
    print(f"  wrote {fname}")


# --------------------------------------------------------------------------- #
# 3. The story (run in this order, Commit ON, gate 0.90)
# --------------------------------------------------------------------------- #
INVOICES = [
    # fname, vendor, invoice_no, inv_date, due_date, po, total, desc
    ("01_abc_happy.pdf", "abc technologies", "ACME-2026-001", "2026-06-08", "2026-07-08",
     "PO-456", 30000, "Cloud infrastructure - June (part 1)"),

    ("02_abc_near_duplicate.pdf", "abc technologies", "ACME-2026-019", "2026-06-18", "2026-07-18",
     "PO-456", 30000, "Cloud infrastructure services"),

    ("03_abc_split_po.pdf", "abc technologies", "ACME-2026-002", "2026-06-15", "2026-07-15",
     "PO-456", 30000, "Cloud infrastructure - June (part 2)"),

    ("04_abc_overbilled.pdf", "abc technologies", "ACME-2026-003", "2026-06-20", "2026-07-20",
     "PO-456", 8000, "Additional support hours"),

    ("05_abc_exact_duplicate.pdf", "abc technologies", "ACME-2026-001", "2026-06-08", "2026-07-08",
     "PO-456", 30000, "Cloud infrastructure - June (part 1)"),

    ("06_mega_within_tolerance.pdf", "Mega Industrial", "MEGA-5521", "2026-06-12", "2026-07-12",
     "PO-777", 60900, "Industrial equipment supply"),

    ("07_global_overbilled.pdf", "Global Parts", "GP-7788", "2026-06-12", "2026-07-12",
     "PO-888", 106000, "Spare parts bulk order"),

    ("08_vendor_mismatch.pdf", "Zenith Traders Ltd", "ZN-301", "2026-06-12", "2026-07-12",
     "PO-900", 10000, "Document scanning services"),

    ("09_po_not_found.pdf", "Metro Components pvt ltd", "MC-5001", "2026-06-12", "2026-07-12",
     "PO-7777", 15000, "Mechanical components"),

    ("10_metro_happy_overdue.pdf", "Metro Components pvt ltd", "MC-7001", "2025-11-15", "2025-12-15",
     "4587", 20000, "Mechanical components - Q4"),

    ("11_invoice_number_collision.pdf", "Metro Components pvt ltd", "ACME-2026-001", "2026-06-19", "2026-07-19",
     "4587", 12000, "Fasteners and fittings"),

    ("12_missing_po.pdf", "abc technologies", "MISC-900", "2026-06-12", "2026-07-12",
     None, 10000, "Ad-hoc consulting"),
]


def main():
    DATA.mkdir(exist_ok=True)
    print("Resetting PO master + ledger to clean starting state...")
    reset_tables()
    print("Generating story invoices:")
    for fname, vendor, inv, idate, ddate, po, total, desc in INVOICES:
        make_invoice(fname, vendor, inv, idate, ddate, po, total, desc)
    print(f"\nDone. {len(INVOICES)} invoices in {DATA}/  (tables reset).")


if __name__ == "__main__":
    main()
