"""
Quick smoke-test / demo runner for the Ingestion + Extraction pipeline.

Usage:
    python run_extraction.py <path-to-invoice>

    Set ANTHROPIC_API_KEY in your environment first.

Prints a structured summary of the extraction result and any clarification
requests that would be shown to an analyst.
"""

import json
import sys
from pathlib import Path

from schemas import InvoiceCase
from agents import IngestionAgent, ExtractionAgent
from agents.extraction import merge_analyst_input


def run(filepath: str) -> InvoiceCase:
    case = InvoiceCase(source_filename=filepath)

    print(f"\n{'='*60}")
    print(f" Case {case.case_id}")
    print(f" File: {Path(filepath).name}")
    print(f"{'='*60}")

    # Step 1 — Ingestion
    ingestion = IngestionAgent()
    ingestion.execute(case)
    _print_trace_step(case.trace[-1])

    # Step 2 — Extraction (LLM)
    extraction = ExtractionAgent()
    extraction.execute(case)
    _print_trace_step(case.trace[-1])

    # Report extraction result
    print(f"\n--- Extracted Fields ---")
    _print_extracted(case)

    # Report clarifications
    if case.clarifications:
        print(f"\n--- Clarification Requests ({len(case.clarifications)}) ---")
        _print_clarifications(case)
    else:
        print("\n All fields extracted with high confidence — no clarifications needed.")

    print(f"\nExtraction Status: {case.extraction_status.value.upper()}")

    if case.is_extraction_blocked:
        print("  !! Pipeline BLOCKED — analyst must resolve critical fields before processing continues.")

    return case


def _print_trace_step(step) -> None:
    elapsed = (step.ended_at - step.started_at).total_seconds() if step.ended_at else 0
    icon = "✓" if step.status == "ok" else "✗"
    print(f"\n[{icon}] {step.agent} ({elapsed:.1f}s)")
    if step.summary:
        print(f"    {step.summary}")
    if step.error:
        print(f"    ERROR: {step.error[:300]}")


def _print_extracted(case) -> None:
    inv = case.extracted
    rows = [
        ("Document type",    case.document_type.value,                 None),
        ("Invoice #",        inv.invoice_number.value,                 inv.invoice_number.confidence),
        ("Invoice date",     inv.invoice_date.value,                   inv.invoice_date.confidence),
        ("Due date",         inv.due_date.value,                       inv.due_date.confidence),
        ("Vendor",           inv.vendor_name.value,                    inv.vendor_name.confidence),
        ("Currency",         inv.currency.value,                       inv.currency.confidence),
        ("Subtotal",         inv.subtotal.value,                       inv.subtotal.confidence),
        ("Tax",              inv.tax_total.value,                      inv.tax_total.confidence),
        ("Discount",         inv.discount_total.value,                 inv.discount_total.confidence),
        ("Total",            inv.total_amount.value,                   inv.total_amount.confidence),
        ("Payment terms",    inv.payment_terms.value,                  inv.payment_terms.confidence),
        ("PO refs",          [f.value for f in inv.po_references],     None),
    ]
    for label, value, conf in rows:
        conf_str = f"  [{conf:.0%}]" if conf is not None else ""
        flag = " ⚠" if conf is not None and conf < 0.85 else ""
        print(f"  {label:<18} {str(value) if value is not None else '—'}{conf_str}{flag}")

    if inv.line_items:
        print(f"\n  Line items ({len(inv.line_items)}):")
        for i, li in enumerate(inv.line_items):
            desc  = li.description.value or "—"
            qty   = li.quantity.value or "?"
            price = li.unit_price.value or "?"
            total = li.line_total.value or "?"
            print(f"    {i+1}. {desc[:40]:<40} qty:{qty}  @{price}  = {total}")


def _print_clarifications(case) -> None:
    for req in case.clarifications:
        crit = req.criticality.value.upper()
        icon = "!!" if req.criticality.value == "critical" else " >"
        print(f"\n  {icon} [{crit}] {req.field_label}")
        print(f"     Reason:    {req.reason}")
        if req.auto_extracted_value:
            print(f"     Auto-got:  '{req.auto_extracted_value}' ({req.auto_confidence:.0%} confidence)")
        else:
            print(f"     Auto-got:  — (not found)")
        if req.evidence:
            print(f"     Evidence:  \"{req.evidence[:80]}\"")

    print()
    print("  To resolve: set req.analyst_value = '...' and req.status = ANALYST_PROVIDED,")
    print("  then call merge_analyst_input(case) to fold values back in.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_extraction.py <invoice_file>")
        sys.exit(1)
    run(sys.argv[1])
