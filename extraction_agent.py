"""
ExtractionAgent — the LLM stage.

Takes an ExtractedDocument (text + OCR confidence from the extractors) and calls
OpenAI with structured outputs to read the content and pull out the useful
invoice fields. Each field comes back with its own confidence + evidence
snippet; the document-level OCR confidence is carried through so the downstream
decision layer can see BOTH how well the text was read AND how confidently each
field was interpreted.

Model: openai/gpt-oss-120b served on Groq (OpenAI-compatible endpoint, so we
reuse the OpenAI SDK's structured-output `.parse` helper). The GROQ_API_KEY is
read from a .env file (or the environment).
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from extractors import ExtractedDocument
from llm_client import DEFAULT_MODEL, get_client

_MODEL = DEFAULT_MODEL

_SYSTEM = (
    "You are an accounts-payable analyst. Read the supplied document text (produced "
    "by OCR or a PDF text layer) and extract these fields:\n"
    "  - vendor_name: the supplier / seller issuing the invoice\n"
    "  - invoice_number: the invoice's own identifier\n"
    "  - po_numbers: every purchase-order (PO) reference found, as a list\n"
    "  - invoice_date: the date the invoice was issued\n"
    "  - due_date: the date payment is due\n"
    "  - billing_period_start / billing_period_end: the service/billing period the "
    "invoice covers — often labelled 'service period', 'billing period', 'from'/'to', "
    "or a date range. Leave null if the invoice is a one-off with no period.\n"
    "  - currency, subtotal, tax_total, total_amount\n"
    "  - line_items: for each row, its description, quantity, unit price, line total\n\n"
    "Rules:\n"
    "- Copy values exactly as written; do not normalise, convert, or compute them.\n"
    "- For dates, prefer ISO format (YYYY-MM-DD) only when the original is unambiguous; "
    "otherwise copy it verbatim.\n"
    "- If a field is absent, set value to null and confidence to 0.0.\n"
    "- confidence (0.0-1.0) reflects how clearly the field is labelled AND how clean "
    "the surrounding text looks.\n"
    "- The text's overall OCR confidence is given below; when it is low, be more "
    "cautious and lower your per-field confidence accordingly.\n"
    "- evidence: copy the short text snippet that supports each value.\n"
    "Return the answer using the required structured schema — nothing else."
)


# --------------------------------------------------------------------------- #
# Output schema (pinned via structured outputs → validated, typed)
# --------------------------------------------------------------------------- #

class ExtractedField(BaseModel):
    value: str | None = Field(description="Value exactly as it appears, or null if absent")
    confidence: float = Field(description="0.0-1.0 confidence this value is correct")
    evidence: str | None = Field(description="Text snippet supporting this value")


class LineItem(BaseModel):
    description: ExtractedField
    quantity: ExtractedField
    unit_price: ExtractedField
    line_total: ExtractedField


class InvoiceFields(BaseModel):
    vendor_name: ExtractedField
    invoice_number: ExtractedField
    po_numbers: list[ExtractedField]
    invoice_date: ExtractedField
    due_date: ExtractedField
    billing_period_start: ExtractedField
    billing_period_end: ExtractedField
    currency: ExtractedField
    subtotal: ExtractedField
    tax_total: ExtractedField
    total_amount: ExtractedField
    line_items: list[LineItem]


@dataclass
class ExtractionResult:
    """The agent's output: structured fields + provenance carried through."""
    fields: InvoiceFields
    ocr_confidence: float       # from the extractor stage
    source_format: str
    source_path: str
    model: str

    @property
    def needs_review(self) -> bool:
        """Flag if OCR was weak or any critical field came back low-confidence."""
        critical = [
            self.fields.invoice_number,
            self.fields.total_amount,
            self.fields.vendor_name,
        ]
        return self.ocr_confidence < 0.80 or any(f.confidence < 0.80 for f in critical)


class ExtractionAgent:
    def __init__(self, model: str = _MODEL, api_key: str | None = None, max_tokens: int = 4096):
        self.model = model
        self.max_tokens = max_tokens
        self._client = get_client(api_key)

    def run(self, doc: ExtractedDocument) -> ExtractionResult:
        user_content = (
            f"OCR/text confidence: {doc.confidence:.0%}\n"
            f"Source format: {doc.fmt.value}\n\n"
            f"DOCUMENT TEXT:\n\n{doc.text}"
        )
        completion = self._client.beta.chat.completions.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_content},
            ],
            response_format=InvoiceFields,
        )
        message = completion.choices[0].message
        if message.refusal:
            raise RuntimeError(f"Model refused extraction: {message.refusal}")
        return ExtractionResult(
            fields=message.parsed,
            ocr_confidence=doc.confidence,
            source_format=doc.fmt.value,
            source_path=doc.source_path,
            model=self.model,
        )


if __name__ == "__main__":
    import sys

    from extractors import extract

    target = sys.argv[1] if len(sys.argv) > 1 else "data/invoice_01_happy.pdf"
    doc = extract(target)
    print(f"[extract] {doc.fmt.value} | confidence={doc.confidence:.0%} | {len(doc.text)} chars")

    result = ExtractionAgent().run(doc)
    f = result.fields
    print(f"[agent]   model={result.model} | needs_review={result.needs_review}\n")
    for label, ef in [
        ("Vendor", f.vendor_name), ("Invoice #", f.invoice_number),
        ("Invoice date", f.invoice_date), ("Due date", f.due_date),
        ("Period start", f.billing_period_start), ("Period end", f.billing_period_end),
        ("Total", f.total_amount),
    ]:
        print(f"  {label:14s} {str(ef.value):<26} [{ef.confidence:4.0%}]")
    for i, po in enumerate(f.po_numbers):
        print(f"  {'PO #' + str(i+1):14s} {str(po.value):<26} [{po.confidence:4.0%}]")
