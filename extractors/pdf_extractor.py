"""PdfExtractor — text-layer extraction with per-page OCR fallback for scans."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from .base import BaseExtractor, DocFormat, ExtractedDocument, MIN_CHARS_PER_PAGE, PageText
from .image_extractor import ImageExtractor


class PdfExtractor(BaseExtractor):
    """Read each page's native text layer; OCR image-only pages via ImageExtractor."""
    fmt = DocFormat.PDF

    def __init__(self, ocr: ImageExtractor | None = None, dpi: int = 200):
        # Composition: a scanned PDF page is just an image, so reuse ImageExtractor.
        self.ocr = ocr or ImageExtractor()
        self.dpi = dpi

    def extract(self, path) -> ExtractedDocument:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        doc = fitz.open(str(path))
        pages: list[PageText] = []
        try:
            for i, page in enumerate(doc):
                native = page.get_text().strip()
                if len(native) >= MIN_CHARS_PER_PAGE:
                    pages.append(PageText(page=i + 1, text=native, confidence=1.0, mode="native"))
                else:
                    # Image-only / scanned page → render and OCR
                    mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                    text, conf = self.ocr.ocr_pixmap(page.get_pixmap(matrix=mat))
                    pages.append(PageText(page=i + 1, text=text, confidence=conf, mode="ocr"))
        finally:
            doc.close()

        return ExtractedDocument(source_path=str(path), fmt=self.fmt, pages=pages)
