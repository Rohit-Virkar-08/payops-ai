"""Format detection and dispatch."""

from __future__ import annotations

from pathlib import Path

from .base import BaseExtractor, DocFormat, ExtractedDocument, IMAGE_SUFFIXES
from .image_extractor import ImageExtractor
from .pdf_extractor import PdfExtractor


def detect_format(path) -> DocFormat:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return DocFormat.PDF
    if suffix in IMAGE_SUFFIXES:
        return DocFormat.IMAGE
    raise ValueError(f"Unsupported file type: {suffix}")


def extract(path) -> ExtractedDocument:
    """Detect format and run the matching extractor."""
    fmt = detect_format(path)
    extractor: BaseExtractor = PdfExtractor() if fmt == DocFormat.PDF else ImageExtractor()
    return extractor.extract(path)
