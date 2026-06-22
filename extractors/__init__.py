"""
extractors — format detection + text extraction layer for the AP pipeline.

    from extractors import extract, ImageExtractor, PdfExtractor, ExtractedDocument

`extract(path)` detects the format and dispatches to the right extractor.
Both extractors return an ExtractedDocument (text + confidence + per-page
provenance), which the ExtractionAgent then sends to the LLM.
"""

from .base import (
    DocFormat,
    ExtractedDocument,
    PageText,
    BaseExtractor,
)
from .image_extractor import ImageExtractor
from .pdf_extractor import PdfExtractor
from .router import detect_format, extract

__all__ = [
    "DocFormat",
    "ExtractedDocument",
    "PageText",
    "BaseExtractor",
    "ImageExtractor",
    "PdfExtractor",
    "detect_format",
    "extract",
]
