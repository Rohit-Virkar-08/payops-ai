"""
Back-compat shim. The extractors now live in the `extractors` package, split
into image_extractor.py and pdf_extractor.py. Prefer importing from there:

    from extractors import extract, ImageExtractor, PdfExtractor, ExtractedDocument

This module re-exports them so older imports keep working.
"""

from extractors import (  # noqa: F401
    DocFormat,
    ExtractedDocument,
    PageText,
    BaseExtractor,
    ImageExtractor,
    PdfExtractor,
    detect_format,
    extract,
)

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
