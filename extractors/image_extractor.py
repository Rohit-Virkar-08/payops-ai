"""ImageExtractor — OCR-based text extraction for image files."""

from __future__ import annotations

from pathlib import Path

import cv2
import fitz  # PyMuPDF — only for the ocr_pixmap helper used by PdfExtractor
import numpy as np

from .base import BaseExtractor, DocFormat, ExtractedDocument, PageText, ocr_array


class ImageExtractor(BaseExtractor):
    """OCR every page of an image file via RapidOCR."""
    fmt = DocFormat.IMAGE

    def __init__(self, upscale_to: int = 1600):
        # Upscale small images so small fonts stay legible to the detector.
        self.upscale_to = upscale_to

    def _read(self, path: Path) -> np.ndarray:
        # Unicode-safe read (cv2.imread chokes on non-ASCII Windows paths).
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        h, w = img.shape[:2]
        if max(h, w) < self.upscale_to:
            scale = self.upscale_to / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return img

    def extract(self, path) -> ExtractedDocument:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        text, conf = ocr_array(self._read(path))
        return ExtractedDocument(
            source_path=str(path),
            fmt=self.fmt,
            pages=[PageText(page=1, text=text, confidence=conf, mode="ocr")],
        )

    def ocr_pixmap(self, pix: "fitz.Pixmap") -> tuple[str, float]:
        """OCR a rendered PDF page (used by PdfExtractor for scanned pages)."""
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)
        return ocr_array(img)
