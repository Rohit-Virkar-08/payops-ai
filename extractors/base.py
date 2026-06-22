"""
Shared types and the OCR engine used by both extractors.

ExtractedDocument is the contract every extractor returns and the LLM agent
consumes: combined text + a confidence score + per-page provenance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from rapidocr_onnxruntime import RapidOCR

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"}

# Below this many characters on a PDF page, assume it's a scanned/image page
# with no real text layer and fall back to OCR.
MIN_CHARS_PER_PAGE = 50


class DocFormat(str, Enum):
    PDF = "pdf"
    IMAGE = "image"


@dataclass
class PageText:
    """Text extracted from one page, with provenance."""
    page: int
    text: str
    confidence: float          # 0.0–1.0
    mode: str                  # "native" (PDF text layer) | "ocr"


@dataclass
class ExtractedDocument:
    """Uniform output of any extractor — the handoff to the LLM agent."""
    source_path: str
    fmt: DocFormat
    pages: list[PageText] = field(default_factory=list)

    REVIEW_THRESHOLD = 0.80

    @property
    def text(self) -> str:
        """All page text joined with page-break markers."""
        return "\n\n--- PAGE BREAK ---\n\n".join(p.text for p in self.pages)

    @property
    def confidence(self) -> float:
        """Mean page confidence (length-weighted so blank pages don't skew it)."""
        weighted = [(p.confidence, max(len(p.text), 1)) for p in self.pages]
        if not weighted:
            return 0.0
        total_w = sum(w for _, w in weighted)
        return sum(c * w for c, w in weighted) / total_w

    @property
    def needs_review(self) -> bool:
        return self.confidence < self.REVIEW_THRESHOLD


class BaseExtractor(ABC):
    """Common contract: a path in, an ExtractedDocument out."""
    fmt: DocFormat

    @abstractmethod
    def extract(self, path) -> ExtractedDocument:
        ...


# --------------------------------------------------------------------------- #
# OCR engine — shared singleton (loading the ONNX models is the expensive part)
# --------------------------------------------------------------------------- #

_engine: RapidOCR | None = None


def get_engine() -> RapidOCR:
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    return _engine


def ocr_array(img: np.ndarray) -> tuple[str, float]:
    """Run OCR on a BGR image array → (text in reading order, mean confidence)."""
    raw, _ = get_engine()(img)
    if not raw:
        return "", 0.0

    items = []  # (top, left, text, score)
    for box, text, score in raw:
        t = str(text).strip()
        if not t:
            continue
        top = min(p[1] for p in box)
        left = min(p[0] for p in box)
        items.append((top, left, t, float(score)))

    # Reading order: cluster into rows by vertical position, then left→right.
    items.sort(key=lambda it: (round(it[0] / 15), it[1]))
    text = "\n".join(it[2] for it in items)
    conf = sum(it[3] for it in items) / len(items)
    return text, conf
