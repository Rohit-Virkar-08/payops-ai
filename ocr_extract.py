"""
ocr_extract — free, offline OCR for invoice images.

Uses RapidOCR (ONNXRuntime + PaddleOCR models, pip-installable, no system
binary, no API key). Given an image path it returns:
  - full_text:  all recognized text, ordered roughly top-to-bottom / left-to-right
  - lines:      per-line text + bounding box + confidence (for provenance)

This mirrors the project's provenance model: every piece of text keeps the
bbox + confidence it came from, so downstream agents can trace any value back
to where it sat on the page.

Why RapidOCR over Tesseract here: Tesseract needs a system binary installed
(not present on this machine); RapidOCR ships its models via pip and runs on
CPU offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

# Single shared engine — loading the ONNX models is the expensive part.
_engine: RapidOCR | None = None


def _get_engine() -> RapidOCR:
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    return _engine


@dataclass
class OcrLine:
    text: str
    confidence: float
    bbox: list[list[float]]  # 4 corner points [[x,y], ...]

    @property
    def top(self) -> float:
        return min(p[1] for p in self.bbox)

    @property
    def left(self) -> float:
        return min(p[0] for p in self.bbox)


@dataclass
class OcrResult:
    full_text: str
    lines: list[OcrLine] = field(default_factory=list)
    mean_confidence: float = 0.0


def _preprocess(path: Path) -> np.ndarray:
    """Light cleanup that helps OCR on photographed / low-quality scans:
    grayscale + adaptive threshold + upscale small images."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    h, w = img.shape[:2]
    # Upscale small images so small fonts are legible to the detector.
    if max(h, w) < 1600:
        scale = 1600 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img


def extract_text(image_path: str | Path, preprocess: bool = True) -> OcrResult:
    """Run OCR on an invoice image and return ordered text + provenance."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)

    engine = _get_engine()
    img = _preprocess(path) if preprocess else str(path)

    raw, _elapsed = engine(img)
    if not raw:
        return OcrResult(full_text="", lines=[], mean_confidence=0.0)

    lines = [
        OcrLine(text=str(text).strip(), confidence=float(score), bbox=box)
        for box, text, score in raw
        if str(text).strip()
    ]
    # Reading order: group into rows by vertical position, then left-to-right.
    lines.sort(key=lambda ln: (round(ln.top / 15), ln.left))

    full_text = "\n".join(ln.text for ln in lines)
    mean_conf = sum(ln.confidence for ln in lines) / len(lines) if lines else 0.0
    return OcrResult(full_text=full_text, lines=lines, mean_confidence=mean_conf)


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else None
    if not target:
        # Default to the first image in data/ for a quick smoke test.
        imgs = sorted(Path("data").glob("*.png"))
        target = str(imgs[0]) if imgs else None
    if not target:
        print("Usage: python ocr_extract.py <image_path>")
        raise SystemExit(1)

    result = extract_text(target)
    print(f"# {target}")
    print(f"# {len(result.lines)} lines | mean confidence {result.mean_confidence:.2%}\n")
    print(result.full_text)
