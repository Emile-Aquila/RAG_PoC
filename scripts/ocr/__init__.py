"""OCR extraction and persistent caching services."""

from .cache import OCR_CACHE_DIR, OcrCache, OcrCacheResult
from .service import (
    MIN_OCR_DOCUMENT_LENGTH,
    OCR_ENGINE_NAME,
    DoclingRapidOcrEngine,
    OcrDocument,
    OcrEngine,
    OcrPage,
    sha256_file,
)

__all__ = [
    "MIN_OCR_DOCUMENT_LENGTH",
    "OCR_CACHE_DIR",
    "OCR_ENGINE_NAME",
    "DoclingRapidOcrEngine",
    "OcrCache",
    "OcrCacheResult",
    "OcrDocument",
    "OcrEngine",
    "OcrPage",
    "sha256_file",
]
