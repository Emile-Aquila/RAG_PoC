"""Persist engine-neutral OCR output in a JSON cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .service import (
    MIN_OCR_DOCUMENT_LENGTH,
    OcrDocument,
    OcrEngine,
    OcrPage,
    sha256_file,
)

try:
    from ..rag_core import PROJECT_DIR, RagError
except ImportError:  # pragma: no cover - direct script execution
    from rag_core import PROJECT_DIR, RagError


OCR_CACHE_DIR = PROJECT_DIR / "ocr_cache"
CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OcrCacheResult:
    document: OcrDocument
    cache_hit: bool
    warning: str | None = None


class OcrCache:
    def __init__(self, cache_dir: Path = OCR_CACHE_DIR) -> None:
        self.cache_dir = cache_dir

    def _cache_path(
        self, pdf_path: Path, source_sha256: str, settings_digest: str
    ) -> Path:
        source_id = hashlib.sha256(pdf_path.name.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / source_id / f"{source_sha256}-{settings_digest}.json"

    def get_or_create(
        self,
        pdf_path: Path,
        engine: OcrEngine,
        on_miss: Callable[[], None] | None = None,
    ) -> OcrCacheResult:
        try:
            source_sha256 = sha256_file(pdf_path)
        except OSError as exc:
            raise RagError(f"PDFを読み取れません: {pdf_path.name} ({exc})") from exc

        cache_path = self._cache_path(
            pdf_path, source_sha256, engine.settings_digest
        )
        cached = self._load(
            cache_path,
            expected_source=pdf_path.name,
            expected_source_sha256=source_sha256,
            expected_engine_version=engine.engine_version,
            expected_settings_digest=engine.settings_digest,
        )
        if cached is not None:
            return OcrCacheResult(document=cached, cache_hit=True)

        if on_miss is not None:
            on_miss()
        document = engine.extract(pdf_path)
        if document.source_sha256 != source_sha256:
            raise RagError(
                f"OCR処理中にPDFが変更されました: {pdf_path.name}。再実行してください。"
            )
        if document.settings_digest != engine.settings_digest:
            raise RagError("OCRエンジン設定と結果の設定IDが一致しません。")

        warning = self._store(cache_path, document)
        return OcrCacheResult(
            document=document,
            cache_hit=False,
            warning=warning,
        )

    def _load(
        self,
        path: Path,
        *,
        expected_source: str,
        expected_source_sha256: str,
        expected_engine_version: str,
        expected_settings_digest: str,
    ) -> OcrDocument | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
                return None
            document_data = payload["document"]
            if document_data["source"] != expected_source:
                return None
            if document_data["source_sha256"] != expected_source_sha256:
                return None
            if document_data["engine_version"] != expected_engine_version:
                return None
            if document_data["settings_digest"] != expected_settings_digest:
                return None

            pages = tuple(
                OcrPage(
                    page_number=int(page["page_number"]),
                    markdown=str(page["markdown"]),
                )
                for page in document_data["pages"]
            )
            page_count = int(document_data["page_count"])
            if page_count <= 0 or len(pages) != page_count:
                return None
            if tuple(page.page_number for page in pages) != tuple(
                range(1, page_count + 1)
            ):
                return None
            if (
                sum(len(page.markdown.strip()) for page in pages)
                < MIN_OCR_DOCUMENT_LENGTH
            ):
                return None
            return OcrDocument(
                source=str(document_data["source"]),
                source_sha256=str(document_data["source_sha256"]),
                page_count=page_count,
                engine=str(document_data["engine"]),
                engine_version=str(document_data["engine_version"]),
                settings_digest=str(document_data["settings_digest"]),
                pages=pages,
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    def _store(self, path: Path, document: OcrDocument) -> str | None:
        payload: dict[str, Any] = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "document": asdict(document),
        }
        temporary_path: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(payload, temporary, ensure_ascii=False, sort_keys=True)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
            return None
        except OSError as exc:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            return f"OCRキャッシュを保存できませんでした: {path} ({exc})"
