"""Run full-document OCR with Docling and the RapidOCR backend."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import pymupdf

try:
    from ..rag_core import RagError
except ImportError:  # pragma: no cover - direct script execution
    from rag_core import RagError


MIN_OCR_DOCUMENT_LENGTH = 200
OCR_ENGINE_NAME = "docling+rapidocr"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


@dataclass(frozen=True)
class OcrPage:
    page_number: int
    markdown: str


@dataclass(frozen=True)
class OcrDocument:
    source: str
    source_sha256: str
    page_count: int
    engine: str
    engine_version: str
    settings_digest: str
    pages: tuple[OcrPage, ...]


class OcrEngine(Protocol):
    @property
    def engine_version(self) -> str: ...

    @property
    def settings_digest(self) -> str: ...

    def extract(self, pdf_path: Path) -> OcrDocument: ...


class DoclingRapidOcrEngine:
    """Lazy, reusable Docling converter configured for full-page RapidOCR."""

    def __init__(
        self,
        converter_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._converter_factory = converter_factory or self._build_converter
        self._converter: Any | None = None
        self._engine_version = (
            f"docling={_package_version('docling')};"
            f"rapidocr={_package_version('rapidocr')}"
        )
        self._settings: dict[str, Any] = {
            "backend": "onnxruntime",
            "do_table_structure": True,
            "language": ["en"],
            "mode": "full_page",
            "scale": 3.0,
            "table_cell_matching": True,
            "text_score": 0.5,
            "use_cls": True,
            "use_det": True,
            "use_rec": True,
        }
        settings_json = json.dumps(
            {
                "engine": OCR_ENGINE_NAME,
                "engine_version": self._engine_version,
                "settings": self._settings,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._settings_digest = hashlib.sha256(settings_json.encode("utf-8")).hexdigest()

    @property
    def engine_version(self) -> str:
        return self._engine_version

    @property
    def settings_digest(self) -> str:
        return self._settings_digest

    @property
    def settings(self) -> dict[str, Any]:
        return dict(self._settings)

    def _build_converter(self) -> Any:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                OcrMode,
                PdfPipelineOptions,
                RapidOcrOptions,
                TableStructureOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except Exception as exc:
            raise RagError(
                "Docling/RapidOCRを初期化できません。"
                "`uv sync`でOCR依存関係をインストールしてください。"
            ) from exc

        ocr_options = RapidOcrOptions(
            mode=OcrMode.FULL_PAGE,
            backend="onnxruntime",
            lang=["en"],
            scale=3.0,
            text_score=0.5,
            use_det=True,
            use_cls=True,
            use_rec=True,
            print_verbose=False,
        )
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.ocr_options = ocr_options
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True
        )
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def _get_converter(self) -> Any:
        if self._converter is None:
            self._converter = self._converter_factory()
        return self._converter

    def extract(self, pdf_path: Path) -> OcrDocument:
        try:
            with pymupdf.open(pdf_path) as pdf:
                page_count = len(pdf)
            if page_count == 0:
                raise RagError(f"PDFにページがありません: {pdf_path.name}")

            converted = self._get_converter().convert(source=pdf_path)
            document = converted.document
            docling_pages = getattr(document, "pages", None)
            if docling_pages is not None and len(docling_pages) != page_count:
                raise RagError(
                    f"OCR結果のページ数が一致しません: {pdf_path.name} "
                    f"(PDF={page_count}, OCR={len(docling_pages)})"
                )

            pages = tuple(
                OcrPage(
                    page_number=page_number,
                    markdown=str(
                        document.export_to_markdown(
                            page_no=page_number,
                            traverse_pictures=True,
                            image_placeholder="",
                        )
                    ).strip(),
                )
                for page_number in range(1, page_count + 1)
            )
        except RagError:
            raise
        except Exception as exc:
            raise RagError(f"全文OCRに失敗しました: {pdf_path.name} ({exc})") from exc

        if tuple(page.page_number for page in pages) != tuple(
            range(1, page_count + 1)
        ):
            raise RagError(f"OCR結果のページ番号が不正です: {pdf_path.name}")
        total_text = sum(len(page.markdown.strip()) for page in pages)
        if total_text < MIN_OCR_DOCUMENT_LENGTH:
            raise RagError(
                f"OCR結果が短すぎます: {pdf_path.name} ({total_text}文字)"
            )

        try:
            source_sha256 = sha256_file(pdf_path)
        except OSError as exc:
            raise RagError(
                f"OCR後にPDFを読み取れません: {pdf_path.name} ({exc})"
            ) from exc

        return OcrDocument(
            source=pdf_path.name,
            source_sha256=source_sha256,
            page_count=page_count,
            engine=OCR_ENGINE_NAME,
            engine_version=self.engine_version,
            settings_digest=self.settings_digest,
            pages=pages,
        )
