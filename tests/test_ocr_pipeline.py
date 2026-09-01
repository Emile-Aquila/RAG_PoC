from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import pymupdf

from scripts.ocr import (
    DoclingRapidOcrEngine,
    OcrCache,
    OcrDocument,
    OcrPage,
    sha256_file,
)
from scripts.rag_core import RagError


def _make_pdf(path: Path, page_count: int = 2) -> None:
    with pymupdf.open() as document:
        for index in range(page_count):
            page = document.new_page()
            page.insert_text((72, 72), f"Page {index + 1} test text")
        document.save(path)


class OcrServiceTest(unittest.TestCase):
    def test_extracts_every_page_and_reuses_converter(self) -> None:
        class FakeDocument:
            def __init__(self) -> None:
                self.pages = {1: object(), 2: object()}

            def export_to_markdown(self, **kwargs: object) -> str:
                page_number = int(kwargs["page_no"])  # type: ignore[arg-type]
                self.last_kwargs = kwargs
                return f"# Page {page_number}\n" + "recognized text " * 20

        class FakeConverter:
            def __init__(self) -> None:
                self.calls = 0
                self.document = FakeDocument()

            def convert(self, **_: object) -> object:
                self.calls += 1
                return type("Result", (), {"document": self.document})()

        converter = FakeConverter()
        factory_calls = 0

        def factory() -> FakeConverter:
            nonlocal factory_calls
            factory_calls += 1
            return converter

        engine = DoclingRapidOcrEngine(converter_factory=factory)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.pdf"
            second = Path(directory) / "second.pdf"
            _make_pdf(first)
            _make_pdf(second)
            first_result = engine.extract(first)
            engine.extract(second)

        self.assertEqual(factory_calls, 1)
        self.assertEqual(converter.calls, 2)
        self.assertEqual(first_result.page_count, 2)
        self.assertEqual([page.page_number for page in first_result.pages], [1, 2])
        self.assertIn("# Page 1", first_result.pages[0].markdown)
        self.assertTrue(converter.document.last_kwargs["traverse_pictures"])
        self.assertEqual(converter.document.last_kwargs["image_placeholder"], "")
        self.assertEqual(engine.settings["mode"], "full_page")
        self.assertEqual(engine.settings["backend"], "onnxruntime")
        self.assertEqual(engine.settings["language"], ["en"])

    def test_rejects_page_count_mismatch(self) -> None:
        class FakeDocument:
            def __init__(self) -> None:
                self.pages = {1: object()}

        class FakeConverter:
            def convert(self, **_: object) -> object:
                return type("Result", (), {"document": FakeDocument()})()

        engine = DoclingRapidOcrEngine(converter_factory=FakeConverter)
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "two-pages.pdf"
            _make_pdf(pdf, page_count=2)
            with self.assertRaisesRegex(RagError, "ページ数が一致"):
                engine.extract(pdf)

    @unittest.skipUnless(
        os.environ.get("RUN_OCR_INTEGRATION") == "1",
        "RUN_OCR_INTEGRATION=1 のときだけ実OCRを実行します。",
    )
    def test_real_docling_rapidocr_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scanned_pdf = root / "scanned.pdf"
            with pymupdf.open() as output:
                for page_number in (1, 2):
                    with pymupdf.open() as source:
                        page = source.new_page(width=595, height=842)
                        lines = [
                            f"INTEGRATION PAGE {page_number}",
                            *(f"Scientific evidence line {index}." for index in range(1, 16)),
                        ]
                        page.insert_textbox(
                            pymupdf.Rect(60, 70, 535, 700),
                            "\n".join(lines),
                            fontsize=14,
                            lineheight=1.3,
                        )
                        pixmap = page.get_pixmap(dpi=216, alpha=False)
                    target = output.new_page(width=595, height=842)
                    target.insert_image(target.rect, stream=pixmap.tobytes("png"))
                output.save(scanned_pdf)

            cache = OcrCache(root / "cache")
            engine = DoclingRapidOcrEngine()
            first = cache.get_or_create(scanned_pdf, engine)
            second = cache.get_or_create(scanned_pdf, engine)

        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(first.document.page_count, 2)
        self.assertIn("INTEGRATION", first.document.pages[0].markdown.upper())
        self.assertIn("PAGE 2", first.document.pages[1].markdown.upper())


class OcrCacheTest(unittest.TestCase):
    def test_cache_hit_and_content_change_invalidation(self) -> None:
        class FakeEngine:
            engine_version = "fake=1"
            settings_digest = "settings-a"

            def __init__(self) -> None:
                self.calls = 0

            def extract(self, path: Path) -> OcrDocument:
                self.calls += 1
                return OcrDocument(
                    source=path.name,
                    source_sha256=sha256_file(path),
                    page_count=1,
                    engine="fake",
                    engine_version=self.engine_version,
                    settings_digest=self.settings_digest,
                    pages=(OcrPage(1, "recognized text " * 20),),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "paper.pdf"
            pdf.write_bytes(b"first PDF content")
            engine = FakeEngine()
            cache = OcrCache(root / "cache")

            first = cache.get_or_create(pdf, engine)
            second = cache.get_or_create(pdf, engine)
            pdf.write_bytes(b"changed PDF content")
            third = cache.get_or_create(pdf, engine)
            engine.settings_digest = "settings-b"
            fourth = cache.get_or_create(pdf, engine)

        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertFalse(third.cache_hit)
        self.assertFalse(fourth.cache_hit)
        self.assertEqual(engine.calls, 3)

    def test_corrupt_cache_is_rebuilt(self) -> None:
        class FakeEngine:
            engine_version = "fake=1"
            settings_digest = "settings-a"

            def __init__(self) -> None:
                self.calls = 0

            def extract(self, path: Path) -> OcrDocument:
                self.calls += 1
                return OcrDocument(
                    path.name,
                    sha256_file(path),
                    1,
                    "fake",
                    self.engine_version,
                    self.settings_digest,
                    (OcrPage(1, "recognized text " * 20),),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "paper.pdf"
            pdf.write_bytes(b"PDF content")
            engine = FakeEngine()
            cache = OcrCache(root / "cache")
            first = cache.get_or_create(pdf, engine)
            paths = list((root / "cache").rglob("*.json"))
            self.assertEqual(len(paths), 1)
            paths[0].write_text("{broken", encoding="utf-8")
            second = cache.get_or_create(pdf, engine)
            json.loads(paths[0].read_text(encoding="utf-8"))

        self.assertFalse(first.cache_hit)
        self.assertFalse(second.cache_hit)
        self.assertEqual(engine.calls, 2)

    def test_cache_write_warning_keeps_in_memory_result(self) -> None:
        class FakeEngine:
            engine_version = "fake=1"
            settings_digest = "settings-a"

            def extract(self, path: Path) -> OcrDocument:
                return OcrDocument(
                    path.name,
                    sha256_file(path),
                    1,
                    "fake",
                    self.engine_version,
                    self.settings_digest,
                    (OcrPage(1, "recognized text " * 20),),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "paper.pdf"
            pdf.write_bytes(b"PDF content")
            cache = OcrCache(root / "cache")
            cache._store = lambda *_: "write failed"  # type: ignore[method-assign]
            result = cache.get_or_create(pdf, FakeEngine())

        self.assertFalse(result.cache_hit)
        self.assertEqual(result.warning, "write failed")
        self.assertEqual(result.document.page_count, 1)

if __name__ == "__main__":
    unittest.main()
