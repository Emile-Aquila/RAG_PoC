from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import query_rag
from scripts.answer_generator import Evidence, PaperEvidence
from scripts.ocr import OcrCacheResult, OcrDocument, OcrPage
from scripts.rag_core import TOP_K, RagError, SearchResult


def _ocr_document(source: str) -> OcrDocument:
    return OcrDocument(
        source=source,
        source_sha256="hash",
        page_count=1,
        engine="fake",
        engine_version="fake=1",
        settings_digest="settings",
        pages=(OcrPage(1, "recognized text " * 20),),
    )


class QueryPipelineTest(unittest.TestCase):
    def test_top_k_is_two(self) -> None:
        self.assertEqual(TOP_K, 2)

    def test_rejects_pdf_path_outside_pdf_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_dir = root / "pdfs"
            pdf_dir.mkdir()
            outside = root / "outside.pdf"
            outside.write_bytes(b"pdf")

            with self.assertRaisesRegex(RagError, "不正なPDFパス"):
                query_rag._resolve_pdf_path("../outside.pdf", pdf_dir)

    def test_lists_top_k_then_continues_after_one_ocr_failure(self) -> None:
        results = [
            SearchResult(1, "Paper A", "a.pdf", "abstract a", 0.9),
            SearchResult(2, "Paper B", "b.pdf", "abstract b", 0.8),
        ]
        document = _ocr_document("a.pdf")

        class FakeCache:
            def get_or_create(self, path: Path, engine: object, on_miss: object) -> OcrCacheResult:
                if path.name == "b.pdf":
                    raise RagError("OCR failed")
                on_miss()  # type: ignore[operator]
                return OcrCacheResult(document=document, cache_hit=False)

        paper = PaperEvidence(
            title="Paper A",
            source="a.pdf",
            answerable=True,
            summary="summary",
            evidence=(Evidence(1, "fact"),),
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.object(query_rag, "_parse_question", return_value="question"),
            patch.object(query_rag, "open_collection", return_value=object()),
            patch.object(query_rag, "create_ollama_client", return_value=object()),
            patch.object(query_rag, "ensure_ollama_models"),
            patch.object(query_rag, "embed_query", return_value=[0.1]),
            patch.object(query_rag, "search_all", return_value=results),
            patch.object(query_rag, "_resolve_pdf_path", side_effect=lambda source: Path(source)),
            patch.object(query_rag, "OcrCache", return_value=FakeCache()),
            patch.object(query_rag, "DoclingRapidOcrEngine", return_value=object()),
            patch.object(query_rag, "analyze_paper", return_value=paper) as analyze,
            patch.object(query_rag, "generate_answer", return_value="final answer") as generate,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = query_rag.main()

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertLess(output.index("ベクトル検索結果"), output.index("[OCR START]"))
        self.assertIn("[OCR DONE] Paper A", output)
        self.assertIn("回答に利用できなかったTop-k論文", output)
        self.assertIn("final answer", output)
        self.assertIn("[OCR FAILED] Paper B", stderr.getvalue())
        analyze.assert_called_once()
        generate.assert_called_once()

    def test_all_ocr_failures_skip_llm_answer(self) -> None:
        results = [SearchResult(1, "Paper A", "a.pdf", "abstract", 0.9)]

        class FailingCache:
            def get_or_create(self, *_: object, **__: object) -> OcrCacheResult:
                raise RagError("OCR failed")

        stderr = io.StringIO()
        with (
            patch.object(query_rag, "_parse_question", return_value="question"),
            patch.object(query_rag, "open_collection", return_value=object()),
            patch.object(query_rag, "create_ollama_client", return_value=object()),
            patch.object(query_rag, "ensure_ollama_models"),
            patch.object(query_rag, "embed_query", return_value=[0.1]),
            patch.object(query_rag, "search_all", return_value=results),
            patch.object(query_rag, "_resolve_pdf_path", return_value=Path("a.pdf")),
            patch.object(query_rag, "OcrCache", return_value=FailingCache()),
            patch.object(query_rag, "DoclingRapidOcrEngine", return_value=object()),
            patch.object(query_rag, "generate_answer") as generate,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = query_rag.main()

        self.assertEqual(exit_code, 1)
        self.assertIn("すべて失敗", stderr.getvalue())
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
