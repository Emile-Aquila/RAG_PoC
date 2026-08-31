from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from script.rag_core import (
    MIN_ABSTRACT_LENGTH,
    PDF_DIR,
    RagError,
    ensure_ollama_models,
    load_paper_abstracts,
    open_collection,
)


class AbstractExtractionTest(unittest.TestCase):
    def test_extracts_all_current_pdfs(self) -> None:
        pdf_count = len(list(PDF_DIR.glob("*.pdf")))
        papers = load_paper_abstracts()

        self.assertEqual(pdf_count, 8)
        self.assertEqual(len(papers), pdf_count)
        self.assertEqual(len({paper.paper_id for paper in papers}), pdf_count)
        for paper in papers:
            with self.subTest(source=paper.source):
                self.assertGreaterEqual(len(paper.abstract), MIN_ABSTRACT_LENGTH)
                self.assertFalse(paper.abstract.lower().startswith("abstract"))
                self.assertIsNone(
                    re.search(r"(?:^|\s)(?:1|i)[.]?\s+introduction\b", paper.abstract, re.I)
                )
                for noise in ("arXiv:", "Published in", "Figure 1:", "Correspondence to"):
                    self.assertNotIn(noise, paper.abstract)

    def test_worldvla_uses_unlabeled_summary(self) -> None:
        paper = next(
            paper
            for paper in load_paper_abstracts()
            if paper.source.startswith("WorldVLA-")
        )
        self.assertTrue(paper.abstract.startswith("We present WorldVLA"))
        self.assertNotIn("Date: June", paper.abstract)

    def test_missing_database_error_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RagError, "build_db.py"):
                open_collection(Path(directory))

    def test_offline_ollama_error_is_actionable(self) -> None:
        class OfflineClient:
            def list(self) -> None:
                raise ConnectionError("offline")

        with self.assertRaisesRegex(RagError, "ollama serve"):
            ensure_ollama_models(OfflineClient(), ["model:tag"])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
