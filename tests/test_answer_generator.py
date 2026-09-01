from __future__ import annotations

import json
import unittest
from typing import cast

from scripts.answer_generator import (
    LLM_NUM_PREDICT,
    PAPER_ANALYSIS_MODEL,
    Evidence,
    PaperEvidence,
    analyze_paper,
    generate_answer,
)
from scripts.ocr import OcrDocument, OcrPage
from scripts.rag_core import RagError


class FakeClient:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected LLM call")
        payload = self.responses.pop(0)
        return {"message": {"content": json.dumps(payload, ensure_ascii=False)}}


def _document(*pages: str) -> OcrDocument:
    return OcrDocument(
        source="paper.pdf",
        source_sha256="hash",
        page_count=len(pages),
        engine="engine",
        engine_version="version",
        settings_digest="settings",
        pages=tuple(
            OcrPage(page_number=index, markdown=text)
            for index, text in enumerate(pages, start=1)
        ),
    )


class WholePaperAnswerTest(unittest.TestCase):
    def test_entire_paper_is_analyzed_in_one_call(self) -> None:
        document = _document("A" * 13_000, "B" * 13_000)
        client = FakeClient(
            [
                {"answerable": False, "summary": "none", "evidence": []},
            ]
        )

        paper = analyze_paper(client, "question", "Paper", document)  # type: ignore[arg-type]

        self.assertEqual(len(client.calls), 1)
        self.assertFalse(paper.answerable)
        self.assertEqual(paper.evidence, ())
        call = client.calls[0]
        self.assertEqual(call["model"], PAPER_ANALYSIS_MODEL)
        self.assertFalse(call["think"])
        self.assertEqual(call["options"]["num_ctx"], 65536)  # type: ignore[index]
        self.assertEqual(  # type: ignore[index]
            call["options"]["num_predict"], LLM_NUM_PREDICT
        )
        messages = cast(list[dict[str, str]], call["messages"])
        paper_prompt = messages[1]["content"]
        self.assertIn("[Page 1]\n" + "A" * 13_000, paper_prompt)
        self.assertIn("[Page 2]\n" + "B" * 13_000, paper_prompt)
        self.assertLess(paper_prompt.index("[Page 1]"), paper_prompt.index("[Page 2]"))

    def test_paper_and_final_answer_keep_page_citation(self) -> None:
        analysis_client = FakeClient(
            [
                {
                    "answerable": True,
                    "summary": "Future prediction supports control.",
                    "evidence": [
                        {"page_number": 1, "fact": "It predicts future states."}
                    ],
                },
            ]
        )
        paper = analyze_paper(
            analysis_client,
            "How is prediction used?",
            "Paper A",
            _document("recognized text " * 30),
        )  # type: ignore[arg-type]
        final_client = FakeClient(
            [{"answer": "Prediction supports control [Paper A, p.1]."}]
        )

        answer = generate_answer(
            final_client,
            "How is prediction used?",
            [paper],
        )  # type: ignore[arg-type]

        self.assertEqual(answer, "Prediction supports control [Paper A, p.1].")
        self.assertEqual(paper.evidence[0].page_number, 1)
        self.assertEqual(
            final_client.calls[0]["options"]["num_predict"],  # type: ignore[index]
            LLM_NUM_PREDICT,
        )

    def test_invalid_paper_page_retries_then_fails(self) -> None:
        invalid = {
            "answerable": True,
            "summary": "invalid",
            "evidence": [{"page_number": 99, "fact": "not grounded"}],
        }
        client = FakeClient([invalid, invalid])

        with self.assertRaisesRegex(RagError, "構造化応答"):
            analyze_paper(
                client,
                "question",
                "Paper",
                _document("recognized text " * 30),
            )  # type: ignore[arg-type]

        self.assertEqual(len(client.calls), 2)
        first_messages = cast(list[dict[str, str]], client.calls[0]["messages"])
        self.assertIn("JSON Schema", first_messages[-1]["content"])
        self.assertIn('"summary"', first_messages[-1]["content"])
        self.assertIn('"page_number"', first_messages[-1]["content"])

        retry_messages = cast(list[dict[str, str]], client.calls[1]["messages"])
        self.assertEqual(retry_messages[-2]["role"], "assistant")
        self.assertIn('"page_number": 99', retry_messages[-2]["content"])
        self.assertIn("Validation error", retry_messages[-1]["content"])
        self.assertIn("許可されていないページです: 99", retry_messages[-1]["content"])
        self.assertIn("JSON Schema", retry_messages[-1]["content"])

    def test_final_answer_rejects_unknown_citation(self) -> None:
        paper = PaperEvidence(
            title="Paper A",
            source="paper.pdf",
            answerable=True,
            summary="summary",
            evidence=(Evidence(1, "fact"),),
        )
        client = FakeClient(
            [
                {"answer": "Unsupported [Paper A, p.9]."},
                {"answer": "Still unsupported [Paper A, p.9]."},
            ]
        )

        with self.assertRaisesRegex(RagError, "最終回答生成"):
            generate_answer(client, "question", [paper])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
