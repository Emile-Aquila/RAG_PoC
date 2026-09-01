"""Generate page-grounded answers from whole-paper OCR documents."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import ollama

try:
    from .ocr import OcrDocument
    from .rag_core import LLM_MODEL, RagError
except ImportError:  # pragma: no cover - direct script execution
    from ocr import OcrDocument
    from rag_core import LLM_MODEL, RagError


LLM_CONTEXT_SIZE = 65_536
LLM_NUM_PREDICT = 5_000
PAPER_ANALYSIS_MODEL = "qwen3.5:9b-mlx"
_CITATION = re.compile(r"\[(?P<title>[^\[\]]+), p\.(?P<page>\d+)\]")

_EVIDENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "page_number": {"type": "integer"},
        "fact": {"type": "string"},
    },
    "required": ["page_number", "fact"],
    "additionalProperties": False,
}
_PAPER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "summary": {"type": "string"},
        "evidence": {"type": "array", "items": _EVIDENCE_SCHEMA},
    },
    "required": ["answerable", "summary", "evidence"],
    "additionalProperties": False,
}
_FINAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Evidence:
    page_number: int
    fact: str


@dataclass(frozen=True)
class PaperEvidence:
    title: str
    source: str
    answerable: bool
    summary: str
    evidence: tuple[Evidence, ...]


def _response_content(response: Any) -> str:
    message = getattr(response, "message", None)
    content = getattr(message, "content", None)
    if content is None and isinstance(response, dict):
        content = (response.get("message") or {}).get("content")
    return str(content or "").strip()


def _parse_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise TypeError("JSONオブジェクトではありません。")
    return parsed


def _structured_output_instruction(
    schema: dict[str, Any], example: dict[str, Any]
) -> str:
    return (
        "Return exactly one JSON object and no Markdown or explanatory text. "
        "Include every required property, preserve the property types, and do not "
        "add properties that are absent from the schema. Each evidence item must be "
        "an object, never a string.\n\n"
        "JSON Schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "Illustrative shape example (do not copy its placeholder facts):\n"
        f"{json.dumps(example, ensure_ascii=False, indent=2)}"
    )


def _chat_json(
    client: ollama.Client,
    *,
    messages: list[dict[str, str]],
    model: str,
    schema: dict[str, Any],
    example: dict[str, Any],
    num_predict: int,
    validator: Callable[[dict[str, Any]], Any],
    stage: str,
) -> Any:
    output_instruction = _structured_output_instruction(schema, example)
    retry_messages = [
        *messages,
        {"role": "user", "content": output_instruction},
    ]
    last_error: Exception | None = None
    for attempt in range(2):
        content = ""
        try:
            response = client.chat(
                model=model,
                messages=retry_messages,
                format=schema,
                think=False,
                options={
                    "temperature": 0,
                    "num_ctx": LLM_CONTEXT_SIZE,
                    "num_predict": num_predict,
                },
            )
            content = _response_content(response)
            if not content:
                raise ValueError("空の応答です。")
            return validator(_parse_json(content))
        except Exception as exc:  # noqa: BLE001 - retry all client/validation failures
            last_error = exc
            if attempt == 0:
                if content:
                    retry_messages.append(
                        {"role": "assistant", "content": content}
                    )
                retry_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous response failed validation. Correct that "
                            "response without changing or inventing evidence. "
                            f"Validation error: {exc}\n\n{output_instruction}"
                        ),
                    }
                )

    raise RagError(f"{stage}の構造化応答を検証できませんでした: {last_error}")


def _read_evidence(data: Any, allowed_pages: set[int]) -> tuple[Evidence, ...]:
    if not isinstance(data, list):
        raise TypeError("evidenceが配列ではありません。")
    evidence: list[Evidence] = []
    for item in data:
        if not isinstance(item, dict):
            raise TypeError("evidence要素がオブジェクトではありません。")
        page_number = item.get("page_number")
        fact = item.get("fact")
        if isinstance(page_number, bool) or not isinstance(page_number, int):
            raise TypeError("page_numberが整数ではありません。")
        if page_number not in allowed_pages:
            raise ValueError(f"許可されていないページです: {page_number}")
        if not isinstance(fact, str) or not fact.strip():
            raise ValueError("根拠本文が空です。")
        evidence.append(Evidence(page_number=page_number, fact=fact.strip()))
    return tuple(evidence)


def _format_ocr_document(document: OcrDocument) -> str:
    return "\n\n".join(
        f"[Page {page.page_number}]\n{page.markdown}" for page in document.pages
    )


def analyze_paper(
    client: ollama.Client,
    question: str,
    title: str,
    document: OcrDocument,
) -> PaperEvidence:
    allowed_pages = {page.page_number for page in document.pages}
    if not allowed_pages:
        raise RagError(f"OCR文書にページがありません: {document.source}")

    def validate(payload: dict[str, Any]) -> PaperEvidence:
        answerable = payload.get("answerable")
        summary = payload.get("summary")
        if not isinstance(answerable, bool):
            raise TypeError("answerableが真偽値ではありません。")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("summaryが空です。")
        evidence = _read_evidence(payload.get("evidence"), allowed_pages)
        if answerable != bool(evidence):
            raise ValueError("answerableとevidenceの有無が一致しません。")
        return PaperEvidence(
            title=title,
            source=document.source,
            answerable=answerable,
            summary=summary.strip(),
            evidence=evidence,
        )

    return _chat_json(
        client,
        messages=[
            {
                "role": "system",
                "content": (
                    "Analyze the complete OCR text of one research paper in light of "
                    "the user's question. Read all supplied pages and use only this OCR "
                    "text; do not use outside knowledge or the paper abstract. Extract "
                    "question-relevant facts, remove duplicates, and attach the exact "
                    "page_number shown in the OCR text to every fact. If the complete "
                    "paper contains no relevant evidence, set answerable=false and "
                    "evidence=[]."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\nPaper:\n{title}\n\n"
                    f"Complete OCR text:\n{_format_ocr_document(document)}"
                ),
            },
        ],
        model=PAPER_ANALYSIS_MODEL,
        schema=_PAPER_SCHEMA,
        example={
            "answerable": True,
            "summary": "Question-focused summary supported by the complete paper.",
            "evidence": [
                {
                    "page_number": min(allowed_pages),
                    "fact": "A question-relevant fact supported by this exact page.",
                }
            ],
        },
        num_predict=LLM_NUM_PREDICT,
        validator=validate,
        stage=f"論文全文分析 ({title})",
    )


def generate_answer(
    client: ollama.Client,
    question: str,
    papers: Sequence[PaperEvidence],
) -> str:
    if not papers:
        raise RagError("回答生成に利用できるOCR済み論文がありません。")

    allowed_citations = {
        (paper.title, evidence.page_number)
        for paper in papers
        for evidence in paper.evidence
    }

    def validate(payload: dict[str, Any]) -> str:
        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answerが空です。")
        citations = [
            (match.group("title"), int(match.group("page")))
            for match in _CITATION.finditer(answer)
        ]
        invalid = [citation for citation in citations if citation not in allowed_citations]
        if invalid:
            raise ValueError(f"存在しない引用です: {invalid}")
        if allowed_citations and not citations:
            raise ValueError("根拠がある回答にページ引用がありません。")
        if not allowed_citations and citations:
            raise ValueError("根拠がない回答に引用があります。")
        return answer.strip()

    paper_payload = [asdict(paper) for paper in papers]
    if allowed_citations:
        example_title, example_page = min(allowed_citations)
        answer_example = {
            "answer": (
                "A supported statement "
                f"[{example_title}, p.{example_page}]."
            )
        }
    else:
        answer_example = {
            "answer": "The supplied OCR evidence is insufficient to answer the question."
        }
    return _chat_json(
        client,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Answer only from the supplied "
                    "paper evidence; do not use outside knowledge or abstracts. Cite "
                    "supporting claims exactly as [Paper title, p.N]. Use only titles "
                    "and page numbers present in the evidence. If evidence is "
                    "insufficient, say so explicitly. Answer in the same language as "
                    "the user's question."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\nPaper evidence in Top-k order:\n"
                    + json.dumps(paper_payload, ensure_ascii=False)
                ),
            },
        ],
        model=LLM_MODEL,
        schema=_FINAL_SCHEMA,
        example=answer_example,
        num_predict=LLM_NUM_PREDICT,
        validator=validate,
        stage="最終回答生成",
    )
