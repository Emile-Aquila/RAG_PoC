"""Search abstracts, OCR top papers, and answer from page-grounded evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .answer_generator import (
        PAPER_ANALYSIS_MODEL,
        PaperEvidence,
        analyze_paper,
        generate_answer,
    )
    from .ocr import DoclingRapidOcrEngine, OcrCache, OcrDocument
    from .rag_core import (
        EMBEDDING_MODEL,
        LLM_MODEL,
        PDF_DIR,
        TOP_K,
        RagError,
        create_ollama_client,
        embed_query,
        ensure_ollama_models,
        open_collection,
        search_all,
    )
except ImportError:  # pragma: no cover - direct script execution
    from answer_generator import (
        PAPER_ANALYSIS_MODEL,
        PaperEvidence,
        analyze_paper,
        generate_answer,
    )
    from ocr import DoclingRapidOcrEngine, OcrCache, OcrDocument
    from rag_core import (
        EMBEDDING_MODEL,
        LLM_MODEL,
        PDF_DIR,
        TOP_K,
        RagError,
        create_ollama_client,
        embed_query,
        ensure_ollama_models,
        open_collection,
        search_all,
    )


def _resolve_pdf_path(source: str, pdf_dir: Path = PDF_DIR) -> Path:
    try:
        root = pdf_dir.resolve(strict=True)
        candidate = (root / source).resolve(strict=True)
    except OSError as exc:
        raise RagError(f"Top-k論文のPDFが見つかりません: {source}") from exc
    if candidate.parent != root or candidate.suffix.casefold() != ".pdf":
        raise RagError(f"不正なPDFパスです: {source}")
    return candidate


def _parse_question() -> str:
    parser = argparse.ArgumentParser(
        description="Abstractのベクトル検索結果を表示し、Ollamaで回答します。"
    )
    parser.add_argument("question", nargs="*", help="検索・質問するテキスト")
    args = parser.parse_args()
    question = " ".join(args.question).strip()
    if question:
        return question

    try:
        return input("質問を入力してください: ").strip()
    except EOFError:
        return ""


def main() -> int:
    question = _parse_question()
    if not question:
        print("エラー: 質問が空です。質問文を入力してください。", file=sys.stderr)
        return 1

    try:
        collection = open_collection()
        client = create_ollama_client()
        ensure_ollama_models(
            client,
            list(dict.fromkeys([EMBEDDING_MODEL, PAPER_ANALYSIS_MODEL, LLM_MODEL])),
        )
        query_embedding = embed_query(client, question)
        results = search_all(collection, query_embedding)
        selected = results[: min(TOP_K, len(results))]

        print("\n=== ベクトル検索結果（cosine類似度） ===")
        for result in results:
            marker = "[SELECTED]" if result.rank <= len(selected) else "          "
            print(
                f"{result.rank:>2}. {marker} 類似度={result.similarity:.4f} "
                f"| {result.title}"
            )

        print("\n=== OCR処理 ===", flush=True)
        cache = OcrCache()
        engine = DoclingRapidOcrEngine()
        ocr_documents: list[tuple[str, OcrDocument]] = []
        failed_sources: list[tuple[str, str]] = []

        for result in selected:
            try:
                pdf_path = _resolve_pdf_path(result.source)

                def announce_miss(title: str = result.title) -> None:
                    print(f"[OCR START] {title}", flush=True)

                cached = cache.get_or_create(
                    pdf_path,
                    engine,
                    on_miss=announce_miss,
                )
                if cached.cache_hit:
                    print(
                        f"[CACHE HIT] {result.title} "
                        f"({cached.document.page_count}ページ)",
                        flush=True,
                    )
                else:
                    print(
                        f"[OCR DONE] {result.title} "
                        f"({cached.document.page_count}ページ)",
                        flush=True,
                    )
                if cached.warning:
                    print(f"警告: {cached.warning}", file=sys.stderr)
                ocr_documents.append((result.title, cached.document))
            except RagError as exc:
                reason = str(exc)
                print(f"[OCR FAILED] {result.title}: {reason}", file=sys.stderr)
                failed_sources.append((result.title, reason))

        if not ocr_documents:
            raise RagError("Top-k論文の全文OCRがすべて失敗しました。")

        print("\n=== OCR全文の論文単位分析 ===", flush=True)
        paper_evidence: list[PaperEvidence] = []
        for title, document in ocr_documents:
            print(f"[LLM ANALYZE] {title}", flush=True)
            paper_evidence.append(analyze_paper(client, question, title, document))

        print("\n=== Ollama回答 ===")
        print(generate_answer(client, question, paper_evidence))
        if failed_sources:
            print("\n=== 回答に利用できなかったTop-k論文 ===")
            for title, reason in failed_sources:
                print(f"- {title}: {reason}")
        return 0
    except RagError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n中断しました。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
