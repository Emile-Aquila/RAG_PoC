"""Search all paper abstracts and answer with the top matches."""

from __future__ import annotations

import argparse
import sys

from rag_core import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
    RagError,
    answer_question,
    create_ollama_client,
    embed_query,
    ensure_ollama_models,
    open_collection,
    search_all,
)


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
        ensure_ollama_models(client, [EMBEDDING_MODEL, LLM_MODEL])
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

        print("\n=== Ollama回答 ===")
        print(answer_question(client, question, selected))
        return 0
    except RagError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n中断しました。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
