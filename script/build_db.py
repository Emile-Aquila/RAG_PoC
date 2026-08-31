"""Build the persistent ChromaDB from paper abstracts."""

from __future__ import annotations

import sys

from rag_core import (
    DB_DIR,
    EMBEDDING_MODEL,
    RagError,
    create_ollama_client,
    embed_texts,
    ensure_ollama_models,
    load_paper_abstracts,
    rebuild_collection,
)


def main() -> int:
    try:
        papers = load_paper_abstracts()
        print(f"PDFからAbstractを抽出しました: {len(papers)}件")
        for paper in papers:
            print(f"  - {paper.title} ({len(paper.abstract)}文字)")

        client = create_ollama_client()
        ensure_ollama_models(client, [EMBEDDING_MODEL])
        print(f"Embeddingを生成します: {EMBEDDING_MODEL}")
        embeddings = embed_texts(client, [paper.abstract for paper in papers])

        collection = rebuild_collection(papers, embeddings)
        print(f"ChromaDBを構築しました: {DB_DIR}")
        print(f"コレクション: {collection.name} / 登録件数: {collection.count()}")
        return 0
    except RagError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n中断しました。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
