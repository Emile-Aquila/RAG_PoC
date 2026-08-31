"""Abstract-based RAG core utilities."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import chromadb
import ollama
import pymupdf


PROJECT_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_DIR / "pdfs"
DB_DIR = PROJECT_DIR / "db"
COLLECTION_NAME = "paper_abstracts"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "qwen3.5:9b-mlx"
TOP_K = 3
MIN_ABSTRACT_LENGTH = 200
QUERY_INSTRUCTION = (
    "Retrieve research paper abstracts that are relevant to and help answer "
    "the user's question."
)


class RagError(RuntimeError):
    """An expected, user-actionable RAG error."""


@dataclass(frozen=True)
class PaperAbstract:
    paper_id: str
    title: str
    source: str
    abstract: str
    page_count: int


@dataclass(frozen=True)
class SearchResult:
    rank: int
    title: str
    source: str
    abstract: str
    similarity: float


_ABSTRACT_HEADING = re.compile(
    r"(?im)^\s*abstract\s*(?:[—–:-]\s*)?"
)
_SECTION_HEADING = re.compile(
    r"(?im)^\s*(?:"
    r"index\s+terms?|key\s*words?|keywords?"
    r"|(?:1|i)[.\s]*introduction"
    r"|introduction"
    r")\s*(?:[—–:.-]\s*)?"
)
_FALLBACK_START = re.compile(
    r"^(?:we\s+(?:present|introduce|propose)|this\s+(?:paper|work)|in\s+this\s+paper)\b",
    re.IGNORECASE,
)
_FALLBACK_STOP = re.compile(
    r"^(?:date\s*:|code\s*:|correspondence\s*:|keywords?\b|index\s+terms?\b|"
    r"(?:1|i)[.\s]*introduction\b|introduction\b)",
    re.IGNORECASE,
)
_BLOCK_NOISE = re.compile(
    r"^(?:arxiv:|published\s+in\b|figure\s+\d|table\s+\d|"
    r"[∗*†].*(?:contribution|author)|reviewed\s+on\b|https?://)",
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[a-z])", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_labeled_abstract(document: pymupdf.Document) -> str | None:
    collected: list[str] = []
    heading_found = False

    for page_index in range(min(3, len(document))):
        for block in document[page_index].get_text("blocks"):
            raw_text = str(block[4])
            if not heading_found:
                heading = _ABSTRACT_HEADING.search(raw_text)
                if heading is None:
                    continue
                heading_found = True
                raw_text = raw_text[heading.end() :]

            next_section = _SECTION_HEADING.search(raw_text)
            if next_section is not None:
                before_section = _normalize_text(raw_text[: next_section.start()])
                if before_section:
                    collected.append(before_section)
                return _normalize_text(" ".join(collected))

            block_text = _normalize_text(raw_text)
            if not block_text or block_text.isdigit() or _BLOCK_NOISE.match(block_text):
                continue
            collected.append(block_text)

    if not heading_found or not collected:
        return None
    return _normalize_text(" ".join(collected))


def _extract_unlabeled_abstract(page: pymupdf.Page) -> str | None:
    blocks = page.get_text("blocks")
    collected: list[str] = []
    collecting = False

    for block in blocks:
        block_text = _normalize_text(str(block[4]))
        if not block_text:
            continue

        if not collecting:
            if _FALLBACK_START.match(block_text):
                collecting = True
                collected.append(block_text)
            continue

        if _FALLBACK_STOP.match(block_text):
            break
        collected.append(block_text)

    if not collected:
        return None
    return _normalize_text(" ".join(collected))


def extract_abstract(pdf_path: Path) -> tuple[str, int]:
    """Extract and normalize one abstract, raising on ambiguous/invalid input."""
    try:
        with pymupdf.open(pdf_path) as document:
            page_count = len(document)
            if page_count == 0:
                raise RagError(f"PDFにページがありません: {pdf_path.name}")

            abstract = _extract_labeled_abstract(document)
            if abstract is None:
                abstract = _extract_unlabeled_abstract(document[0])
    except RagError:
        raise
    except Exception as exc:
        raise RagError(f"PDFを読み取れません: {pdf_path.name} ({exc})") from exc

    if abstract is None:
        raise RagError(f"Abstractを特定できません: {pdf_path.name}")
    if len(abstract) < MIN_ABSTRACT_LENGTH:
        raise RagError(
            f"抽出したAbstractが短すぎます: {pdf_path.name} "
            f"({len(abstract)}文字)"
        )
    return abstract, page_count


def load_paper_abstracts(pdf_dir: Path = PDF_DIR) -> list[PaperAbstract]:
    if not pdf_dir.is_dir():
        raise RagError(f"PDFディレクトリがありません: {pdf_dir}")

    pdf_paths = sorted(pdf_dir.glob("*.pdf"), key=lambda path: path.name.casefold())
    if not pdf_paths:
        raise RagError(f"PDFが見つかりません: {pdf_dir}")

    papers: list[PaperAbstract] = []
    for pdf_path in pdf_paths:
        abstract, page_count = extract_abstract(pdf_path)
        source = pdf_path.name
        paper_id = hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]
        papers.append(
            PaperAbstract(
                paper_id=paper_id,
                title=pdf_path.stem,
                source=source,
                abstract=abstract,
                page_count=page_count,
            )
        )
    return papers


def create_ollama_client() -> ollama.Client:
    return ollama.Client(host="http://127.0.0.1:11434")


def _model_names(response: Any) -> set[str]:
    models = getattr(response, "models", None)
    if models is None and isinstance(response, dict):
        models = response.get("models", [])

    names: set[str] = set()
    for model in models or []:
        name = getattr(model, "model", None)
        if name is None and isinstance(model, dict):
            name = model.get("model") or model.get("name")
        if name:
            names.add(str(name))
    return names


def ensure_ollama_models(client: ollama.Client, required: Sequence[str]) -> None:
    try:
        names = _model_names(client.list())
    except Exception as exc:
        raise RagError(
            "Ollamaサーバーへ接続できません。別ターミナルで "
            "`ollama serve` を起動してください。"
        ) from exc

    missing = [model for model in required if model not in names]
    if missing:
        commands = " / ".join(f"ollama pull {model}" for model in missing)
        raise RagError(
            f"必要なOllamaモデルがありません: {', '.join(missing)}。"
            f"次を実行してください: {commands}"
        )


def embed_texts(client: ollama.Client, texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        raise RagError("Embedding対象のテキストがありません。")
    try:
        response = client.embed(model=EMBEDDING_MODEL, input=list(texts))
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None and isinstance(response, dict):
            embeddings = response.get("embeddings")
    except Exception as exc:
        raise RagError(
            f"{EMBEDDING_MODEL} によるEmbedding生成に失敗しました: {exc}"
        ) from exc

    if embeddings is None or len(embeddings) != len(texts):
        raise RagError("Ollamaから期待した件数のEmbeddingが返されませんでした。")
    return [list(vector) for vector in embeddings]


def rebuild_collection(
    papers: Sequence[PaperAbstract],
    embeddings: Sequence[Sequence[float]],
    db_dir: Path = DB_DIR,
) -> chromadb.Collection:
    if len(papers) != len(embeddings):
        raise RagError("論文数とEmbedding数が一致しません。")
    if not papers:
        raise RagError("DBへ保存する論文がありません。")

    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))
    try:
        client.get_collection(COLLECTION_NAME, embedding_function=None)
    except Exception as exc:
        if exc.__class__.__name__ not in {"NotFoundError", "InvalidCollectionException"}:
            raise
    else:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={
            "embedding_model": EMBEDDING_MODEL,
            "content": "paper_abstracts",
        },
    )
    collection.add(
        ids=[paper.paper_id for paper in papers],
        embeddings=[list(vector) for vector in embeddings],
        documents=[paper.abstract for paper in papers],
        metadatas=[
            {
                "title": paper.title,
                "source": paper.source,
                "embedding_model": EMBEDDING_MODEL,
                "abstract_char_count": len(paper.abstract),
                "page_count": paper.page_count,
            }
            for paper in papers
        ],
    )
    return collection


def open_collection(db_dir: Path = DB_DIR) -> chromadb.Collection:
    if not db_dir.is_dir() or not (db_dir / "chroma.sqlite3").exists():
        raise RagError(
            f"ChromaDBがありません: {db_dir}。先に `uv run python script/build_db.py` "
            "を実行してください。"
        )

    try:
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    except Exception as exc:
        raise RagError(
            f"コレクション `{COLLECTION_NAME}` がありません。"
            "`uv run python script/build_db.py` で再構築してください。"
        ) from exc

    metadata = collection.metadata or {}
    stored_model = metadata.get("embedding_model")
    if stored_model != EMBEDDING_MODEL:
        raise RagError(
            f"DBのEmbeddingモデルが一致しません: {stored_model!r}。"
            "`uv run python script/build_db.py` で再構築してください。"
        )
    if collection.count() == 0:
        raise RagError("ChromaDBのコレクションが空です。DBを再構築してください。")
    return collection


def embed_query(client: ollama.Client, question: str) -> list[float]:
    query = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {question}"
    return embed_texts(client, [query])[0]


def search_all(
    collection: chromadb.Collection, query_embedding: Sequence[float]
) -> list[SearchResult]:
    count = collection.count()
    raw = collection.query(
        query_embeddings=[list(query_embedding)],
        n_results=count,
        include=["documents", "metadatas", "distances"],
    )

    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    if not (len(documents) == len(metadatas) == len(distances) == count):
        raise RagError("ChromaDBから不完全な検索結果が返されました。")

    results: list[SearchResult] = []
    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances, strict=True), start=1
    ):
        if document is None or metadata is None:
            raise RagError("検索結果に文書またはメタデータがありません。")
        results.append(
            SearchResult(
                rank=rank,
                title=str(metadata["title"]),
                source=str(metadata["source"]),
                abstract=str(document),
                similarity=1.0 - float(distance),
            )
        )
    return results


def answer_question(
    client: ollama.Client, question: str, selected: Sequence[SearchResult]
) -> str:
    context = "\n\n".join(
        f"[{result.title}]\n{result.abstract}" for result in selected
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant. Answer only from the supplied paper "
                "abstracts. If the abstracts do not contain enough information, say so "
                "explicitly and do not use outside knowledge. Cite supporting paper titles "
                "in square brackets. Answer in the same language as the user's question."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nRetrieved abstracts:\n{context}",
        },
    ]
    try:
        response = client.chat(
            model=LLM_MODEL,
            messages=messages,
            think=False,
            options={"temperature": 0.2},
        )
        message = getattr(response, "message", None)
        content = getattr(message, "content", None)
        if content is None and isinstance(response, dict):
            content = (response.get("message") or {}).get("content")
    except Exception as exc:
        raise RagError(f"{LLM_MODEL} による回答生成に失敗しました: {exc}") from exc

    answer = str(content or "").strip()
    if not answer:
        raise RagError("Ollamaから空の回答が返されました。")
    return answer
