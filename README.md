# RAG_PoC

`./pdfs` にある論文PDFのAbstractを検索し、上位論文の全文OCRを根拠として
ローカルのOllamaで回答する小規模なRAGです。

処理の流れは次のとおりです。

```text
PDF → Abstract抽出 → qwen3-embedding:8b → ChromaDB
    → 全論文の類似度表示 → 上位2論文
    → Docling + RapidOCRによる全ページOCR
    → 論文ごとの全文分析 → Ollama回答
```

## 簡単な使い方

1. 検索したい論文PDFを `./pdfs` に配置します。
2. 依存関係をインストールします。

   ```bash
   uv sync
   ```

3. 別のターミナルで `ollama serve` を実行します。
4. 次のコマンドで検索DBを作成します。

   ```bash
   uv run python scripts/build_db.py
   ```

5. 質問文を指定して検索します。

   ```bash
   uv run python scripts/query_rag.py "質問文"
   ```

PDFを追加・削除した場合は、手順3を再実行してDBを更新してください。


## DB構築

`./pdfs` の全PDFからAbstractを抽出してEmbeddingを生成し、`./db` の
`paper_abstracts` コレクションを再構築します。

```bash
uv run python scripts/build_db.py
```

再実行時はこのコレクションだけを作り直すため、削除されたPDFがDBに残ったり、
同じ論文が重複登録されたりしません。

## 質問

質問をコマンドライン引数として渡します。

```bash
uv run python scripts/query_rag.py "将来状態の予測をロボット制御に使う研究を説明してください"
```

引数を省略すると対話的に質問を入力できます。

```bash
uv run python scripts/query_rag.py
```

検索時にはDB内の全論文についてcosine類似度を表示し、回答生成に使う上位2件へ
`[SELECTED]` を付けます。その後、上位論文をDoclingのRapidOCRバックエンドで
逐次的に全ページOCRします。各論文のOCR全文は`[Page N]`でページ番号を保持したまま
1回のLLM呼び出しで質問に沿って分析され、その論文の根拠付き要約になります。上位2件の
論文別要約から最終回答を生成します。LLMのコンテキスト上限は65,536トークンです。回答中の根拠は
`[論文タイトル, p.ページ番号]`形式で表示されます。

## OCRキャッシュ

OCR結果は `./ocr_cache` にJSONとして保存されます。同じPDFとOCR設定で再検索した
場合は `[CACHE HIT]` と表示され、Doclingを再実行しません。PDF内容、Doclingまたは
RapidOCRのバージョン、言語や解像度などの設定が変わると、自動的に新しいキャッシュを
生成します。

初回のOCRではDoclingがレイアウト解析モデルなどを取得する場合があるため、ネットワーク
接続が必要で、2回目以降より時間がかかります。上位論文の一部でOCRに失敗した場合は
警告を表示して成功した論文だけで回答し、すべて失敗した場合は回答を生成しません。

## テスト

```bash
uv run python -m unittest discover -s tests -v
```

通常テストはOCRエンジンをモックするためモデルを必要としません。実際の
Docling＋RapidOCRを使う任意の統合テストは次のように実行します。

```bash
RUN_OCR_INTEGRATION=1 uv run python -m unittest \
  tests.test_ocr_pipeline.OcrServiceTest.test_real_docling_rapidocr_and_cache -v
```
