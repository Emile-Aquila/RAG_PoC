# RAG_PoC

`./pdfs` にある論文PDFのAbstractを検索し、ローカルのOllamaで回答する小規模なRAGです。

処理の流れは次のとおりです。

```text
PDF → Abstract抽出 → qwen3-embedding → ChromaDB
    → 全論文の類似度表示 → 上位3論文 → qwen3.5
```

## 簡単な使い方

1. 検索したい論文PDFを `./pdfs` に配置します。
2. 別のターミナルで `ollama serve` を実行します。
3. 次のコマンドで検索DBを作成します。

   ```bash
   uv run python script/build_db.py
   ```

4. 質問文を指定して検索します。

   ```bash
   uv run python script/query_rag.py "質問文"
   ```

PDFを追加・削除した場合は、手順3を再実行してDBを更新してください。


## DB構築

`./pdfs` の全PDFからAbstractを抽出してEmbeddingを生成し、`./db` の
`paper_abstracts` コレクションを再構築します。

```bash
uv run python script/build_db.py
```

再実行時はこのコレクションだけを作り直すため、削除されたPDFがDBに残ったり、
同じ論文が重複登録されたりしません。

## 質問

質問をコマンドライン引数として渡します。

```bash
uv run python script/query_rag.py "将来状態の予測をロボット制御に使う研究を説明してください"
```

引数を省略すると対話的に質問を入力できます。

```bash
uv run python script/query_rag.py
```

検索時にはDB内の全論文についてcosine類似度を表示し、回答生成に使う上位3件へ
`[SELECTED]` を付けます。回答は選択されたAbstractだけを根拠として生成されます。

## テスト

```bash
uv run python -m unittest discover -s tests -v
```
