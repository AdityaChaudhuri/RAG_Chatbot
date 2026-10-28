# Mr. Summarizer

Ask questions about financial news and get short answers that cite their sources.

Under the hood it is a **hybrid retrieval-augmented generation (RAG)** system. Three different retrievers search the news, their results are merged with **Reciprocal Rank Fusion**, a **cross-encoder** reranks the shortlist, and an LLM writes an answer using only the retrieved text. One of the three retrievers walks a **knowledge graph** of companies, people, places and events built from the articles.

Data: about 200k Reuters financial news articles from 2018 ([BEE-spoke-data/financial-news-articles-filtered](https://huggingface.co/datasets/BEE-spoke-data/financial-news-articles-filtered)). A few thousand is enough for a demo.

## How it works

```
                     question
                        |
     +------------------+------------------+
     |                  |                  |
 keyword search     vector search      graph search
 (Postgres FTS)     (pgvector)         (knowledge graph)
 exact words,       meaning,           entities in the question
 tickers, names     paraphrases        + their strongest neighbours
     |                  |                  |
     +--------- Reciprocal Rank Fusion ----+
                        |
               top 20 candidates
                        |
           cross-encoder rerank -> top 6
                        |
         LLM (Claude Haiku or local Ollama)
                        |
          answer with [1][2] citations, streamed
```

### Ingestion (`mr_summarizer/ingest.py`)

1. **Clean** each article: strip Reuters page boilerplate such as "Updated 5 minutes ago" (`text.clean`).
2. **Chunk** into ~180-word windows of whole sentences with ~40 words of overlap, so a fact that crosses a boundary still appears intact in one chunk (`text.chunk`).
3. **Embed** every chunk with `BAAI/bge-small-en-v1.5` (384 dimensions, runs on CPU) and store it in a pgvector column with an HNSW index.
4. **Full-text index**: Postgres generates a `tsvector` column for each chunk automatically, indexed with GIN.
5. **Build the knowledge graph**: spaCy NER finds entities (ORG, PERSON, GPE, PRODUCT, EVENT) in each chunk. Names are normalised ("the Federal Reserve's" becomes `federal reserve`) so the same entity becomes one node. Two entities get an edge when they appear in the same chunk, and the edge weight counts how many chunks they share.

### Retrieval (`mr_summarizer/retrieval.py`)

| Retriever | Good at | Weak at |
|---|---|---|
| Keyword (Postgres full-text) | exact names, tickers, rare terms | synonyms, paraphrases |
| Vector (pgvector cosine) | meaning: "profit warning" finds "cut its earnings outlook" | exact identifiers, numbers |
| Graph (entity co-occurrence) | pulling in connected context: a question about a company also surfaces its CEO, regulators, competitors | questions with no named entities |

Each returns its top 30 chunk ids. **RRF** (`fusion.py`) merges them using rank positions only, because the raw scores are on incompatible scales:

```
rrf(chunk) = sum over retrievers of 1 / (60 + rank)
```

A chunk ranked well by two or three retrievers rises to the top. The top 20 then go through a **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which reads the question and passage together. It is much more accurate than comparing embeddings, but too slow to run over the whole corpus, which is why it only reranks the shortlist.

### Generation (`mr_summarizer/llm.py`)

The top 6 chunks are numbered and sent to the LLM with instructions to use only those sources, cite every factual sentence, and say so when the sources do not answer the question. The answer streams to the browser over Server-Sent Events, and the sources are sent first so the page can show them immediately.

## Design decisions

- **Why three retrievers instead of one?** Each fails in different ways (table above). Fusing them raises recall without having to tune score weights.
- **Why RRF instead of a weighted score?** Keyword scores, cosine distances and graph counts are not comparable, and weights would need retuning for every dataset. RRF only needs ranks and has one robust constant.
- **Why OR the full-text terms?** Postgres `plainto_tsquery` ANDs every word, so a full question like "What did Goldman Sachs say about rates?" matches almost nothing. Swapping `&` for `|` and ranking with `ts_rank_cd` rewards chunks that contain more of the terms.
- **Why co-occurrence edges instead of LLM-extracted relations?** It is free, fast, deterministic and easy to explain. The trade-off is that edges say two entities are related, not *how*. Typed relations (for example `CEO_OF`, `ACQUIRED`) extracted by an LLM are the main planned upgrade.
- **Why match query entities by string instead of running NER on the question?** spaCy is unreliable on short, lowercase questions. Matching known graph keys as whole phrases, longest first, is more robust.
- **Why Postgres for everything?** Vectors, full text and the graph live in one database, so retrieval is a handful of SQL queries with no extra services to run.

## Run it

You need Python 3.10 to 3.13 and a Postgres database with pgvector. Pick one option for the database.

**Option A: free cloud database (Neon), easiest on Windows**

1. Sign up at [neon.com](https://neon.com), create a project (choose the region closest to you), and copy the connection string. It looks like `postgresql://user:pass@ep-xxx.region.aws.neon.tech/neondb?sslmode=require`.
2. You will paste it as `DATABASE_URL` in `.env` in step 2 below, and step 3 creates the tables.

**Option B: local database with Docker**

Run `docker compose up -d`. The schema is created automatically on first start.

**Then:**

```bash
# 1. Python environment
python -m venv .venv
.venv\Scripts\activate              # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Settings
copy .env.example .env               # Mac/Linux: cp .env.example .env
# edit .env: DATABASE_URL, plus your LLM details (below)

# 3. Tables (Neon only; Docker does this for you)
python -m mr_summarizer.init_db

# 4. Load and index news. The first run downloads the models (about 1 GB).
python -m mr_summarizer.ingest --limit 500     # try small first, then --limit 3000

# 5. Start the app, then open http://localhost:8000
uvicorn mr_summarizer.api:app --reload
```

### Choosing an LLM

- **Claude (default):** create an API key in the [Claude Console](https://platform.claude.com) and set `ANTHROPIC_API_KEY` in `.env`. The default model is Claude Haiku 5.5, which costs a fraction of a cent per question.
- **Ollama (free, local):** install [Ollama](https://ollama.com), run `ollama pull llama3.2`, and set `LLM_PROVIDER=ollama`.

## API

| Endpoint | What it returns |
|---|---|
| `GET /` | the web page |
| `GET /ask?q=...` | the answer as Server-Sent Events: `sources`, then `token` events, then `done` |
| `GET /search?q=...` | retrieval only: the final chunks with each retriever's rank, RRF score and rerank score |
| `GET /graph/{name}` | an entity and its most strongly connected neighbours |

`/search` is the quickest way to see the hybrid retrieval working: each hit lists which retrievers found it and at what rank.

## Tests

```bash
pytest
```

The tests cover cleaning, chunking, entity normalisation, graph edge counting, query entity matching, RRF and prompt construction. They run without a database or any models.

## Project layout

```
mr_summarizer/
  config.py      settings from .env
  db.py          Postgres connection (pgvector-aware)
  text.py        cleaning and sentence-aware chunking
  graph.py       entity normalisation, co-occurrence edges, query entity matching
  models.py      embedding model, cross-encoder, spaCy (lazy-loaded)
  init_db.py     creates the tables in a cloud database
  ingest.py      load -> clean -> chunk -> embed -> NER -> graph (batched writes)
  fusion.py      Reciprocal Rank Fusion
  retrieval.py   keyword, vector and graph retrievers + fusion + rerank
  llm.py         prompt and streaming from Claude or Ollama
  api.py         FastAPI app
sql/schema.sql   tables and indexes, including the graph tables
static/          single-page web UI
tests/           unit tests
```

## Roadmap

- [ ] RAGAS evaluation of faithfulness and answer relevance on a set of hand-written questions
- [ ] Retrieval ablation: recall@k for each retriever alone versus fused, to measure what the graph adds
- [ ] Typed relation extraction with an LLM so edges carry meaning (`CEO_OF`, `ACQUIRED`, `REGULATES`)
- [ ] Article dates, so answers can reason about time

## History

The first version of this repo (May 2026) was a PDF question-answering design built on Supabase and Gemini. It is preserved in the git history. Version 2 refocuses on financial news and adds the knowledge graph.
