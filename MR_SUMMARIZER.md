# Mr.Summarizer
### Document Intelligence Platform

> Upload any PDF. Ask anything. Get precise, cited answers — powered by a hybrid GraphRAG pipeline with knowledge graph extraction, multi-query retrieval, ML re-ranking, and Gemini 2.0 Flash generation.

---

## What It Does

Mr.Summarizer turns any PDF into a queryable knowledge base. Upload a legal contract, research paper, technical manual, financial report — anything. The system processes it, builds a knowledge graph of entities and relationships, and lets you ask plain English questions about it.

The key difference from a standard document chatbot: it understands not just what the document says, but how things inside it connect. Ask a relational question like *"How does the liability clause connect to the indemnification schedule?"* and the system traverses the graph to assemble the answer rather than just searching for keywords.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      FRONTEND                           │
│   Next.js 14  ·  Tailwind CSS  ·  shadcn/ui  ·  Vercel │
│   PDF upload  ·  Chat UI  ·  Document library           │
└───────────────────────┬─────────────────────────────────┘
                        │ REST / Server-Sent Events
┌───────────────────────▼─────────────────────────────────┐
│                   BACKEND  (FastAPI · Railway)           │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              INGESTION PIPELINE                  │   │
│  │  PyMuPDF       → extract text, preserve pages   │   │
│  │  spaCy trf     → named entity recognition       │   │
│  │  BART-MNLI     → zero-shot document classification│  │
│  │  Semantic chunker → cosine boundary detection   │   │
│  │  BGE-base-en   → 768-dim local embeddings       │   │
│  │  Supabase      → store chunks + vectors + graph │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              RETRIEVAL PIPELINE                  │   │
│  │  Gemini Flash  → multi-query variant generation  │   │
│  │  hybrid_search stored procedure:                 │   │
│  │    ├─ pgvector ANN  (dense / semantic)           │   │
│  │    ├─ PostgreSQL FTS (sparse / lexical)          │   │
│  │    └─ RRF merge     (Reciprocal Rank Fusion)     │   │
│  │  ms-marco cross-encoder → re-ranking             │   │
│  │  Gemini Flash  → contextual compression          │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              GENERATION                          │   │
│  │  Gemini 2.0 Flash · grounded · streaming SSE    │   │
│  │  Prompt: compressed context + query + history   │   │
│  └──────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                   SUPABASE (PostgreSQL)                  │
│  pgvector · pg_trgm · unaccent · RLS · partitioning     │
│  Tables: documents · chunks · chat_sessions             │
│          chat_history (partitioned) · entities          │
│          relationships                                  │
│  Features: triggers · stored procedures                 │
│            materialized views · GIN indexes             │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Next.js 14, Tailwind CSS, shadcn/ui |
| Backend | FastAPI (Python) |
| LLM | Gemini 2.0 Flash (Google AI free tier) |
| Embeddings | BGE-base-en-v1.5 — local, 768-dim, no API key |
| NER | spaCy `en_core_web_trf` |
| Relation Extraction | REBEL (`Babelscape/rebel-large`) |
| Doc Classification | `facebook/bart-large-mnli` (zero-shot) |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Database | Supabase (PostgreSQL + pgvector) |
| Auth | Supabase Auth |
| File Storage | Supabase Storage |
| Evaluation | RAGAS |
| Backend Deploy | Railway |
| Frontend Deploy | Vercel |

---

## SQL Highlights

The retrieval engine is a PostgreSQL stored procedure that runs two independent strategies and merges them with Reciprocal Rank Fusion:

```sql
-- hybrid_search: pgvector ANN + FTS → RRF merge
WITH
vector_hits AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> query_embedding) AS v_rank
    FROM chunks WHERE user_id = target_user_id
    ORDER BY embedding <=> query_embedding LIMIT top_k * 2
),
fts_hits AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(fts_vector, query, 32) DESC) AS f_rank
    FROM chunks, websearch_to_tsquery('summarizer_fts', query_text) AS query
    WHERE fts_vector @@ query LIMIT top_k * 2
),
rrf AS (
    SELECT
        COALESCE(v.id, f.id) AS chunk_id,
        COALESCE(1.0 / (60 + v.v_rank), 0.0) +
        COALESCE(1.0 / (60 + f.f_rank), 0.0) AS rrf_score
    FROM vector_hits v FULL OUTER JOIN fts_hits f ON v.id = f.id
)
SELECT * FROM rrf ORDER BY rrf_score DESC LIMIT top_k;
```

Other SQL features used: triggers (FTS sync, chunk stats, session auto-title), materialized views with `RANK()`, `SUM() OVER`, `LAG()`, `PERCENT_RANK()` window functions, LATERAL joins, monthly table partitioning, GIN indexes on JSONB and tsvector, and row-level security on every table.

---

## ML Pipeline

**Ingestion (once per upload):**
1. PyMuPDF extracts text page-by-page with page number provenance
2. spaCy `en_core_web_trf` runs NER — people, orgs, locations, dates stored as JSONB
3. BART-large-MNLI classifies document type (legal / academic / financial / technical / general)
4. Semantic chunker splits on cosine similarity drops between adjacent sentences
5. BGE-base-en-v1.5 embeds each chunk locally — 768-dim, no API cost

**Retrieval (per query):**
1. Gemini generates 4 query variants to improve recall across different phrasings
2. Each variant hits `hybrid_search` — pgvector ANN + PostgreSQL FTS, merged with RRF
3. Top pool is re-ranked by `ms-marco-MiniLM-L-6-v2` cross-encoder
4. Gemini strips irrelevant sentences from each top chunk before generation

---

## Build Status

| Phase | Status |
|---|---|
| Database — tables, indexes, triggers, stored procedures, views, RLS | ✅ Done |
| Ingestion pipeline — parse, NER, classify, chunk, embed, store | ✅ Done |
| Retrieval pipeline — hybrid search, multi-query, reranker, compressor | ✅ Done |
| Generation — Gemini 2.0 Flash streaming | ✅ Done |
| API — FastAPI routes for documents and chat | ✅ Done |
| Frontend — Next.js, auth, chat UI, document library | ⬜ In progress |
| Evaluation + deployment — RAGAS, Vercel, Railway | ⬜ Not started |

---

## Environment Variables

```env
GEMINI_API_KEY=

SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=
DATABASE_URL=

NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
```

---

## Project Structure

```
RAG_Chatbot/
├── sql/
│   ├── 001_schema.sql       — tables, indexes, partitions, extensions
│   ├── 002_triggers.sql     — FTS sync, chunk stats, session auto-title
│   ├── 003_procedures.sql   — hybrid_search (RRF), get_session_context
│   ├── 004_views.sql        — document_stats (materialized), analytics
│   └── 005_rls.sql          — row-level security + grants
├── backend/
│   └── app/
│       ├── config.py
│       ├── main.py
│       ├── db/client.py
│       ├── ingestion/
│       │   ├── parser.py
│       │   ├── chunker.py
│       │   ├── ner.py
│       │   ├── classifier.py
│       │   ├── embedder.py
│       │   └── pipeline.py
│       ├── retrieval/
│       │   ├── search.py
│       │   ├── multi_query.py
│       │   ├── reranker.py
│       │   └── compressor.py
│       ├── generation/
│       │   ├── prompts.py
│       │   └── gemini.py
│       └── api/routes/
│           ├── documents.py
│           └── chat.py
├── .env.example
├── MR_SUMMARIZER.md
└── backend/requirements.txt
```
