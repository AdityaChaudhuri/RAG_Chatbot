# Mr.Summarizer
### Intelligent Document Intelligence Platform

> Upload any PDF. Ask anything. Get precise, cited answers — powered by a production-grade RAG pipeline with hybrid retrieval, ML re-ranking, and a cloud-native PostgreSQL backbone.

---

## Overview

Mr.Summarizer is a full-stack Retrieval-Augmented Generation (RAG) web application that transforms static PDF documents into interactive, queryable knowledge bases. Users upload documents, ask natural language questions, and receive accurate, grounded answers synthesised by Claude. Every response is traceable back to the source chunks that generated it.

The project is deliberately engineered to demonstrate two technical pillars simultaneously:

- **Advanced SQL** — hybrid retrieval, window functions, stored procedures, triggers, materialized views, RLS, and partitioned tables inside PostgreSQL
- **Production ML** — semantic chunking, NER extraction, document classification, multi-query retrieval, cross-encoder re-ranking, and a RAGAS evaluation harness

---

## Project Goals

| Goal | Implementation |
|---|---|
| Answer questions from PDF content | RAG pipeline with pgvector similarity search |
| Summarise entire documents | Full-document context compression + Claude |
| Save chat history per user | Partitioned `chat_history` table with RLS |
| Multi-tenant document libraries | Row-Level Security — users see only their data |
| Demonstrate SQL depth | Hybrid search stored proc, triggers, window functions, CTEs |
| Demonstrate ML depth | Chunking, NER, classification, re-ranking, RAGAS eval |
| Deployable to the internet | Vercel (frontend) + Railway (backend) + Supabase (cloud DB) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      FRONTEND                           │
│   Next.js 14 (App Router)  ·  Tailwind CSS  ·  Vercel  │
│   PDF upload  ·  Chat UI  ·  Conversation history       │
└───────────────────────┬─────────────────────────────────┘
                        │ REST / Server-Sent Events
┌───────────────────────▼─────────────────────────────────┐
│                      BACKEND                            │
│              FastAPI (Python)  ·  Railway               │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              INGESTION PIPELINE                  │   │
│  │  1. PDF parse         (PyMuPDF)                  │   │
│  │  2. NER extraction    (spaCy)                    │   │
│  │  3. Doc classification (scikit-learn / HF)       │   │
│  │  4. Semantic chunking  (custom similarity model) │   │
│  │  5. Embedding          (Voyage AI)               │   │
│  │  6. Store chunks + vectors + metadata            │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              RETRIEVAL PIPELINE                  │   │
│  │  1. Multi-query generation  (Claude)             │   │
│  │  2. Hybrid search stored procedure               │   │
│  │     ├─ pgvector ANN search  (dense)              │   │
│  │     ├─ PostgreSQL FTS       (sparse / BM25)      │   │
│  │     └─ RRF merge            (Reciprocal Rank Fusion)│
│  │  3. Cross-encoder re-ranking (sentence-transformers)│
│  │  4. Contextual compression   (LangChain)         │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              GENERATION                          │   │
│  │  Claude claude-sonnet-4-6 via Anthropic SDK      │   │
│  │  Prompt: compressed context + query + history    │   │
│  │  Response streaming via SSE                      │   │
│  └──────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                   SUPABASE (PostgreSQL)                  │
│                                                         │
│  Tables:  documents, chunks, chat_sessions,             │
│           chat_history (partitioned), users             │
│  Extensions: pgvector, pg_trgm, unaccent               │
│  Features: RLS, triggers, stored procedures,            │
│            materialized views, GIN indexes              │
└─────────────────────────────────────────────────────────┘
```

---

## Database Schema (SQL Showcase)

### Extensions & Setup

```sql
CREATE EXTENSION IF NOT EXISTS vector;        -- pgvector
CREATE EXTENSION IF NOT EXISTS pg_trgm;       -- fuzzy text matching
CREATE EXTENSION IF NOT EXISTS unaccent;      -- normalise text for FTS

-- Custom text search configuration
CREATE TEXT SEARCH CONFIGURATION summarizer_fts (COPY = english);
ALTER TEXT SEARCH CONFIGURATION summarizer_fts
  ALTER MAPPING FOR hword, compound_hword, hword_part, word, asciiword
  WITH unaccent, english_stem;
```

### Core Tables

```sql
-- Users (managed by Supabase Auth)
CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL,
    file_url        TEXT NOT NULL,
    doc_type        TEXT,                      -- ML classification output
    entity_tags     JSONB DEFAULT '{}',        -- NER output: {people, orgs, dates}
    chunk_count     INT DEFAULT 0,
    avg_chunk_len   FLOAT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Document chunks with dual-index for hybrid search
CREATE TABLE chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    content         TEXT NOT NULL,
    embedding       VECTOR(1024),              -- Voyage AI dimensions
    fts_vector      TSVECTOR,                  -- Full-text search index
    metadata        JSONB DEFAULT '{}',        -- page_num, section, chunk_index
    token_count     INT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Optimised indexes
CREATE INDEX idx_chunks_embedding ON chunks
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX idx_chunks_fts ON chunks
    USING GIN (fts_vector);

CREATE INDEX idx_chunks_metadata ON chunks
    USING GIN (metadata jsonb_path_ops);

CREATE INDEX idx_chunks_document_id ON chunks (document_id);

-- Chat sessions
CREATE TABLE chat_sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
    title       TEXT,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

-- Chat history — PARTITIONED BY MONTH for scalability
CREATE TABLE chat_history (
    id              UUID NOT NULL DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    source_chunks   JSONB,                     -- which chunks were cited
    retrieval_score FLOAT,                     -- top chunk relevance score
    latency_ms      INT,
    created_at      TIMESTAMPTZ DEFAULT now()
) PARTITION BY RANGE (created_at);

-- Create monthly partitions
CREATE TABLE chat_history_2026_01 PARTITION OF chat_history
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE chat_history_2026_02 PARTITION OF chat_history
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
-- (auto-created monthly via cron or migration)
```

### Triggers

```sql
-- Auto-populate fts_vector on chunk insert/update
CREATE OR REPLACE FUNCTION update_chunk_fts()
RETURNS TRIGGER AS $$
BEGIN
    NEW.fts_vector := to_tsvector('summarizer_fts', NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_chunk_fts
    BEFORE INSERT OR UPDATE OF content ON chunks
    FOR EACH ROW EXECUTE FUNCTION update_chunk_fts();

-- Auto-increment document chunk_count
CREATE OR REPLACE FUNCTION update_document_chunk_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE documents
    SET
        chunk_count   = chunk_count + 1,
        avg_chunk_len = (
            SELECT AVG(token_count) FROM chunks
            WHERE document_id = NEW.document_id
        )
    WHERE id = NEW.document_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_document_chunk_stats
    AFTER INSERT ON chunks
    FOR EACH ROW EXECUTE FUNCTION update_document_chunk_stats();

-- Auto-update session updated_at on new message
CREATE OR REPLACE FUNCTION touch_session()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE chat_sessions SET updated_at = now()
    WHERE id = NEW.session_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_touch_session
    AFTER INSERT ON chat_history
    FOR EACH ROW EXECUTE FUNCTION touch_session();
```

### Hybrid Search Stored Procedure

```sql
-- Core retrieval: pgvector ANN + FTS merged via Reciprocal Rank Fusion
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding  VECTOR(1024),
    query_text       TEXT,
    target_user_id   UUID,
    target_doc_id    UUID DEFAULT NULL,
    top_k            INT  DEFAULT 20,
    rrf_k            INT  DEFAULT 60
)
RETURNS TABLE (
    chunk_id    UUID,
    content     TEXT,
    metadata    JSONB,
    rrf_score   FLOAT
)
LANGUAGE SQL STABLE AS $$
    WITH
    -- Dense retrieval: cosine similarity via pgvector
    vector_results AS (
        SELECT
            id,
            content,
            metadata,
            ROW_NUMBER() OVER (ORDER BY embedding <=> query_embedding) AS vector_rank
        FROM chunks
        WHERE user_id = target_user_id
          AND (target_doc_id IS NULL OR document_id = target_doc_id)
        ORDER BY embedding <=> query_embedding
        LIMIT top_k * 2
    ),
    -- Sparse retrieval: PostgreSQL full-text search
    fts_results AS (
        SELECT
            id,
            content,
            metadata,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank_cd(fts_vector, query, 32) DESC
            ) AS fts_rank
        FROM chunks,
             to_tsquery('summarizer_fts', query_text) AS query
        WHERE user_id = target_user_id
          AND (target_doc_id IS NULL OR document_id = target_doc_id)
          AND fts_vector @@ query
        ORDER BY ts_rank_cd(fts_vector, query, 32) DESC
        LIMIT top_k * 2
    ),
    -- Reciprocal Rank Fusion merge
    rrf_merged AS (
        SELECT
            COALESCE(v.id, f.id)             AS chunk_id,
            COALESCE(v.content, f.content)   AS content,
            COALESCE(v.metadata, f.metadata) AS metadata,
            COALESCE(1.0 / (rrf_k + v.vector_rank), 0) +
            COALESCE(1.0 / (rrf_k + f.fts_rank),    0) AS rrf_score
        FROM vector_results v
        FULL OUTER JOIN fts_results f ON v.id = f.id
    )
    SELECT chunk_id, content, metadata, rrf_score
    FROM rrf_merged
    ORDER BY rrf_score DESC
    LIMIT top_k;
$$;
```

### Materialized View & Window Functions

```sql
-- Document analytics — materialized for fast dashboard queries
CREATE MATERIALIZED VIEW document_stats AS
SELECT
    d.id                                                 AS document_id,
    d.filename,
    d.doc_type,
    d.user_id,
    COUNT(DISTINCT cs.id)                                AS total_sessions,
    COUNT(ch.id)                                         AS total_messages,
    AVG(ch.retrieval_score)                              AS avg_retrieval_score,
    AVG(ch.latency_ms)                                   AS avg_latency_ms,
    -- Window: rank documents by usage per user
    RANK() OVER (
        PARTITION BY d.user_id
        ORDER BY COUNT(ch.id) DESC
    )                                                    AS usage_rank,
    -- Window: 7-day rolling message count
    SUM(COUNT(ch.id)) OVER (
        PARTITION BY d.id
        ORDER BY DATE_TRUNC('day', ch.created_at)
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    )                                                    AS rolling_7d_messages,
    MAX(ch.created_at)                                   AS last_queried_at
FROM documents d
LEFT JOIN chat_sessions cs ON cs.document_id = d.id
LEFT JOIN chat_history  ch ON ch.session_id  = cs.id
GROUP BY d.id, d.filename, d.doc_type, d.user_id, DATE_TRUNC('day', ch.created_at);

CREATE UNIQUE INDEX ON document_stats (document_id);

-- Refresh on a schedule (cron or Supabase edge function)
-- REFRESH MATERIALIZED VIEW CONCURRENTLY document_stats;
```

### Row-Level Security

```sql
ALTER TABLE documents     ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks        ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_history  ENABLE ROW LEVEL SECURITY;

-- Users can only read/write their own rows
CREATE POLICY user_isolation ON documents
    FOR ALL USING (user_id = auth.uid());

CREATE POLICY user_isolation ON chunks
    FOR ALL USING (user_id = auth.uid());

CREATE POLICY user_isolation ON chat_sessions
    FOR ALL USING (user_id = auth.uid());

CREATE POLICY user_isolation ON chat_history
    FOR ALL USING (user_id = auth.uid());
```

---

## ML Pipeline

### 1. Semantic Chunking
Rather than splitting at fixed token boundaries, chunks are split when cosine similarity between consecutive sentences drops below a threshold — keeping semantically coherent units together.

```
sentences → embed each → sliding window similarity →
split where similarity < threshold → merge small chunks
```

### 2. Named Entity Recognition (spaCy)
On ingest, spaCy's `en_core_web_trf` model extracts:
- **PERSON** — authors, referenced individuals
- **ORG** — companies, institutions
- **DATE / TIME** — temporal references
- **GPE** — locations

Stored as JSONB in `documents.entity_tags`, enabling SQL queries like:
```sql
SELECT * FROM documents
WHERE entity_tags @> '{"ORG": ["OpenAI"]}';
```

### 3. Document Classification
A lightweight scikit-learn classifier (TF-IDF + LogisticRegression, or a HuggingFace zero-shot model) assigns each document a type: `legal`, `academic`, `financial`, `technical`, `general`. This routes to document-type-specific retrieval strategies and prompt templates.

### 4. Multi-Query Retrieval
For each user question, Claude generates 3–5 semantically varied reformulations. Each variant runs through `hybrid_search`. Results are deduplicated by chunk ID and merged, dramatically improving recall.

```
user query → Claude → [variant_1, variant_2, variant_3]
                            ↓ parallel retrieval
                       deduplicated union of chunks
```

### 5. Cross-Encoder Re-ranking
The top 20 chunks from RRF are re-scored by a `sentence-transformers` cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Unlike bi-encoders, a cross-encoder reads the query and chunk together, producing higher-fidelity relevance scores. Top 5 are passed to Claude.

```
top_20 chunks → cross-encoder(query, chunk) → sorted scores → top_5
```

### 6. Contextual Compression
Each of the top 5 chunks is compressed: irrelevant sentences stripped, leaving only the parts that directly address the query. Reduces prompt token usage and noise.

### 7. RAGAS Evaluation Harness
An offline evaluation suite using the RAGAS framework measures:

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Answer Relevancy** | Does the answer address the question? |
| **Context Precision** | Are the retrieved chunks actually relevant? |
| **Context Recall** | Were all relevant chunks retrieved? |

A test dataset of question/ground-truth pairs is run nightly against the pipeline to track quality over time.

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Next.js 14, Tailwind CSS, shadcn/ui |
| Backend | FastAPI, Python 3.12 |
| LLM | Claude claude-sonnet-4-6 (Anthropic SDK) |
| Embeddings | Voyage AI (`voyage-3`) |
| RAG Framework | LangChain |
| NER | spaCy (`en_core_web_trf`) |
| Re-ranking | sentence-transformers |
| PDF Parsing | PyMuPDF |
| Database | PostgreSQL 16 via Supabase |
| Vector Search | pgvector |
| Auth | Supabase Auth |
| File Storage | Supabase Storage |
| Evaluation | RAGAS |
| Frontend Deploy | Vercel |
| Backend Deploy | Railway |
| CI/CD | GitHub Actions |
| Repo | https://github.com/AdityaChaudhuri/RAG_Chatbot |

---

## Feature Roadmap

### Phase 1 — Core Pipeline
- [x] Architecture & schema design
- [ ] Database schema, triggers, stored procedures
- [ ] PDF ingestion + semantic chunking
- [ ] NER extraction + document classification
- [ ] Embedding + hybrid search stored procedure
- [ ] Cross-encoder re-ranking
- [ ] Claude integration with streaming

### Phase 2 — Web App
- [ ] Supabase Auth (email + Google OAuth)
- [ ] PDF upload + library management
- [ ] Chat UI with SSE streaming
- [ ] Conversation history sidebar
- [ ] Source chunk citations in UI

### Phase 3 — Intelligence & Analytics
- [ ] Document analytics dashboard (materialized view)
- [ ] RAGAS evaluation harness
- [ ] Multi-document querying (query across all user PDFs)
- [ ] Summarisation mode (full-document)
- [ ] Entity-based document filtering

### Phase 4 — Production
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Monthly partition auto-creation
- [ ] Materialized view refresh cron
- [ ] Rate limiting + usage quotas
- [ ] Deployment (Vercel + Railway)

---

## Project Structure

```
mr-summarizer/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI routes
│   │   ├── ingestion/        # PDF parsing, chunking, NER, classification
│   │   ├── retrieval/        # Hybrid search, re-ranking, compression
│   │   ├── generation/       # Claude integration, prompt templates
│   │   └── db/               # Supabase client, SQL queries
│   ├── eval/                 # RAGAS evaluation harness
│   └── requirements.txt
├── frontend/
│   ├── app/                  # Next.js App Router
│   ├── components/           # Chat, PDF upload, history sidebar
│   └── package.json
├── sql/
│   ├── 001_schema.sql        # Tables, partitions, indexes
│   ├── 002_triggers.sql      # FTS trigger, stats trigger, session trigger
│   ├── 003_procedures.sql    # hybrid_search stored procedure
│   ├── 004_views.sql         # Materialized views
│   └── 005_rls.sql           # Row-level security policies
├── MR_SUMMARIZER.md          # This file
└── .env.example
```

---

## Environment Variables

```env
# Anthropic
ANTHROPIC_API_KEY=

# Voyage AI
VOYAGE_API_KEY=

# Supabase
SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=
DATABASE_URL=

# App
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
```

---

## Why Mr.Summarizer Demonstrates Technical Depth

**SQL:**
The retrieval engine is not a simple `SELECT ... ORDER BY embedding <=> $1 LIMIT 5`. It is a stored procedure that executes two independent retrieval strategies, merges them with a mathematically grounded fusion algorithm, and returns results through PostgreSQL's query planner — while RLS enforces multi-tenant isolation transparently at the database layer, triggers maintain derived state automatically, and a partitioned table with a materialized view powers a real-time analytics dashboard.

**ML:**
The pipeline goes well beyond prompt engineering. Documents are semantically segmented, linguistically tagged, and type-classified before a single vector is stored. At query time, the system generates multiple query perspectives, fuses dense and sparse retrieval signals, then applies a cross-attention re-ranker that understands query-chunk interactions rather than treating them independently. The whole pipeline is measured against academic RAG benchmarks using RAGAS — making quality regressions detectable and the system continuously improvable.
