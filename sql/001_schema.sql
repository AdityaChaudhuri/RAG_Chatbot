-- Mr.Summarizer — schema
-- Run once against your Supabase project to set up all tables and indexes.

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;       -- pgvector: ANN similarity search
CREATE EXTENSION IF NOT EXISTS pg_trgm;      -- trigram matching for fuzzy text
CREATE EXTENSION IF NOT EXISTS unaccent;     -- strip accents before FTS indexing
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- UUID generation fallback

-- Custom FTS config: strips accents then applies English stemming.
-- Using a named config means we get consistent behaviour across insert and search.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_ts_config WHERE cfgname = 'summarizer_fts') THEN
        CREATE TEXT SEARCH CONFIGURATION summarizer_fts (COPY = english);
        ALTER TEXT SEARCH CONFIGURATION summarizer_fts
            ALTER MAPPING FOR hword, compound_hword, hword_part, word, asciiword
            WITH unaccent, english_stem;
    END IF;
END
$$;

-- One row per uploaded PDF.
CREATE TABLE IF NOT EXISTS documents (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    filename      TEXT        NOT NULL,
    file_url      TEXT        NOT NULL,
    doc_type      TEXT        CHECK (doc_type IN ('legal', 'academic', 'financial', 'technical', 'general')),
    entity_tags   JSONB       NOT NULL DEFAULT '{}',  -- {people:[], orgs:[], dates:[], locations:[]}
    chunk_count   INT         NOT NULL DEFAULT 0,
    avg_chunk_len FLOAT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_user_id
    ON documents (user_id, created_at DESC);

-- Lets us query documents by extracted entity: WHERE entity_tags @> '{"orgs": ["Anthropic"]}'
CREATE INDEX IF NOT EXISTS idx_documents_entity_tags
    ON documents USING GIN (entity_tags jsonb_path_ops);

-- One row per semantic chunk. Carries both a dense vector and a tsvector
-- so hybrid_search can hit a single table for both retrieval paths.
CREATE TABLE IF NOT EXISTS chunks (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id   UUID        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    user_id       UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    content       TEXT        NOT NULL,
    embedding     VECTOR(768),                         -- BGE-base-en-v1.5 output dimension
    fts_vector    TSVECTOR,                            -- populated by trigger in 002_triggers.sql
    metadata      JSONB       NOT NULL DEFAULT '{}',   -- {page_nums:[], chunk_index:int}
    token_count   INT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- IVFFlat for approximate nearest-neighbour search (cosine distance).
-- lists=100 handles up to ~1M vectors well; tune upward at scale.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON chunks USING GIN (fts_vector);

CREATE INDEX IF NOT EXISTS idx_chunks_metadata
    ON chunks USING GIN (metadata jsonb_path_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON chunks (document_id);

-- Composite index covers the most common query pattern: user isolation + document scope
CREATE INDEX IF NOT EXISTS idx_chunks_user_doc
    ON chunks (user_id, document_id);

-- Groups messages into a conversation, optionally scoped to a document.
CREATE TABLE IF NOT EXISTS chat_sessions (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    document_id UUID        REFERENCES documents(id) ON DELETE SET NULL,
    title       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id
    ON chat_sessions (user_id, updated_at DESC);

-- Partitioned by month so old data can be archived without touching active rows.
CREATE TABLE IF NOT EXISTS chat_history (
    id              UUID        NOT NULL DEFAULT gen_random_uuid(),
    session_id      UUID        NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_id         UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    role            TEXT        NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT        NOT NULL,
    source_chunks   JSONB,       -- chunk UUIDs cited in this response
    retrieval_score FLOAT,       -- RRF score of the top retrieved chunk
    latency_ms      INT,         -- end-to-end response latency
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
) PARTITION BY RANGE (created_at);

CREATE TABLE IF NOT EXISTS chat_history_2026_01 PARTITION OF chat_history FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE IF NOT EXISTS chat_history_2026_02 PARTITION OF chat_history FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE IF NOT EXISTS chat_history_2026_03 PARTITION OF chat_history FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE IF NOT EXISTS chat_history_2026_04 PARTITION OF chat_history FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE IF NOT EXISTS chat_history_2026_05 PARTITION OF chat_history FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE IF NOT EXISTS chat_history_2026_06 PARTITION OF chat_history FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE IF NOT EXISTS chat_history_2026_07 PARTITION OF chat_history FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');
CREATE TABLE IF NOT EXISTS chat_history_2026_08 PARTITION OF chat_history FOR VALUES FROM ('2026-08-01') TO ('2026-09-01');
CREATE TABLE IF NOT EXISTS chat_history_2026_09 PARTITION OF chat_history FOR VALUES FROM ('2026-09-01') TO ('2026-10-01');
CREATE TABLE IF NOT EXISTS chat_history_2026_10 PARTITION OF chat_history FOR VALUES FROM ('2026-10-01') TO ('2026-11-01');
CREATE TABLE IF NOT EXISTS chat_history_2026_11 PARTITION OF chat_history FOR VALUES FROM ('2026-11-01') TO ('2026-12-01');
CREATE TABLE IF NOT EXISTS chat_history_2026_12 PARTITION OF chat_history FOR VALUES FROM ('2026-12-01') TO ('2027-01-01');

-- Indexes on the parent table propagate to all partitions automatically (PG 11+)
CREATE INDEX IF NOT EXISTS idx_chat_history_session
    ON chat_history (session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_chat_history_user
    ON chat_history (user_id, created_at DESC);
