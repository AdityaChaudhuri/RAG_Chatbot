-- =============================================================================
-- Mr.Summarizer — 003_procedures.sql
-- Stored procedures: hybrid search with Reciprocal Rank Fusion
-- =============================================================================

-- ---------------------------------------------------------------------------
-- hybrid_search
--
-- Executes two independent retrieval strategies against the chunks table,
-- then merges their ranked result lists using Reciprocal Rank Fusion (RRF).
--
-- Strategy A — Dense retrieval (semantic):
--   pgvector ANN search using cosine distance (<=>). Finds chunks whose
--   meaning is close to the query embedding even when keywords differ.
--
-- Strategy B — Sparse retrieval (lexical):
--   PostgreSQL full-text search via tsvector / ts_rank_cd. Finds chunks
--   that share exact or stemmed terms with the query — high precision for
--   named entities, technical terms, and short queries.
--
-- Fusion — Reciprocal Rank Fusion:
--   RRF score = 1/(k + rank_A) + 1/(k + rank_B)
--   where k=60 dampens the effect of very high ranks. A chunk that appears
--   near the top of BOTH lists scores highest; a chunk missing from one list
--   still gets partial credit.
--
-- Parameters:
--   query_embedding  — 1024-d Voyage AI embedding of the user query
--   query_text       — raw query string for FTS (tsquery is built inside)
--   target_user_id   — enforces row-level user isolation at the procedure level
--   target_doc_id    — optional: restrict to a single document
--   top_k            — final number of chunks to return (default 20)
--   rrf_k            — RRF damping constant (default 60, per original paper)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding  VECTOR(1024),
    query_text       TEXT,
    target_user_id   UUID,
    target_doc_id    UUID    DEFAULT NULL,
    top_k            INT     DEFAULT 20,
    rrf_k            INT     DEFAULT 60
)
RETURNS TABLE (
    chunk_id      UUID,
    content       TEXT,
    metadata      JSONB,
    document_id   UUID,
    vector_rank   BIGINT,
    fts_rank      BIGINT,
    rrf_score     FLOAT
)
LANGUAGE SQL
STABLE          -- no side-effects; allows the planner to cache / inline
SECURITY DEFINER  -- runs as owner so RLS on chunks is still enforced via user_id filter
AS $$
    WITH
    -- -----------------------------------------------------------------------
    -- CTE 1: Dense retrieval — top (top_k * 2) by cosine distance
    -- Fetching 2× top_k gives RRF more material to work with before the
    -- final LIMIT; compensates for chunks that appear only in one list.
    -- -----------------------------------------------------------------------
    vector_hits AS (
        SELECT
            id,
            content,
            metadata,
            document_id,
            ROW_NUMBER() OVER (ORDER BY embedding <=> query_embedding) AS v_rank
        FROM chunks
        WHERE user_id = target_user_id
          AND (target_doc_id IS NULL OR document_id = target_doc_id)
          AND embedding IS NOT NULL
        ORDER BY embedding <=> query_embedding
        LIMIT top_k * 2
    ),

    -- -----------------------------------------------------------------------
    -- CTE 2: Sparse retrieval — FTS with cover-density ranking
    -- ts_rank_cd weights by the proximity of matching terms (cover density),
    -- which is more robust than simple ts_rank for multi-word queries.
    -- websearch_to_tsquery handles natural language input gracefully
    -- (no syntax errors on bare user input unlike plainto_tsquery).
    -- -----------------------------------------------------------------------
    fts_hits AS (
        SELECT
            id,
            content,
            metadata,
            document_id,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank_cd(fts_vector, query, 32) DESC
            ) AS f_rank
        FROM
            chunks,
            websearch_to_tsquery('summarizer_fts', query_text) AS query
        WHERE user_id = target_user_id
          AND (target_doc_id IS NULL OR document_id = target_doc_id)
          AND fts_vector @@ query
        ORDER BY ts_rank_cd(fts_vector, query, 32) DESC
        LIMIT top_k * 2
    ),

    -- -----------------------------------------------------------------------
    -- CTE 3: RRF merge
    -- FULL OUTER JOIN ensures chunks from either list are represented.
    -- COALESCE handles chunks that appear in only one list (rank = infinity
    -- in the missing list, contributing 0 to the RRF sum).
    -- -----------------------------------------------------------------------
    rrf AS (
        SELECT
            COALESCE(v.id,          f.id)          AS chunk_id,
            COALESCE(v.content,     f.content)     AS content,
            COALESCE(v.metadata,    f.metadata)    AS metadata,
            COALESCE(v.document_id, f.document_id) AS document_id,
            COALESCE(v.v_rank, (top_k * 2 + 1)::BIGINT) AS vector_rank,
            COALESCE(f.f_rank, (top_k * 2 + 1)::BIGINT) AS fts_rank,
            COALESCE(1.0 / (rrf_k + v.v_rank), 0.0) +
            COALESCE(1.0 / (rrf_k + f.f_rank), 0.0) AS rrf_score
        FROM vector_hits v
        FULL OUTER JOIN fts_hits f ON v.id = f.id
    )

    SELECT
        chunk_id,
        content,
        metadata,
        document_id,
        vector_rank,
        fts_rank,
        rrf_score
    FROM rrf
    ORDER BY rrf_score DESC
    LIMIT top_k;
$$;

-- ---------------------------------------------------------------------------
-- get_session_context
--
-- Returns the N most recent messages for a session, ordered oldest-first,
-- ready to be passed as the messages array to the Claude API.
-- Uses a CTE + window function so we get the tail of the history without
-- a subquery that forces a full partition scan.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION get_session_context(
    p_session_id UUID,
    p_limit      INT DEFAULT 20
)
RETURNS TABLE (
    role       TEXT,
    content    TEXT,
    created_at TIMESTAMPTZ
)
LANGUAGE SQL
STABLE AS $$
    WITH ranked AS (
        SELECT
            role,
            content,
            created_at,
            ROW_NUMBER() OVER (ORDER BY created_at DESC) AS rn
        FROM chat_history
        WHERE session_id = p_session_id
    )
    SELECT role, content, created_at
    FROM   ranked
    WHERE  rn <= p_limit
    ORDER  BY created_at ASC;
$$;

-- ---------------------------------------------------------------------------
-- document_retrieval_stats
--
-- Ad-hoc analytics: for a given document, return per-query retrieval metrics
-- using window functions to rank, percentile, and diff latency across sessions.
-- Useful for evaluating pipeline quality without leaving SQL.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION document_retrieval_stats(p_document_id UUID)
RETURNS TABLE (
    session_id        UUID,
    message_count     BIGINT,
    avg_score         FLOAT,
    avg_latency_ms    FLOAT,
    score_percentile  FLOAT,
    latency_vs_avg_ms FLOAT
)
LANGUAGE SQL
STABLE AS $$
    WITH session_metrics AS (
        SELECT
            cs.id                         AS session_id,
            COUNT(ch.id)                  AS message_count,
            AVG(ch.retrieval_score)       AS avg_score,
            AVG(ch.latency_ms)            AS avg_latency_ms
        FROM chat_sessions cs
        JOIN chat_history  ch ON ch.session_id = cs.id
        WHERE cs.document_id = p_document_id
          AND ch.role = 'assistant'
        GROUP BY cs.id
    )
    SELECT
        session_id,
        message_count,
        avg_score,
        avg_latency_ms,
        PERCENT_RANK() OVER (ORDER BY avg_score)       AS score_percentile,
        avg_latency_ms - AVG(avg_latency_ms) OVER ()   AS latency_vs_avg_ms
    FROM session_metrics
    ORDER BY avg_score DESC;
$$;
