-- Mr.Summarizer — stored procedures

-- hybrid_search
--
-- Runs two independent retrieval strategies then merges them with
-- Reciprocal Rank Fusion (RRF score = 1/(k + rank_A) + 1/(k + rank_B)).
--
-- Vector path: cosine ANN via pgvector — finds semantically similar chunks
--   even when the user's exact words don't appear in the document.
-- FTS path: full-text search via tsvector — high precision for named entities,
--   technical terms, and short keyword-style queries.
--
-- A chunk that ranks near the top of both lists scores highest.
-- A chunk appearing in only one list still gets partial credit via FULL OUTER JOIN.
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding  VECTOR(768),  -- 768-d BGE-base-en-v1.5 embedding of the query
    query_text       TEXT,
    target_user_id   UUID,
    target_doc_id    UUID  DEFAULT NULL,
    top_k            INT   DEFAULT 20,
    rrf_k            INT   DEFAULT 60   -- damping constant from the original RRF paper
)
RETURNS TABLE (
    chunk_id     UUID,
    content      TEXT,
    metadata     JSONB,
    document_id  UUID,
    vector_rank  BIGINT,
    fts_rank     BIGINT,
    rrf_score    FLOAT
)
LANGUAGE SQL
STABLE
SECURITY DEFINER  -- runs as owner; user isolation is enforced by the target_user_id filter
AS $$
    WITH
    -- Fetch 2× top_k from each path so RRF has enough overlap to work with.
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

    fts_hits AS (
        SELECT
            id,
            content,
            metadata,
            document_id,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(fts_vector, query, 32) DESC) AS f_rank
        FROM
            chunks,
            websearch_to_tsquery('summarizer_fts', query_text) AS query
        WHERE user_id = target_user_id
          AND (target_doc_id IS NULL OR document_id = target_doc_id)
          AND fts_vector @@ query
        ORDER BY ts_rank_cd(fts_vector, query, 32) DESC
        LIMIT top_k * 2
    ),

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

    SELECT chunk_id, content, metadata, document_id, vector_rank, fts_rank, rrf_score
    FROM rrf
    ORDER BY rrf_score DESC
    LIMIT top_k;
$$;

-- get_session_context
--
-- Returns the N most recent messages for a session, oldest-first,
-- ready to pass as the messages array to the Gemini API.
CREATE OR REPLACE FUNCTION get_session_context(
    p_session_id UUID,
    p_limit      INT DEFAULT 20
)
RETURNS TABLE (
    role       TEXT,
    content    TEXT,
    created_at TIMESTAMPTZ
)
LANGUAGE SQL STABLE AS $$
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

-- document_retrieval_stats
--
-- Per-session retrieval metrics for a given document.
-- score_percentile and latency_vs_avg use window functions so you can
-- compare each session's quality against all sessions in one query.
CREATE OR REPLACE FUNCTION document_retrieval_stats(p_document_id UUID)
RETURNS TABLE (
    session_id        UUID,
    message_count     BIGINT,
    avg_score         FLOAT,
    avg_latency_ms    FLOAT,
    score_percentile  FLOAT,
    latency_vs_avg_ms FLOAT
)
LANGUAGE SQL STABLE AS $$
    WITH session_metrics AS (
        SELECT
            cs.id                    AS session_id,
            COUNT(ch.id)             AS message_count,
            AVG(ch.retrieval_score)  AS avg_score,
            AVG(ch.latency_ms)       AS avg_latency_ms
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
        PERCENT_RANK() OVER (ORDER BY avg_score)     AS score_percentile,
        avg_latency_ms - AVG(avg_latency_ms) OVER () AS latency_vs_avg_ms
    FROM session_metrics
    ORDER BY avg_score DESC;
$$;
