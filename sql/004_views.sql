-- =============================================================================
-- Mr.Summarizer — 004_views.sql
-- Materialized views and analytics views
-- =============================================================================

-- ---------------------------------------------------------------------------
-- document_stats  (MATERIALIZED)
--
-- Pre-computed per-document analytics for the dashboard. Refreshed
-- concurrently so reads are never blocked during refresh.
--
-- Window functions used:
--   RANK()       — rank each document by usage within that user's library
--   SUM() OVER   — 7-day rolling message count (sliding ROWS frame)
--   LAG()        — change in usage vs. the previous day
-- ---------------------------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS document_stats AS
WITH daily_counts AS (
    -- Aggregate message counts by document and calendar day
    SELECT
        cs.document_id,
        DATE_TRUNC('day', ch.created_at)  AS day,
        COUNT(ch.id)                       AS daily_messages,
        AVG(ch.retrieval_score)            AS daily_avg_score,
        AVG(ch.latency_ms)                 AS daily_avg_latency
    FROM chat_sessions cs
    JOIN chat_history  ch ON ch.session_id = cs.id
    WHERE ch.role = 'assistant'
    GROUP BY cs.document_id, DATE_TRUNC('day', ch.created_at)
),
rolling AS (
    SELECT
        document_id,
        day,
        daily_messages,
        daily_avg_score,
        daily_avg_latency,
        -- 7-day rolling sum (inclusive of current day)
        SUM(daily_messages) OVER (
            PARTITION BY document_id
            ORDER BY day
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        )                                               AS rolling_7d_messages,
        -- Day-over-day change in message volume
        daily_messages - LAG(daily_messages, 1, 0) OVER (
            PARTITION BY document_id ORDER BY day
        )                                               AS day_over_day_delta
    FROM daily_counts
),
totals AS (
    SELECT
        d.id                                            AS document_id,
        d.user_id,
        d.filename,
        d.doc_type,
        d.entity_tags,
        d.chunk_count,
        d.created_at                                    AS uploaded_at,
        COUNT(DISTINCT cs.id)                           AS total_sessions,
        COUNT(ch.id)                                    AS total_messages,
        AVG(ch.retrieval_score)                         AS avg_retrieval_score,
        AVG(ch.latency_ms)                              AS avg_latency_ms,
        MAX(ch.created_at)                              AS last_queried_at
    FROM documents d
    LEFT JOIN chat_sessions cs ON cs.document_id = d.id
    LEFT JOIN chat_history  ch ON ch.session_id  = cs.id AND ch.role = 'assistant'
    GROUP BY d.id, d.user_id, d.filename, d.doc_type,
             d.entity_tags, d.chunk_count, d.created_at
)
SELECT
    t.document_id,
    t.user_id,
    t.filename,
    t.doc_type,
    t.entity_tags,
    t.chunk_count,
    t.uploaded_at,
    t.total_sessions,
    t.total_messages,
    t.avg_retrieval_score,
    t.avg_latency_ms,
    t.last_queried_at,

    -- Rank within the user's library: most-queried document = rank 1
    RANK() OVER (
        PARTITION BY t.user_id
        ORDER BY t.total_messages DESC
    )                                                   AS usage_rank,

    -- Percentile within all documents globally (for benchmarking)
    PERCENT_RANK() OVER (
        ORDER BY t.avg_retrieval_score
    )                                                   AS score_percentile,

    -- Most recent rolling 7d count (join to latest day)
    COALESCE(r.rolling_7d_messages, 0)                 AS rolling_7d_messages,
    COALESCE(r.day_over_day_delta,  0)                 AS day_over_day_delta

FROM totals t
LEFT JOIN LATERAL (
    SELECT rolling_7d_messages, day_over_day_delta
    FROM   rolling
    WHERE  document_id = t.document_id
    ORDER  BY day DESC
    LIMIT  1
) r ON true;

-- Unique index required for CONCURRENT refresh
CREATE UNIQUE INDEX IF NOT EXISTS idx_document_stats_id
    ON document_stats (document_id);

CREATE INDEX IF NOT EXISTS idx_document_stats_user
    ON document_stats (user_id, usage_rank);

-- ---------------------------------------------------------------------------
-- user_activity_summary  (regular VIEW — always fresh)
--
-- Per-user aggregate: total documents, total queries, avg quality score.
-- Uses CTEs and window functions to add a rolling 30-day query count
-- and rank users by engagement (useful for platform-level analytics).
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW user_activity_summary AS
WITH base AS (
    SELECT
        u.id                                            AS user_id,
        COUNT(DISTINCT d.id)                            AS document_count,
        COUNT(ch.id)                                    AS total_queries,
        AVG(ch.retrieval_score)                         AS avg_score,
        MAX(ch.created_at)                              AS last_active_at
    FROM auth.users u
    LEFT JOIN documents     d  ON d.user_id  = u.id
    LEFT JOIN chat_sessions cs ON cs.document_id = d.id
    LEFT JOIN chat_history  ch ON ch.session_id  = cs.id AND ch.role = 'assistant'
    GROUP BY u.id
)
SELECT
    user_id,
    document_count,
    total_queries,
    avg_score,
    last_active_at,
    RANK() OVER (ORDER BY total_queries DESC)           AS engagement_rank,
    NTILE(4) OVER (ORDER BY total_queries DESC)         AS engagement_quartile
FROM base;

-- ---------------------------------------------------------------------------
-- chunk_quality_report  (regular VIEW)
--
-- For each document, shows the distribution of retrieval scores across its
-- chunks using percentile functions — identifies which chunks are consistently
-- surfaced (high score) vs. rarely retrieved (potential quality issues).
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW chunk_quality_report AS
SELECT
    c.document_id,
    d.filename,
    c.id                                                AS chunk_id,
    c.token_count,
    c.metadata ->> 'chunk_index'                        AS chunk_index,
    -- Average score this chunk received when retrieved
    AVG(
        (elem ->> 'score')::FLOAT
    )                                                   AS avg_score,
    COUNT(*)                                            AS times_retrieved,
    -- Rank within document by retrieval frequency
    RANK() OVER (
        PARTITION BY c.document_id
        ORDER BY COUNT(*) DESC
    )                                                   AS retrieval_rank
FROM chunks c
JOIN documents d ON d.id = c.document_id
-- Unnest source_chunks JSONB arrays from assistant messages
LEFT JOIN LATERAL (
    SELECT jsonb_array_elements(ch.source_chunks) AS elem
    FROM chat_history ch
    WHERE ch.role = 'assistant'
      AND ch.source_chunks IS NOT NULL
) cited ON (cited.elem ->> 'chunk_id') = c.id::TEXT
GROUP BY c.document_id, d.filename, c.id, c.token_count, c.metadata;

-- ---------------------------------------------------------------------------
-- Refresh helper — call this from a Supabase Edge Function cron job
-- REFRESH MATERIALIZED VIEW CONCURRENTLY document_stats;
-- ---------------------------------------------------------------------------
