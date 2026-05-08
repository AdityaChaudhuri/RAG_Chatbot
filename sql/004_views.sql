-- Mr.Summarizer — views

-- document_stats (MATERIALIZED)
--
-- Pre-computed analytics for the document library page. Refreshed concurrently
-- so reads are never blocked during refresh.
--
-- Window functions:
--   RANK()        — each document's position in the user's library by usage
--   SUM() OVER    — 7-day rolling message count
--   LAG()         — day-over-day change in message volume
--   PERCENT_RANK()— where this document sits globally by retrieval score
CREATE MATERIALIZED VIEW IF NOT EXISTS document_stats AS
WITH daily_counts AS (
    SELECT
        cs.document_id,
        DATE_TRUNC('day', ch.created_at) AS day,
        COUNT(ch.id)                     AS daily_messages,
        AVG(ch.retrieval_score)          AS daily_avg_score,
        AVG(ch.latency_ms)               AS daily_avg_latency
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
        SUM(daily_messages) OVER (
            PARTITION BY document_id
            ORDER BY day
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_7d_messages,
        daily_messages - LAG(daily_messages, 1, 0) OVER (
            PARTITION BY document_id ORDER BY day
        ) AS day_over_day_delta
    FROM daily_counts
),
totals AS (
    SELECT
        d.id            AS document_id,
        d.user_id,
        d.filename,
        d.doc_type,
        d.entity_tags,
        d.chunk_count,
        d.created_at    AS uploaded_at,
        COUNT(DISTINCT cs.id)   AS total_sessions,
        COUNT(ch.id)            AS total_messages,
        AVG(ch.retrieval_score) AS avg_retrieval_score,
        AVG(ch.latency_ms)      AS avg_latency_ms,
        MAX(ch.created_at)      AS last_queried_at
    FROM documents d
    LEFT JOIN chat_sessions cs ON cs.document_id = d.id
    LEFT JOIN chat_history  ch ON ch.session_id  = cs.id AND ch.role = 'assistant'
    GROUP BY d.id, d.user_id, d.filename, d.doc_type, d.entity_tags, d.chunk_count, d.created_at
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
    RANK() OVER (PARTITION BY t.user_id ORDER BY t.total_messages DESC) AS usage_rank,
    PERCENT_RANK() OVER (ORDER BY t.avg_retrieval_score)                AS score_percentile,
    COALESCE(r.rolling_7d_messages, 0)                                  AS rolling_7d_messages,
    COALESCE(r.day_over_day_delta,  0)                                  AS day_over_day_delta
FROM totals t
LEFT JOIN LATERAL (
    SELECT rolling_7d_messages, day_over_day_delta
    FROM   rolling
    WHERE  document_id = t.document_id
    ORDER  BY day DESC
    LIMIT  1
) r ON true;

-- Required for CONCURRENT refresh (needs a unique key to diff rows)
CREATE UNIQUE INDEX IF NOT EXISTS idx_document_stats_id  ON document_stats (document_id);
CREATE INDEX        IF NOT EXISTS idx_document_stats_user ON document_stats (user_id, usage_rank);

-- user_activity_summary
--
-- Per-user totals for platform-level analytics.
-- engagement_rank and engagement_quartile let you segment users by activity.
CREATE OR REPLACE VIEW user_activity_summary AS
WITH base AS (
    SELECT
        u.id                    AS user_id,
        COUNT(DISTINCT d.id)    AS document_count,
        COUNT(ch.id)            AS total_queries,
        AVG(ch.retrieval_score) AS avg_score,
        MAX(ch.created_at)      AS last_active_at
    FROM auth.users u
    LEFT JOIN documents     d  ON d.user_id     = u.id
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
    RANK()  OVER (ORDER BY total_queries DESC) AS engagement_rank,
    NTILE(4) OVER (ORDER BY total_queries DESC) AS engagement_quartile
FROM base;

-- chunk_quality_report
--
-- Shows how often each chunk is retrieved and what score it typically receives.
-- Chunks with a high retrieval_rank but low avg_score may be poor quality.
-- Uses a LATERAL unnest to join each assistant message's source_chunks array
-- back to the chunks table without needing a separate junction table.
CREATE OR REPLACE VIEW chunk_quality_report AS
SELECT
    c.document_id,
    d.filename,
    c.id                             AS chunk_id,
    c.token_count,
    c.metadata ->> 'chunk_index'     AS chunk_index,
    AVG((elem ->> 'score')::FLOAT)   AS avg_score,
    COUNT(*)                         AS times_retrieved,
    RANK() OVER (
        PARTITION BY c.document_id
        ORDER BY COUNT(*) DESC
    )                                AS retrieval_rank
FROM chunks c
JOIN documents d ON d.id = c.document_id
LEFT JOIN LATERAL (
    SELECT jsonb_array_elements(ch.source_chunks) AS elem
    FROM chat_history ch
    WHERE ch.role = 'assistant'
      AND ch.source_chunks IS NOT NULL
) cited ON (cited.elem ->> 'chunk_id') = c.id::TEXT
GROUP BY c.document_id, d.filename, c.id, c.token_count, c.metadata;

-- To refresh the materialized view (run from a Supabase cron job or manually):
-- REFRESH MATERIALIZED VIEW CONCURRENTLY document_stats;
