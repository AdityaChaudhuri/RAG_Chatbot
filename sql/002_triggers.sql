-- =============================================================================
-- Mr.Summarizer — 002_triggers.sql
-- Trigger functions and triggers for automatic state maintenance
-- =============================================================================

-- ---------------------------------------------------------------------------
-- TRIGGER 1: Auto-populate fts_vector on chunk insert/update
--
-- Runs BEFORE INSERT OR UPDATE so the tsvector is always in sync with content.
-- Uses the custom summarizer_fts config for accent-normalised English stemming.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_chunk_fts()
RETURNS TRIGGER
LANGUAGE plpgsql AS $$
BEGIN
    NEW.fts_vector := to_tsvector('summarizer_fts', NEW.content);
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_chunk_fts ON chunks;
CREATE TRIGGER trg_chunk_fts
    BEFORE INSERT OR UPDATE OF content
    ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_chunk_fts();

-- ---------------------------------------------------------------------------
-- TRIGGER 2: Keep documents.chunk_count and avg_chunk_len accurate
--
-- Fires AFTER INSERT on chunks. Recalculates avg_chunk_len with a subquery
-- so we get the true average across all chunks for that document without
-- having to load them all into application memory.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_document_chunk_stats()
RETURNS TRIGGER
LANGUAGE plpgsql AS $$
BEGIN
    UPDATE documents
    SET
        chunk_count   = chunk_count + 1,
        avg_chunk_len = (
            SELECT AVG(token_count)
            FROM   chunks
            WHERE  document_id = NEW.document_id
        )
    WHERE id = NEW.document_id;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_document_chunk_stats ON chunks;
CREATE TRIGGER trg_document_chunk_stats
    AFTER INSERT
    ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_document_chunk_stats();

-- ---------------------------------------------------------------------------
-- TRIGGER 3: Touch chat_sessions.updated_at on every new message
--
-- Keeps the session list ordered by most-recently-active without requiring
-- the application layer to issue a separate UPDATE on each message insert.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION touch_session_on_message()
RETURNS TRIGGER
LANGUAGE plpgsql AS $$
BEGIN
    UPDATE chat_sessions
    SET    updated_at = now()
    WHERE  id = NEW.session_id;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_touch_session ON chat_history;
CREATE TRIGGER trg_touch_session
    AFTER INSERT
    ON chat_history
    FOR EACH ROW
    EXECUTE FUNCTION touch_session_on_message();

-- ---------------------------------------------------------------------------
-- TRIGGER 4: Auto-set chat session title from first user message
--
-- Fires AFTER INSERT on chat_history. If the session has no title yet and
-- the new message is from the user, truncate the message text and use it
-- as the session title — saves a round-trip from the application layer.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION auto_title_session()
RETURNS TRIGGER
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.role = 'user' THEN
        UPDATE chat_sessions
        SET    title = LEFT(NEW.content, 60)
        WHERE  id    = NEW.session_id
          AND  title IS NULL;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_auto_title_session ON chat_history;
CREATE TRIGGER trg_auto_title_session
    AFTER INSERT
    ON chat_history
    FOR EACH ROW
    EXECUTE FUNCTION auto_title_session();
