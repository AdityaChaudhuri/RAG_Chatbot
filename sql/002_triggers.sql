-- Mr.Summarizer — triggers
-- Automatic state maintenance so the app layer doesn't have to.

-- Keep fts_vector in sync with chunk content at insert/update time.
CREATE OR REPLACE FUNCTION update_chunk_fts()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.fts_vector := to_tsvector('summarizer_fts', NEW.content);
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_chunk_fts ON chunks;
CREATE TRIGGER trg_chunk_fts
    BEFORE INSERT OR UPDATE OF content ON chunks
    FOR EACH ROW EXECUTE FUNCTION update_chunk_fts();

-- Increment chunk_count and recalculate avg_chunk_len after each chunk insert.
-- AVG is recomputed in SQL rather than in application memory to stay accurate.
CREATE OR REPLACE FUNCTION update_document_chunk_stats()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    UPDATE documents
    SET
        chunk_count   = chunk_count + 1,
        avg_chunk_len = (SELECT AVG(token_count) FROM chunks WHERE document_id = NEW.document_id)
    WHERE id = NEW.document_id;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_document_chunk_stats ON chunks;
CREATE TRIGGER trg_document_chunk_stats
    AFTER INSERT ON chunks
    FOR EACH ROW EXECUTE FUNCTION update_document_chunk_stats();

-- Bump session updated_at on every new message so the session list stays
-- ordered by most recently active without an extra UPDATE from the app.
CREATE OR REPLACE FUNCTION touch_session_on_message()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    UPDATE chat_sessions SET updated_at = now() WHERE id = NEW.session_id;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_touch_session ON chat_history;
CREATE TRIGGER trg_touch_session
    AFTER INSERT ON chat_history
    FOR EACH ROW EXECUTE FUNCTION touch_session_on_message();

-- Use the first user message as the session title if none has been set yet.
CREATE OR REPLACE FUNCTION auto_title_session()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
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
    AFTER INSERT ON chat_history
    FOR EACH ROW EXECUTE FUNCTION auto_title_session();
