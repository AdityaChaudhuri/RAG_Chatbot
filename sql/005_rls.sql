-- =============================================================================
-- Mr.Summarizer — 005_rls.sql
-- Row-Level Security policies — multi-tenant isolation
--
-- Every user can only SELECT / INSERT / UPDATE / DELETE their own rows.
-- auth.uid() is the Supabase helper that returns the JWT sub claim.
-- Service-role calls (backend) bypass RLS automatically.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- documents
-- ---------------------------------------------------------------------------
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY documents_select ON documents
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY documents_insert ON documents
    FOR INSERT WITH CHECK (user_id = auth.uid());

CREATE POLICY documents_update ON documents
    FOR UPDATE USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());

CREATE POLICY documents_delete ON documents
    FOR DELETE USING (user_id = auth.uid());

-- ---------------------------------------------------------------------------
-- chunks
-- ---------------------------------------------------------------------------
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;

CREATE POLICY chunks_select ON chunks
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY chunks_insert ON chunks
    FOR INSERT WITH CHECK (user_id = auth.uid());

-- Chunks are immutable after ingestion — no UPDATE policy
CREATE POLICY chunks_delete ON chunks
    FOR DELETE USING (user_id = auth.uid());

-- ---------------------------------------------------------------------------
-- chat_sessions
-- ---------------------------------------------------------------------------
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;

CREATE POLICY sessions_select ON chat_sessions
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY sessions_insert ON chat_sessions
    FOR INSERT WITH CHECK (user_id = auth.uid());

CREATE POLICY sessions_update ON chat_sessions
    FOR UPDATE USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());

CREATE POLICY sessions_delete ON chat_sessions
    FOR DELETE USING (user_id = auth.uid());

-- ---------------------------------------------------------------------------
-- chat_history  (partitioned — policies apply to all child partitions)
-- ---------------------------------------------------------------------------
ALTER TABLE chat_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY history_select ON chat_history
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY history_insert ON chat_history
    FOR INSERT WITH CHECK (user_id = auth.uid());

-- Chat history is immutable — no UPDATE or DELETE policies.
-- Hard deletes happen only via cascade from session deletion.

-- ---------------------------------------------------------------------------
-- Supabase storage bucket policy (reference — applied via Supabase dashboard
-- or Management API, not SQL, but documented here for completeness)
--
-- Bucket: "documents"
-- Policy: authenticated users can INSERT and SELECT objects under their
--         own user_id prefix:  {user_id}/*
--
-- storage.objects INSERT:  (bucket_id = 'documents') AND
--                          ((storage.foldername(name))[1] = auth.uid()::text)
--
-- storage.objects SELECT:  (bucket_id = 'documents') AND
--                          ((storage.foldername(name))[1] = auth.uid()::text)
-- ---------------------------------------------------------------------------

-- ---------------------------------------------------------------------------
-- Grant execute on procedures to authenticated role
-- hybrid_search is SECURITY DEFINER so it runs as the owner (bypasses RLS
-- on chunks) but we still filter by target_user_id inside the function body,
-- providing equivalent isolation without exposing RLS to the planner.
-- ---------------------------------------------------------------------------
GRANT EXECUTE ON FUNCTION hybrid_search(VECTOR, TEXT, UUID, UUID, INT, INT)
    TO authenticated;

GRANT EXECUTE ON FUNCTION get_session_context(UUID, INT)
    TO authenticated;

GRANT EXECUTE ON FUNCTION document_retrieval_stats(UUID)
    TO authenticated;
