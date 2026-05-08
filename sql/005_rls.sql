-- Mr.Summarizer — row-level security
-- Every user can only read and write their own rows. auth.uid() returns the
-- JWT sub claim that Supabase sets from the logged-in user's session.
-- Service-role calls from the backend bypass RLS automatically.

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY documents_select ON documents FOR SELECT USING (user_id = auth.uid());
CREATE POLICY documents_insert ON documents FOR INSERT WITH CHECK (user_id = auth.uid());
CREATE POLICY documents_update ON documents FOR UPDATE USING (user_id = auth.uid()) WITH CHECK (user_id = auth.uid());
CREATE POLICY documents_delete ON documents FOR DELETE USING (user_id = auth.uid());

ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;

CREATE POLICY chunks_select ON chunks FOR SELECT USING (user_id = auth.uid());
CREATE POLICY chunks_insert ON chunks FOR INSERT WITH CHECK (user_id = auth.uid());
-- Chunks are immutable after ingestion — no UPDATE policy needed.
CREATE POLICY chunks_delete ON chunks FOR DELETE USING (user_id = auth.uid());

ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;

CREATE POLICY sessions_select ON chat_sessions FOR SELECT USING (user_id = auth.uid());
CREATE POLICY sessions_insert ON chat_sessions FOR INSERT WITH CHECK (user_id = auth.uid());
CREATE POLICY sessions_update ON chat_sessions FOR UPDATE USING (user_id = auth.uid()) WITH CHECK (user_id = auth.uid());
CREATE POLICY sessions_delete ON chat_sessions FOR DELETE USING (user_id = auth.uid());

-- Policies on the parent table apply to all monthly partitions automatically.
ALTER TABLE chat_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY history_select ON chat_history FOR SELECT USING (user_id = auth.uid());
CREATE POLICY history_insert ON chat_history FOR INSERT WITH CHECK (user_id = auth.uid());
-- Chat history is append-only. Hard deletes happen only via cascade from session deletion.

-- hybrid_search is SECURITY DEFINER so it runs as the table owner and bypasses
-- RLS on chunks — but we still filter by target_user_id inside the function body,
-- which gives the same isolation guarantee without exposing RLS to the query planner.
GRANT EXECUTE ON FUNCTION hybrid_search(VECTOR(768), TEXT, UUID, UUID, INT, INT) TO authenticated;
GRANT EXECUTE ON FUNCTION get_session_context(UUID, INT)                          TO authenticated;
GRANT EXECUTE ON FUNCTION document_retrieval_stats(UUID)                          TO authenticated;

-- Supabase Storage bucket policy (applied via dashboard, documented here for reference)
--
-- Bucket: "documents"
-- INSERT: (bucket_id = 'documents') AND ((storage.foldername(name))[1] = auth.uid()::text)
-- SELECT: (bucket_id = 'documents') AND ((storage.foldername(name))[1] = auth.uid()::text)
