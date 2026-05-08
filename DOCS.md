# Mr.Summarizer — Code Documentation

Every file explained so you know the codebase from top to bottom.

---

## Table of Contents

1. [SQL Layer](#sql-layer)
   - [001_schema.sql](#001_schemasql)
   - [002_triggers.sql](#002_triggerssql)
   - [003_procedures.sql](#003_proceduressql)
   - [004_views.sql](#004_viewssql)
   - [005_rls.sql](#005_rlssql)
2. [Backend Core](#backend-core)
   - [config.py](#configpy)
   - [db/client.py](#dbclientpy)
   - [main.py](#mainpy)
3. [Ingestion Pipeline](#ingestion-pipeline)
   - [ingestion/parser.py](#ingestionparserpy)
   - [ingestion/chunker.py](#ingestionchunkerpy)
   - [ingestion/ner.py](#ingestionnerpy)
   - [ingestion/classifier.py](#ingestionclassifierpy)
   - [ingestion/embedder.py](#ingestionembedderpy)
   - [ingestion/pipeline.py](#ingestionpipelinepy)
4. [Retrieval Pipeline](#retrieval-pipeline)
   - [retrieval/search.py](#retrievalsearchpy)
   - [retrieval/multi_query.py](#retrievalmulti_querypy)
   - [retrieval/reranker.py](#retrievalrerankerpy)
   - [retrieval/compressor.py](#retrievalcompressorpy)
5. [Generation Layer](#generation-layer)
   - [generation/prompts.py](#generationpromptspy)
   - [generation/gemini.py](#generationgeminipy)
6. [API Routes](#api-routes)
   - [api/routes/documents.py](#apiroutesdocumentspy)
   - [api/routes/chat.py](#apirouteschatpy)

---

# SQL Layer

Lives in `sql/` and runs once against your Supabase project in order (001 → 005). Defines the entire data model, search engine, and access control.

---

## 001_schema.sql

Creates every table, the custom text-search configuration, and all indexes.

---

### Extensions

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```
Loads **pgvector** — adds a `VECTOR(n)` column type and distance operators for approximate nearest-neighbour search. We store 768-dimensional embeddings and query them with cosine distance. `IF NOT EXISTS` makes every statement safe to re-run.

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```
**Trigram** extension. Breaks strings into 3-character substrings for fuzzy text matching. Used as a fallback when a user misspells a technical term.

```sql
CREATE EXTENSION IF NOT EXISTS unaccent;
```
Strips diacritic marks before indexing, so "Résumé" and "Resume" match in full-text search. Plugged into the custom FTS config below.

```sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```
Provides `gen_random_uuid()` as a fallback. PostgreSQL 13+ includes it natively but the extension guarantees it in all Supabase environments.

---

### Custom FTS Configuration

```sql
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
```
`DO $$...$$` is a PL/pgSQL anonymous block — lets us write procedural logic outside a function. The `IF NOT EXISTS` workaround is needed because `CREATE TEXT SEARCH CONFIGURATION` has no native guard clause.

The config chains two dictionaries: `unaccent` (strip accents) then `english_stem` (reduce words to their root). So "Résumé" and "resumed" both match a search for "resume".

---

### documents table

```sql
id UUID PRIMARY KEY DEFAULT gen_random_uuid()
```
UUIDs throughout instead of serial integers — globally unique, no information leakage about row count.

```sql
user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE
```
Foreign key to Supabase's built-in auth table. `ON DELETE CASCADE` — delete a user and all their documents disappear automatically. No orphaned rows possible.

```sql
doc_type TEXT CHECK (doc_type IN ('legal','academic','financial','technical','general'))
```
The classifier's output. The `CHECK` constraint is a database-level guard against classifier bugs — only valid categories can be stored.

```sql
entity_tags JSONB NOT NULL DEFAULT '{}'
```
`JSONB` is PostgreSQL's binary JSON. Supports indexing and containment operators (`@>`). Stores NER output: `{"people": ["John"], "orgs": ["Anthropic"]}`. `DEFAULT '{}'` means no NULLs — new rows start with an empty object.

```sql
chunk_count INT NOT NULL DEFAULT 0
```
Starts at 0. The `trg_document_chunk_stats` trigger increments it on every chunk insert — no need for the app to run `COUNT(*)` queries.

---

### documents indexes

```sql
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents (user_id, created_at DESC);
```
Composite B-tree. Covers `SELECT * FROM documents WHERE user_id = X ORDER BY created_at DESC` entirely — no sort step needed because `DESC` matches the index direction.

```sql
CREATE INDEX IF NOT EXISTS idx_documents_entity_tags ON documents USING GIN (entity_tags jsonb_path_ops);
```
GIN (Generalised Inverted Index) on the JSONB column. Indexes every key-value pair inside the JSON, so `entity_tags @> '{"orgs": ["OpenAI"]}'` is answered by an index lookup, not a row scan. `jsonb_path_ops` is a smaller operator class optimised specifically for `@>` containment queries.

---

### chunks table

```sql
user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE
```
Deliberately denormalised — `user_id` is also on `documents`. Storing it here lets `hybrid_search` filter by user without joining to `documents`, keeping the hot retrieval path to a single table.

```sql
embedding VECTOR(768)
```
A 768-dimensional float array matching the output of BGE-base-en-v1.5. `NULL` is allowed briefly but our pipeline always writes the embedding before finishing.

```sql
fts_vector TSVECTOR
```
Pre-processed tokenised representation of `content`, built automatically by the `trg_chunk_fts` trigger. The app never writes this column directly.

---

### chunks indexes

```sql
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```
**IVFFlat** (Inverted File with Flat compression) is an Approximate Nearest Neighbour index. It clusters all vectors into `lists = 100` Voronoi cells at build time. At query time it searches only the closest cells, not the entire table. `vector_cosine_ops` measures cosine distance — appropriate for normalised vectors because it captures angular similarity regardless of magnitude. `lists = 100` handles up to ~1M vectors well.

```sql
CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING GIN (fts_vector);
```
GIN on the tsvector column. When you run `fts_vector @@ query`, PostgreSQL uses the GIN index to find matching rows instantly instead of reading every row.

```sql
CREATE INDEX IF NOT EXISTS idx_chunks_user_doc ON chunks (user_id, document_id);
```
Composite index for the most common retrieval pattern: `WHERE user_id = X AND document_id = Y`. `user_id` is the leading column because it filters the most rows.

---

### chat_sessions table

```sql
document_id UUID REFERENCES documents(id) ON DELETE SET NULL
```
`ON DELETE SET NULL` — if a document is deleted, sessions survive with their history intact but `document_id` is cleared. Compare to chunks which use `CASCADE` (chunks without a document are useless).

---

### chat_history — Partitioned Table

```sql
CREATE TABLE IF NOT EXISTS chat_history (...) PARTITION BY RANGE (created_at);
```
Partitioning splits one logical table into physical segments by month. Benefits:
- Queries filtered by date only scan the relevant partition, not the whole table.
- Old months can be dropped instantly with `DROP TABLE chat_history_2026_01`.
- Each partition is vacuumed independently — less lock contention.

```sql
CREATE TABLE IF NOT EXISTS chat_history_2026_01 PARTITION OF chat_history FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
```
The upper bound is exclusive (standard half-open interval). Indexes created on the parent table propagate automatically to all partitions in PG 11+.

---

## 002_triggers.sql

Trigger functions that maintain derived state automatically so the application layer doesn't have to.

---

### Trigger 1 — update_chunk_fts

```sql
BEGIN
    NEW.fts_vector := to_tsvector('summarizer_fts', NEW.content);
    RETURN NEW;
END;
```
`NEW` is the row being inserted or updated. We assign the computed tsvector back to `fts_vector` before the write hits disk. `RETURN NEW` is required for `BEFORE` triggers — returning the (possibly modified) row tells PostgreSQL to proceed.

`BEFORE INSERT OR UPDATE OF content` — fires before any INSERT, and before any UPDATE that touches the `content` column specifically. Not every UPDATE, which keeps it efficient.

---

### Trigger 2 — update_document_chunk_stats

```sql
UPDATE documents
SET
    chunk_count   = chunk_count + 1,
    avg_chunk_len = (SELECT AVG(token_count) FROM chunks WHERE document_id = NEW.document_id)
WHERE id = NEW.document_id;
```
`chunk_count + 1` is O(1). The `AVG` subquery is a recalculation — it runs after the new chunk exists in the table (this is an `AFTER` trigger), so it includes the new row. Simpler than an incremental formula and accurate.

---

### Trigger 3 — touch_session_on_message

Every new message bumps the session's `updated_at` so the session list stays ordered by most-recently-active without a separate UPDATE from the application.

---

### Trigger 4 — auto_title_session

```sql
IF NEW.role = 'user' THEN
    UPDATE chat_sessions
    SET    title = LEFT(NEW.content, 60)
    WHERE  id    = NEW.session_id
      AND  title IS NULL;
END IF;
```
Only fires for user messages (not assistant responses). `AND title IS NULL` means the title is set exactly once — from the first user message. No subsequent messages can overwrite it.

---

## 003_procedures.sql

---

### hybrid_search

The most important function in the codebase. Runs two independent retrieval strategies and merges their ranked results with Reciprocal Rank Fusion (RRF).

```sql
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding  VECTOR(768),
    query_text       TEXT,
    target_user_id   UUID,
    target_doc_id    UUID  DEFAULT NULL,
    top_k            INT   DEFAULT 20,
    rrf_k            INT   DEFAULT 60
)
```
- `query_embedding` — 768-d BGE-base-en-v1.5 embedding of the user's question.
- `query_text` — raw text for the full-text search path. Both are needed because vector and keyword search operate on different representations of the query.
- `target_doc_id DEFAULT NULL` — if provided, restricts to one document; NULL searches all the user's documents.
- `rrf_k DEFAULT 60` — damping constant from the original RRF paper. Higher values flatten score differences; lower values amplify the top ranks.

```sql
LANGUAGE SQL STABLE SECURITY DEFINER
```
`LANGUAGE SQL` — the function body is plain SQL CTEs. SQL functions can be inlined by the query planner, making them faster than PL/pgSQL equivalents.

`STABLE` — reads but does not modify the database. Lets the planner cache results within a transaction.

`SECURITY DEFINER` — runs as the table owner (bypasses RLS on chunks). User isolation is enforced by the `target_user_id` filter inside the body — equivalent isolation without exposing RLS to the planner.

---

#### CTE 1: vector_hits

```sql
ROW_NUMBER() OVER (ORDER BY embedding <=> query_embedding) AS v_rank
...
LIMIT top_k * 2
```
`<=>` is pgvector's cosine distance operator. `ROW_NUMBER()` assigns a rank to each result. Fetching `top_k * 2` gives the RRF merge more candidates — a chunk that ranks low in vector search but high in FTS would be missed if we only fetched `top_k`.

`AND embedding IS NOT NULL` — safety guard. A chunk without an embedding can't be ranked and would cause an error.

---

#### CTE 2: fts_hits

```sql
FROM chunks, websearch_to_tsquery('summarizer_fts', query_text) AS query
WHERE fts_vector @@ query
ORDER BY ts_rank_cd(fts_vector, query, 32) DESC
```
`websearch_to_tsquery` handles natural language input gracefully — phrases, AND logic, and it never throws on unusual characters the way `to_tsquery` does.

`@@` is the full-text match operator, accelerated by the GIN index on `fts_vector`.

`ts_rank_cd` (cover density ranking) weights term matches that appear close together in the text more highly than scattered matches. The `32` argument normalises by document length so longer documents don't automatically outrank shorter ones.

The cross join `FROM chunks, websearch_to_tsquery(...) AS query` is just a way to alias the query object so it's reusable across the SELECT and WHERE clauses.

---

#### CTE 3: rrf (Reciprocal Rank Fusion)

```sql
FULL OUTER JOIN fts_hits f ON v.id = f.id
```
`FULL OUTER JOIN` includes rows from **either** list, not just rows in both. A chunk that ranks #1 in vector search but wasn't found by FTS still appears.

```sql
COALESCE(v.v_rank, (top_k * 2 + 1)::BIGINT) AS vector_rank,
COALESCE(1.0 / (rrf_k + v.v_rank), 0.0) +
COALESCE(1.0 / (rrf_k + f.f_rank), 0.0) AS rrf_score
```
If a chunk only appears in one list, its rank in the missing list is `top_k * 2 + 1` (one beyond the last fetched rank), giving it a small non-zero contribution: `1 / (60 + 41) ≈ 0.01`.

**The RRF formula: score = 1/(k + rank_A) + 1/(k + rank_B)**
- Rank #1 in both lists: `1/61 + 1/61 = 0.033` — highest possible.
- Rank #1 in vector only: `1/61 + 0 = 0.016`.
- Rank #20 in both: `1/80 + 1/80 = 0.025` — still competitive.

RRF works because it only uses ranks, not raw scores — so vector distances (0–2) and FTS scores (0–1) are always comparable.

---

### get_session_context

```sql
WITH ranked AS (
    SELECT role, content, created_at,
           ROW_NUMBER() OVER (ORDER BY created_at DESC) AS rn
    FROM chat_history WHERE session_id = p_session_id
)
SELECT role, content, created_at FROM ranked WHERE rn <= p_limit ORDER BY created_at ASC;
```
`ROW_NUMBER() ... DESC` numbers rows newest-first. `WHERE rn <= p_limit` keeps only the most recent N. `ORDER BY ... ASC` re-sorts oldest-first for the final output because the Gemini API expects messages in chronological order.

---

### document_retrieval_stats

```sql
PERCENT_RANK() OVER (ORDER BY avg_score)     AS score_percentile,
avg_latency_ms - AVG(avg_latency_ms) OVER () AS latency_vs_avg_ms
```
`PERCENT_RANK()` returns a value 0–1 representing how this session's quality compares to all sessions for this document. `AVG(...) OVER ()` with an empty window clause aggregates across all rows — subtracting it shows how far above or below average each session's latency is.

---

## 004_views.sql

---

### document_stats (Materialized View)

Stores its result set as a physical table — instant to read, refreshed explicitly. `REFRESH MATERIALIZED VIEW CONCURRENTLY` rebuilds in the background without blocking reads. A unique index on `document_id` is required for concurrent refresh.

```sql
SUM(daily_messages) OVER (
    PARTITION BY document_id
    ORDER BY day
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
) AS rolling_7d_messages
```
Sliding window: sums the current day and the 6 days before it, per document. `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` is exactly 7 rows.

```sql
daily_messages - LAG(daily_messages, 1, 0) OVER (PARTITION BY document_id ORDER BY day) AS day_over_day_delta
```
`LAG(column, 1, 0)` returns the previous row's value, with `0` as default if there's no prior row. Subtracting it gives the day-over-day change in usage.

```sql
RANK() OVER (PARTITION BY t.user_id ORDER BY t.total_messages DESC) AS usage_rank
```
Ranks each document within its user's library by query count. `PARTITION BY user_id` means user A's rank-1 document is independent of user B's. `RANK()` allows ties — two documents with equal counts both get rank 1, next rank is 3.

```sql
LEFT JOIN LATERAL (
    SELECT rolling_7d_messages, day_over_day_delta
    FROM rolling WHERE document_id = t.document_id
    ORDER BY day DESC LIMIT 1
) r ON true
```
`LATERAL` lets the subquery reference `t.document_id` from the outer query — without it you can't use outer columns inside a FROM subquery. `ORDER BY day DESC LIMIT 1` picks only the most recent day's rolling stats per document.

---

### user_activity_summary

```sql
NTILE(4) OVER (ORDER BY total_queries DESC) AS engagement_quartile
```
Divides users into 4 equal-sized buckets by query count. Quartile 1 = top 25% most active. Useful for segmentation without verbose CASE WHEN logic.

---

### chunk_quality_report

Joins each assistant message's `source_chunks` JSONB array back to the `chunks` table via a LATERAL unnest — tracks which chunks are actually retrieved and how they score over time. Chunks with high retrieval frequency but low average score are candidates for re-chunking.

---

## 005_rls.sql

Row-Level Security ensures every user can only read and write their own rows. `auth.uid()` returns the UUID from the current user's JWT — set automatically by Supabase when a user logs in. Service-role calls from the backend bypass RLS automatically.

```sql
CREATE POLICY documents_select ON documents FOR SELECT USING (user_id = auth.uid());
```
`USING` filters visible rows. A user who queries without matching rows sees zero results, not an error.

```sql
CREATE POLICY documents_insert ON documents FOR INSERT WITH CHECK (user_id = auth.uid());
```
`WITH CHECK` validates the row being written. A user cannot insert a document with someone else's `user_id` — the database rejects it.

Chunks have no UPDATE policy — they are immutable after ingestion. Chat history has no DELETE policy — messages can only be removed by deleting the parent session, which cascades. Both are intentional.

```sql
GRANT EXECUTE ON FUNCTION hybrid_search(VECTOR(768), TEXT, UUID, UUID, INT, INT) TO authenticated;
```
Even though `hybrid_search` is `SECURITY DEFINER`, the `authenticated` role still needs explicit permission to call it — otherwise logged-in users get "permission denied" from Supabase's RPC endpoint.

---

# Backend Core

---

## config.py

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    gemini_api_key: str
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    database_url: str
    ...
```
`pydantic_settings` loads configuration from environment variables and `.env` files, and validates types at startup. If `GEMINI_API_KEY` is missing, the app fails immediately with a clear error instead of crashing at runtime when the key is first used.

Each attribute maps to an environment variable of the same name (uppercased). `str` fields are required; fields with defaults (like `chunk_similarity_threshold: float = 0.5`) are optional and can be overridden via environment variables without a code change.

```python
model_config = SettingsConfigDict(env_file=".env", extra="ignore")
```
`extra="ignore"` silently drops unknown environment variables — no errors on shared CI environments that have unrelated keys.

---

## db/client.py

```python
_client: Client | None = None

def get_supabase() -> Client:
    global _client
    if _client is None:
        _client = create_client(settings.supabase_url, settings.supabase_service_role_key)
    return _client

supabase: Client = get_supabase()
```
Lazy singleton — the client is only created when first accessed, not at import time. `supabase_service_role_key` bypasses RLS; this is intentional for the backend which needs to write rows on behalf of users. This key must never reach the frontend.

`supabase` at module level is a convenience alias — every other module does `from app.db.client import supabase`.

---

## main.py

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://mr-summarizer.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
Without CORS headers, browsers block JavaScript on `localhost:3000` from calling the backend at a different origin. The origins list is explicit — using `"*"` would be insecure in production because it allows any site to make credentialed requests.

```python
@app.get("/health")
def health():
    return {"status": "ok", "service": "mr-summarizer"}
```
Railway and other platforms poll this endpoint before routing traffic to the instance. Returning `status: ok` is the minimal signal they need.

---

# Ingestion Pipeline

Runs once per document upload. Takes a raw PDF and ends with embedded, indexed chunks in the database.

---

## ingestion/parser.py

```python
import fitz  # PyMuPDF
```
`fitz` is the Python binding for MuPDF. Despite the import name, it's installed as `pymupdf`. It handles complex PDF layouts, multi-column text, and embedded fonts better than alternatives like `pdfplumber`.

```python
for page_num, page in enumerate(doc, start=1):
    text = page.get_text("text")
    cleaned = text.strip()
    if cleaned:
        pages.append(...)
```
`enumerate(doc, start=1)` gives 1-based page numbers matching what users see in a PDF viewer. `if cleaned:` skips pages that are entirely images (scanned docs), page-break markers, or blank pages — empty chunks harm retrieval quality.

```python
def full_text(pages: list[dict]) -> str:
    return "\n\n".join(p["content"] for p in pages)
```
Joins all pages with double newlines. Used by NER and the classifier, which need the whole document as one string.

---

## ingestion/chunker.py

```python
_ENCODER = tiktoken.get_encoding("cl100k_base")
```
`cl100k_base` is the BPE vocabulary used by GPT-4 and most modern LLMs. Using it for token counting gives accurate estimates of how many tokens a chunk will consume in the model's context window.

```python
_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
```
A lightweight model used only for detecting topic boundaries, not for retrieval. It's 6× faster than the retrieval model and accurate enough for comparing adjacent sentences.

```python
def _cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0
```
`1e-8` guards against division by zero for zero-vectors. Since we use `normalize_embeddings=True` below, all vectors are unit-length — `dot(a, b)` equals cosine similarity directly. The full formula is kept for correctness with any input.

```python
boundaries = {0}
for i, sim in enumerate(similarities):
    if sim < threshold:
        boundaries.add(i + 1)
boundaries.add(len(sentences))
```
`boundaries` is a Python set — automatically unique and unordered. We seed it with `{0}` (the start) and `len(sentences)` (the end) unconditionally, then add interior split points where similarity drops. `sorted()` gives boundary positions in order for slicing.

The merge pass uses a buffer pattern: accumulate chunks until the buffer reaches `min_tok` tokens, then flush and start a new buffer. This prevents very short chunks (like a section heading that became its own boundary) from being indexed alone.

---

## ingestion/ner.py

```python
_LABEL_MAP = {
    "PERSON": "people",
    "GPE": "locations",
    "LOC": "locations",
    ...
}
```
Maps spaCy's internal label names to cleaner JSONB keys. `GPE` (Geo-Political Entity) and `LOC` both map to `"locations"` — merging them avoids a confusing split between "San Francisco" (GPE) and "the Amazon rainforest" (LOC).

```python
_MAX_CHARS = 100_000
doc = nlp(text[:_MAX_CHARS])
```
spaCy's transformer model processes text in windows — beyond ~100k characters, memory usage grows steeply and accuracy degrades. 100k characters covers most of a typical PDF's most entity-rich sections (introduction, key sections).

```python
entities[key].add(cleaned)
```
`entities` is a `defaultdict(set)`. Using a set means "Anthropic" appearing 50 times in a document only appears once in the output. `sorted(v)` converts sets to lists before returning — JSON arrays have a defined order; sets don't.

---

## ingestion/classifier.py

```python
_CLASSIFIER = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
```
BART-large-MNLI frames classification as natural language inference: "Does this text entail it is about [legal documents]?" The model scores each label as a probability of entailment — no task-specific training data needed.

`device=-1` forces CPU. For one classification per upload (not per query), CPU is fast enough and avoids GPU memory overhead.

```python
result = _get_classifier()(sample, _LABELS, multi_label=False)
return result["labels"][0]
```
`multi_label=False` — one label wins. `result["labels"]` is sorted by probability descending, so `[0]` is the top result. Only the first 2000 characters are passed — the BART model has a 1024-token limit anyway, and the beginning of a document (abstract, intro) is the most informative for classification.

---

## ingestion/embedder.py

```python
_MODEL_NAME = "BAAI/bge-base-en-v1.5"
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
```
BGE models are trained with an instruction prefix on the query side — omitting it at query time degrades retrieval accuracy by ~2–3%. Documents are embedded without any prefix. This asymmetry is intentional; always use `embed_documents` at ingestion time and `embed_query` at search time.

`_BATCH_SIZE = 32` is conservative — safe on CPU with limited RAM. Increase on machines with more memory for faster ingestion.

```python
def embed_documents(texts: list[str]) -> list[list[float]]:
    embeddings: np.ndarray = _get_model().encode(
        texts, batch_size=_BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False
    )
    return embeddings.tolist()
```
`normalize_embeddings=True` L2-normalises each vector to unit length. Required for cosine similarity to work correctly — un-normalised vectors would cause magnitude to skew results. `.tolist()` converts numpy arrays to plain Python lists for JSON serialisation.

---

## ingestion/pipeline.py

```python
async def ingest_document(file_path, user_id, filename, file_url) -> dict:
```
`async def` — yields control to the event loop during I/O, so FastAPI can serve other requests while a document is being ingested.

```python
entity_tags, doc_type = await asyncio.gather(
    loop.run_in_executor(None, extract_entities, text),
    loop.run_in_executor(None, classify_document, text),
)
```
NER and classification are CPU-bound synchronous functions. `run_in_executor` moves each to a thread pool so they don't block the async event loop. `asyncio.gather` runs both concurrently — cuts the analysis phase roughly in half.

```python
doc_result = supabase.table("documents").insert({...}).execute()
document_id: str = doc_result.data[0]["id"]
```
The document row is inserted before chunks so foreign keys have a target. `.data[0]["id"]` retrieves the UUID the database generated for this row.

```python
for i in range(0, len(records), _CHUNK_INSERT_BATCH):
    supabase.table("chunks").insert(records[i : i + _CHUNK_INSERT_BATCH]).execute()
```
Chunks are inserted in batches of 100 to stay within Supabase's ~1 MB request body limit. A document with 500 chunks would exceed the limit in a single request.

---

# Retrieval Pipeline

Four stages at query time: multi-query retrieval → hybrid SQL search → cross-encoder re-ranking → contextual compression. Each stage narrows and refines the candidate pool.

---

## retrieval/search.py

```python
def hybrid_search(query, user_id, document_id=None, top_k=None) -> list[dict]:
    query_vector = embed_query(query)
    result = supabase.rpc("hybrid_search", {...}).execute()
    return result.data or []
```
`embed_query` applies the BGE query prefix before embedding — the asymmetric prefix is what makes retrieval accurate. `supabase.rpc()` calls the PostgreSQL stored procedure via Supabase's PostgREST API. The `VECTOR(768)` parameter is serialised as a JSON array of 768 floats.

---

## retrieval/multi_query.py

```python
def _generate_variants(query: str, n: int) -> list[str]:
    response = _get_model().generate_content(
        f"Generate {n} semantically varied reformulations of this search query..."
    )
    lines = response.text.strip().splitlines()
    return [line.strip() for line in lines if line.strip()]
```
Uses Gemini 2.0 Flash to generate `n` rephrased versions of the query. Different phrasings hit different vocabulary in the documents — a question about "termination notice" might also need to match chunks that say "contract cancellation policy".

```python
best: dict[str, dict] = {}
for variant in all_queries:
    chunks = hybrid_search(variant, user_id, document_id, top_k=k)
    for chunk in chunks:
        cid = chunk["chunk_id"]
        if cid not in best or chunk["rrf_score"] > best[cid]["rrf_score"]:
            best[cid] = chunk
```
`best` is keyed by `chunk_id`. Each chunk is only kept at its highest RRF score across all variants. This deduplication-with-best-score approach gives the reranker the most optimistic view of each chunk's relevance before it makes its final call.

---

## retrieval/reranker.py

```python
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```
MS MARCO is Microsoft's large-scale passage-ranking dataset (real Bing queries + human relevance labels). `MiniLM-L-6` is a distilled 6-layer model — much smaller than the full cross-encoder while retaining ~95% of its accuracy.

**Why a cross-encoder is more accurate than a bi-encoder:**
A bi-encoder (BGE) encodes the query and chunk independently — they never see each other. A cross-encoder concatenates query + chunk into a single input and produces one relevance score. The model can reason about the interaction between them, which is more accurate but slower (one forward pass per pair). We apply it only to the top-K pool, not all chunks.

```python
pairs = [(query, chunk["content"]) for chunk in chunks]
scores: list[float] = _get_model().predict(pairs).tolist()
```
`model.predict(pairs)` batches all pairs in one call — more efficient than calling per-pair. `.tolist()` converts numpy to a Python list.

```python
return [{**chunk, "rerank_score": score} for score, chunk in scored[:k]]
```
`{**chunk, "rerank_score": score}` spreads all existing chunk keys into a new dict and adds `rerank_score` — non-destructive, the original chunk is not mutated.

---

## retrieval/compressor.py

```python
def compress_chunks(query: str, chunks: list[dict]) -> list[dict]:
    for chunk in chunks:
        response = _get_model().generate_content(
            f"Question: {query}\n\n"
            f"Document excerpt:\n{chunk['content']}\n\n"
            "Extract only the sentences from the excerpt that directly help "
            "answer the question. Preserve exact wording. "
            "If nothing is relevant, reply with an empty response."
        )
        text = response.text.strip()
        if text:
            compressed.append({**chunk, "content": text})
```
"Preserve exact wording" ensures citations remain verifiable. "If nothing is relevant, reply with an empty response" lets us drop chunks that the cross-encoder passed but that genuinely have no answer content — the `if text:` check is the filter. This is the last defence against hallucination: if no chunk compresses to anything useful, Gemini receives no context and the system prompt tells it to say so.

---

# Generation Layer

---

## generation/prompts.py

```python
_DOC_TYPE_INSTRUCTIONS: dict[str, str] = {
    "legal": "Be precise about obligations, rights, and liabilities. Use the exact terminology...",
    "financial": "Be precise with figures, dates, and financial terminology. Do not round numbers.",
    ...
}
```
Different document types need different generation behaviour. For financial documents Gemini must not round numbers. For legal documents a paraphrase could change the legal meaning. These instructions are prepended to the user's query.

```python
context_blocks = "\n\n---\n\n".join(
    "[Page {pages}]\n{content}".format(...)
    for chunk in chunks
)
```
Each chunk is formatted with its page number(s) as a header. The `---` separator makes chunk boundaries clear in the prompt. Page numbers enable Gemini to write citations like "According to page 3..." that the frontend can eventually highlight.

The prompt structure is: `instruction → context → question → grounding constraint`. Putting the constraint last ("Answer based only on the document context above") keeps it nearest to where Gemini starts generating.

---

## generation/gemini.py

```python
import google.generativeai as genai

CHAT_MODEL = "gemini-2.0-flash"
```
Gemini 2.0 Flash is Google's fast general-purpose model — free tier, 1M token context window, suitable for production at scale.

```python
def _get_model() -> genai.GenerativeModel:
    global _MODEL
    if _MODEL is None:
        genai.configure(api_key=settings.gemini_api_key)
        _MODEL = genai.GenerativeModel(CHAT_MODEL, system_instruction=system_prompt())
    return _MODEL
```
Lazy singleton — model is initialised on first call. `system_instruction` is Gemini's dedicated slot for the system prompt, kept separate from the message history.

```python
def _convert_history(history: list[dict]) -> list[dict]:
    converted = []
    for msg in history:
        role = "model" if msg["role"] == "assistant" else msg["role"]
        if role in ("user", "model"):
            converted.append({"role": role, "parts": [{"text": msg["content"]}]})
    return converted
```
Gemini uses "model" where the OpenAI format uses "assistant". This converts the history fetched from the database (OpenAI-style) to the format Gemini's API expects.

```python
def stream_answer(query, chunks, history, doc_type="general") -> Generator[str, None, None]:
    chat = model.start_chat(history=gemini_history)
    response = chat.send_message(prompt, stream=True, generation_config=...)
    for chunk in response:
        if chunk.text:
            yield chunk.text
```
`start_chat` creates a session with the conversation history. Streaming yields tokens as they arrive — the first token reaches the user's browser within ~200ms rather than waiting for the full response. The caller (`chat.py`) iterates over yielded tokens and sends them via SSE.

```python
def summarise(chunks: list[dict], doc_type: str = "general") -> str:
    response = model.generate_content(prompt, generation_config=...)
    return response.text
```
Summarisation is blocking (not streaming) because it's triggered by a button click where the user expects to wait, and the full text needs to be returned as a single JSON response.

---

# API Routes

---

## api/routes/documents.py

```python
def _require_user(x_user_id: str | None) -> str:
    if not x_user_id:
        raise HTTPException(401, "X-User-Id header required")
    return x_user_id
```
Every route calls this guard. In production this would be replaced by JWT middleware — `X-User-Id` is a placeholder that's easy to swap later.

```python
with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
    tmp.write(content)
    tmp_path = tmp.name
try:
    result = await ingest_document(tmp_path, ...)
finally:
    os.unlink(tmp_path)
```
PyMuPDF requires a file path, not an in-memory buffer. The `try/finally` guarantees the temp file is deleted even if ingestion throws an exception — prevents disk leaks.

```python
result = supabase.table("document_stats").select("*").eq("user_id", user_id).order("usage_rank").execute()
```
The document list queries the materialized view `document_stats`, not the raw `documents` table. The frontend gets pre-computed `usage_rank`, `rolling_7d_messages`, and `avg_retrieval_score` at zero aggregation cost.

```python
.order("metadata->>chunk_index")
```
`->>` extracts a JSONB field as text. This orders chunks in their original document order (chunk 0, 1, 2...) before passing them to Gemini for summarisation — a summary from randomly ordered chunks would be incoherent.

---

## api/routes/chat.py

```python
pool = multi_query_retrieve(req.query, user_id, doc_id)
top = rerank(req.query, pool)
context = compress_chunks(req.query, top)
```
The three-stage retrieval pipeline in 3 lines:
1. `multi_query_retrieve` — generates variants, retrieves for each, deduplicates (~20–80 candidates)
2. `rerank` — cross-encoder scores, keeps top 5
3. `compress_chunks` — strips irrelevant sentences from the top 5

```python
async def event_stream() -> AsyncGenerator[str, None]:
    tokens: list[str] = []
    for token in stream_answer(req.query, context, history, doc_type):
        tokens.append(token)
        yield f"data: {json.dumps({'token': token})}\n\n"
```
**Server-Sent Events (SSE) format**: each message prefixed with `data: ` and terminated with `\n\n`. The browser's `EventSource` API parses this automatically and fires an event per token. `json.dumps` wraps each token so special characters like `\n` don't break the SSE framing.

```python
    full_response = "".join(tokens)
    supabase.table("chat_history").insert({
        "role": "assistant",
        "content": full_response,
        "source_chunks": cited_ids,
        "retrieval_score": top_score,
        "latency_ms": latency_ms,
    }).execute()
    yield "data: [DONE]\n\n"
```
The assistant message is persisted after the full response is assembled — we need the complete text to store it. `"".join(tokens)` is efficient Python (much faster than `+=` in a loop). `source_chunks` stores the chunk UUIDs that fed this response; the `chunk_quality_report` view uses these to track which chunks are actually useful over time.

`"data: [DONE]\n\n"` is the SSE termination signal — the frontend closes the connection and stops showing the loading indicator.

```python
return StreamingResponse(event_stream(), media_type="text/event-stream")
```
`StreamingResponse` wraps the async generator and sets `Content-Type: text/event-stream`. FastAPI and uvicorn flush each `yield` to the TCP socket as it arrives instead of buffering the full response.

---

*End of documentation.*
