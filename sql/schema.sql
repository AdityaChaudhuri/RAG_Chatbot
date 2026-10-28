-- Mr. Summarizer schema
-- Runs automatically the first time the Docker database starts.
-- To reset everything: docker compose down -v && docker compose up -d

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per news article
CREATE TABLE IF NOT EXISTS articles (
    id          SERIAL PRIMARY KEY,
    source_id   TEXT UNIQUE NOT NULL,      -- stable id from the dataset (we use the URL)
    title       TEXT NOT NULL,
    url         TEXT,
    body        TEXT NOT NULL
);

-- Articles are split into overlapping chunks; retrieval works on chunks.
-- Each chunk carries two search representations:
--   embedding : dense vector for semantic search (pgvector)
--   tsv       : full-text index for keyword search (Postgres built-in)
CREATE TABLE IF NOT EXISTS chunks (
    id           SERIAL PRIMARY KEY,
    article_id   INT NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,
    content      TEXT NOT NULL,
    embedding    vector(384),             -- must match the embedding model's dimension
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (article_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin
    ON chunks USING gin (tsv);

-- ---------------------------------------------------------------
-- Knowledge graph
--   nodes : entities (companies, people, places, products, events)
--   edges : two entities mentioned in the same chunk, weighted by
--           how many chunks they share
--   chunk_entities links each node back to the text it came from
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entities (
    id      SERIAL PRIMARY KEY,
    key     TEXT NOT NULL,                -- normalised lowercase name used for matching
    name    TEXT NOT NULL,                -- display name (first form seen)
    label   TEXT NOT NULL,                -- spaCy label: ORG, PERSON, GPE, PRODUCT, EVENT
    UNIQUE (key, label)
);

CREATE TABLE IF NOT EXISTS chunk_entities (
    chunk_id   INT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    entity_id  INT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (chunk_id, entity_id)
);
CREATE INDEX IF NOT EXISTS chunk_entities_entity ON chunk_entities (entity_id);

CREATE TABLE IF NOT EXISTS entity_edges (
    src     INT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    dst     INT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    weight  INT NOT NULL DEFAULT 1,
    PRIMARY KEY (src, dst),
    CHECK (src < dst)                     -- store each undirected edge once
);
CREATE INDEX IF NOT EXISTS entity_edges_dst ON entity_edges (dst);
