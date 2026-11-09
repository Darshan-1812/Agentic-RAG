-- Phase 2 foundation for Phase 3/4 retrieval. No embedding generation or BM25
-- query code is implemented in this phase; only the persistent structures are
-- created so later phases do not require an incompatible schema rewrite.

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    embedding vector(1024) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chunk_embeddings_chunk_workspace_unique UNIQUE (chunk_id, workspace_id),
    CONSTRAINT chunk_embeddings_chunk_fk
        FOREIGN KEY (chunk_id, workspace_id)
        REFERENCES document_chunks (chunk_id, workspace_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS chunk_embeddings_workspace_idx
    ON chunk_embeddings (workspace_id);

CREATE INDEX IF NOT EXISTS chunk_embeddings_hnsw_idx
    ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ParadeDB pg_search BM25 index. The key_field must be a column in the index.
CREATE INDEX IF NOT EXISTS document_chunks_bm25_idx
    ON document_chunks USING bm25 (chunk_id, content)
    WITH (key_field = 'chunk_id');
