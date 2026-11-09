-- CalibRAG Member 2 core ingestion schema.
-- Every tenant-owned table carries workspace_id so retrieval queries can enforce
-- isolation in SQL rather than filtering after the fact.

CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id TEXT PRIMARY KEY,
    name TEXT,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT workspaces_id_nonempty CHECK (btrim(workspace_id) <> '')
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id UUID PRIMARY KEY,
    workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    doc_title TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    byte_size BIGINT NOT NULL,
    page_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'ready',
    storage_path TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT documents_hash_format CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT documents_byte_size_nonnegative CHECK (byte_size >= 0),
    CONSTRAINT documents_page_count_nonnegative CHECK (page_count >= 0),
    CONSTRAINT documents_status_valid CHECK (
        status IN ('pending', 'processing', 'ready', 'failed', 'deleted')
    ),
    CONSTRAINT documents_workspace_hash_unique UNIQUE (workspace_id, content_hash),
    -- Referenced by the composite (doc_id, workspace_id) foreign keys below.
    CONSTRAINT documents_doc_workspace_unique UNIQUE (doc_id, workspace_id)
);

CREATE INDEX IF NOT EXISTS documents_workspace_created_idx
    ON documents (workspace_id, created_at DESC);
CREATE INDEX IF NOT EXISTS documents_workspace_status_idx
    ON documents (workspace_id, status);

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id UUID NOT NULL,
    workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    page_number INTEGER NOT NULL DEFAULT 1,
    section_hierarchy TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    has_table BOOLEAN NOT NULL DEFAULT FALSE,
    table_markdown TEXT,
    char_count INTEGER NOT NULL,
    token_count_est INTEGER NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'pdf',
    content_hash TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT document_chunks_content_nonempty CHECK (btrim(content) <> ''),
    CONSTRAINT document_chunks_index_nonnegative CHECK (chunk_index >= 0),
    CONSTRAINT document_chunks_page_valid CHECK (page_number >= 1),
    CONSTRAINT document_chunks_char_count_nonnegative CHECK (char_count >= 0),
    CONSTRAINT document_chunks_token_count_nonnegative CHECK (token_count_est >= 0),
    CONSTRAINT document_chunks_chunk_hash_format CHECK (
        content_hash IS NULL OR content_hash ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT document_chunks_doc_workspace_fk
        FOREIGN KEY (doc_id, workspace_id)
        REFERENCES documents (doc_id, workspace_id)
        ON DELETE CASCADE,
    CONSTRAINT document_chunks_doc_index_unique
        UNIQUE (doc_id, workspace_id, chunk_index),
    -- Referenced by chunk_embeddings so workspace_id cannot be swapped.
    CONSTRAINT document_chunks_chunk_workspace_unique
        UNIQUE (chunk_id, workspace_id)
);

CREATE INDEX IF NOT EXISTS document_chunks_workspace_doc_idx
    ON document_chunks (workspace_id, doc_id, chunk_index);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id UUID PRIMARY KEY,
    workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
    doc_id UUID,
    source_filename TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    error TEXT,
    stats JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    CONSTRAINT ingestion_jobs_status_valid CHECK (
        status IN ('queued', 'running', 'completed', 'failed', 'cancelled')
    )
);

CREATE INDEX IF NOT EXISTS ingestion_jobs_workspace_status_idx
    ON ingestion_jobs (workspace_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS ingestion_jobs_workspace_doc_idx
    ON ingestion_jobs (workspace_id, doc_id);
