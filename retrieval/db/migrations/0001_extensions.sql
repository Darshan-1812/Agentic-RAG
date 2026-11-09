-- Phase 2 foundation: required PostgreSQL extensions.
-- Use a ParadeDB PostgreSQL 17 image; vanilla postgres:17 does not ship pg_search.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;
