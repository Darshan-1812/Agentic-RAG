"""Database foundation for the Member 2 ingestion/retrieval layer."""
from retrieval.db.connection import (
    Database,
    DatabaseHealth,
    DatabaseHealthError,
    DatabaseUnavailable,
    ExtensionInfo,
)
from retrieval.db.migrate import (
    DEFAULT_MIGRATIONS_DIR,
    Migration,
    MigrationError,
    MigrationRun,
    apply_migrations,
    discover_migrations,
    get_schema_status,
    migration_checksums,
)
from retrieval.db.repository import IngestionRepository, WorkspaceRequiredError

__all__ = [
    "Database",
    "DatabaseHealth",
    "DatabaseHealthError",
    "DatabaseUnavailable",
    "ExtensionInfo",
    "DEFAULT_MIGRATIONS_DIR",
    "Migration",
    "MigrationError",
    "MigrationRun",
    "apply_migrations",
    "discover_migrations",
    "get_schema_status",
    "migration_checksums",
    "IngestionRepository",
    "WorkspaceRequiredError",
]
