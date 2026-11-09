"""Small, dependency-free SQL migration runner.

Migrations are numbered ``*.sql`` files in ``retrieval/db/migrations``. Each
file is applied once inside its own transaction and recorded in
``schema_migrations``. Checksums prevent silent edits to already-applied files.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from retrieval.db.connection import Database


DEFAULT_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
MIGRATION_LOCK_ID = 0x43414C49  # "CALI"
_MIGRATION_FILE_RE = re.compile(r"^(?P<version>\d+)_(?P<name>[A-Za-z0-9_.-]+)\.sql$")


class MigrationError(RuntimeError):
    """Raised when migrations are malformed or drift from applied state."""


@dataclass(frozen=True)
class Migration:
    version: str
    name: str
    path: Path
    checksum: str

    @property
    def sql(self) -> str:
        return self.path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class MigrationRun:
    applied: Tuple[str, ...] = ()
    already_applied: Tuple[str, ...] = ()

    @property
    def applied_count(self) -> int:
        return len(self.applied)


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_migrations(
    migrations_dir: Optional[Union[Path, str]] = None,
) -> List[Migration]:
    """Return migrations sorted by numeric version, or an empty list."""
    directory = Path(migrations_dir) if migrations_dir else DEFAULT_MIGRATIONS_DIR
    if not directory.is_dir():
        return []

    migrations: List[Migration] = []
    seen: Dict[str, Path] = {}
    for path in sorted(directory.glob("*.sql")):
        match = _MIGRATION_FILE_RE.match(path.name)
        if not match:
            raise MigrationError(
                f"Migration filename must look like 0001_name.sql: {path.name}"
            )
        version = match.group("version")
        if version in seen:
            raise MigrationError(
                f"Duplicate migration version {version}: {seen[version]} and {path}"
            )
        seen[version] = path
        migrations.append(
            Migration(
                version=version,
                name=match.group("name"),
                path=path,
                checksum=_checksum(path),
            )
        )

    migrations.sort(key=lambda item: int(item.version))
    return migrations


async def _ensure_migration_table(connection) -> None:
    await connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            checksum CHAR(64) NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


async def apply_migrations(
    database: Database,
    migrations_dir: Optional[Union[Path, str]] = None,
) -> MigrationRun:
    """Apply pending migrations and return a summary.

    The runner takes a PostgreSQL advisory lock so two worker processes cannot
    apply the same migration concurrently.
    """
    migrations = discover_migrations(migrations_dir)
    if database.is_connected is False:
        await database.connect()

    applied_versions: List[str] = []
    skipped_versions: List[str] = []

    async with database.acquire() as connection:
        await _ensure_migration_table(connection)
        await connection.execute("SELECT pg_advisory_lock($1)", MIGRATION_LOCK_ID)
        try:
            rows = await connection.fetch(
                "SELECT version, checksum FROM schema_migrations ORDER BY version"
            )
            applied = {str(row["version"]): str(row["checksum"]) for row in rows}

            for migration in migrations:
                expected = applied.get(migration.version)
                if expected is not None:
                    if expected != migration.checksum:
                        raise MigrationError(
                            f"Applied migration {migration.version}_{migration.name} "
                            "has been modified; create a new migration instead"
                        )
                    skipped_versions.append(migration.version)
                    continue

                async with connection.transaction():
                    await connection.execute(migration.sql)
                    await connection.execute(
                        "INSERT INTO schema_migrations (version, name, checksum) "
                        "VALUES ($1, $2, $3)",
                        migration.version,
                        migration.name,
                        migration.checksum,
                    )
                applied_versions.append(migration.version)
        finally:
            await connection.execute("SELECT pg_advisory_unlock($1)", MIGRATION_LOCK_ID)

    return MigrationRun(
        applied=tuple(applied_versions),
        already_applied=tuple(skipped_versions),
    )


async def get_schema_status(
    database: Database,
) -> Dict[str, object]:
    """Return applied migration versions and the maximum schema version."""
    if database.is_connected is False:
        await database.connect()
    async with database.acquire() as connection:
        exists = await connection.fetchval(
            "SELECT EXISTS ("
            "  SELECT 1 FROM information_schema.tables "
            "  WHERE table_schema = 'public' AND table_name = 'schema_migrations'"
            ")"
        )
        if not exists:
            return {"schema_version": None, "applied": ()}
        rows = await connection.fetch(
            "SELECT version, name, checksum, applied_at "
            "FROM schema_migrations ORDER BY version"
        )
    return {
        "schema_version": str(rows[-1]["version"]) if rows else None,
        "applied": tuple(dict(row) for row in rows),
    }


def migration_checksums(
    migrations_dir: Optional[Union[Path, str]] = None,
) -> Sequence[Tuple[str, str]]:
    """Return ``(version, checksum)`` pairs for tooling and tests."""
    return tuple(
        (migration.version, migration.checksum)
        for migration in discover_migrations(migrations_dir)
    )
