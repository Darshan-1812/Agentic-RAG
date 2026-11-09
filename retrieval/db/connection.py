"""Async PostgreSQL connection management for Member 2.

The module imports ``asyncpg`` lazily so the rest of the repository can be
inspected, imported, and unit-tested without installing the database driver.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterable, Optional, Tuple

from retrieval.config import ConfigurationError, RetrievalSettings


class DatabaseUnavailable(RuntimeError):
    """Raised when the database driver or server cannot be reached."""


class DatabaseHealthError(RuntimeError):
    """Raised when a required database extension or schema is missing."""


@dataclass(frozen=True)
class ExtensionInfo:
    name: str
    installed: bool
    installed_version: Optional[str] = None
    available: bool = False
    default_version: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.installed


@dataclass(frozen=True)
class DatabaseHealth:
    connected: bool
    server_version: str = ""
    database: str = ""
    user: str = ""
    schema_version: Optional[str] = None
    extensions: Dict[str, ExtensionInfo] = field(default_factory=dict)

    @property
    def missing_extensions(self) -> Tuple[str, ...]:
        return tuple(
            name for name, info in self.extensions.items() if not info.installed
        )

    @property
    def ok(self) -> bool:
        return self.connected and not self.missing_extensions


def _import_asyncpg():
    try:
        import asyncpg  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DatabaseUnavailable(
            "asyncpg is not installed. Install retrieval/requirements-ingestion.txt "
            "before using the retrieval database layer."
        ) from exc
    return asyncpg


class Database:
    """Small wrapper around an asyncpg pool."""

    def __init__(self, settings: RetrievalSettings):
        self.settings = settings
        self._pool: Any = None

    @property
    def is_connected(self) -> bool:
        return self._pool is not None

    @property
    def pool(self) -> Any:
        if self._pool is None:
            raise DatabaseUnavailable("Database is not connected; call connect() first")
        return self._pool

    async def connect(self) -> "Database":
        if self._pool is not None:
            return self
        asyncpg = _import_asyncpg()
        try:
            self._pool = await asyncpg.create_pool(
                dsn=self.settings.database_dsn,
                min_size=self.settings.db_pool_min_size,
                max_size=self.settings.db_pool_max_size,
                command_timeout=self.settings.db_command_timeout,
                statement_cache_size=self.settings.db_statement_cache_size,
                server_settings={"application_name": "calibrag-retrieval"},
            )
        except (OSError, ConfigurationError, asyncpg.PostgresError) as exc:
            raise DatabaseUnavailable(f"Unable to connect to PostgreSQL: {exc}") from exc
        return self

    async def close(self) -> None:
        if self._pool is None:
            return
        pool, self._pool = self._pool, None
        await pool.close()

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        async with self.pool.acquire() as connection:
            yield connection

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Any]:
        async with self.acquire() as connection:
            async with connection.transaction():
                yield connection

    async def health(
        self,
        required_extensions: Iterable[str] = ("vector", "pg_search"),
    ) -> DatabaseHealth:
        """Inspect server version, installed extensions, and schema version.

        This method connects on demand and does not raise for missing extensions;
        inspect ``DatabaseHealth.ok`` or call :meth:`assert_healthy` instead.
        """
        if self._pool is None:
            await self.connect()

        async with self.acquire() as connection:
            server = await connection.fetchrow(
                "SELECT version() AS version, current_database() AS database, "
                "current_user AS user"
            )
            extension_rows = await connection.fetch(
                "SELECT extname, extversion FROM pg_extension"
            )
            available_rows = await connection.fetch(
                "SELECT name, default_version, installed_version "
                "FROM pg_available_extensions"
            )
            schema_exists = await connection.fetchval(
                "SELECT EXISTS ("
                "  SELECT 1 FROM information_schema.tables "
                "  WHERE table_schema = 'public' AND table_name = 'schema_migrations'"
                ")"
            )
            schema_version = None
            if schema_exists:
                schema_version = await connection.fetchval(
                    "SELECT max(version) FROM schema_migrations"
                )

        installed = {
            str(row["extname"]): str(row["extversion"]) for row in extension_rows
        }
        available = {
            str(row["name"]): (
                str(row["default_version"]) if row["default_version"] else None
            )
            for row in available_rows
        }

        extensions: Dict[str, ExtensionInfo] = {}
        for name in required_extensions:
            extensions[name] = ExtensionInfo(
                name=name,
                installed=name in installed,
                installed_version=installed.get(name),
                available=name in available,
                default_version=available.get(name),
            )

        return DatabaseHealth(
            connected=True,
            server_version=str(server["version"]) if server else "",
            database=str(server["database"]) if server else "",
            user=str(server["user"]) if server else "",
            schema_version=str(schema_version) if schema_version else None,
            extensions=extensions,
        )

    async def assert_healthy(
        self,
        required_extensions: Iterable[str] = ("vector", "pg_search"),
    ) -> DatabaseHealth:
        health = await self.health(required_extensions=required_extensions)
        if health.missing_extensions:
            missing = ", ".join(health.missing_extensions)
            hint = (
                "Start the ParadeDB PostgreSQL 17 service and run the retrieval "
                "migrations; missing extension(s): " + missing
            )
            raise DatabaseHealthError(hint)
        return health

    async def wait_until_ready(
        self,
        attempts: int = 10,
        delay_seconds: float = 1.0,
        required_extensions: Iterable[str] = ("vector", "pg_search"),
    ) -> DatabaseHealth:
        last_error: Optional[Exception] = None
        for attempt in range(1, max(1, attempts) + 1):
            try:
                return await self.assert_healthy(required_extensions)
            except (DatabaseUnavailable, DatabaseHealthError) as exc:
                last_error = exc
                if attempt < attempts:
                    await asyncio.sleep(delay_seconds)
        raise DatabaseUnavailable(
            f"PostgreSQL did not become ready after {attempts} attempts: {last_error}"
        )

    async def __aenter__(self) -> "Database":
        return await self.connect()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
