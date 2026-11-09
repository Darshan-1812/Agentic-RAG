"""CLI: apply retrieval migrations and inspect database health.

Usage:
    python -m retrieval.db
    python -m retrieval.db --migrate
    python -m retrieval.db --health
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict

from retrieval.config import RetrievalSettings
from retrieval.db.connection import Database, DatabaseHealthError
from retrieval.db.migrate import apply_migrations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CalibRAG retrieval database utility")
    parser.add_argument("--migrate", action="store_true", help="apply pending migrations")
    parser.add_argument("--health", action="store_true", help="print extension/schema health")
    parser.add_argument("--env-file", default=None, help="optional dotenv file path")
    return parser


async def _run(apply: bool, health: bool, env_file: str | None) -> int:
    settings = RetrievalSettings.from_env(env_file=env_file)
    database = Database(settings)
    await database.connect()
    try:
        if apply:
            result = await apply_migrations(database)
            print(
                json.dumps(
                    {
                        "applied": list(result.applied),
                        "already_applied": list(result.already_applied),
                    },
                    indent=2,
                )
            )
        if health:
            report = await database.health()
            payload = asdict(report)
            payload["ok"] = report.ok
            payload["missing_extensions"] = list(report.missing_extensions)
            print(json.dumps(payload, indent=2, default=str))
            if report.missing_extensions:
                return 1
    except DatabaseHealthError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        await database.close()
    return 0


def main() -> int:
    args = _parser().parse_args()
    apply = args.migrate or not args.health
    health = args.health or not args.migrate
    return asyncio.run(_run(apply, health, args.env_file))


if __name__ == "__main__":
    raise SystemExit(main())
