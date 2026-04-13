#!/usr/bin/env python3
"""Migration script from v2.x to v3.2

Migrates data from memory table to atom/entity/reference tables.

Usage:
    # Dry run (validate without executing)
    uv run python scripts/migrate_v2_to_v3.2.py --dry-run

    # Execute migration
    uv run python scripts/migrate_v2_to_v3.2.py --execute

    # With custom batch size
    uv run python scripts/migrate_v2_to_v3.2.py --execute --batch-size 500
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import Any, Optional

from surrealdb import Surreal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Migration error"""

    pass


class V2ToV3Migration:
    """Migration from v2.x to v3.2"""

    def __init__(
        self,
        url: str = "ws://localhost:18002",
        namespace: str = "memory_ns",
        database: str = "memory_db",
        username: str = "root",
        password: str = "root",
        batch_size: int = 100,
        dry_run: bool = True,
    ):
        self._url = url
        self._namespace = namespace
        self._database = database
        self._username = username
        self._password = password
        self._batch_size = batch_size
        self._dry_run = dry_run
        self._db: Optional[Surreal] = None
        self._stats = {
            "memory_records": 0,
            "atoms_created": 0,
            "entities_created": 0,
            "references_created": 0,
            "errors": 0,
        }

    async def connect(self) -> None:
        """Connect to SurrealDB"""
        self._db = Surreal(self._url)
        await self._db.connect()
        await self._db.signin({"user": self._username, "pass": self._password})
        await self._db.use(self._namespace, self._database)
        logger.info("[Migration] Connected to %s/%s", self._namespace, self._database)

    async def close(self) -> None:
        """Close connection"""
        if self._db:
            await self._db.close()
            self._db = None
            logger.info("[Migration] Disconnected")

    async def validate_schema(self) -> bool:
        """Validate v3.2 schema exists"""
        try:
            result = await self._db.query("INFO FOR DB")
            tables = result.get("tables", {})

            required_tables = ["atom", "entity", "reference", "performance_log", "session_state"]
            missing = [t for t in required_tables if t not in tables]

            if missing:
                logger.error("[Migration] Missing tables: %s", missing)
                return False

            logger.info("[Migration] Schema validation passed")
            return True
        except Exception as e:
            logger.error("[Migration] Schema validation failed: %s", e)
            return False

    async def count_memory_records(self) -> int:
        """Count memory records to migrate"""
        try:
            result = await self._db.query("SELECT count() FROM memory GROUP BY count")
            count = result[0]["count"] if result else 0
            self._stats["memory_records"] = count
            logger.info("[Migration] Found %d memory records", count)
            return count
        except Exception as e:
            logger.error("[Migration] Failed to count records: %s", e)
            return 0

    async def migrate_memory_to_atom(self, memory_record: dict[str, Any]) -> Optional[str]:
        """Migrate a memory record to atom"""
        try:
            atom_data = {
                "tenant_id": memory_record.get("tenant_id", "default"),
                "type": "fragment",
                "content": memory_record.get("content", ""),
                "metadata": {
                    "source": "memory_migration",
                    "original_id": str(memory_record.get("id")),
                },
                "fingerprint": memory_record.get("content_hash"),
            }

            if self._dry_run:
                logger.debug("[Migration] Would create atom: %s", atom_data["content"][:50])
                return "dry_run_id"

            result = await self._db.create("atom", atom_data)
            self._stats["atoms_created"] += 1
            return result[0]["id"] if result else None
        except Exception as e:
            logger.error("[Migration] Failed to migrate atom: %s", e)
            self._stats["errors"] += 1
            return None

    async def migrate_batch(self, offset: int) -> bool:
        """Migrate a batch of records"""
        try:
            query = f"SELECT * FROM memory LIMIT {self._batch_size} START {offset}"
            records = await self._db.query(query)

            if not records:
                return False

            for record in records:
                await self.migrate_memory_to_atom(record)

            return True
        except Exception as e:
            logger.error("[Migration] Batch failed at offset %d: %s", offset, e)
            return False

    async def run_migration(self) -> dict[str, Any]:
        """Run the migration"""
        logger.info("[Migration] Starting v2.x to v3.2 migration")
        logger.info("[Migration] Dry run: %s", self._dry_run)

        start_time = datetime.now()

        try:
            await self.connect()

            # Validate schema
            if not await self.validate_schema():
                raise MigrationError("Schema validation failed")

            # Count records
            total_records = await self.count_memory_records()

            if total_records == 0:
                logger.info("[Migration] No records to migrate")
                return self._stats

            # Migrate in batches
            offset = 0
            while offset < total_records:
                logger.info("[Migration] Processing batch at offset %d", offset)

                if not await self.migrate_batch(offset):
                    break

                offset += self._batch_size

                # Progress report every 10 batches
                if (offset // self._batch_size) % 10 == 0:
                    logger.info("[Migration] Progress: %d/%d records", offset, total_records)

            # Record migration in schema_version
            if not self._dry_run:
                await self._db.create(
                    "schema_version",
                    {
                        "version": "3.2.0-migrated",
                        "description": f"Migrated from v2.x: {self._stats['atoms_created']} atoms created",
                        "applied_at": datetime.now().isoformat(),
                    },
                )

        except Exception as e:
            logger.error("[Migration] Migration failed: %s", e)
            raise MigrationError(str(e)) from e

        finally:
            await self.close()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("[Migration] Completed in %.2f seconds", duration)
        logger.info("[Migration] Stats: %s", json.dumps(self._stats, indent=2))

        return self._stats

    async def rollback(self) -> None:
        """Rollback migration (delete migrated data)"""
        logger.warning("[Migration] Rolling back migration")

        try:
            await self.connect()

            if self._dry_run:
                logger.info("[Migration] Would delete migrated atoms")
            else:
                # Delete atoms created by migration
                await self._db.query("DELETE atom WHERE metadata.source = 'memory_migration'")
                logger.info("[Migration] Rollback completed")

        finally:
            await self.close()


def main():
    parser = argparse.ArgumentParser(description="Migrate from v2.x to v3.2")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute migration (default is dry-run)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for migration (default: 100)",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback migration",
    )
    parser.add_argument(
        "--url",
        default="ws://localhost:18002",
        help="SurrealDB URL",
    )
    parser.add_argument(
        "--namespace",
        default="memory_ns",
        help="Namespace",
    )
    parser.add_argument(
        "--database",
        default="memory_db",
        help="Database",
    )

    args = parser.parse_args()

    migration = V2ToV3Migration(
        url=args.url,
        namespace=args.namespace,
        database=args.database,
        batch_size=args.batch_size,
        dry_run=not args.execute,
    )

    try:
        if args.rollback:
            asyncio.run(migration.rollback())
        else:
            stats = asyncio.run(migration.run_migration())

            # Exit with error code if there were errors
            if stats["errors"] > 0:
                logger.error("[Migration] Completed with %d errors", stats["errors"])
                sys.exit(1)

            logger.info("[Migration] Success!")

    except MigrationError as e:
        logger.error("[Migration] Failed: %s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("[Migration] Interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
