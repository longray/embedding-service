"""Tests for v2 to v3.2 migration script"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from migrate_v2_to_v32 import V2ToV3Migration, MigrationError


class TestV2ToV3MigrationInitialization:
    """Test V2ToV3Migration initialization"""

    def test_basic_initialization(self):
        """Test basic initialization"""
        migration = V2ToV3Migration(
            url="ws://localhost:18002",
            namespace="test_ns",
            database="test_db",
            batch_size=50,
            dry_run=False,
        )

        assert migration._url == "ws://localhost:18002"
        assert migration._namespace == "test_ns"
        assert migration._database == "test_db"
        assert migration._batch_size == 50
        assert migration._dry_run is False
        assert migration._db is None

    def test_default_initialization(self):
        """Test initialization with default values"""
        migration = V2ToV3Migration()

        assert migration._url == "ws://localhost:18002"
        assert migration._namespace == "memory_ns"
        assert migration._database == "memory_db"
        assert migration._batch_size == 100
        assert migration._dry_run is True


class TestV2ToV3MigrationConnect:
    """Test V2ToV3Migration connect method"""

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection"""
        migration = V2ToV3Migration()

        with patch("migrate_v2_to_v32.Surreal") as mock_surreal:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.signin = AsyncMock()
            mock_instance.use = AsyncMock()
            mock_surreal.return_value = mock_instance

            await migration.connect()

            assert migration._db is not None
            mock_instance.connect.assert_called_once()
            mock_instance.signin.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing connection"""
        migration = V2ToV3Migration()

        with patch("migrate_v2_to_v32.Surreal") as mock_surreal:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.signin = AsyncMock()
            mock_instance.use = AsyncMock()
            mock_instance.close = AsyncMock()
            mock_surreal.return_value = mock_instance

            await migration.connect()
            await migration.close()

            mock_instance.close.assert_called_once()
            assert migration._db is None


class TestV2ToV3MigrationValidation:
    """Test V2ToV3Migration validation methods"""

    @pytest.mark.asyncio
    async def test_validate_schema_success(self):
        """Test schema validation success"""
        migration = V2ToV3Migration()

        with patch("migrate_v2_to_v32.Surreal") as mock_surreal:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.signin = AsyncMock()
            mock_instance.use = AsyncMock()
            mock_instance.query = AsyncMock(
                return_value={
                    "tables": {
                        "atom": {},
                        "entity": {},
                        "reference": {},
                        "performance_log": {},
                        "session_state": {},
                    }
                }
            )
            mock_surreal.return_value = mock_instance

            await migration.connect()
            result = await migration.validate_schema()

            assert result is True

    @pytest.mark.asyncio
    async def test_validate_schema_missing_tables(self):
        """Test schema validation with missing tables"""
        migration = V2ToV3Migration()

        with patch("migrate_v2_to_v32.Surreal") as mock_surreal:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.signin = AsyncMock()
            mock_instance.use = AsyncMock()
            mock_instance.query = AsyncMock(
                return_value={
                    "tables": {"atom": {}}  # Missing other tables
                }
            )
            mock_surreal.return_value = mock_instance

            await migration.connect()
            result = await migration.validate_schema()

            assert result is False


class TestV2ToV3MigrationCount:
    """Test V2ToV3Migration count methods"""

    @pytest.mark.asyncio
    async def test_count_memory_records(self):
        """Test counting memory records"""
        migration = V2ToV3Migration()

        with patch("migrate_v2_to_v32.Surreal") as mock_surreal:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.signin = AsyncMock()
            mock_instance.use = AsyncMock()
            mock_instance.query = AsyncMock(return_value=[{"count": 100}])
            mock_surreal.return_value = mock_instance

            await migration.connect()
            count = await migration.count_memory_records()

            assert count == 100
            assert migration._stats["memory_records"] == 100


class TestV2ToV3MigrationMigrate:
    """Test V2ToV3Migration migrate methods"""

    @pytest.mark.asyncio
    async def test_migrate_memory_to_atom_dry_run(self):
        """Test migrating memory to atom in dry-run mode"""
        migration = V2ToV3Migration(dry_run=True)

        with patch("migrate_v2_to_v32.Surreal") as mock_surreal:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.signin = AsyncMock()
            mock_instance.use = AsyncMock()
            mock_surreal.return_value = mock_instance

            await migration.connect()

            memory_record = {
                "id": "memory:abc123",
                "tenant_id": "test",
                "content": "Test content",
                "content_hash": "hash123",
            }

            result = await migration.migrate_memory_to_atom(memory_record)

            assert result == "dry_run_id"

    @pytest.mark.asyncio
    async def test_migrate_batch(self):
        """Test migrating a batch"""
        migration = V2ToV3Migration()

        with patch("migrate_v2_to_v32.Surreal") as mock_surreal:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.signin = AsyncMock()
            mock_instance.use = AsyncMock()
            mock_instance.query = AsyncMock(
                return_value=[
                    {"id": "memory:1", "content": "Test 1"},
                    {"id": "memory:2", "content": "Test 2"},
                ]
            )
            mock_surreal.return_value = mock_instance

            await migration.connect()
            result = await migration.migrate_batch(0)

            assert result is True


class TestV2ToV3MigrationRollback:
    """Test V2ToV3Migration rollback"""

    @pytest.mark.asyncio
    async def test_rollback_dry_run(self):
        """Test rollback in dry-run mode"""
        migration = V2ToV3Migration(dry_run=True)

        with patch("migrate_v2_to_v32.Surreal") as mock_surreal:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.signin = AsyncMock()
            mock_instance.use = AsyncMock()
            mock_surreal.return_value = mock_instance

            await migration.rollback()

            # In dry-run, should not execute delete
            mock_instance.query.assert_not_called()


import asyncio
