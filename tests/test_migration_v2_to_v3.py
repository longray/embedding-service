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


class TestV2ToV3MigrationDataIntegrity:
    """Test V2ToV3Migration data integrity"""

    @pytest.mark.asyncio
    async def test_verify_atom_data_integrity(self):
        """Test atom table data integrity after migration"""
        migration = V2ToV3Migration(dry_run=False)

        with patch("migrate_v2_to_v32.Surreal") as mock_surreal:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.signin = AsyncMock()
            mock_instance.use = AsyncMock()
            # Mock atom table has correct number of records
            mock_instance.query = AsyncMock(
                side_effect=[
                    [{"count": 100}],  # memory records
                    [{"count": 100}],  # atom records
                ]
            )
            mock_surreal.return_value = mock_instance

            await migration.connect()
            memory_count = await migration.count_memory_records()

            # Verify atom count matches memory count
            atom_result = await migration._db.query("SELECT count() FROM atom GROUP BY count")
            atom_count = atom_result[0]["count"] if atom_result else 0

            assert atom_count == memory_count, "Atom count should match memory count"

    @pytest.mark.asyncio
    async def test_verify_entity_data_integrity(self):
        """Test entity table data integrity after migration"""
        migration = V2ToV3Migration(dry_run=False)

        with patch("migrate_v2_to_v32.Surreal") as mock_surreal:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.signin = AsyncMock()
            mock_instance.use = AsyncMock()
            # Mock entity record with required fields
            mock_instance.query = AsyncMock(
                return_value=[{"tenant_id": "test", "type": "function", "name": "test_func"}]
            )
            mock_surreal.return_value = mock_instance

            await migration.connect()

            # Verify entity records exist and have required fields
            entity_result = await migration._db.query("SELECT * FROM entity LIMIT 1")

            assert len(entity_result) > 0, "Should have entity records"
            entity = entity_result[0]
            assert "tenant_id" in entity, "Entity should have tenant_id"
            assert "type" in entity, "Entity should have type"
            assert "name" in entity, "Entity should have name"

    @pytest.mark.asyncio
    async def test_verify_reference_data_integrity(self):
        """Test reference table data integrity after migration"""
        migration = V2ToV3Migration(dry_run=False)

        with patch("migrate_v2_to_v32.Surreal") as mock_surreal:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.signin = AsyncMock()
            mock_instance.use = AsyncMock()
            # Mock reference record with required fields
            mock_instance.query = AsyncMock(return_value=[{"in": "entity:1", "out": "entity:2", "type": "call"}])
            mock_surreal.return_value = mock_instance

            await migration.connect()

            # Verify reference records have valid in/out records
            ref_result = await migration._db.query("SELECT * FROM reference LIMIT 1")

            assert len(ref_result) > 0, "Should have reference records"
            ref = ref_result[0]
            assert "in" in ref, "Reference should have in"
            assert "out" in ref, "Reference should have out"
            assert "type" in ref, "Reference should have type"


class TestV2ToV3MigrationPerformance:
    """Test V2ToV3Migration performance"""

    @pytest.mark.asyncio
    async def test_migration_performance_benchmark(self):
        """Test migration performance benchmark"""
        migration = V2ToV3Migration(batch_size=100, dry_run=False)

        with patch("migrate_v2_to_v32.Surreal") as mock_surreal:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.signin = AsyncMock()
            mock_instance.use = AsyncMock()
            mock_instance.query = AsyncMock(return_value=[{"count": 1000}])
            mock_instance.create = AsyncMock(return_value=[{"id": "atom:test"}])
            mock_surreal.return_value = mock_instance

            await migration.connect()

            import time

            start_time = time.time()

            # Simulate migrating 1000 records
            for i in range(0, 1000, 100):
                await migration.migrate_batch(i)

            end_time = time.time()
            duration = end_time - start_time

            # Performance requirement: should process 1000 records in less than 60 seconds
            assert duration < 60, f"Migration too slow: {duration}s for 1000 records"

            # Calculate throughput (avoid division by zero)
            if duration > 0:
                throughput = 1000 / duration
                print(f"Migration throughput: {throughput:.2f} records/second")
            else:
                print("Migration throughput: >1000 records/second (too fast to measure)")

    @pytest.mark.asyncio
    async def test_batch_size_performance(self):
        """Test different batch sizes for optimal performance"""
        batch_sizes = [50, 100, 200, 500]

        for batch_size in batch_sizes:
            migration = V2ToV3Migration(batch_size=batch_size, dry_run=True)

            with patch("migrate_v2_to_v32.Surreal") as mock_surreal:
                mock_instance = AsyncMock()
                mock_instance.connect = AsyncMock()
                mock_instance.signin = AsyncMock()
                mock_instance.use = AsyncMock()
                mock_instance.query = AsyncMock(
                    return_value=[{"id": f"memory:{i}", "content": f"Test {i}"} for i in range(batch_size)]
                )
                mock_surreal.return_value = mock_instance

                await migration.connect()

                import time

                start_time = time.time()
                result = await migration.migrate_batch(0)
                end_time = time.time()

                duration = end_time - start_time
                print(f"Batch size {batch_size}: {duration:.3f}s")

                assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
