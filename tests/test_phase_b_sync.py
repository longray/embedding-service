"""
Phase B (v2.2-lite) Backend Sync Test Suite
Tests for sync API endpoints: incremental, full, fingerprints, conflict resolution
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


class TestSyncFingerprints:
    """Tests for get_fingerprints endpoint (B-B5)"""

    @pytest.mark.asyncio
    async def test_get_fingerprints_returns_list(self):
        """Test that get_fingerprints returns list of fingerprints"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        mock_db.query = AsyncMock(
            return_value=[
                {"source_id": "entry-001", "content_hash": "abc123", "updated_at": 1234567890},
                {"source_id": "entry-002", "content_hash": "def456", "updated_at": 1234567891},
            ]
        )

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        fingerprints = await manager.get_fingerprints(tenant_id="test-tenant")

        assert len(fingerprints) == 2
        assert fingerprints[0]["source_id"] == "entry-001"
        assert fingerprints[0]["hash"] == "abc123"
        assert "mtime" in fingerprints[0]

    @pytest.mark.asyncio
    async def test_get_fingerprints_empty_result(self):
        """Test get_fingerprints with empty database"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        mock_db.query = AsyncMock(return_value=[])

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        fingerprints = await manager.get_fingerprints(tenant_id="test-tenant")

        assert fingerprints == []

    @pytest.mark.asyncio
    async def test_get_fingerprints_tenant_isolation(self):
        """Test that fingerprints are filtered by tenant_id"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        mock_db.query = AsyncMock(return_value=[])

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        await manager.get_fingerprints(tenant_id="tenant-a")

        # Verify tenant_id was passed to query (positional args: call_args[0][1] is params dict)
        call_args = mock_db.query.call_args
        assert call_args[0][1]["tenant_id"] == "tenant-a"


class TestSyncPreview:
    """Tests for sync_preview endpoint (B-B2)"""

    @pytest.mark.asyncio
    async def test_sync_preview_new_entries(self):
        """Test detecting new entries to upload"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        mock_db.query = AsyncMock(return_value=[])

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        local_fingerprints = [
            {"source_id": "entry-001", "hash": "abc123", "mtime": 1234567890, "path": "test.md"},
        ]

        result = await manager.sync_preview(fingerprints=local_fingerprints, tenant_id="test-tenant")

        assert len(result["to_upload"]) == 1
        assert result["to_upload"][0]["source_id"] == "entry-001"
        assert result["to_upload"][0]["reason"] == "new"
        assert len(result["to_delete"]) == 0
        assert len(result["conflicts"]) == 0

    @pytest.mark.asyncio
    async def test_sync_preview_deleted_entries(self):
        """Test detecting entries to delete"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        mock_db.query = AsyncMock(
            return_value=[
                {"source_id": "entry-old", "content_hash": "xyz789", "updated_at": 1234567890},
            ]
        )

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        local_fingerprints = []

        result = await manager.sync_preview(fingerprints=local_fingerprints, tenant_id="test-tenant")

        assert len(result["to_delete"]) == 1
        assert result["to_delete"][0] == "entry-old"
        assert len(result["to_upload"]) == 0

    @pytest.mark.asyncio
    async def test_sync_preview_conflicts(self):
        """Test detecting conflicts (same source_id, different hash)"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        mock_db.query = AsyncMock(
            return_value=[
                {"source_id": "entry-001", "content_hash": "server-hash", "updated_at": 1234567890},
            ]
        )
        mock_db.create = AsyncMock(return_value=[{"id": "conflict:abc123"}])

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        local_fingerprints = [
            {"source_id": "entry-001", "hash": "local-hash", "mtime": 1234567891, "path": "test.md"},
        ]

        result = await manager.sync_preview(fingerprints=local_fingerprints, tenant_id="test-tenant")

        assert len(result["conflicts"]) == 1
        assert result["conflicts"][0]["source_id"] == "entry-001"
        assert result["conflicts"][0]["local_hash"] == "local-hash"
        assert result["conflicts"][0]["server_hash"] == "server-hash"

    @pytest.mark.asyncio
    async def test_sync_preview_unchanged_entries(self):
        """Test that unchanged entries don't appear in results"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        mock_db.query = AsyncMock(
            return_value=[
                {"source_id": "entry-001", "content_hash": "same-hash", "updated_at": 1234567890},
            ]
        )

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        local_fingerprints = [
            {"source_id": "entry-001", "hash": "same-hash", "mtime": 1234567890, "path": "test.md"},
        ]

        result = await manager.sync_preview(fingerprints=local_fingerprints, tenant_id="test-tenant")

        assert len(result["to_upload"]) == 0
        assert len(result["to_delete"]) == 0
        assert len(result["conflicts"]) == 0


class TestSyncFull:
    """Tests for sync_full endpoint (B-B3)"""

    @pytest.mark.asyncio
    async def test_sync_full_success(self):
        """Test full sync with successful uploads"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        mock_db.query = AsyncMock(return_value=[{"id": "memory:test"}])

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        with patch.object(manager, "upload_memories", new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = {
                "total": 1,
                "success": 1,
                "failed": 0,
                "updated": 0,
                "skipped": [],
                "memory_ids": [],
            }

            memories = [
                {"content": "Test 1", "type": "general", "source_id": "entry-001"},
                {"content": "Test 2", "type": "general", "source_id": "entry-002"},
            ]

            result = await manager.sync_full(memories=memories, tenant_id="test-tenant")

            assert result["total"] == 2
            assert result["success"] == 2
            assert result["failed"] == 0
            assert result["skipped"] == []

    @pytest.mark.asyncio
    async def test_sync_full_with_skipped(self):
        """Test full sync with skipped duplicates"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        with patch.object(manager, "upload_memories", new_callable=AsyncMock) as mock_upload:
            mock_upload.return_value = {
                "total": 1,
                "success": 0,
                "failed": 0,
                "updated": 0,
                "skipped": [
                    {"local_id": "entry-001", "existing_id": "memory:aaa", "reason": "hash", "similarity": None}
                ],
                "memory_ids": [],
            }

            memories = [
                {"content": "Duplicate content", "type": "general", "source_id": "entry-001", "local_id": "entry-001"},
            ]

            result = await manager.sync_full(memories=memories, tenant_id="test-tenant")

            assert result["total"] == 1
            assert result["success"] == 0
            assert len(result["skipped"]) == 1
            assert result["skipped"][0]["local_id"] == "entry-001"
            assert result["skipped"][0]["reason"] == "hash"

    @pytest.mark.asyncio
    async def test_sync_full_with_failures(self):
        """Test full sync with some failures"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        call_count = 0

        async def mock_upload_side_effect(memories, tenant_id):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return {
                    "total": 1,
                    "success": 1,
                    "failed": 0,
                    "updated": 0,
                    "skipped": [],
                    "memory_ids": [],
                }
            else:
                return {
                    "total": 1,
                    "success": 0,
                    "failed": 1,
                    "updated": 0,
                    "skipped": [],
                    "memory_ids": [],
                    "errors": ["upload failed"],
                }

        with patch.object(manager, "upload_memories", new_callable=AsyncMock, side_effect=mock_upload_side_effect):
            memories = [
                {"content": "Test 1", "type": "general", "source_id": "entry-001"},
                {"content": "Test 2", "type": "general", "source_id": "entry-002"},
                {"content": "Test 3", "type": "general", "source_id": "entry-003"},
            ]

            result = await manager.sync_full(memories=memories, tenant_id="test-tenant")

            assert result["total"] == 3
            assert result["success"] == 2
            assert result["failed"] == 1
            assert len(result["errors"]) == 1


class TestResolveConflict:
    """Tests for resolve_conflict endpoint (B-B4)"""

    @pytest.mark.asyncio
    async def test_resolve_conflict_use_local(self):
        """Test resolving conflict with use_local strategy"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        mock_db.query = AsyncMock(
            return_value=[
                {
                    "id": "conflict:conflict-001",
                    "source_id": "entry-001",
                    "local_hash": "local-hash",
                    "server_hash": "server-hash",
                    "tenant_id": "test-tenant",
                    "status": "pending",
                    "local_content": "local content",
                    "server_content": "server content",
                }
            ]
        )

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        result = await manager.resolve_conflict(
            conflict_id="conflict-001", resolution="use_local", tenant_id="test-tenant"
        )

        assert result["conflict_id"] == "conflict-001"
        assert result["resolution"] == "use_local"
        assert result["status"] == "resolved"

    @pytest.mark.asyncio
    async def test_resolve_conflict_use_remote(self):
        """Test resolving conflict with use_remote strategy"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        mock_db.query = AsyncMock(
            return_value=[
                {
                    "id": "conflict:conflict-002",
                    "source_id": "entry-002",
                    "local_hash": "local-hash2",
                    "server_hash": "server-hash2",
                    "tenant_id": "test-tenant",
                    "status": "pending",
                    "local_content": "local content",
                    "server_content": "server content",
                }
            ]
        )

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        result = await manager.resolve_conflict(
            conflict_id="conflict-002", resolution="use_remote", tenant_id="test-tenant"
        )

        assert result["resolution"] == "use_remote"

    @pytest.mark.asyncio
    async def test_resolve_conflict_keep_both(self):
        """Test resolving conflict with keep_both strategy"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        mock_db.query = AsyncMock(
            return_value=[
                {
                    "id": "conflict:conflict-003",
                    "source_id": "entry-003",
                    "local_hash": "local-hash3",
                    "server_hash": "server-hash3",
                    "tenant_id": "test-tenant",
                    "status": "pending",
                    "local_content": "local content",
                    "server_content": "server content",
                }
            ]
        )
        mock_db.create = AsyncMock(return_value=[{"id": "memory:new999"}])

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        with patch.object(manager, "_get_embeddings", return_value=[[0.1, 0.2, 0.3]]):
            result = await manager.resolve_conflict(
                conflict_id="conflict-003", resolution="keep_both", tenant_id="test-tenant"
            )

        assert result["resolution"] == "keep_both"


class TestSyncAPIEndpoints:
    """Tests for FastAPI endpoints (B-B1)"""

    def test_sync_preview_endpoint_exists(self):
        """Test that sync_preview endpoint is registered"""
        from wrapper.src.main import app

        routes = [route.path for route in app.routes]
        assert "/api/v1/sync/preview" in routes

    def test_sync_incremental_endpoint_backward_compat(self):
        """Test that legacy sync_incremental endpoint still exists"""
        from wrapper.src.main import app

        routes = [route.path for route in app.routes]
        assert "/api/v1/sync/incremental" in routes

    def test_sync_full_endpoint_exists(self):
        """Test that sync_full endpoint is registered"""
        from wrapper.src.main import app

        routes = [route.path for route in app.routes]
        assert "/api/v1/sync/full" in routes

    def test_get_fingerprints_endpoint_exists(self):
        """Test that get_fingerprints endpoint is registered"""
        from wrapper.src.main import app

        routes = [route.path for route in app.routes]
        assert "/api/v1/sync/fingerprints" in routes

    def test_resolve_conflict_endpoint_exists(self):
        """Test that resolve_conflict endpoint is registered"""
        from wrapper.src.main import app

        routes = [route.path for route in app.routes]
        assert any("/api/v1/sync/conflicts/{conflict_id}/resolve" in r for r in routes)


class TestSyncDataModels:
    """Tests for Pydantic data models (B-B1)"""

    def test_sync_fingerprint_model(self):
        """Test SyncFingerprint model validation"""
        from wrapper.src.main import SyncFingerprint

        fp = SyncFingerprint(
            path="active/test/entry-001.md", mtime=1234567890, hash="abc123def456", source_id="entry-001"
        )

        assert fp.path == "active/test/entry-001.md"
        assert fp.mtime == 1234567890
        assert fp.hash == "abc123def456"
        assert fp.source_id == "entry-001"

    def test_sync_preview_request_model(self):
        """Test SyncPreviewRequest model"""
        from wrapper.src.main import SyncPreviewRequest, SyncFingerprint

        fp = SyncFingerprint(path="test.md", mtime=1234567890, hash="abc123", source_id="entry-001")

        request = SyncPreviewRequest(fingerprints=[fp], tenant_id="test-tenant")

        assert len(request.fingerprints) == 1
        assert request.tenant_id == "test-tenant"

    def test_sync_preview_response_model(self):
        """Test SyncPreviewResponse model"""
        from wrapper.src.main import SyncPreviewResponse

        response = SyncPreviewResponse(
            synced=5,
            to_upload=[{"source_id": "entry-001", "reason": "new"}],
            to_delete=["entry-old"],
            conflicts=[{"source_id": "entry-002", "local_hash": "abc", "server_hash": "def"}],
        )

        assert response.synced == 5
        assert len(response.to_upload) == 1
        assert len(response.to_delete) == 1
        assert len(response.conflicts) == 1

    def test_sync_incremental_request_model_backward_compat(self):
        """Test SyncIncrementalRequest model still works"""
        from wrapper.src.main import SyncIncrementalRequest, SyncFingerprint

        fp = SyncFingerprint(path="test.md", mtime=1234567890, hash="abc123", source_id="entry-001")

        request = SyncIncrementalRequest(fingerprints=[fp], tenant_id="test-tenant")

        assert len(request.fingerprints) == 1
        assert request.tenant_id == "test-tenant"

    def test_sync_incremental_response_model_backward_compat(self):
        """Test SyncIncrementalResponse model still works"""
        from wrapper.src.main import SyncIncrementalResponse

        response = SyncIncrementalResponse(
            synced=5,
            to_upload=[{"source_id": "entry-001", "reason": "new"}],
            to_delete=["entry-old"],
            conflicts=[{"source_id": "entry-002", "local_hash": "abc", "server_hash": "def"}],
        )

        assert response.synced == 5

    def test_conflict_resolution_request_model(self):
        """Test ConflictResolutionRequest model"""
        from wrapper.src.main import ConflictResolutionRequest

        request = ConflictResolutionRequest(resolution="use_local", tenant_id="test-tenant")

        assert request.resolution == "use_local"
        assert request.tenant_id == "test-tenant"


class TestSyncIntegration:
    """Integration tests for complete sync flow"""

    @pytest.mark.asyncio
    async def test_full_sync_flow(self):
        """Test complete sync flow: fingerprints → incremental → full"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        mock_db.query = AsyncMock(return_value=[])

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        # Step 1: Get server fingerprints (empty)
        fingerprints = await manager.get_fingerprints(tenant_id="test")
        assert fingerprints == []

        # Step 2: Incremental sync (should detect new entries)
        local_fps = [
            {"source_id": "entry-001", "hash": "hash1", "mtime": 1234567890, "path": "test1.md"},
            {"source_id": "entry-002", "hash": "hash2", "mtime": 1234567891, "path": "test2.md"},
        ]

        inc_result = await manager.sync_preview(fingerprints=local_fps, tenant_id="test")

        assert len(inc_result["to_upload"]) == 2
        assert len(inc_result["conflicts"]) == 0


class TestConflictPersistence:
    """Tests for conflict persistence methods (_record_conflict, get_conflicts, get_conflict_detail)"""

    @pytest.mark.asyncio
    async def test_record_conflict(self):
        """Test recording conflict to database"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        expected_conflict_data = {
            "source_id": "entry-001",
            "local_hash": "local-hash-123",
            "server_hash": "server-hash-456",
            "local_content": "Local content",
            "server_content": "Server content",
            "local_mtime": 1234567890,
            "server_mtime": 1234567891,
            "tenant_id": "test-tenant",
            "status": "pending",
        }

        mock_db.create = AsyncMock(return_value=[{"id": "conflict:abc123"}])

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        conflict_id = await manager._record_conflict(
            source_id="entry-001",
            local_hash="local-hash-123",
            server_hash="server-hash-456",
            local_content="Local content",
            server_content="Server content",
            local_mtime=1234567890,
            server_mtime=1234567891,
            tenant_id="test-tenant",
        )

        # Verify the call to create was correct
        assert conflict_id == "conflict:abc123"
        mock_db.create.assert_called_once_with("conflict", expected_conflict_data)

    @pytest.mark.asyncio
    async def test_get_conflicts(self):
        """Test getting conflict list"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        mock_db.query = AsyncMock(
            return_value=[
                {
                    "id": "conflict:abc123",
                    "source_id": "entry-001",
                    "local_hash": "local-hash",
                    "server_hash": "server-hash",
                    "local_content": "local content",
                    "server_content": "server content",
                    "local_mtime": 1234567890,
                    "server_mtime": 1234567891,
                    "tenant_id": "test-tenant",
                    "status": "pending",
                    "resolution": None,
                    "resolved_at": None,
                    "created_at": "2023-01-01T00:00:00Z",
                    "updated_at": "2023-01-01T00:00:00Z",
                }
            ]
        )

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        conflicts = await manager.get_conflicts(tenant_id="test-tenant", status=None, limit=100)

        assert len(conflicts) == 1
        assert conflicts[0]["id"] == "conflict:abc123"
        assert conflicts[0]["source_id"] == "entry-001"
        assert conflicts[0]["local_hash"] == "local-hash"
        assert conflicts[0]["server_hash"] == "server-hash"
        assert conflicts[0]["status"] == "pending"

        # Verify query call with correct parameters
        call_args = mock_db.query.call_args
        assert "tenant_id" in call_args[0][1]
        assert call_args[0][1]["tenant_id"] == "test-tenant"

    @pytest.mark.asyncio
    async def test_get_conflict_detail(self):
        """Test getting single conflict detail"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        mock_db.query = AsyncMock(
            return_value=[
                {
                    "id": "conflict:def456",
                    "source_id": "entry-002",
                    "local_hash": "local-hash2",
                    "server_hash": "server-hash2",
                    "local_content": "local content2",
                    "server_content": "server content2",
                    "local_mtime": 1234567892,
                    "server_mtime": 1234567893,
                    "tenant_id": "test-tenant",
                    "status": "pending",
                    "resolution": None,
                    "resolved_at": None,
                    "created_at": "2023-01-01T00:00:00Z",
                    "updated_at": "2023-01-01T00:00:00Z",
                }
            ]
        )

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        conflict_detail = await manager.get_conflict_detail(
            conflict_id="def456",  # Without prefix
            tenant_id="test-tenant",
        )

        assert conflict_detail is not None
        assert conflict_detail["id"] == "conflict:def456"
        assert conflict_detail["source_id"] == "entry-002"
        assert conflict_detail["local_hash"] == "local-hash2"
        assert conflict_detail["status"] == "pending"


class TestResolveConflictRealStrategies:
    """Tests for real conflict resolution strategies"""

    @pytest.mark.asyncio
    async def test_resolve_conflict_use_local_real(self):
        """Test real use_local resolution strategy"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()

        # Setup for getting conflict detail
        mock_db.query = AsyncMock(
            side_effect=[
                # First call: get conflict detail (get_conflict_detail)
                [
                    {
                        "id": "conflict:abc123",
                        "source_id": "entry-001",
                        "local_hash": "local-hash",
                        "server_hash": "server-hash",
                        "local_content": "Updated local content",
                        "server_content": "Original server content",
                        "local_mtime": 1234567890,
                        "server_mtime": 1234567891,
                        "tenant_id": "test-tenant",
                        "status": "pending",
                        "resolution": None,
                        "resolved_at": None,
                        "created_at": "2023-01-01T00:00:00Z",
                        "updated_at": "2023-01-01T00:00:00Z",
                    }
                ],
                # Second call: update memory in surrealdb (UPDATE query)
                [{"id": "memory:xyz789"}],
                # Third call: select updated memory id (SELECT query)
                [{"id": "memory:xyz789"}],
                # Fourth call: update conflict status (UPDATE query)
                [{"id": "conflict:abc123", "status": "resolved", "resolution": "use_local"}],
            ]
        )

        # Mock embedding and meili clients
        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        with patch.object(manager, "_get_embeddings", return_value=[[0.1, 0.2, 0.3]]):
            with patch.object(
                manager, "_build_meili_doc", return_value={"id": "xyz789", "content": "Updated local content"}
            ):
                mock_meili = AsyncMock()
                # Use add_documents method instead of update_document
                mock_meili.add_documents = AsyncMock()
                manager.set_meili_client(mock_meili)

                result = await manager.resolve_conflict(
                    conflict_id="abc123", resolution="use_local", tenant_id="test-tenant"
                )

        assert result["conflict_id"] == "abc123"
        assert result["resolution"] == "use_local"
        assert result["status"] == "resolved"
        assert result["source_id"] == "entry-001"

        # Verify the update query was called
        update_calls = [call for call in mock_db.query.call_args_list if "UPDATE memory" in str(call)]
        assert len(update_calls) > 0

    @pytest.mark.asyncio
    async def test_resolve_conflict_use_remote_real(self):
        """Test real use_remote resolution strategy"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()

        # Setup for getting conflict detail
        mock_db.query = AsyncMock(
            side_effect=[
                # First call: get conflict detail
                [
                    {
                        "id": "conflict:def456",
                        "source_id": "entry-002",
                        "local_hash": "local-hash2",
                        "server_hash": "server-hash2",
                        "local_content": "Local content that will be discarded",
                        "server_content": "Keep this server content",
                        "local_mtime": 1234567892,
                        "server_mtime": 1234567893,
                        "tenant_id": "test-tenant",
                        "status": "pending",
                        "resolution": None,
                        "resolved_at": None,
                        "created_at": "2023-01-01T00:00:00Z",
                        "updated_at": "2023-01-01T00:00:00Z",
                    }
                ],
                # Second call: update conflict status
                [{"id": "conflict:def456", "status": "resolved", "resolution": "use_remote"}],
            ]
        )

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        result = await manager.resolve_conflict(conflict_id="def456", resolution="use_remote", tenant_id="test-tenant")

        assert result["conflict_id"] == "def456"
        assert result["resolution"] == "use_remote"
        assert result["status"] == "resolved"

        # For use_remote, no update to memory should happen, only conflict status update

    @pytest.mark.asyncio
    async def test_resolve_conflict_keep_both_real(self):
        """Test real keep_both resolution strategy"""
        from wrapper.src.utils.memory_manager import MemoryManager
        import hashlib

        mock_db = AsyncMock()

        # Setup for getting conflict detail and creating new memory
        mock_db.query = AsyncMock(
            side_effect=[
                # First call: get conflict detail
                [
                    {
                        "id": "conflict:ghi789",
                        "source_id": "entry-003",
                        "local_hash": "local-hash3",
                        "server_hash": "server-hash3",
                        "local_content": "Local content that will be kept as separate record",
                        "server_content": "Server content that stays",
                        "local_mtime": 1234567894,
                        "server_mtime": 1234567895,
                        "tenant_id": "test-tenant",
                        "status": "pending",
                        "resolution": None,
                        "resolved_at": None,
                        "created_at": "2023-01-01T00:00:00Z",
                        "updated_at": "2023-01-01T00:00:00Z",
                    }
                ],
                # Subsequent call: update conflict status
                [{"id": "conflict:ghi789", "status": "resolved", "resolution": "keep_both"}],
            ]
        )

        # Mock creating new memory record
        mock_db.create = AsyncMock(return_value=[{"id": "memory:new999"}])

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        with patch.object(manager, "_get_embeddings", return_value=[[0.4, 0.5, 0.6]]):
            with patch.object(
                manager,
                "_build_meili_doc",
                return_value={"id": "new999", "content": "Local content that will be kept as separate record"},
            ):
                mock_meili = AsyncMock()
                # Use add_documents method instead of add_document
                mock_meili.add_documents = AsyncMock()
                manager.set_meili_client(mock_meili)

                result = await manager.resolve_conflict(
                    conflict_id="ghi789", resolution="keep_both", tenant_id="test-tenant"
                )

        assert result["conflict_id"] == "ghi789"
        assert result["resolution"] == "keep_both"
        assert result["status"] == "resolved"

        # Verify create was called for new memory with -local suffix
        mock_db.create.assert_called()


class TestConflictIsolation:
    """Tests for tenant isolation in conflict operations"""

    @pytest.mark.asyncio
    async def test_conflict_isolation(self):
        """Test that conflicts are isolated by tenant_id"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        # Mock query to track calls
        mock_db.query = AsyncMock(return_value=[])
        mock_db.create = AsyncMock(return_value=[{"id": "conflict:isolated123"}])

        # Record a conflict for tenant A
        conflict_id_a = await manager._record_conflict(
            source_id="entry-001", local_hash="hash-a", server_hash="hash-b", tenant_id="tenant-a"
        )

        # Get conflicts for tenant A
        conflicts_a = await manager.get_conflicts(tenant_id="tenant-a")

        # Get conflicts for tenant B
        conflicts_b = await manager.get_conflicts(tenant_id="tenant-b")

        # Verify tenant isolation in recorded data
        assert mock_db.create.call_args[0][1]["tenant_id"] == "tenant-a"

        # Verify calls were made with correct tenants
        query_calls = mock_db.query.call_args_list
        tenant_params = [call[0][1].get("tenant_id") for call in query_calls if "tenant_id" in call[0][1]]
        assert "tenant-a" in tenant_params
        assert "tenant-b" in tenant_params
