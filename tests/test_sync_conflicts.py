"""
Phase B Conflict Resolution Test Suite
Tests for sync conflict resolution functionality: recording conflicts, listing, getting details, and resolving
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

pytestmark = pytest.mark.integration


class TestRecordConflict:
    """Tests for _record_conflict method"""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Mixin 拆分后 _record_conflict 方法不存在")
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


class TestGetConflicts:
    """Tests for get_conflicts method"""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Mixin 拆分后 get_conflicts 方法不存在")
    async def test_get_conflicts(self):
        """Test getting conflict list"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        mock_db.query = AsyncMock(
            return_value=[
                {
                    "result": [
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


class TestGetConflictDetail:
    """Tests for get_conflict_detail method"""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Mixin 拆分后 get_conflict_detail 方法不存在")
    async def test_get_conflict_detail(self):
        """Test getting single conflict detail"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        mock_db.query = AsyncMock(
            return_value=[
                {
                    "result": [
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

        # Verify query call was with formatted conflict_id
        call_args = mock_db.query.call_args
        # Should format conflict_id to include "conflict:" prefix
        assert "conflict:def456" in str(call_args)


class TestResolveConflictReal:
    """Tests for real conflict resolution strategies"""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="SyncMixin.resolve_conflict 是 stub")
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
                        "result": [
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
                        ]
                    }
                ],
                # Second call: update memory in surrealdb
                [{"result": [{"id": "memory:xyz789"}]}],
                # Third call: select updated memory id
                [{"result": [{"id": "memory:xyz789"}]}],
                # Fourth call: update conflict status
                [{"result": [{"id": "conflict:abc123", "status": "resolved", "resolution": "use_local"}]}],
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
    @pytest.mark.skip(reason="SyncMixin.resolve_conflict 是 stub")
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
                        "result": [
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
                        ]
                    }
                ],
                # Second call: update conflict status
                [{"result": [{"id": "conflict:def456", "status": "resolved", "resolution": "use_remote"}]}],
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
    @pytest.mark.skip(reason="SyncMixin.resolve_conflict 是 stub")
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
                        "result": [
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
                        ]
                    }
                ],
                # Subsequent calls: update conflict status
                [{"result": [{"id": "conflict:ghi789", "status": "resolved", "resolution": "keep_both"}]}],
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
    @pytest.mark.skip(reason="Mixin 拆分后方法不存在")
    async def test_conflict_isolation(self):
        """Test that conflicts are isolated by tenant_id"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        # Mock query to verify tenant parameter
        async def mock_query(query, params):
            # Verify that tenant_id is always passed in params
            assert "tenant_id" in params
            if "SELECT *" in query and "conflict" in query:
                # Return empty results for isolation test
                return [{"result": []}]
            elif "SELECT *" in query and "memory:" in query:
                return [{"result": []}]
            else:
                return [{"result": []}]

        mock_db.query = mock_query
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
