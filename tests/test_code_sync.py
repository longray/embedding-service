"""代码同步功能单元测试 (BL-CA-07/08)

覆盖:
- BL-CA-07: sync_code_fingerprints API
- BL-CA-08: code 类型 upsert 逻辑

运行方式:
    uv run pytest tests/test_code_sync.py -v
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


# ==================== BL-CA-07: sync_code_fingerprints ====================


class TestSyncCodeFingerprints:
    """测试代码指纹同步功能"""

    @pytest.fixture
    def memory_manager_mocked(self):
        """创建带有 mock _db_query 的 MemoryManager"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )
        # Mock _extract_records 方法
        manager._extract_records = MagicMock(return_value=[])
        return manager, mock_db

    @pytest.mark.asyncio
    async def test_all_missing(self, memory_manager_mocked):
        """场景1: 全新文件，全部 missing"""
        manager, mock_db = memory_manager_mocked

        # 服务端无记录
        mock_db.query = AsyncMock(return_value=[])
        manager._extract_records = MagicMock(return_value=[])

        local_fingerprints = [
            {"path": "src/a.js", "hash": "h1", "symbols_hash": "s1", "mtime": 1000, "size": 100},
            {"path": "src/b.js", "hash": "h2", "symbols_hash": "s2", "mtime": 2000, "size": 200},
        ]

        result = await manager.sync_code_fingerprints(
            fingerprints=local_fingerprints,
            project_id="test-project",
            tenant_id="default",
        )

        assert len(result["missing"]) == 2
        assert "src/a.js" in result["missing"]
        assert "src/b.js" in result["missing"]
        assert len(result["changed"]) == 0
        assert len(result["unchanged"]) == 0
        assert len(result["conflicts"]) == 0

    @pytest.mark.asyncio
    async def test_all_unchanged(self, memory_manager_mocked):
        """场景2: 完全一致，全部 unchanged"""
        manager, mock_db = memory_manager_mocked

        # 服务端有相同记录
        server_records = [
            {
                "id": "mem:1",
                "content_hash": "h1",
                "mtime": 1000,
                "metadata": {"file_path": "src/a.js", "symbols_hash": "s1"},
            }
        ]
        mock_db.query = AsyncMock(return_value=server_records)
        manager._extract_records = MagicMock(return_value=server_records)

        local_fingerprints = [
            {"path": "src/a.js", "hash": "h1", "symbols_hash": "s1", "mtime": 1000, "size": 100},
        ]

        result = await manager.sync_code_fingerprints(
            fingerprints=local_fingerprints,
            project_id="test-project",
            tenant_id="default",
        )

        assert len(result["unchanged"]) == 1
        assert "src/a.js" in result["unchanged"]
        assert len(result["changed"]) == 0
        assert len(result["missing"]) == 0

    @pytest.mark.asyncio
    async def test_content_changed(self, memory_manager_mocked):
        """场景3: 内容变更，reason=content_modified"""
        manager, mock_db = memory_manager_mocked

        server_records = [
            {
                "id": "mem:1",
                "content_hash": "old_hash",
                "mtime": 1000,
                "metadata": {"file_path": "src/a.js", "symbols_hash": "s1"},
            }
        ]
        mock_db.query = AsyncMock(return_value=server_records)
        manager._extract_records = MagicMock(return_value=server_records)

        local_fingerprints = [
            {"path": "src/a.js", "hash": "new_hash", "symbols_hash": "s1", "mtime": 2000, "size": 100},
        ]

        result = await manager.sync_code_fingerprints(
            fingerprints=local_fingerprints,
            project_id="test-project",
            tenant_id="default",
        )

        assert len(result["changed"]) == 1
        assert result["changed"][0]["path"] == "src/a.js"
        assert result["changed"][0]["reason"] == "content_modified"
        assert result["changed"][0]["server_mtime"] == 1000

    @pytest.mark.asyncio
    async def test_symbols_changed(self, memory_manager_mocked):
        """场景4: 仅符号变更，reason=symbols_modified"""
        manager, mock_db = memory_manager_mocked

        server_records = [
            {
                "id": "mem:1",
                "content_hash": "h1",
                "mtime": 1000,
                "metadata": {"file_path": "src/a.js", "symbols_hash": "old_symbols"},
            }
        ]
        mock_db.query = AsyncMock(return_value=server_records)
        manager._extract_records = MagicMock(return_value=server_records)

        local_fingerprints = [
            {"path": "src/a.js", "hash": "h1", "symbols_hash": "new_symbols", "mtime": 1000, "size": 100},
        ]

        result = await manager.sync_code_fingerprints(
            fingerprints=local_fingerprints,
            project_id="test-project",
            tenant_id="default",
        )

        assert len(result["changed"]) == 1
        assert result["changed"][0]["reason"] == "symbols_modified"

    @pytest.mark.asyncio
    async def test_mtime_conflict(self, memory_manager_mocked):
        """场景5: mtime 冲突，归入 conflicts"""
        manager, mock_db = memory_manager_mocked

        server_records = [
            {
                "id": "mem:1",
                "content_hash": "old_hash",
                "mtime": 2000,  # 服务端更新
                "metadata": {"file_path": "src/a.js", "symbols_hash": "s1"},
            }
        ]
        mock_db.query = AsyncMock(return_value=server_records)
        manager._extract_records = MagicMock(return_value=server_records)

        local_fingerprints = [
            {"path": "src/a.js", "hash": "new_hash", "symbols_hash": "s1", "mtime": 1000, "size": 100},
        ]

        result = await manager.sync_code_fingerprints(
            fingerprints=local_fingerprints,
            project_id="test-project",
            tenant_id="default",
        )

        assert len(result["conflicts"]) == 1
        assert result["conflicts"][0]["path"] == "src/a.js"
        assert result["conflicts"][0]["local_mtime"] == 1000
        assert result["conflicts"][0]["server_mtime"] == 2000

    @pytest.mark.asyncio
    async def test_mixed_scenarios(self, memory_manager_mocked):
        """场景6: 混合场景"""
        manager, mock_db = memory_manager_mocked

        server_records = [
            {
                "id": "mem:1",
                "content_hash": "h1",
                "mtime": 1000,
                "metadata": {"file_path": "src/unchanged.js", "symbols_hash": "s1"},
            },
            {
                "id": "mem:2",
                "content_hash": "old_h2",
                "mtime": 1000,
                "metadata": {"file_path": "src/changed.js", "symbols_hash": "s2"},
            },
            {
                "id": "mem:3",
                "content_hash": "old_h3",
                "mtime": 2000,
                "metadata": {"file_path": "src/conflict.js", "symbols_hash": "s3"},
            },
        ]
        mock_db.query = AsyncMock(return_value=server_records)
        manager._extract_records = MagicMock(return_value=server_records)

        local_fingerprints = [
            {"path": "src/unchanged.js", "hash": "h1", "symbols_hash": "s1", "mtime": 1000, "size": 100},
            {"path": "src/changed.js", "hash": "new_h2", "symbols_hash": "s2", "mtime": 2000, "size": 100},
            {"path": "src/conflict.js", "hash": "new_h3", "symbols_hash": "s3", "mtime": 1000, "size": 100},
            {"path": "src/missing.js", "hash": "h4", "symbols_hash": "s4", "mtime": 1000, "size": 100},
        ]

        result = await manager.sync_code_fingerprints(
            fingerprints=local_fingerprints,
            project_id="test-project",
            tenant_id="default",
        )

        assert len(result["unchanged"]) == 1
        assert len(result["changed"]) == 1
        assert len(result["conflicts"]) == 1
        assert len(result["missing"]) == 1

    @pytest.mark.asyncio
    async def test_empty_fingerprints(self, memory_manager_mocked):
        """场景7: 空列表输入"""
        manager, mock_db = memory_manager_mocked

        mock_db.query = AsyncMock(return_value=[])
        manager._extract_records = MagicMock(return_value=[])

        result = await manager.sync_code_fingerprints(
            fingerprints=[],
            project_id="test-project",
            tenant_id="default",
        )

        assert len(result["missing"]) == 0
        assert len(result["changed"]) == 0
        assert len(result["unchanged"]) == 0
        assert len(result["conflicts"]) == 0

    @pytest.mark.asyncio
    async def test_fingerprint_without_path(self, memory_manager_mocked):
        """场景8: 无 path 字段的指纹应被跳过"""
        manager, mock_db = memory_manager_mocked

        mock_db.query = AsyncMock(return_value=[])
        manager._extract_records = MagicMock(return_value=[])

        local_fingerprints = [
            {"path": "", "hash": "h1", "symbols_hash": "s1", "mtime": 1000, "size": 100},
            {"hash": "h2", "symbols_hash": "s2", "mtime": 1000, "size": 100},  # 无 path
        ]

        result = await manager.sync_code_fingerprints(
            fingerprints=local_fingerprints,
            project_id="test-project",
            tenant_id="default",
        )

        assert len(result["missing"]) == 0  # 两个都被跳过


# ==================== BL-CA-08: code upsert ====================


class TestCodeUpsert:
    """测试代码文件 upsert 逻辑"""

    @pytest.fixture
    def memory_manager_mocked(self):
        """创建带有 mock _db_query 和 _update_memory 的 MemoryManager"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )
        # Mock _update_memory
        manager._update_memory = AsyncMock()
        return manager, mock_db

    @pytest.mark.asyncio
    async def test_code_file_upsert_existing(self, memory_manager_mocked):
        """场景1: 已存在的代码文件，执行 UPDATE"""
        manager, mock_db = memory_manager_mocked

        # 模拟已存在的记录
        existing_records = [{"id": "mem:existing123"}]
        mock_db.query = AsyncMock(return_value=existing_records)
        manager._extract_records = MagicMock(return_value=existing_records)

        memories = [
            {
                "content": "console.log('hello')",
                "type": "code",
                "metadata": {"file_path": "src/index.js"},
                "project_id": "test-project",
            }
        ]

        # Mock embeddings
        with patch.object(manager, "_get_embeddings", return_value=[[0.1, 0.2, 0.3]]):
            result = await manager.upload_memories(memories, tenant_id="default")

        # 验证调用了 _update_memory
        manager._update_memory.assert_called_once()
        assert result["updated"] == 1
        assert result["success"] == 1

    @pytest.mark.asyncio
    async def test_code_file_insert_new(self, memory_manager_mocked):
        """场景2: 新代码文件，跳过 upsert 走正常流程"""
        manager, mock_db = memory_manager_mocked

        # 模拟 code upsert 查询返回空（表示是新文件）
        mock_db.query = AsyncMock(return_value=[])
        manager._extract_records = MagicMock(return_value=[])

        memories = [
            {
                "content": "console.log('hello')",
                "type": "code",
                "metadata": {"file_path": "src/new.js"},
                "project_id": "test-project",
            }
        ]

        with patch.object(manager, "_get_embeddings", return_value=[[0.1, 0.2, 0.3]]):
            # 只验证没有报错，且没有调用 _update_memory（因为是新文件）
            result = await manager.upload_memories(memories, tenant_id="default")

        # 验证 _update_memory 没有被调用（因为是新文件，不是更新）
        manager._update_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_code_type_no_upsert(self, memory_manager_mocked):
        """场景3: 非 code 类型，不走 upsert 逻辑"""
        manager, mock_db = memory_manager_mocked

        # 非 code 类型应该直接走原有流程
        mock_db.query = AsyncMock(return_value=[])  # content_hash 检查
        manager._extract_records = MagicMock(return_value=[])

        memories = [
            {
                "content": "This is a note",
                "type": "general",
                "metadata": {},
                "project_id": "test-project",
            }
        ]

        mock_db.create = AsyncMock(return_value={"id": "mem:new456"})
        with patch.object(manager, "_get_embeddings", return_value=[[0.1, 0.2, 0.3]]):
            result = await manager.upload_memories(memories, tenant_id="default")

        # 应该走普通流程，不调用 code upsert 的查询
        # code upsert 的查询是 "SELECT id, metadata FROM memory WHERE type = 'code'..."
        call_args_list = [str(call) for call in mock_db.query.call_args_list]
        code_upsert_query = "type = 'code'"
        assert not any(code_upsert_query in str(call) for call in call_args_list)

    @pytest.mark.asyncio
    async def test_code_without_file_path(self, memory_manager_mocked):
        """场景4: code 类型但无 file_path，走原有流程"""
        manager, mock_db = memory_manager_mocked

        # 无 file_path 应该跳过 upsert
        mock_db.query = AsyncMock(return_value=[])  # content_hash 检查
        manager._extract_records = MagicMock(return_value=[])

        memories = [
            {
                "content": "console.log('hello')",
                "type": "code",
                "metadata": {},  # 无 file_path
                "project_id": "test-project",
            }
        ]

        mock_db.create = AsyncMock(return_value={"id": "mem:new789"})
        with patch.object(manager, "_get_embeddings", return_value=[[0.1, 0.2, 0.3]]):
            result = await manager.upload_memories(memories, tenant_id="default")

        # 应该跳过 code upsert，走普通流程
        # _update_memory 不应该被调用（因为没有 file_path）
        manager._update_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_upsert_updates_correct_record(self, memory_manager_mocked):
        """场景5: 验证更新的是正确的记录 ID"""
        manager, mock_db = memory_manager_mocked

        existing_records = [{"id": "mem:correct_id"}]
        mock_db.query = AsyncMock(return_value=existing_records)
        manager._extract_records = MagicMock(return_value=existing_records)

        memories = [
            {
                "content": "console.log('updated')",
                "type": "code",
                "metadata": {"file_path": "src/index.js"},
                "project_id": "test-project",
            }
        ]

        with patch.object(manager, "_get_embeddings", return_value=[[0.1, 0.2, 0.3]]):
            await manager.upload_memories(memories, tenant_id="default")

        # 验证 _update_memory 被调用时传入正确的 ID
        call_args = manager._update_memory.call_args
        assert call_args[0][0] == "mem:correct_id"
