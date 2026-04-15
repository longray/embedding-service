"""Lookup API 端点测试

验证 BL-T-3: Lookup API 测试

注意：由于直接调用 router 函数时 FastAPI Query 对象的行为与通过 HTTP 调用时不同，
这些测试使用模拟来验证基本逻辑。完整的集成测试需要使用 HTTP 客户端。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from wrapper.src.routers.lookup import lookup_memory


class TestLookupBySourceId:
    """source_id 查询测试"""

    @pytest.fixture
    def mock_memory_manager(self):
        """模拟 MemoryManager"""
        mm = MagicMock()
        mm.lookup_by_source_id = AsyncMock(
            return_value=[
                {
                    "id": "memory:123",
                    "source_id": "source-abc-123",
                    "file_path": "src/main.py",
                    "project_id": "test-project",
                    "type": "code",
                    "content_hash": "abc123",
                    "created_at": "2026-04-15T10:00:00Z",
                    "updated_at": "2026-04-15T10:00:00Z",
                }
            ]
        )
        mm.lookup_by_hash = AsyncMock(return_value=[])
        mm.lookup_by_file_path = AsyncMock(return_value=[])
        return mm

    @pytest.mark.asyncio
    async def test_lookup_by_source_id_success(self, mock_memory_manager):
        """测试正常 source_id 查询"""
        with patch("wrapper.src.routers.lookup.state.memory_manager", mock_memory_manager):
            result = await lookup_memory(source_id="source-abc-123")

        assert result["found"] is True
        mock_memory_manager.lookup_by_source_id.assert_called_once()

    @pytest.mark.asyncio
    async def test_lookup_by_source_id_not_found(self, mock_memory_manager):
        """测试查询不存在的 source_id"""
        mock_memory_manager.lookup_by_source_id = AsyncMock(return_value=[])

        with patch("wrapper.src.routers.lookup.state.memory_manager", mock_memory_manager):
            result = await lookup_memory(source_id="nonexistent-source")

        assert result["found"] is False
        assert "未找到匹配的记忆" in result["message"]


class TestLookupErrorHandling:
    """错误处理测试"""

    @pytest.mark.asyncio
    async def test_lookup_memory_manager_not_initialized(self):
        """测试 MemoryManager 未初始化"""
        with patch("wrapper.src.routers.lookup.state.memory_manager", None):
            with pytest.raises(HTTPException) as exc_info:
                await lookup_memory(source_id="source-123")

        assert exc_info.value.status_code == 503
        assert "MemoryManager未初始化" in exc_info.value.detail


class TestLookupMultiTenant:
    """多租户隔离测试"""

    @pytest.fixture
    def mock_memory_manager(self):
        """模拟 MemoryManager"""
        mm = MagicMock()
        mm.lookup_by_source_id = AsyncMock(return_value=[{"id": "memory:123"}])
        mm.lookup_by_hash = AsyncMock(return_value=[])
        mm.lookup_by_file_path = AsyncMock(return_value=[])
        return mm

    @pytest.mark.asyncio
    async def test_lookup_different_tenant(self, mock_memory_manager):
        """测试不同 tenant_id"""
        with patch("wrapper.src.routers.lookup.state.memory_manager", mock_memory_manager):
            result = await lookup_memory(
                source_id="source-123",
                tenant_id="tenant-a",
            )

        call_args = mock_memory_manager.lookup_by_source_id.call_args[1]
        assert call_args["tenant_id"] == "tenant-a"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
