"""PrecomputeService 初始化测试

验证 BL-C-1: PrecomputeService — 初始化资源实现
"""

import pytest
from unittest.mock import MagicMock

from wrapper.src.services.precompute import PrecomputeService


class TestPrecomputeServiceInitialization:
    """测试 PrecomputeService 初始化"""

    @pytest.fixture
    def mock_db(self):
        """模拟数据库连接"""
        return MagicMock()

    @pytest.fixture
    def precompute_service(self, mock_db):
        """预计算服务实例"""
        return PrecomputeService(
            db=mock_db,
            tenant_id="test",
            max_concurrent=3,
            timeout_seconds=60.0,
        )

    @pytest.mark.asyncio
    async def test_start_initializes_code_parser(self, precompute_service):
        """测试 start() 初始化 tree-sitter 解析器"""
        await precompute_service.start()

        assert precompute_service._code_parser is not None
        assert precompute_service._running is True

        await precompute_service.stop()

    @pytest.mark.asyncio
    async def test_start_raises_error_without_db(self):
        """测试 start() 在没有数据库连接时抛出错误"""
        service = PrecomputeService(db=None, tenant_id="test")

        with pytest.raises(RuntimeError, match="数据库连接未提供"):
            await service.start()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, precompute_service):
        """测试 start() 幂等性"""
        await precompute_service.start()
        await precompute_service.start()  # 第二次调用应该被忽略

        assert precompute_service._running is True

        await precompute_service.stop()

    @pytest.mark.asyncio
    async def test_stop_cleans_code_parser(self, precompute_service):
        """测试 stop() 清理 tree-sitter 解析器"""
        await precompute_service.start()
        assert precompute_service._code_parser is not None

        await precompute_service.stop()

        assert precompute_service._code_parser is None
        assert precompute_service._running is False

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, precompute_service):
        """测试 stop() 幂等性"""
        await precompute_service.start()
        await precompute_service.stop()
        await precompute_service.stop()  # 第二次调用应该被忽略

        assert precompute_service._running is False

    def test_initialization_sets_attributes(self, mock_db):
        """测试初始化设置属性"""
        service = PrecomputeService(
            db=mock_db,
            tenant_id="test-tenant",
            max_concurrent=10,
            timeout_seconds=120.0,
        )

        assert service._db == mock_db
        assert service._tenant_id == "test-tenant"
        assert service._running is False
        assert service._code_parser is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
