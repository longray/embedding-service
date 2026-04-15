"""PrecomputeService 资源清理测试

验证 BL-C-2: PrecomputeService — 资源清理实现
"""

import pytest
import pytest_asyncio
from unittest.mock import MagicMock

from wrapper.src.services.precompute import PrecomputeService


class TestPrecomputeServiceCleanup:
    """测试 PrecomputeService 资源清理"""

    @pytest.fixture
    def mock_db(self):
        """模拟数据库连接"""
        return MagicMock()

    @pytest_asyncio.fixture
    async def running_service(self, mock_db):
        """运行中的服务实例"""
        service = PrecomputeService(
            db=mock_db,
            tenant_id="test",
            max_concurrent=3,
        )
        await service.start()
        yield service
        if service._running:
            await service.stop()

    @pytest.mark.asyncio
    async def test_stop_cleans_code_parser(self, running_service):
        """测试 stop() 清理 tree-sitter 解析器"""
        service = running_service
        assert service._code_parser is not None

        await service.stop()

        assert service._code_parser is None
        assert service._running is False

    @pytest.mark.asyncio
    async def test_stop_stops_performance_monitor(self, running_service):
        """测试 stop() 停止性能监控器"""
        import tracemalloc

        service = running_service

        await service.stop()

        assert not tracemalloc.is_tracing()

    @pytest.mark.asyncio
    async def test_stop_cleans_concurrency_resources(self, running_service):
        """测试 stop() 清理并发控制资源"""
        service = running_service

        # 添加一些处理中任务
        service._concurrency_control._processing.add("task1")
        service._concurrency_control._processing.add("task2")

        await service.stop()

        assert len(service._concurrency_control._processing) == 0

    @pytest.mark.asyncio
    async def test_stop_cleans_queue(self, running_service):
        """测试 stop() 清理队列"""
        service = running_service

        # 添加一些队列任务
        await service._concurrency_control._queue.put("item1")
        await service._concurrency_control._queue.put("item2")

        await service.stop()

        assert service._concurrency_control._queue.empty()

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, mock_db):
        """测试 stop() 幂等性"""
        service = PrecomputeService(db=mock_db, tenant_id="test")
        await service.start()
        await service.stop()
        await service.stop()  # 第二次调用应该被忽略

        assert service._running is False

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, mock_db):
        """测试 stop() 在服务未运行时"""
        service = PrecomputeService(db=mock_db, tenant_id="test")

        # 不应该抛出异常
        await service.stop()

        assert service._running is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
