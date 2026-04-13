"""PrecomputeService 单元测试

测试范围：
- PrecomputeService 基础功能
- 生命周期管理
- Tenant 隔离
- 批量处理

运行方式：
    uv run pytest tests/test_precompute_service.py -v
"""

import pytest
from unittest.mock import MagicMock

from wrapper.src.services.precompute import PrecomputeService


class TestPrecomputeService:
    """PrecomputeService 单元测试"""

    @pytest.fixture
    def mock_db(self):
        """创建 mock 数据库"""
        return MagicMock()

    @pytest.fixture
    def service(self, mock_db):
        """创建 PrecomputeService 实例"""
        return PrecomputeService(db=mock_db, tenant_id="default")

    def test_initialization(self, mock_db):
        """测试初始化"""
        service = PrecomputeService(db=mock_db, tenant_id="test-tenant")

        assert service.tenant_id == "test-tenant"
        assert service.is_running is False
        assert service.db == mock_db

    @pytest.mark.asyncio
    async def test_start(self, service):
        """测试启动服务"""
        await service.start()

        assert service.is_running is True

    @pytest.mark.asyncio
    async def test_start_already_running(self, service):
        """测试重复启动"""
        await service.start()
        await service.start()  # 不应报错

        assert service.is_running is True

    @pytest.mark.asyncio
    async def test_stop(self, service):
        """测试停止服务"""
        await service.start()
        await service.stop()

        assert service.is_running is False

    @pytest.mark.asyncio
    async def test_stop_not_running(self, service):
        """测试停止未运行的服务"""
        await service.stop()  # 不应报错

        assert service.is_running is False

    @pytest.mark.asyncio
    async def test_process_batch(self, service):
        """测试处理批次"""
        await service.start()

        batch = [
            {"file_path": "test1.py", "content": "def foo(): pass"},
            {"file_path": "test2.py", "content": "def bar(): pass"},
        ]

        result = await service.process_batch(batch)

        assert result["tenant_id"] == "default"
        assert result["processed_count"] == 2
        assert "symbols" in result
        assert "call_relations" in result

    @pytest.mark.asyncio
    async def test_process_batch_not_running(self, service):
        """测试未启动时处理批次"""
        batch = [{"file_path": "test.py", "content": ""}]

        with pytest.raises(RuntimeError, match="PrecomputeService 未启动"):
            await service.process_batch(batch)

    @pytest.mark.asyncio
    async def test_health_check_running(self, service):
        """测试健康检查（运行中）"""
        await service.start()

        health = await service.health_check()

        assert health["tenant_id"] == "default"
        assert health["is_running"] is True
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_stopped(self, service):
        """测试健康检查（已停止）"""
        health = await service.health_check()

        assert health["tenant_id"] == "default"
        assert health["is_running"] is False
        assert health["status"] == "stopped"


class TestPrecomputeServiceTenantIsolation:
    """Tenant 隔离测试"""

    @pytest.mark.asyncio
    async def test_tenant_isolation(self):
        """测试 tenant 隔离"""
        db1 = MagicMock()
        db2 = MagicMock()

        service1 = PrecomputeService(db=db1, tenant_id="tenant-1")
        service2 = PrecomputeService(db=db2, tenant_id="tenant-2")

        assert service1.tenant_id == "tenant-1"
        assert service2.tenant_id == "tenant-2"
        assert service1.tenant_id != service2.tenant_id

        await service1.start()
        await service2.start()

        assert service1.is_running is True
        assert service2.is_running is True

        batch = [{"file_path": "test.py", "content": ""}]

        result1 = await service1.process_batch(batch)
        result2 = await service2.process_batch(batch)

        assert result1["tenant_id"] == "tenant-1"
        assert result2["tenant_id"] == "tenant-2"


class TestPrecomputeServiceLifecycle:
    """生命周期测试"""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """测试完整生命周期"""
        mock_db = MagicMock()
        service = PrecomputeService(db=mock_db, tenant_id="test")

        assert service.is_running is False

        await service.start()
        assert service.is_running is True

        batch = [{"file_path": "test.py", "content": "def test(): pass"}]
        result = await service.process_batch(batch)
        assert result["processed_count"] == 1

        await service.stop()
        assert service.is_running is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
