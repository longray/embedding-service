"""PerformanceMonitor 持久化测试

测试范围：
- 性能指标保存到数据库
- 性能指标从数据库查询
- 批量持久化
- 平均指标计算

运行方式：
    uv run pytest tests/test_performance_persistence.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from wrapper.src.services.performance_monitor import PerformanceMonitor, PerformanceMetrics


class TestPerformancePersistence:
    """性能指标持久化测试"""

    @pytest.fixture
    def mock_db(self):
        """创建 mock 数据库"""
        db = AsyncMock()
        db.query = AsyncMock()
        return db

    @pytest.fixture
    def monitor_with_db(self, mock_db):
        """创建带 DB 的 PerformanceMonitor 实例"""
        return PerformanceMonitor(tenant_id="default", db=mock_db)

    @pytest.fixture
    def monitor_without_db(self):
        """创建不带 DB 的 PerformanceMonitor 实例"""
        return PerformanceMonitor(tenant_id="default", db=None)

    def test_init_with_db(self, mock_db):
        """测试带 DB 初始化"""
        monitor = PerformanceMonitor(tenant_id="test", db=mock_db)
        assert monitor.db is mock_db
        assert monitor.tenant_id == "test"

    def test_init_without_db(self):
        """测试不带 DB 初始化"""
        monitor = PerformanceMonitor(tenant_id="test", db=None)
        assert monitor.db is None

    @pytest.mark.asyncio
    async def test_save_to_db_success(self, monitor_with_db, mock_db):
        """测试成功保存指标到 DB"""
        mock_db.query = AsyncMock(return_value=[])

        metric = PerformanceMetrics(
            operation="test_op",
            duration_ms=100.0,
            memory_mb=50.0,
            metadata={"key": "value"},
        )

        result = await monitor_with_db.save_to_db(metric)

        assert result is True
        mock_db.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_to_db_no_db(self, monitor_without_db):
        """测试无 DB 时保存失败"""
        metric = PerformanceMetrics(
            operation="test_op",
            duration_ms=100.0,
            memory_mb=50.0,
        )

        result = await monitor_without_db.save_to_db(metric)

        assert result is False

    @pytest.mark.asyncio
    async def test_save_to_db_error(self, monitor_with_db, mock_db):
        """测试 DB 错误时保存失败"""
        mock_db.query = AsyncMock(side_effect=Exception("DB Error"))

        metric = PerformanceMetrics(
            operation="test_op",
            duration_ms=100.0,
            memory_mb=50.0,
        )

        result = await monitor_with_db.save_to_db(metric)

        assert result is False

    @pytest.mark.asyncio
    async def test_persist_all_metrics(self, monitor_with_db, mock_db):
        """测试批量持久化指标"""
        # 记录一些指标
        monitor_with_db.record("op1", 100.0, 50.0)
        monitor_with_db.record("op2", 200.0, 100.0)

        mock_db.query = AsyncMock(return_value=[])

        result = await monitor_with_db.persist_all_metrics()

        assert result["success"] == 2
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_persist_all_metrics_no_db(self, monitor_without_db):
        """测试无 DB 时批量持久化失败"""
        monitor_without_db.record("op1", 100.0, 50.0)

        result = await monitor_without_db.persist_all_metrics()

        assert result["success"] == 0
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_query_metrics_from_db(self, monitor_with_db, mock_db):
        """测试从 DB 查询指标"""
        mock_db.query = AsyncMock(
            return_value=[
                {
                    "operation": "test_op",
                    "duration_ms": 100.0,
                    "memory_mb": 50.0,
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ]
        )

        result = await monitor_with_db.query_metrics_from_db(
            operation="test_op",
            limit=10,
        )

        assert len(result) == 1
        assert result[0]["operation"] == "test_op"

    @pytest.mark.asyncio
    async def test_query_metrics_from_db_no_db(self, monitor_without_db):
        """测试无 DB 时查询返回空列表"""
        result = await monitor_without_db.query_metrics_from_db()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_average_metrics_from_db(self, monitor_with_db, mock_db):
        """测试从 DB 获取平均指标"""
        mock_db.query = AsyncMock(
            return_value=[
                {
                    "avg_duration_ms": 150.0,
                    "avg_memory_mb": 75.0,
                }
            ]
        )

        result = await monitor_with_db.get_average_metrics_from_db(
            operation="test_op",
            hours=24,
        )

        assert result["avg_duration_ms"] == 150.0
        assert result["avg_memory_mb"] == 75.0

    @pytest.mark.asyncio
    async def test_get_average_metrics_from_db_no_db(self, monitor_without_db):
        """测试无 DB 时获取平均指标返回默认值"""
        result = await monitor_without_db.get_average_metrics_from_db()

        assert result["avg_duration_ms"] == 0.0
        assert result["avg_memory_mb"] == 0.0


class TestPerformancePersistenceEdgeCases:
    """性能指标持久化边界情况测试"""

    @pytest.mark.asyncio
    async def test_query_metrics_with_time_range(self):
        """测试带时间范围查询指标"""
        mock_db = AsyncMock()
        mock_db.query = AsyncMock(return_value=[])

        monitor = PerformanceMonitor(tenant_id="default", db=mock_db)

        await monitor.query_metrics_from_db(
            operation="test_op",
            start_time=1609459200.0,  # 2021-01-01
            end_time=1640995200.0,  # 2022-01-01
            limit=50,
        )

        # 验证查询被调用
        mock_db.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_average_metrics_no_data(self):
        """测试获取平均指标无数据时"""
        mock_db = AsyncMock()
        mock_db.query = AsyncMock(return_value=[])

        monitor = PerformanceMonitor(tenant_id="default", db=mock_db)

        result = await monitor.get_average_metrics_from_db(operation="nonexistent")

        assert result["avg_duration_ms"] == 0.0
        assert result["avg_memory_mb"] == 0.0

    @pytest.mark.asyncio
    async def test_persist_partial_failure(self):
        """测试部分持久化失败"""
        mock_db = AsyncMock()

        # 设置第一个调用成功，第二个失败
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return []
            raise Exception("DB Error")

        mock_db.query = AsyncMock(side_effect=side_effect)

        monitor = PerformanceMonitor(tenant_id="default", db=mock_db)
        monitor.record("op1", 100.0, 50.0)
        monitor.record("op2", 200.0, 100.0)

        result = await monitor.persist_all_metrics()

        assert result["success"] == 1
        assert result["failed"] == 1

    @pytest.mark.asyncio
    async def test_query_metrics_db_error(self):
        """测试查询指标时 DB 错误"""
        mock_db = AsyncMock()
        mock_db.query = AsyncMock(side_effect=Exception("DB Error"))

        monitor = PerformanceMonitor(tenant_id="default", db=mock_db)

        result = await monitor.query_metrics_from_db()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_average_metrics_db_error(self):
        """测试获取平均指标时 DB 错误"""
        mock_db = AsyncMock()
        mock_db.query = AsyncMock(side_effect=Exception("DB Error"))

        monitor = PerformanceMonitor(tenant_id="default", db=mock_db)

        result = await monitor.get_average_metrics_from_db()

        assert result["avg_duration_ms"] == 0.0
        assert result["avg_memory_mb"] == 0.0
