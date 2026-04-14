"""ConcurrencyControl 队列持久化测试

测试范围：
- 队列状态保存到数据库
- 队列状态从数据库恢复
- 任务状态更新
- 队列状态清除

运行方式：
    uv run pytest tests/test_queue_persistence.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from wrapper.src.services.concurrency_control import ConcurrencyControl, TaskInfo


class TestQueuePersistence:
    """队列持久化测试"""

    @pytest.fixture
    def mock_db(self):
        """创建 mock 数据库"""
        db = AsyncMock()
        db.query = AsyncMock()
        return db

    @pytest.fixture
    def cc_with_db(self, mock_db):
        """创建带 DB 的 ConcurrencyControl 实例"""
        return ConcurrencyControl(
            max_concurrent=5,
            timeout_seconds=30.0,
            max_queue_size=100,
            db=mock_db,
            tenant_id="default",
        )

    @pytest.fixture
    def cc_without_db(self):
        """创建不带 DB 的 ConcurrencyControl 实例"""
        return ConcurrencyControl(
            max_concurrent=5,
            timeout_seconds=30.0,
            max_queue_size=100,
            db=None,
            tenant_id="default",
        )

    def test_init_with_db(self, mock_db):
        """测试带 DB 初始化"""
        cc = ConcurrencyControl(db=mock_db, tenant_id="test")
        assert cc.db is mock_db
        assert cc.tenant_id == "test"

    def test_init_without_db(self):
        """测试不带 DB 初始化"""
        cc = ConcurrencyControl(db=None, tenant_id="test")
        assert cc.db is None

    @pytest.mark.asyncio
    async def test_save_queue_state_empty(self, cc_with_db, mock_db):
        """测试保存空队列状态"""
        mock_db.query = AsyncMock(return_value=[])

        result = await cc_with_db.save_queue_state()

        assert result["saved"] == 0
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_save_queue_state_no_db(self, cc_without_db):
        """测试无 DB 时保存失败"""
        result = await cc_without_db.save_queue_state()

        assert result["saved"] == 0
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_restore_queue_state(self, cc_with_db, mock_db):
        """测试恢复队列状态"""
        mock_db.query = AsyncMock(
            return_value=[
                {
                    "task_id": "task_1",
                    "task_data": {"type": "test"},
                    "status": "pending",
                },
                {
                    "task_id": "task_2",
                    "task_data": {"type": "test"},
                    "status": "pending",
                },
            ]
        )

        async def mock_processor(task_id, task_data):
            return f"processed_{task_id}"

        result = await cc_with_db.restore_queue_state(task_processor=mock_processor)

        assert result == 2

    @pytest.mark.asyncio
    async def test_restore_queue_state_no_db(self, cc_without_db):
        """测试无 DB 时恢复失败"""
        result = await cc_without_db.restore_queue_state()

        assert result == 0

    @pytest.mark.asyncio
    async def test_clear_queue_state_from_db(self, cc_with_db, mock_db):
        """测试清除队列状态"""
        mock_db.query = AsyncMock(return_value=[{"id": "task:1"}, {"id": "task:2"}])

        result = await cc_with_db.clear_queue_state_from_db()

        assert result == 2

    @pytest.mark.asyncio
    async def test_clear_queue_state_no_db(self, cc_without_db):
        """测试无 DB 时清除返回 0"""
        result = await cc_without_db.clear_queue_state_from_db()

        assert result == 0

    @pytest.mark.asyncio
    async def test_update_task_status_in_db(self, cc_with_db, mock_db):
        """测试更新任务状态"""
        mock_db.query = AsyncMock(return_value=[])

        result = await cc_with_db.update_task_status_in_db(
            task_id="task_1",
            status="completed",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_update_task_status_with_error(self, cc_with_db, mock_db):
        """测试更新任务状态为失败"""
        mock_db.query = AsyncMock(return_value=[])

        result = await cc_with_db.update_task_status_in_db(
            task_id="task_1",
            status="failed",
            error_message="Test error",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_update_task_status_no_db(self, cc_without_db):
        """测试无 DB 时更新失败"""
        result = await cc_without_db.update_task_status_in_db(
            task_id="task_1",
            status="completed",
        )

        assert result is False


class TestQueuePersistenceEdgeCases:
    """队列持久化边界情况测试"""

    @pytest.mark.asyncio
    async def test_restore_queue_state_no_processor(self):
        """测试恢复队列状态无处理器"""
        mock_db = AsyncMock()
        mock_db.query = AsyncMock(
            return_value=[
                {"task_id": "task_1", "task_data": {}, "status": "pending"},
            ]
        )

        cc = ConcurrencyControl(db=mock_db, tenant_id="default")

        # 不提供 task_processor
        result = await cc.restore_queue_state(task_processor=None)

        assert result == 0

    @pytest.mark.asyncio
    async def test_restore_queue_state_db_error(self):
        """测试恢复队列状态时 DB 错误"""
        mock_db = AsyncMock()
        mock_db.query = AsyncMock(side_effect=Exception("DB Error"))

        cc = ConcurrencyControl(db=mock_db, tenant_id="default")

        async def mock_processor(task_id, task_data):
            return None

        result = await cc.restore_queue_state(task_processor=mock_processor)

        assert result == 0

    @pytest.mark.asyncio
    async def test_update_task_status_db_error(self):
        """测试更新任务状态时 DB 错误"""
        mock_db = AsyncMock()
        mock_db.query = AsyncMock(side_effect=Exception("DB Error"))

        cc = ConcurrencyControl(db=mock_db, tenant_id="default")

        result = await cc.update_task_status_in_db(
            task_id="task_1",
            status="completed",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_clear_queue_state_db_error(self):
        """测试清除队列状态时 DB 错误"""
        mock_db = AsyncMock()
        mock_db.query = AsyncMock(side_effect=Exception("DB Error"))

        cc = ConcurrencyControl(db=mock_db, tenant_id="default")

        result = await cc.clear_queue_state_from_db()

        assert result == 0
