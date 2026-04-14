"""WebSocket from_offset 集成测试

测试范围：
- MessageQueue 消息入队和查询
- from_offset 查询功能
- 消息过期清理

运行方式：
    uv run pytest tests/test_websocket_from_offset.py -v
"""

import pytest
from datetime import datetime, timedelta

from wrapper.src.websocket import MessageQueue, QueuedMessage


class TestMessageQueue:
    """MessageQueue 集成测试"""

    @pytest.fixture
    def message_queue(self, tmp_path):
        """创建 MessageQueue 实例"""
        queue_file = tmp_path / "test-messages.json"
        queue = MessageQueue(
            queue_file=str(queue_file),
            ttl_days=7,
            max_messages=100,
        )
        return queue

    def test_enqueue_returns_offset(self, message_queue):
        """测试入队返回 offset"""
        offset1 = message_queue.enqueue(
            session_id="sess-1",
            message_type="CREATE",
            data={"id": "test-1", "content": "Hello"},
        )

        assert offset1 == 1

        offset2 = message_queue.enqueue(
            session_id="sess-1",
            message_type="UPDATE",
            data={"id": "test-1", "content": "Hello World"},
        )

        assert offset2 == 2

    def test_get_messages_from_offset(self, message_queue):
        """测试从 offset 查询消息"""
        # 添加多条消息
        for i in range(5):
            message_queue.enqueue(
                session_id="sess-1",
                message_type="CREATE",
                data={"id": f"test-{i}", "content": f"Item {i}"},
            )

        # 从 offset 2 查询（返回 offset 3, 4, 5）
        messages = message_queue.get_messages_from_offset(2)

        assert len(messages) == 3
        assert messages[0].offset == 3
        assert messages[1].offset == 4
        assert messages[2].offset == 5

    def test_get_messages_with_session_filter(self, message_queue):
        """测试按 Session 过滤查询"""
        # 添加不同 Session 的消息
        message_queue.enqueue(
            session_id="sess-1",
            message_type="CREATE",
            data={"id": "test-1"},
        )
        message_queue.enqueue(
            session_id="sess-2",
            message_type="CREATE",
            data={"id": "test-2"},
        )
        message_queue.enqueue(
            session_id="sess-1",
            message_type="UPDATE",
            data={"id": "test-1"},
        )

        # 只查询 sess-1 的消息
        messages = message_queue.get_messages_from_offset(0, session_id="sess-1")

        assert len(messages) == 2
        assert all(m.session_id == "sess-1" for m in messages)

    def test_get_messages_with_limit(self, message_queue):
        """测试查询限制数量"""
        # 添加 10 条消息
        for i in range(10):
            message_queue.enqueue(
                session_id="sess-1",
                message_type="CREATE",
                data={"id": f"test-{i}"},
            )

        # 限制返回 3 条
        messages = message_queue.get_messages_from_offset(0, limit=3)

        assert len(messages) == 3

    def test_mark_delivered(self, message_queue):
        """测试标记已送达"""
        offset = message_queue.enqueue(
            session_id="sess-1",
            message_type="CREATE",
            data={"id": "test-1"},
        )

        # 标记为已送达
        result = message_queue.mark_delivered(offset)

        assert result is True

        # 获取未送达消息
        undelivered = message_queue.get_undelivered_messages("sess-1")
        assert len(undelivered) == 0

    def test_get_undelivered_messages(self, message_queue):
        """测试获取未送达消息"""
        # 添加消息
        offset1 = message_queue.enqueue(
            session_id="sess-1",
            message_type="CREATE",
            data={"id": "test-1"},
        )
        message_queue.enqueue(
            session_id="sess-1",
            message_type="UPDATE",
            data={"id": "test-1"},
        )

        # 标记第一条为已送达
        message_queue.mark_delivered(offset1)

        # 获取未送达消息
        undelivered = message_queue.get_undelivered_messages("sess-1")

        assert len(undelivered) == 1
        assert undelivered[0].message_type == "UPDATE"

    def test_get_last_offset(self, message_queue):
        """测试获取最后 offset"""
        assert message_queue.get_last_offset() == 0

        for i in range(5):
            message_queue.enqueue(
                session_id="sess-1",
                message_type="CREATE",
                data={"id": f"test-{i}"},
            )

        assert message_queue.get_last_offset() == 5

    def test_get_message_count(self, message_queue):
        """测试获取消息数量"""
        # 添加 sess-1 的消息
        for i in range(3):
            message_queue.enqueue(
                session_id="sess-1",
                message_type="CREATE",
                data={"id": f"test-{i}"},
            )

        # 添加 sess-2 的消息
        for i in range(2):
            message_queue.enqueue(
                session_id="sess-2",
                message_type="CREATE",
                data={"id": f"test-{i}"},
            )

        assert message_queue.get_message_count() == 5
        assert message_queue.get_message_count("sess-1") == 3
        assert message_queue.get_message_count("sess-2") == 2

    def test_clear_session_messages(self, message_queue):
        """测试清除 Session 消息"""
        # 添加消息
        for i in range(3):
            message_queue.enqueue(
                session_id="sess-1",
                message_type="CREATE",
                data={"id": f"test-{i}"},
            )
        for i in range(2):
            message_queue.enqueue(
                session_id="sess-2",
                message_type="CREATE",
                data={"id": f"test-{i}"},
            )

        # 清除 sess-1 的消息
        cleared = message_queue.clear_session_messages("sess-1")

        assert cleared == 3
        assert message_queue.get_message_count("sess-1") == 0
        assert message_queue.get_message_count("sess-2") == 2

    def test_max_messages_limit(self, message_queue):
        """测试最大消息数量限制"""
        # 添加超过限制的消息
        for i in range(150):
            message_queue.enqueue(
                session_id="sess-1",
                message_type="CREATE",
                data={"id": f"test-{i}"},
            )

        # 应该只保留最新的 100 条
        assert message_queue.get_message_count() == 100


class TestMessageQueuePersistence:
    """MessageQueue 持久化测试"""

    def test_messages_persisted_to_file(self, tmp_path):
        """测试消息持久化到文件"""
        queue_file = tmp_path / "test-messages.json"

        # 创建队列并添加消息
        queue1 = MessageQueue(queue_file=str(queue_file))
        queue1.enqueue(
            session_id="sess-1",
            message_type="CREATE",
            data={"id": "test-1", "content": "Hello"},
        )

        # 创建新队列实例（从文件加载）
        queue2 = MessageQueue(queue_file=str(queue_file))

        # 验证消息已加载
        messages = queue2.get_messages_from_offset(0)
        assert len(messages) == 1
        assert messages[0].data["content"] == "Hello"

    def test_offset_persisted(self, tmp_path):
        """测试 offset 持久化"""
        queue_file = tmp_path / "test-messages.json"

        queue1 = MessageQueue(queue_file=str(queue_file))
        for i in range(5):
            queue1.enqueue(
                session_id="sess-1",
                message_type="CREATE",
                data={"id": f"test-{i}"},
            )

        queue2 = MessageQueue(queue_file=str(queue_file))
        assert queue2.get_last_offset() == 5

        # 新消息应该继续递增
        offset = queue2.enqueue(
            session_id="sess-1",
            message_type="CREATE",
            data={"id": "test-5"},
        )
        assert offset == 6


class TestMessageQueueEdgeCases:
    """MessageQueue 边界情况测试"""

    @pytest.fixture
    def message_queue(self, tmp_path):
        """创建 MessageQueue 实例"""
        queue_file = tmp_path / "test-messages.json"
        queue = MessageQueue(queue_file=str(queue_file))
        return queue

    def test_get_messages_from_nonexistent_offset(self, message_queue):
        """测试从不存在的 offset 查询"""
        # 添加消息
        message_queue.enqueue(
            session_id="sess-1",
            message_type="CREATE",
            data={"id": "test-1"},
        )

        # 从大于最大 offset 的位置查询
        messages = message_queue.get_messages_from_offset(100)

        assert len(messages) == 0

    def test_mark_nonexistent_delivered(self, message_queue):
        """测试标记不存在的消息为已送达"""
        result = message_queue.mark_delivered(999)

        assert result is False

    def test_empty_queue_operations(self, message_queue):
        """测试空队列操作"""
        assert message_queue.get_last_offset() == 0
        assert message_queue.get_message_count() == 0
        assert len(message_queue.get_messages_from_offset(0)) == 0
        assert len(message_queue.get_undelivered_messages("sess-1")) == 0
