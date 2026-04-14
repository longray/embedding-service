"""WebSocket 消息持久化测试

验证 MessageQueue 的消息持久化功能：
- 消息队列持久化存储
- 服务重启后恢复未确认消息
- 消息过期清理机制

使用方式:
    uv run pytest tests/test_websocket_persistent.py -v
"""

import json
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Import message_queue module
import sys

sys.path.insert(0, str(Path(__file__).parent / ".." / "wrapper" / "src" / "websocket"))

from message_queue import QueuedMessage, MessageQueue


class TestQueuedMessage:
    """测试 QueuedMessage 数据类"""

    def test_create_message(self):
        """测试创建消息"""
        msg = QueuedMessage(
            offset=1,
            session_id="test-session",
            message_type="test",
            data={"key": "value"},
            timestamp=datetime.utcnow().isoformat(),
            delivered=False,
        )

        assert msg.offset == 1
        assert msg.session_id == "test-session"
        assert msg.message_type == "test"
        assert msg.data == {"key": "value"}
        assert msg.delivered is False

    def test_to_dict(self):
        """测试转换为字典"""
        msg = QueuedMessage(
            offset=1,
            session_id="test-session",
            message_type="test",
            data={"key": "value"},
            timestamp="2024-01-01T00:00:00",
            delivered=True,
        )

        d = msg.to_dict()
        assert d["offset"] == 1
        assert d["session_id"] == "test-session"
        assert d["message_type"] == "test"
        assert d["data"] == {"key": "value"}
        assert d["timestamp"] == "2024-01-01T00:00:00"
        assert d["delivered"] is True

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "offset": 1,
            "session_id": "test-session",
            "message_type": "test",
            "data": {"key": "value"},
            "timestamp": "2024-01-01T00:00:00",
            "delivered": True,
        }

        msg = QueuedMessage.from_dict(data)
        assert msg.offset == 1
        assert msg.session_id == "test-session"
        assert msg.message_type == "test"
        assert msg.data == {"key": "value"}
        assert msg.delivered is True

    def test_from_dict_default_delivered(self):
        """测试从字典创建（默认 delivered）"""
        data = {
            "offset": 1,
            "session_id": "test-session",
            "message_type": "test",
            "data": {},
            "timestamp": "2024-01-01T00:00:00",
        }

        msg = QueuedMessage.from_dict(data)
        assert msg.delivered is False


class TestMessageQueueBasic:
    """测试 MessageQueue 基础功能"""

    def test_init_default(self):
        """测试默认初始化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "test-queue.json"
            mq = MessageQueue(queue_file=str(queue_file))

            assert mq.queue_file == queue_file
            assert mq.ttl_days == 7
            assert mq.get_message_count() == 0
            assert mq.get_last_offset() == 0

    def test_init_custom_params(self):
        """测试自定义参数初始化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "test-queue.json"
            mq = MessageQueue(
                queue_file=str(queue_file),
                ttl_days=14,
                max_messages=5000,
            )

            assert mq.ttl_days == 14

    def test_enqueue_single(self):
        """测试单条消息入队"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "test-queue.json"
            mq = MessageQueue(queue_file=str(queue_file))

            offset = mq.enqueue(
                session_id="session-1",
                message_type="test",
                data={"key": "value"},
            )

            assert offset == 1
            assert mq.get_message_count() == 1
            assert mq.get_last_offset() == 1

    def test_enqueue_multiple(self):
        """测试多条消息入队"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "test-queue.json"
            mq = MessageQueue(queue_file=str(queue_file))

            offset1 = mq.enqueue("session-1", "type1", {"data": 1})
            offset2 = mq.enqueue("session-1", "type2", {"data": 2})
            offset3 = mq.enqueue("session-2", "type1", {"data": 3})

            assert offset1 == 1
            assert offset2 == 2
            assert offset3 == 3
            assert mq.get_message_count() == 3

    def test_get_messages_from_offset(self):
        """测试从 offset 查询消息"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "test-queue.json"
            mq = MessageQueue(queue_file=str(queue_file))

            # 添加消息
            for i in range(5):
                mq.enqueue("session-1", "test", {"index": i})

            # 查询从 offset 2 开始的消息
            messages = mq.get_messages_from_offset(from_offset=2)

            assert len(messages) == 3  # offset 3, 4, 5
            assert messages[0].offset == 3
            assert messages[1].offset == 4
            assert messages[2].offset == 5

    def test_get_messages_with_session_filter(self):
        """测试按 session 过滤查询"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "test-queue.json"
            mq = MessageQueue(queue_file=str(queue_file))

            mq.enqueue("session-1", "test", {"data": 1})
            mq.enqueue("session-2", "test", {"data": 2})
            mq.enqueue("session-1", "test", {"data": 3})

            messages = mq.get_messages_from_offset(from_offset=0, session_id="session-1")

            assert len(messages) == 2
            assert all(m.session_id == "session-1" for m in messages)

    def test_mark_delivered(self):
        """测试标记消息已送达"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "test-queue.json"
            mq = MessageQueue(queue_file=str(queue_file))

            mq.enqueue("session-1", "test", {"data": 1})
            mq.enqueue("session-1", "test", {"data": 2})

            # 标记第一条消息已送达
            result = mq.mark_delivered(1)
            assert result is True

            # 获取未送达消息
            undelivered = mq.get_undelivered_messages("session-1")
            assert len(undelivered) == 1
            assert undelivered[0].offset == 2

    def test_mark_delivered_not_found(self):
        """测试标记不存在的消息"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "test-queue.json"
            mq = MessageQueue(queue_file=str(queue_file))

            result = mq.mark_delivered(999)
            assert result is False

    def test_get_undelivered_messages(self):
        """测试获取未送达消息"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "test-queue.json"
            mq = MessageQueue(queue_file=str(queue_file))

            mq.enqueue("session-1", "test", {"data": 1})
            mq.enqueue("session-1", "test", {"data": 2})
            mq.enqueue("session-2", "test", {"data": 3})

            # 标记第一条已送达
            mq.mark_delivered(1)

            # 获取 session-1 的未送达消息
            undelivered = mq.get_undelivered_messages("session-1")
            assert len(undelivered) == 1
            assert undelivered[0].offset == 2

    def test_clear_session_messages(self):
        """测试清除 session 消息"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "test-queue.json"
            mq = MessageQueue(queue_file=str(queue_file))

            mq.enqueue("session-1", "test", {"data": 1})
            mq.enqueue("session-2", "test", {"data": 2})
            mq.enqueue("session-1", "test", {"data": 3})

            cleared = mq.clear_session_messages("session-1")
            assert cleared == 2
            assert mq.get_message_count() == 1


class TestMessageQueuePersistence:
    """测试消息队列持久化"""

    def test_persistence_save_and_load(self):
        """测试持久化保存和加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "test-queue.json"

            # 创建队列并添加消息
            mq1 = MessageQueue(queue_file=str(queue_file))
            mq1.enqueue("session-1", "test", {"data": "persistent"})
            mq1.enqueue("session-2", "test", {"data": "also persistent"})

            # 创建新队列实例（模拟重启）
            mq2 = MessageQueue(queue_file=str(queue_file))

            assert mq2.get_message_count() == 2
            assert mq2.get_last_offset() == 2

            messages = mq2.get_messages_from_offset(from_offset=0)
            assert messages[0].data == {"data": "persistent"}
            assert messages[1].data == {"data": "also persistent"}

    def test_persistence_delivered_state(self):
        """测试持久化已送达状态"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "test-queue.json"

            # 创建队列，添加消息并标记已送达
            mq1 = MessageQueue(queue_file=str(queue_file))
            mq1.enqueue("session-1", "test", {"data": 1})
            mq1.enqueue("session-1", "test", {"data": 2})
            mq1.mark_delivered(1)

            # 创建新队列实例
            mq2 = MessageQueue(queue_file=str(queue_file))

            undelivered = mq2.get_undelivered_messages("session-1")
            assert len(undelivered) == 1
            assert undelivered[0].offset == 2

    def test_persistence_file_format(self):
        """测试持久化文件格式"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "test-queue.json"

            mq = MessageQueue(queue_file=str(queue_file))
            mq.enqueue("session-1", "test", {"key": "value"})

            # 读取文件验证格式
            with open(queue_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert "current_offset" in data
            assert "messages" in data
            assert len(data["messages"]) == 1
            assert data["messages"][0]["session_id"] == "session-1"
            assert data["messages"][0]["data"] == {"key": "value"}


class TestMessageQueueCleanup:
    """测试消息队列清理机制"""

    def test_cleanup_expired_messages(self):
        """测试清理过期消息"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "test-queue.json"

            # 先创建队列并添加消息
            mq = MessageQueue(queue_file=str(queue_file), ttl_days=1)
            mq.enqueue("session-1", "test", {"data": 1})

            # 修改消息时间为 2 天前
            old_time = (datetime.utcnow() - timedelta(days=2)).isoformat()
            mq._messages[0].timestamp = old_time
            mq._save_messages()

            # 直接调用 cleanup 方法（不重新创建实例，避免初始化时自动清理）
            cleared = mq.cleanup()

            assert cleared == 1
            assert mq.get_message_count() == 0

    def test_no_cleanup_recent_messages(self):
        """测试不清理近期消息"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "test-queue.json"

            mq = MessageQueue(queue_file=str(queue_file), ttl_days=7)
            mq.enqueue("session-1", "test", {"data": 1})

            # 清理不应删除任何消息
            cleared = mq.cleanup()

            assert cleared == 0
            assert mq.get_message_count() == 1

    def test_max_messages_limit(self):
        """测试最大消息数量限制"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "test-queue.json"

            mq = MessageQueue(queue_file=str(queue_file), max_messages=3)

            # 添加 5 条消息
            for i in range(5):
                mq.enqueue("session-1", "test", {"index": i})

            # 只保留最近的 3 条
            assert mq.get_message_count() == 3

            messages = mq.get_messages_from_offset(from_offset=0)
            assert messages[0].data == {"index": 2}
            assert messages[2].data == {"index": 4}


class TestMessageQueueEdgeCases:
    """测试边界情况"""

    def test_empty_queue_operations(self):
        """测试空队列操作"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "test-queue.json"
            mq = MessageQueue(queue_file=str(queue_file))

            assert mq.get_message_count() == 0
            assert mq.get_last_offset() == 0

            messages = mq.get_messages_from_offset(from_offset=0)
            assert messages == []

            undelivered = mq.get_undelivered_messages("session-1")
            assert undelivered == []

    def test_nonexistent_queue_file(self):
        """测试不存在的队列文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "nonexistent" / "queue.json"
            mq = MessageQueue(queue_file=str(queue_file))

            assert mq.get_message_count() == 0

            # 添加消息应该能正常工作
            offset = mq.enqueue("session-1", "test", {"data": 1})
            assert offset == 1

    def test_corrupted_queue_file(self):
        """测试损坏的队列文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "corrupted.json"

            # 写入损坏的 JSON
            with open(queue_file, "w") as f:
                f.write("not valid json")

            # 应该能优雅处理
            mq = MessageQueue(queue_file=str(queue_file))
            assert mq.get_message_count() == 0

    def test_large_message_data(self):
        """测试大数据消息"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "test-queue.json"
            mq = MessageQueue(queue_file=str(queue_file))

            large_data = {"key": "x" * 10000}
            offset = mq.enqueue("session-1", "test", large_data)

            assert offset == 1

            messages = mq.get_messages_from_offset(from_offset=0)
            assert messages[0].data == large_data

    def test_special_characters_in_data(self):
        """测试特殊字符数据"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "test-queue.json"
            mq = MessageQueue(queue_file=str(queue_file))

            special_data = {
                "emoji": "🎉🎊",
                "unicode": "中文测试",
                "quotes": 'He said "Hello"',
                "newlines": "line1\nline2",
            }
            offset = mq.enqueue("session-1", "test", special_data)

            assert offset == 1

            # 验证持久化后能正确加载
            mq2 = MessageQueue(queue_file=str(queue_file))
            messages = mq2.get_messages_from_offset(from_offset=0)
            assert messages[0].data == special_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
