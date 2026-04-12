"""WebSocket ACK 确认管理器测试

测试范围：
- AckManager 基础功能
- ACK 成功场景
- 超时重试
- 最大重试后失败

运行方式：
    uv run pytest tests/test_websocket_ack.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock

from wrapper.src.websocket.ack_manager import AckManager


class TestAckManager:
    """AckManager 单元测试"""

    @pytest.fixture
    def ack_manager(self):
        """创建 AckManager 实例"""
        return AckManager(default_timeout=0.1, max_retries=2)

    def test_initialization(self, ack_manager):
        """测试初始化"""
        assert ack_manager._default_timeout == 0.1
        assert ack_manager._max_retries == 2
        assert ack_manager.pending_count == 0

    @pytest.mark.asyncio
    async def test_send_with_ack_success(self, ack_manager):
        """测试成功发送并收到 ACK"""
        send_callback = AsyncMock()
        message = {"type": "test", "data": "hello"}

        async def delayed_ack():
            await asyncio.sleep(0.05)
            for ack_id in list(ack_manager._pending_acks.keys()):
                ack_manager.handle_ack(ack_id)

        asyncio.create_task(delayed_ack())
        result = await ack_manager.send_with_ack(send_callback, message)

        assert result is True
        assert ack_manager.pending_count == 0
        send_callback.assert_called_once()
        call_args = send_callback.call_args[0][0]
        assert "_ackId" in call_args
        assert call_args["type"] == "test"

    @pytest.mark.asyncio
    async def test_send_with_ack_timeout_and_retry(self, ack_manager):
        """测试超时后重试"""
        send_callback = AsyncMock()
        message = {"type": "test"}

        async def delayed_ack():
            await asyncio.sleep(0.25)
            for ack_id in list(ack_manager._pending_acks.keys()):
                ack_manager.handle_ack(ack_id)

        asyncio.create_task(delayed_ack())
        result = await ack_manager.send_with_ack(send_callback, message, timeout=0.1)

        assert result is True
        assert send_callback.call_count >= 2

    @pytest.mark.asyncio
    async def test_send_with_ack_max_retries_exceeded(self, ack_manager):
        """测试超过最大重试次数后失败"""
        send_callback = AsyncMock()
        message = {"type": "test"}

        result = await ack_manager.send_with_ack(send_callback, message, timeout=0.05)

        assert result is False
        assert send_callback.call_count == 3

    @pytest.mark.asyncio
    async def test_handle_ack_unknown_id(self, ack_manager):
        """测试处理未知 ACK ID"""
        result = ack_manager.handle_ack("unknown-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_is_pending(self, ack_manager):
        """测试 is_pending 方法"""
        send_callback = AsyncMock()
        message = {"type": "test"}

        async def slow_send(msg):
            await asyncio.sleep(0.5)

        task = asyncio.create_task(ack_manager.send_with_ack(slow_send, message))
        await asyncio.sleep(0.05)

        assert ack_manager.pending_count == 1
        pending_ack_id = list(ack_manager._pending_acks.keys())[0]
        assert ack_manager.is_pending(pending_ack_id) is True

        ack_manager.handle_ack(pending_ack_id)
        assert ack_manager.is_pending(pending_ack_id) is False

        try:
            await task
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_get_retry_count(self, ack_manager):
        """测试 get_retry_count 方法"""
        assert ack_manager.get_retry_count("non-existent") == 0

        send_callback = AsyncMock()
        message = {"type": "test"}

        async def delayed_ack():
            await asyncio.sleep(0.25)
            for ack_id in list(ack_manager._pending_acks.keys()):
                ack_manager.handle_ack(ack_id)

        asyncio.create_task(delayed_ack())
        await ack_manager.send_with_ack(send_callback, message, timeout=0.1)

        assert ack_manager.pending_count == 0

    @pytest.mark.asyncio
    async def test_clear_all(self, ack_manager):
        """测试 clear_all 方法"""
        send_callback = AsyncMock()
        message = {"type": "test"}

        task = asyncio.create_task(ack_manager.send_with_ack(send_callback, message))
        await asyncio.sleep(0.05)

        assert ack_manager.pending_count == 1

        await ack_manager.clear_all()

        assert ack_manager.pending_count == 0

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_send_callback_failure(self, ack_manager):
        """测试发送回调失败"""
        send_callback = AsyncMock(side_effect=Exception("Send failed"))
        message = {"type": "test"}

        result = await ack_manager.send_with_ack(send_callback, message)

        assert result is False
        send_callback.assert_called_once()


class TestAckManagerConcurrency:
    """并发测试"""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_acks(self):
        """测试多个并发 ACK"""
        ack_manager = AckManager(default_timeout=0.5, max_retries=1)
        send_callback = AsyncMock()
        results = []

        async def send_and_track(msg_id):
            message = {"type": "test", "id": msg_id}
            result = await ack_manager.send_with_ack(send_callback, message)
            results.append((msg_id, result))

        async def ack_all():
            await asyncio.sleep(0.1)
            for ack_id in list(ack_manager._pending_acks.keys()):
                ack_manager.handle_ack(ack_id)

        tasks = [asyncio.create_task(send_and_track(i)) for i in range(5)]
        asyncio.create_task(ack_all())

        await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(result for _, result in results)
        assert send_callback.call_count == 5


class TestAckManagerConfiguration:
    """配置测试"""

    def test_custom_timeout(self):
        """测试自定义超时"""
        ack_manager = AckManager(default_timeout=10.0)
        assert ack_manager._default_timeout == 10.0

    def test_custom_max_retries(self):
        """测试自定义最大重试次数"""
        ack_manager = AckManager(max_retries=5)
        assert ack_manager._max_retries == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
