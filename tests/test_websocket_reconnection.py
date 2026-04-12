"""WebSocket 重连管理器测试

测试范围：
- ReconnectionManager 基础功能
- 指数退避计算
- 随机抖动
- 重连调度

运行方式：
    uv run pytest tests/test_websocket_reconnection.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock

from wrapper.src.websocket.reconnection import ReconnectionManager


class TestReconnectionManager:
    """ReconnectionManager 单元测试"""

    @pytest.fixture
    def reconnection_manager(self):
        """创建 ReconnectionManager 实例"""
        return ReconnectionManager(max_retries=10)

    def test_initialization(self, reconnection_manager):
        """测试初始化"""
        assert reconnection_manager.max_retries == 10
        assert reconnection_manager.current_attempt == 0
        assert reconnection_manager.is_reconnecting is False
        assert reconnection_manager.can_reconnect is True

    def test_calculate_delay_sequence(self, reconnection_manager):
        """测试指数退避序列"""
        expected_delays = [1, 2, 4, 8, 16, 32, 64, 128, 256, 300]

        for i, expected in enumerate(expected_delays):
            delay = reconnection_manager.calculate_delay(i)
            assert delay is not None
            assert expected <= delay <= expected + 1

    def test_calculate_delay_with_jitter(self, reconnection_manager):
        """测试随机抖动"""
        delays = [reconnection_manager.calculate_delay(0) for _ in range(100)]

        for delay in delays:
            assert 1 <= delay <= 2

        unique_delays = len(set(delays))
        assert unique_delays > 1

    def test_calculate_delay_exceeds_max_retries(self, reconnection_manager):
        """测试超过最大重试次数"""
        delay = reconnection_manager.calculate_delay(10)
        assert delay is None

    def test_calculate_delay_negative_attempt(self, reconnection_manager):
        """测试负数重试次数"""
        with pytest.raises(ValueError, match="attempt must be non-negative"):
            reconnection_manager.calculate_delay(-1)

    @pytest.mark.asyncio
    async def test_reset_counter(self, reconnection_manager):
        """测试重置计数器"""
        reconnection_manager._current_attempt = 5
        reconnection_manager._is_reconnecting = True

        reconnection_manager.reset_counter()

        assert reconnection_manager.current_attempt == 0
        assert reconnection_manager.is_reconnecting is False

    @pytest.mark.asyncio
    async def test_schedule_reconnect_success(self, reconnection_manager):
        """测试成功调度重连"""
        callback = AsyncMock()

        result = await reconnection_manager.schedule_reconnect(callback, attempt=0)

        assert result is True
        assert reconnection_manager.is_reconnecting is True
        assert reconnection_manager.current_attempt == 1

        await asyncio.sleep(2.5)

        callback.assert_called_once()
        assert reconnection_manager.is_reconnecting is False

    @pytest.mark.asyncio
    async def test_schedule_reconnect_exceeds_max(self, reconnection_manager):
        """测试超过最大重试次数"""
        callback = AsyncMock()

        result = await reconnection_manager.schedule_reconnect(callback, attempt=10)

        assert result is False
        assert reconnection_manager.is_reconnecting is False

    @pytest.mark.asyncio
    async def test_schedule_reconnect_while_reconnecting(self, reconnection_manager):
        """测试重连进行中再次调度"""
        callback = AsyncMock()

        await reconnection_manager.schedule_reconnect(callback, attempt=0)
        result = await reconnection_manager.schedule_reconnect(callback, attempt=1)

        assert result is False

        await reconnection_manager.cancel_reconnect()

    @pytest.mark.asyncio
    async def test_cancel_reconnect(self, reconnection_manager):
        """测试取消重连"""
        callback = AsyncMock()

        await reconnection_manager.schedule_reconnect(callback, attempt=0)
        assert reconnection_manager.is_reconnecting is True

        await reconnection_manager.cancel_reconnect()

        assert reconnection_manager.is_reconnecting is False
        callback.assert_not_called()

    def test_can_reconnect_property(self, reconnection_manager):
        """测试 can_reconnect 属性"""
        assert reconnection_manager.can_reconnect is True

        reconnection_manager._current_attempt = 9
        assert reconnection_manager.can_reconnect is True

        reconnection_manager._current_attempt = 10
        assert reconnection_manager.can_reconnect is False

    @pytest.mark.asyncio
    async def test_reconnect_callback_failure(self, reconnection_manager):
        """测试重连回调失败"""
        callback = AsyncMock(side_effect=Exception("Connection failed"))

        await reconnection_manager.schedule_reconnect(callback, attempt=0)

        await asyncio.sleep(2.5)

        assert reconnection_manager.is_reconnecting is False


class TestReconnectionBackoffSequence:
    """指数退避序列测试"""

    def test_backoff_sequence_values(self):
        """测试退避序列值"""
        expected = [1, 2, 4, 8, 16, 32, 64, 128, 256, 300]
        assert ReconnectionManager.BACKOFF_SEQUENCE == expected

    def test_backoff_sequence_length(self):
        """测试退避序列长度"""
        assert len(ReconnectionManager.BACKOFF_SEQUENCE) == 10


class TestReconnectionIntegration:
    """重连机制集成测试"""

    @pytest.mark.asyncio
    async def test_full_reconnection_cycle(self):
        """测试完整重连周期"""
        manager = ReconnectionManager(max_retries=3)
        reconnect_count = 0

        async def reconnect_callback():
            nonlocal reconnect_count
            reconnect_count += 1

        for attempt in range(3):
            manager.reset_counter()
            result = await manager.schedule_reconnect(reconnect_callback, attempt=0)
            assert result is True
            await asyncio.sleep(2.5)

        assert reconnect_count == 3

    @pytest.mark.asyncio
    async def test_custom_max_retries(self):
        """测试自定义最大重试次数"""
        manager = ReconnectionManager(max_retries=5)

        assert manager.max_retries == 5

        for i in range(5):
            delay = manager.calculate_delay(i)
            assert delay is not None

        assert manager.calculate_delay(5) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
