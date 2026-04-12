"""WebSocket 心跳机制聚焦测试

测试范围：
- HeartbeatManager 基础功能
- ReliableWebSocketServer 集成
- ping/pong 超时检测

运行方式：
    uv run pytest tests/test_websocket_heartbeat.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from wrapper.src.websocket.heartbeat import HeartbeatManager
from wrapper.src.websocket.reliable_server import ReliableWebSocketServer


class TestHeartbeatManager:
    """HeartbeatManager 单元测试"""

    @pytest.fixture
    def mock_callbacks(self):
        """创建模拟回调函数"""
        return {
            "send_ping": AsyncMock(),
            "on_connection_lost": AsyncMock(),
        }

    @pytest.fixture
    def heartbeat_manager(self, mock_callbacks):
        """创建 HeartbeatManager 实例"""
        return HeartbeatManager(
            send_ping=mock_callbacks["send_ping"],
            on_connection_lost=mock_callbacks["on_connection_lost"],
            interval=0.1,  # 100ms 间隔，加速测试
            timeout=0.05,  # 50ms 超时
            max_missing=2,
        )

    @pytest.mark.asyncio
    async def test_initialization(self, heartbeat_manager, mock_callbacks):
        """测试初始化"""
        assert heartbeat_manager.is_running is False
        assert heartbeat_manager.missing_count == 0
        assert heartbeat_manager.last_pong_time > 0

    @pytest.mark.asyncio
    async def test_start_stop(self, heartbeat_manager):
        """测试启动和停止"""
        await heartbeat_manager.start()
        assert heartbeat_manager.is_running is True

        await heartbeat_manager.stop()
        assert heartbeat_manager.is_running is False

    @pytest.mark.asyncio
    async def test_double_start(self, heartbeat_manager):
        """测试重复启动"""
        await heartbeat_manager.start()
        await heartbeat_manager.start()  # 不应报错
        assert heartbeat_manager.is_running is True

    @pytest.mark.asyncio
    async def test_stop_without_start(self, heartbeat_manager):
        """测试未启动就停止"""
        await heartbeat_manager.stop()  # 不应报错
        assert heartbeat_manager.is_running is False

    @pytest.mark.asyncio
    async def test_pong_received(self, heartbeat_manager):
        """测试 pong 接收处理"""
        await heartbeat_manager.start()

        # 模拟收到 pong
        heartbeat_manager.on_pong_received()
        assert heartbeat_manager.missing_count == 0

        await heartbeat_manager.stop()

    @pytest.mark.asyncio
    async def test_connection_lost_on_timeout(self, mock_callbacks):
        """测试超时触发连接丢失"""
        manager = HeartbeatManager(
            send_ping=mock_callbacks["send_ping"],
            on_connection_lost=mock_callbacks["on_connection_lost"],
            interval=0.05,
            timeout=0.02,
            max_missing=1,  # 1 次未响应就触发
        )

        await manager.start()

        # 等待超时发生
        await asyncio.sleep(0.15)

        # 验证 on_connection_lost 被调用
        mock_callbacks["on_connection_lost"].assert_called_once()

        await manager.stop()

    @pytest.mark.asyncio
    async def test_multiple_missing_pongs(self, mock_callbacks):
        """测试多次未响应才触发连接丢失"""
        manager = HeartbeatManager(
            send_ping=mock_callbacks["send_ping"],
            on_connection_lost=mock_callbacks["on_connection_lost"],
            interval=0.05,
            timeout=0.02,
            max_missing=3,
        )

        await manager.start()

        # 等待部分超时（但不到 3 次）
        await asyncio.sleep(0.1)

        # 此时应该还未触发连接丢失
        assert mock_callbacks["on_connection_lost"].call_count == 0

        # 等待足够时间触发 3 次超时
        await asyncio.sleep(0.15)

        # 验证 on_connection_lost 被调用
        assert mock_callbacks["on_connection_lost"].call_count >= 1

        await manager.stop()


class TestReliableWebSocketServer:
    """ReliableWebSocketServer 集成测试"""

    @pytest.fixture
    def mock_websocket(self):
        """创建模拟 WebSocket"""
        ws = MagicMock()
        ws.accept = AsyncMock()
        ws.close = AsyncMock()
        ws.send_json = AsyncMock()
        ws.send_text = AsyncMock()
        ws.receive_json = AsyncMock()
        ws.receive_text = AsyncMock()
        ws.client = MagicMock(host="127.0.0.1", port=12345)
        return ws

    @pytest.fixture
    def reliable_server(self, mock_websocket):
        """创建 ReliableWebSocketServer 实例"""
        return ReliableWebSocketServer(
            websocket=mock_websocket,
            heartbeat_interval=0.05,  # 缩短间隔加速测试
            heartbeat_timeout=0.03,
            max_missing_pongs=2,
        )

    @pytest.mark.asyncio
    async def test_accept_starts_heartbeat(self, reliable_server, mock_websocket):
        """测试 accept 启动心跳"""
        await reliable_server.accept()

        assert reliable_server.is_connected is True
        mock_websocket.accept.assert_called_once()

        await reliable_server.close()

    @pytest.mark.asyncio
    async def test_close_stops_heartbeat(self, reliable_server, mock_websocket):
        """测试 close 停止心跳"""
        await reliable_server.accept()
        await reliable_server.close()

        assert reliable_server.is_connected is False
        mock_websocket.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_json_when_connected(self, reliable_server, mock_websocket):
        """测试连接状态下发送 JSON"""
        await reliable_server.accept()

        test_data = {"type": "test", "data": "hello"}
        await reliable_server.send_json(test_data)

        mock_websocket.send_json.assert_called_with(test_data)

        await reliable_server.close()

    @pytest.mark.asyncio
    async def test_send_json_when_disconnected(self, reliable_server):
        """测试断开状态下发送 JSON 报错"""
        with pytest.raises(RuntimeError, match="WebSocket 未连接"):
            await reliable_server.send_json({"type": "test"})

    @pytest.mark.asyncio
    async def test_ping_message_format(self, reliable_server, mock_websocket):
        """测试 ping 消息格式 - 验证心跳配置正确传递"""
        await reliable_server.accept()

        try:
            assert reliable_server._heartbeat_manager is not None
            assert reliable_server._heartbeat_manager.is_running is True
            assert reliable_server._heartbeat_manager._interval == 0.05
            assert reliable_server._heartbeat_manager._timeout == 0.03
            assert reliable_server._heartbeat_manager._max_missing == 2
        finally:
            await reliable_server.close()

    @pytest.mark.asyncio
    async def test_pong_response_handling(self, reliable_server, mock_websocket):
        """测试 pong 响应处理 - 验证 on_pong_received 正确设置事件"""
        await reliable_server.accept()

        manager = reliable_server._heartbeat_manager

        try:
            manager._pong_event = asyncio.Event()
            assert not manager._pong_event.is_set()

            manager.on_pong_received()
            assert manager._pong_event.is_set()
            assert manager.last_pong_time > 0
        finally:
            await reliable_server.close()


class TestHeartbeatIntegration:
    """心跳机制集成测试"""

    @pytest.mark.asyncio
    async def test_full_heartbeat_cycle(self):
        """测试完整心跳周期"""
        ping_count = 0
        pong_received = asyncio.Event()

        async def send_ping():
            nonlocal ping_count
            ping_count += 1

        async def on_connection_lost():
            pass

        manager = HeartbeatManager(
            send_ping=send_ping,
            on_connection_lost=on_connection_lost,
            interval=0.1,
            timeout=0.05,
            max_missing=5,
        )

        await manager.start()

        # 模拟收到 pong（防止超时）
        asyncio.create_task(self._send_pong_after(manager, 0.02, pong_received))

        # 等待几次心跳
        await asyncio.sleep(0.25)

        # 验证 ping 被发送
        assert ping_count >= 2

        await manager.stop()

    async def _send_pong_after(self, manager, delay, event):
        await asyncio.sleep(delay)
        manager.on_pong_received()
        event.set()

    @pytest.mark.asyncio
    async def test_configurable_parameters(self):
        """测试可配置参数"""
        custom_interval = 0.2
        custom_timeout = 0.1
        custom_max_missing = 5

        manager = HeartbeatManager(
            send_ping=AsyncMock(),
            on_connection_lost=AsyncMock(),
            interval=custom_interval,
            timeout=custom_timeout,
            max_missing=custom_max_missing,
        )

        assert manager._interval == custom_interval
        assert manager._timeout == custom_timeout
        assert manager._max_missing == custom_max_missing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
