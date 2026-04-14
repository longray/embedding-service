"""WebSocket 集成测试

测试范围：
- ReliableWebSocketServer 与 AckManager 集成
- ACK 消息处理流程
- 发送消息并等待确认

运行方式：
    uv run pytest tests/test_websocket_integration.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from wrapper.src.websocket import ReliableWebSocketServer, AckManager


class TestWebSocketAckIntegration:
    """WebSocket ACK 集成测试"""

    @pytest.fixture
    def mock_websocket(self):
        """创建模拟 WebSocket"""
        ws = AsyncMock()
        ws.client = MagicMock()
        ws.client.host = "127.0.0.1"
        ws.client.port = 12345
        return ws

    @pytest.fixture
    def reliable_server(self, mock_websocket):
        """创建 ReliableWebSocketServer 实例"""
        server = ReliableWebSocketServer(
            websocket=mock_websocket,
            heartbeat_interval=60.0,
            heartbeat_timeout=10.0,
            max_missing_pongs=2,
            ack_timeout=0.1,
            ack_max_retries=2,
        )
        return server

    @pytest.mark.asyncio
    async def test_ack_manager_initialized_on_accept(self, reliable_server, mock_websocket):
        """测试 accept() 时 AckManager 被初始化"""
        assert reliable_server._ack_manager is None

        await reliable_server.accept()

        assert reliable_server._ack_manager is not None
        assert isinstance(reliable_server._ack_manager, AckManager)

        await reliable_server.close()

    @pytest.mark.asyncio
    async def test_send_json_with_ack_without_connection_raises(self, reliable_server):
        """测试未连接时发送带 ACK 的消息抛出异常"""
        with pytest.raises(RuntimeError, match="WebSocket 未连接"):
            await reliable_server.send_json_with_ack({"type": "test"})

    @pytest.mark.asyncio
    async def test_send_json_without_connection_raises(self, reliable_server):
        """测试未连接时发送消息抛出异常"""
        with pytest.raises(RuntimeError, match="WebSocket 未连接"):
            await reliable_server.send_json({"type": "test"})

    @pytest.mark.asyncio
    async def test_ack_manager_cleanup_on_close(self, reliable_server, mock_websocket):
        """测试关闭时清理 AckManager"""
        await reliable_server.accept()
        assert reliable_server._ack_manager is not None

        await reliable_server.close()

        assert not reliable_server._is_connected

    @pytest.mark.asyncio
    async def test_send_json_with_ack_manager_not_initialized_raises(self, reliable_server, mock_websocket):
        """测试 ACK 管理器未初始化时发送带 ACK 的消息抛出异常"""
        reliable_server._is_connected = True

        with pytest.raises(RuntimeError, match="ACK 管理器未初始化"):
            await reliable_server.send_json_with_ack({"type": "test"})


class TestWebSocketIntegrationEdgeCases:
    """WebSocket 集成边界情况测试"""

    @pytest.fixture
    def mock_websocket(self):
        """创建模拟 WebSocket"""
        ws = AsyncMock()
        ws.client = MagicMock()
        ws.client.host = "127.0.0.1"
        ws.client.port = 12345
        return ws

    @pytest.mark.asyncio
    async def test_receive_loop_handles_non_dict_message(self, mock_websocket):
        """测试接收循环处理非字典消息"""
        server = ReliableWebSocketServer(
            websocket=mock_websocket,
            heartbeat_interval=60.0,
            ack_timeout=0.1,
            ack_max_retries=1,
        )

        messages = ["not a dict", {"type": "pong"}]
        msg_iter = iter(messages)

        async def mock_receive_json():
            try:
                return next(msg_iter)
            except StopIteration:
                raise StopAsyncIteration

        mock_websocket.receive_json = mock_receive_json

        await server.accept()
        await server.close()

    @pytest.mark.asyncio
    async def test_receive_loop_handles_unknown_message_type(self, mock_websocket):
        """测试接收循环处理未知消息类型"""
        server = ReliableWebSocketServer(
            websocket=mock_websocket,
            heartbeat_interval=60.0,
            ack_timeout=0.1,
            ack_max_retries=1,
        )

        messages = [{"type": "unknown", "data": "test"}, {"type": "pong"}]
        msg_iter = iter(messages)

        async def mock_receive_json():
            try:
                return next(msg_iter)
            except StopIteration:
                raise StopAsyncIteration

        mock_websocket.receive_json = mock_receive_json

        await server.accept()
        await server.close()

    @pytest.mark.asyncio
    async def test_receive_loop_handles_ack_without_id(self, mock_websocket):
        """测试接收循环处理没有 _ackId 的 ACK 消息"""
        server = ReliableWebSocketServer(
            websocket=mock_websocket,
            heartbeat_interval=60.0,
            ack_timeout=0.1,
            ack_max_retries=1,
        )

        messages = [{"type": "ack"}, {"type": "pong"}]
        msg_iter = iter(messages)

        async def mock_receive_json():
            try:
                return next(msg_iter)
            except StopIteration:
                raise StopAsyncIteration

        mock_websocket.receive_json = mock_receive_json

        await server.accept()
        await server.close()
