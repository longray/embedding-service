"""WebSocket 客户端配置测试

测试范围：
- WebSocket 连接参数支持 mode=diff|full
- 动态切换模式 API
- 向后兼容（默认 full 模式）

运行方式：
    uv run pytest tests/test_websocket_client_config.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from wrapper.src.websocket import ReliableWebSocketServer


class TestWebSocketClientConfig:
    """WebSocket 客户端配置测试"""

    @pytest.fixture
    def mock_websocket(self):
        """创建模拟 WebSocket"""
        ws = AsyncMock()
        ws.client = MagicMock()
        ws.client.host = "127.0.0.1"
        ws.client.port = 12345
        ws.query_params = {}
        return ws

    @pytest.mark.asyncio
    async def test_default_mode_is_full(self, mock_websocket):
        """测试默认模式为 full"""
        server = ReliableWebSocketServer(mock_websocket)

        await server.accept()

        assert server.diff_mode == "full"
        assert server._diff_manager is not None
        assert server._diff_manager.mode == "full"

        await server.close()

    @pytest.mark.asyncio
    async def test_diff_mode_initialization(self, mock_websocket):
        """测试 diff 模式初始化"""
        server = ReliableWebSocketServer(
            mock_websocket,
            diff_mode="diff",
            diff_threshold=30.0,
            diff_min_size=50,
        )

        await server.accept()

        assert server.diff_mode == "diff"
        assert server._diff_manager is not None
        assert server._diff_manager.mode == "diff"
        assert server._diff_manager.threshold == 30.0

        await server.close()

    @pytest.mark.asyncio
    async def test_set_diff_mode_changes_mode(self, mock_websocket):
        """测试设置 diff 模式"""
        server = ReliableWebSocketServer(mock_websocket)

        await server.accept()

        assert server.diff_mode == "full"

        server.set_diff_mode("diff")

        assert server.diff_mode == "diff"
        assert server._diff_manager is not None
        assert server._diff_manager.mode == "diff"

        await server.close()

    @pytest.mark.asyncio
    async def test_send_data_respects_diff_mode(self, mock_websocket):
        """测试发送数据遵循 diff 模式"""
        server = ReliableWebSocketServer(
            mock_websocket,
            diff_mode="full",
        )

        await server.accept()

        data = {"id": "test-1", "content": "Hello"}
        await server.send_data_with_diff("key1", data)

        sent_message = mock_websocket.send_json.call_args[0][0]
        assert sent_message["type"] == "full"
        assert sent_message["data"] == data

        await server.close()


class TestWebSocketBackwardCompatibility:
    """WebSocket 向后兼容性测试"""

    @pytest.fixture
    def mock_websocket(self):
        """创建模拟 WebSocket"""
        ws = AsyncMock()
        ws.client = MagicMock()
        ws.client.host = "127.0.0.1"
        ws.client.port = 12345
        return ws

    @pytest.mark.asyncio
    async def test_no_mode_parameter_defaults_to_full(self, mock_websocket):
        """测试无 mode 参数默认为 full"""
        server = ReliableWebSocketServer(mock_websocket)

        await server.accept()

        assert server.diff_mode == "full"

        await server.close()

    @pytest.mark.asyncio
    async def test_existing_api_unchanged(self, mock_websocket):
        """测试现有 API 不变"""
        server = ReliableWebSocketServer(
            mock_websocket,
            heartbeat_interval=30.0,
            heartbeat_timeout=5.0,
            max_missing_pongs=2,
        )

        await server.accept()

        # 原有功能正常工作
        assert server.is_connected is True
        assert server._heartbeat_manager is not None
        assert server._ack_manager is not None

        await server.close()

    @pytest.mark.asyncio
    async def test_send_json_still_works(self, mock_websocket):
        """测试 send_json 仍然可用"""
        server = ReliableWebSocketServer(mock_websocket)

        await server.accept()

        data = {"type": "test", "content": "Hello"}
        await server.send_json(data)

        mock_websocket.send_json.assert_called_with(data)

        await server.close()


class TestWebSocketSessionRecovery:
    """WebSocket Session 恢复测试"""

    @pytest.fixture
    def mock_websocket(self):
        """创建模拟 WebSocket"""
        ws = AsyncMock()
        ws.client = MagicMock()
        ws.client.host = "127.0.0.1"
        ws.client.port = 12345
        return ws

    @pytest.mark.asyncio
    async def test_create_session_on_connect(self, mock_websocket):
        """测试连接时创建 Session"""
        server = ReliableWebSocketServer(mock_websocket)

        await server.accept()

        session_id = server.create_session()

        assert session_id is not None
        assert session_id.startswith("sess-")
        assert server.session_id == session_id

        await server.close()

    @pytest.mark.asyncio
    async def test_restore_existing_session(self, mock_websocket):
        """测试恢复已有 Session"""
        server1 = ReliableWebSocketServer(mock_websocket)
        await server1.accept()

        session_id = server1.create_session()
        server1.update_message_offset(100)
        await server1.close()

        server2 = ReliableWebSocketServer(mock_websocket)
        await server2.accept()

        result = server2.restore_session(session_id)

        assert result is True
        assert server2.session_id == session_id
        assert server2.message_offset == 100

        await server2.close()
