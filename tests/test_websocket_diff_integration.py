"""WebSocket Diff 集成测试

测试范围：
- ReliableWebSocketServer 与 DiffManager 集成
- diff/full 模式切换
- 状态缓存管理

运行方式：
    uv run pytest tests/test_websocket_diff_integration.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from wrapper.src.websocket import ReliableWebSocketServer, DiffManager


class TestWebSocketDiffIntegration:
    """WebSocket Diff 集成测试"""

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
            diff_mode="diff",
            diff_threshold=50.0,
            diff_min_size=10,
        )
        return server

    @pytest.mark.asyncio
    async def test_diff_manager_initialized_on_accept(self, reliable_server, mock_websocket):
        """测试 accept() 时 DiffManager 被初始化"""
        assert reliable_server._diff_manager is None

        await reliable_server.accept()

        assert reliable_server._diff_manager is not None
        assert isinstance(reliable_server._diff_manager, DiffManager)

        await reliable_server.close()

    @pytest.mark.asyncio
    async def test_send_data_with_diff_full_mode_first_time(self, reliable_server, mock_websocket):
        """测试首次发送数据使用 full 模式"""
        await reliable_server.accept()

        data = {"id": "test-1", "content": "Hello World", "version": 1}

        result = await reliable_server.send_data_with_diff("key1", data)

        assert result is True

        sent_message = mock_websocket.send_json.call_args[0][0]
        assert sent_message["type"] == "full"
        assert sent_message["key"] == "key1"
        assert sent_message["data"] == data

        await reliable_server.close()

    @pytest.mark.asyncio
    async def test_send_data_with_diff_creates_state(self, reliable_server, mock_websocket):
        """测试发送数据后创建状态缓存"""
        await reliable_server.accept()

        data = {"id": "test-1", "content": "Hello World"}
        await reliable_server.send_data_with_diff("key1", data)

        cached_state = reliable_server._diff_manager.get_state("key1")
        assert cached_state is not None
        assert cached_state["content"] == "Hello World"

        await reliable_server.close()

    @pytest.mark.asyncio
    async def test_set_diff_mode_changes_mode(self, reliable_server, mock_websocket):
        """测试设置 diff 模式"""
        await reliable_server.accept()

        assert reliable_server.diff_mode == "diff"

        reliable_server.set_diff_mode("full")

        assert reliable_server.diff_mode == "full"
        assert reliable_server._diff_manager.mode == "full"

        await reliable_server.close()

    @pytest.mark.asyncio
    async def test_update_diff_state_manually(self, reliable_server, mock_websocket):
        """测试手动更新 diff 状态"""
        await reliable_server.accept()

        data = {"id": "test-1", "content": "Initial"}
        reliable_server.update_diff_state("key1", data)

        cached = reliable_server._diff_manager.get_state("key1")
        assert cached["content"] == "Initial"

        await reliable_server.close()

    @pytest.mark.asyncio
    async def test_clear_diff_state_single_key(self, reliable_server, mock_websocket):
        """测试清除单个 key 的 diff 状态"""
        await reliable_server.accept()

        reliable_server.update_diff_state("key1", {"data": 1})
        reliable_server.update_diff_state("key2", {"data": 2})

        reliable_server.clear_diff_state("key1")

        assert reliable_server._diff_manager.get_state("key1") is None
        assert reliable_server._diff_manager.get_state("key2") is not None

        await reliable_server.close()

    @pytest.mark.asyncio
    async def test_clear_diff_state_all(self, reliable_server, mock_websocket):
        """测试清除所有 diff 状态"""
        await reliable_server.accept()

        reliable_server.update_diff_state("key1", {"data": 1})
        reliable_server.update_diff_state("key2", {"data": 2})

        reliable_server.clear_diff_state()

        assert reliable_server._diff_manager.state_count == 0

        await reliable_server.close()

    @pytest.mark.asyncio
    async def test_send_data_without_connection_raises(self, reliable_server):
        """测试未连接时发送数据抛出异常"""
        with pytest.raises(RuntimeError, match="WebSocket 未连接"):
            await reliable_server.send_data_with_diff("key1", {"data": 1})

    @pytest.mark.asyncio
    async def test_set_diff_mode_without_manager_raises(self, reliable_server):
        """测试 DIFF 管理器未初始化时设置模式抛出异常"""
        with pytest.raises(RuntimeError, match="DIFF 管理器未初始化"):
            reliable_server.set_diff_mode("full")

    @pytest.mark.asyncio
    async def test_update_state_without_manager_raises(self, reliable_server):
        """测试 DIFF 管理器未初始化时更新状态抛出异常"""
        with pytest.raises(RuntimeError, match="DIFF 管理器未初始化"):
            reliable_server.update_diff_state("key1", {"data": 1})

    @pytest.mark.asyncio
    async def test_clear_state_without_manager_raises(self, reliable_server):
        """测试 DIFF 管理器未初始化时清除状态抛出异常"""
        with pytest.raises(RuntimeError, match="DIFF 管理器未初始化"):
            reliable_server.clear_diff_state("key1")


class TestWebSocketDiffEdgeCases:
    """WebSocket Diff 边界情况测试"""

    @pytest.fixture
    def mock_websocket(self):
        """创建模拟 WebSocket"""
        ws = AsyncMock()
        ws.client = MagicMock()
        ws.client.host = "127.0.0.1"
        ws.client.port = 12345
        return ws

    @pytest.mark.asyncio
    async def test_send_data_with_metadata(self, mock_websocket):
        """测试发送带元数据的数据"""
        server = ReliableWebSocketServer(
            websocket=mock_websocket,
            heartbeat_interval=60.0,
            diff_mode="full",
        )

        await server.accept()

        data = {"id": "test-1", "content": "Hello"}
        metadata = {"timestamp": 1234567890, "source": "test"}

        await server.send_data_with_diff("key1", data, metadata=metadata)

        sent_message = mock_websocket.send_json.call_args[0][0]
        assert sent_message["metadata"] == metadata

        await server.close()

    @pytest.mark.asyncio
    async def test_default_diff_mode_is_diff(self, mock_websocket):
        """测试默认 diff 模式为 diff"""
        server = ReliableWebSocketServer(
            websocket=mock_websocket,
            heartbeat_interval=60.0,
        )

        await server.accept()

        assert server.diff_mode == "diff"
        assert server._diff_manager is not None
        assert server._diff_manager.mode == "diff"

        await server.close()
