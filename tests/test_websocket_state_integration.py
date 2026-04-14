"""WebSocket State Recovery 集成测试

测试范围：
- ReliableWebSocketServer 与 StateRecoveryManager 集成
- Session 创建和恢复
- Offset 管理

运行方式：
    uv run pytest tests/test_websocket_state_integration.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from wrapper.src.websocket import ReliableWebSocketServer, StateRecoveryManager


class TestWebSocketStateIntegration:
    """WebSocket State Recovery 集成测试"""

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
        )
        return server

    @pytest.mark.asyncio
    async def test_state_recovery_initialized_on_accept(self, reliable_server, mock_websocket):
        """测试 accept() 时 StateRecoveryManager 被初始化"""
        assert reliable_server._state_recovery is None

        await reliable_server.accept()

        assert reliable_server._state_recovery is not None
        assert isinstance(reliable_server._state_recovery, StateRecoveryManager)

        await reliable_server.close()

    @pytest.mark.asyncio
    async def test_create_session_generates_id(self, reliable_server, mock_websocket):
        """测试创建 Session 生成 ID"""
        await reliable_server.accept()

        session_id = reliable_server.create_session()

        assert session_id is not None
        assert session_id.startswith("sess-")
        assert reliable_server.session_id == session_id
        assert reliable_server.message_offset == 0

        await reliable_server.close()

    @pytest.mark.asyncio
    async def test_restore_session_existing(self, reliable_server, mock_websocket):
        """测试恢复已存在的 Session"""
        await reliable_server.accept()

        session_id = reliable_server.create_session()
        reliable_server.update_message_offset(100)

        await reliable_server.close()

        new_server = ReliableWebSocketServer(mock_websocket)
        await new_server.accept()

        result = new_server.restore_session(session_id)

        assert result is True
        assert new_server.session_id == session_id
        assert new_server.message_offset == 100

        await new_server.close()

    @pytest.mark.asyncio
    async def test_restore_session_nonexistent(self, reliable_server, mock_websocket):
        """测试恢复不存在的 Session"""
        await reliable_server.accept()

        result = reliable_server.restore_session("sess-nonexistent-123")

        assert result is False
        assert reliable_server.session_id is None

        await reliable_server.close()

    @pytest.mark.asyncio
    async def test_update_message_offset(self, reliable_server, mock_websocket):
        """测试更新消息 offset"""
        await reliable_server.accept()

        reliable_server.create_session()
        reliable_server.update_message_offset(50)

        assert reliable_server.message_offset == 50

        await reliable_server.close()

    @pytest.mark.asyncio
    async def test_increment_message_offset(self, reliable_server, mock_websocket):
        """测试递增消息 offset"""
        await reliable_server.accept()

        reliable_server.create_session()

        offset1 = reliable_server.increment_message_offset()
        offset2 = reliable_server.increment_message_offset()

        assert offset1 == 1
        assert offset2 == 2
        assert reliable_server.message_offset == 2

        await reliable_server.close()

    @pytest.mark.asyncio
    async def test_state_saved_on_close(self, reliable_server, mock_websocket):
        """测试关闭时保存状态"""
        await reliable_server.accept()

        session_id = reliable_server.create_session()
        reliable_server.update_message_offset(42)

        await reliable_server.close()

        new_server = ReliableWebSocketServer(mock_websocket)
        await new_server.accept()

        result = new_server.restore_session(session_id)

        assert result is True
        assert new_server.message_offset == 42

        await new_server.close()

    @pytest.mark.asyncio
    async def test_create_session_without_manager_raises(self, reliable_server):
        """测试 StateRecoveryManager 未初始化时创建 Session 抛出异常"""
        with pytest.raises(RuntimeError, match="StateRecoveryManager 未初始化"):
            reliable_server.create_session()

    @pytest.mark.asyncio
    async def test_restore_session_without_manager_raises(self, reliable_server):
        """测试 StateRecoveryManager 未初始化时恢复 Session 抛出异常"""
        with pytest.raises(RuntimeError, match="StateRecoveryManager 未初始化"):
            reliable_server.restore_session("sess-test-123")


class TestWebSocketStateEdgeCases:
    """WebSocket State Recovery 边界情况测试"""

    @pytest.fixture
    def mock_websocket(self):
        """创建模拟 WebSocket"""
        ws = AsyncMock()
        ws.client = MagicMock()
        ws.client.host = "127.0.0.1"
        ws.client.port = 12345
        return ws

    @pytest.mark.asyncio
    async def test_multiple_sessions_independent(self, mock_websocket):
        """测试多个 Session 相互独立"""
        server1 = ReliableWebSocketServer(mock_websocket)
        await server1.accept()

        session1 = server1.create_session()
        server1.update_message_offset(100)
        await server1.close()

        server2 = ReliableWebSocketServer(mock_websocket)
        await server2.accept()

        session2 = server2.create_session()
        server2.update_message_offset(200)
        await server2.close()

        server3 = ReliableWebSocketServer(mock_websocket)
        await server3.accept()

        result1 = server3.restore_session(session1)
        assert result1 is True
        assert server3.message_offset == 100

        result2 = server3.restore_session(session2)
        assert result2 is True
        assert server3.message_offset == 200

        await server3.close()

    @pytest.mark.asyncio
    async def test_session_persistence_across_connections(self, mock_websocket):
        """测试 Session 在连接间持久化"""
        server = ReliableWebSocketServer(mock_websocket)
        await server.accept()

        session_id = server.create_session()
        server.increment_message_offset()
        server.increment_message_offset()

        await server.close()

        new_server = ReliableWebSocketServer(mock_websocket)
        await new_server.accept()

        result = new_server.restore_session(session_id)
        assert result is True
        assert new_server.message_offset == 2

        new_server.increment_message_offset()
        await new_server.close()

        final_server = ReliableWebSocketServer(mock_websocket)
        await final_server.accept()

        result = final_server.restore_session(session_id)
        assert result is True
        assert final_server.message_offset == 3

        await final_server.close()
