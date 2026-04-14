"""WebSocket 自动恢复集成测试

测试范围：
- 断线重连后自动恢复 session
- 同步丢失消息（from_offset）
- 恢复后发送 ACK 确认
- 恢复失败进入降级模式

运行方式：
    uv run pytest tests/test_websocket_auto_recovery.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from wrapper.src.websocket import (
    ReliableWebSocketServer,
    MessageQueue,
    StateRecoveryManager,
    AutoRecoveryMixin,
)


class MockServer(ReliableWebSocketServer, AutoRecoveryMixin):
    """模拟服务器，继承 ReliableWebSocketServer 和 AutoRecoveryMixin"""

    pass


class TestAutoRecoveryMixin:
    """AutoRecoveryMixin 集成测试"""

    @pytest.fixture
    def mock_websocket(self, tmp_path):
        """创建模拟 WebSocket"""
        ws = AsyncMock()
        ws.client = MagicMock()
        ws.client.host = "127.0.0.1"
        ws.client.port = 12345
        return ws

    @pytest.fixture
    def server_with_recovery(self, mock_websocket, tmp_path):
        """创建带恢复功能的服务器"""
        server = MockServer(mock_websocket)
        # 手动设置恢复相关属性
        server._auto_recovery_enabled = True
        server._recovery_failed = False
        server._message_queue = MessageQueue(queue_file=str(tmp_path / "messages.json"))
        server._state_recovery = StateRecoveryManager(state_file=str(tmp_path / "state.json"))
        return server

    @pytest.mark.asyncio
    async def test_recover_from_disconnect_success(self, server_with_recovery, mock_websocket):
        """测试成功从断线恢复"""
        # 创建 session 和消息
        await server_with_recovery.accept()
        session_id = server_with_recovery.create_session()

        # 添加一些消息到队列
        await server_with_recovery.queue_message("CREATE", {"id": "test-1"})
        await server_with_recovery.queue_message("UPDATE", {"id": "test-1"})

        # 模拟断开连接
        await server_with_recovery.close()

        # 创建新服务器实例并恢复
        new_server = MockServer(mock_websocket)
        new_server._auto_recovery_enabled = True
        new_server._recovery_failed = False
        new_server._message_queue = server_with_recovery._message_queue
        new_server._state_recovery = server_with_recovery._state_recovery

        await new_server.accept()
        new_server._session_id = session_id

        # 执行恢复
        result = await new_server.recover_from_disconnect()

        assert result is True
        assert new_server.is_recovery_failed() is False

        await new_server.close()

    @pytest.mark.asyncio
    async def test_recover_without_session_fails(self, server_with_recovery):
        """测试无 session 时恢复失败"""
        await server_with_recovery.accept()

        # 不创建 session，直接尝试恢复
        server_with_recovery._session_id = None

        result = await server_with_recovery.recover_from_disconnect()

        assert result is False

        await server_with_recovery.close()

    @pytest.mark.asyncio
    async def test_recover_disabled_returns_false(self, server_with_recovery):
        """测试禁用恢复时返回 False"""
        await server_with_recovery.accept()

        server_with_recovery.disable_auto_recovery()

        result = await server_with_recovery.recover_from_disconnect()

        assert result is False

        await server_with_recovery.close()

    @pytest.mark.asyncio
    async def test_enable_disable_auto_recovery(self, server_with_recovery):
        """测试启用/禁用自动恢复"""
        await server_with_recovery.accept()

        # 默认启用
        assert server_with_recovery.auto_recovery_enabled is True

        # 禁用
        server_with_recovery.disable_auto_recovery()
        assert server_with_recovery.auto_recovery_enabled is False

        # 启用
        server_with_recovery.enable_auto_recovery()
        assert server_with_recovery.auto_recovery_enabled is True

        await server_with_recovery.close()

    @pytest.mark.asyncio
    async def test_queue_message_returns_offset(self, server_with_recovery):
        """测试队列消息返回 offset"""
        await server_with_recovery.accept()
        server_with_recovery.create_session()

        offset = await server_with_recovery.queue_message("CREATE", {"id": "test-1"})

        assert offset is not None
        assert offset > 0

        await server_with_recovery.close()

    @pytest.mark.asyncio
    async def test_queue_message_without_session_returns_none(self, server_with_recovery):
        """测试无 session 时队列消息返回 None"""
        await server_with_recovery.accept()

        # 不创建 session
        server_with_recovery._session_id = None

        offset = await server_with_recovery.queue_message("CREATE", {"id": "test-1"})

        assert offset is None

        await server_with_recovery.close()


class TestAutoRecoveryEdgeCases:
    """自动恢复边界情况测试"""

    @pytest.fixture
    def mock_websocket(self):
        """创建模拟 WebSocket"""
        ws = AsyncMock()
        ws.client = MagicMock()
        ws.client.host = "127.0.0.1"
        ws.client.port = 12345
        return ws

    @pytest.mark.asyncio
    async def test_recovery_failed_flag_set_on_failure(self, mock_websocket, tmp_path):
        """测试恢复失败时设置标志"""
        server = MockServer(mock_websocket)
        server._auto_recovery_enabled = True
        server._recovery_failed = False
        server._message_queue = MessageQueue(queue_file=str(tmp_path / "messages.json"))
        server._state_recovery = StateRecoveryManager(state_file=str(tmp_path / "state.json"))

        await server.accept()

        # 设置一个不存在的 session_id
        server._session_id = "sess-nonexistent-123"

        # 尝试恢复
        result = await server.recover_from_disconnect()

        assert result is False
        assert server.is_recovery_failed() is True

        await server.close()

    @pytest.mark.asyncio
    async def test_recovery_resets_failed_flag_on_success(self, mock_websocket, tmp_path):
        """测试成功恢复后重置失败标志"""
        server = MockServer(mock_websocket)
        server._auto_recovery_enabled = True
        server._recovery_failed = True  # 先设置为失败
        server._message_queue = MessageQueue(queue_file=str(tmp_path / "messages.json"))
        server._state_recovery = StateRecoveryManager(state_file=str(tmp_path / "state.json"))

        await server.accept()
        session_id = server.create_session()

        # 添加消息
        await server.queue_message("CREATE", {"id": "test-1"})

        # 关闭并重新连接
        await server.close()

        # 新服务器恢复
        new_server = MockServer(mock_websocket)
        new_server._auto_recovery_enabled = True
        new_server._recovery_failed = True  # 初始为失败状态
        new_server._message_queue = server._message_queue
        new_server._state_recovery = server._state_recovery

        await new_server.accept()
        new_server._session_id = session_id

        # 恢复
        result = await new_server.recover_from_disconnect()

        assert result is True
        assert new_server.is_recovery_failed() is False

        await new_server.close()


class TestIntegrationWithExistingFeatures:
    """与现有功能集成测试"""

    @pytest.fixture
    def mock_websocket(self):
        """创建模拟 WebSocket"""
        ws = AsyncMock()
        ws.client = MagicMock()
        ws.client.host = "127.0.0.1"
        ws.client.port = 12345
        return ws

    @pytest.mark.asyncio
    async def test_auto_recovery_with_state_recovery(self, mock_websocket, tmp_path):
        """测试自动恢复与状态恢复集成"""
        server = MockServer(mock_websocket)
        server._auto_recovery_enabled = True
        server._message_queue = MessageQueue(queue_file=str(tmp_path / "messages.json"))
        server._state_recovery = StateRecoveryManager(state_file=str(tmp_path / "state.json"))

        await server.accept()

        # 创建 session 并更新 offset
        session_id = server.create_session()
        for _ in range(5):
            server.increment_message_offset()

        # 关闭
        await server.close()

        # 新服务器恢复
        new_server = MockServer(mock_websocket)
        new_server._auto_recovery_enabled = True
        new_server._message_queue = server._message_queue
        new_server._state_recovery = server._state_recovery

        await new_server.accept()
        new_server._session_id = session_id

        # 恢复
        result = await new_server.recover_from_disconnect()

        assert result is True
        assert new_server.message_offset == 5

        await new_server.close()

    @pytest.mark.asyncio
    async def test_auto_recovery_with_message_queue(self, mock_websocket, tmp_path):
        """测试自动恢复与消息队列集成"""
        server = MockServer(mock_websocket)
        server._auto_recovery_enabled = True
        server._message_queue = MessageQueue(queue_file=str(tmp_path / "messages.json"))
        server._state_recovery = StateRecoveryManager(state_file=str(tmp_path / "state.json"))

        await server.accept()
        session_id = server.create_session()

        # 添加多条消息
        for i in range(3):
            await server.queue_message("CREATE", {"id": f"test-{i}"})

        # 关闭
        await server.close()

        # 验证消息在队列中
        assert server._message_queue.get_message_count(session_id) == 3

        # 新服务器恢复
        new_server = MockServer(mock_websocket)
        new_server._auto_recovery_enabled = True
        new_server._message_queue = server._message_queue
        new_server._state_recovery = server._state_recovery

        await new_server.accept()
        new_server._session_id = session_id

        # 恢复
        result = await new_server.recover_from_disconnect()

        assert result is True

        await new_server.close()
