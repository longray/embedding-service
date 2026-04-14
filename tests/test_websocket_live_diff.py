"""WebSocket LIVE SELECT DIFF 集成测试

测试范围：
- LiveDiffHandler 初始化和启动/停止
- 变更处理和 diff 生成
- 状态缓存管理
- 变更合并

运行方式：
    uv run pytest tests/test_websocket_live_diff.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from wrapper.src.websocket import LiveDiffHandler, DiffManager


class MockWebSocketServer:
    """模拟 WebSocket 服务器"""

    def __init__(self):
        self.is_connected = True
        self.sent_messages = []

    async def send_json(self, message):
        self.sent_messages.append(message)


class MockSurrealDB:
    """模拟 SurrealDB 客户端"""

    def __init__(self):
        self.query = AsyncMock(return_value=[{"id": "live-query-123"}])


class TestLiveDiffHandler:
    """LiveDiffHandler 集成测试"""

    @pytest.fixture
    def mock_surrealdb(self):
        """创建模拟 SurrealDB 客户端"""
        return MockSurrealDB()

    @pytest.fixture
    def mock_websocket(self):
        """创建模拟 WebSocket 服务器"""
        return MockWebSocketServer()

    @pytest.fixture
    def diff_manager(self):
        """创建 DiffManager 实例"""
        return DiffManager(mode="diff", threshold=50.0, min_diff_size=10)

    @pytest.fixture
    def live_handler(self, mock_surrealdb, mock_websocket, diff_manager):
        """创建 LiveDiffHandler 实例"""
        handler = LiveDiffHandler(
            surrealdb_client=mock_surrealdb,
            websocket_server=mock_websocket,
            diff_manager=diff_manager,
            table_name="memory",
            merge_interval=0.01,
        )
        return handler

    @pytest.mark.asyncio
    async def test_handler_initialization(self, live_handler):
        """测试处理器初始化"""
        assert live_handler.is_running is False
        assert live_handler.live_query_id is None
        assert live_handler.pending_count == 0
        assert live_handler.cache_count == 0

    @pytest.mark.asyncio
    async def test_start_creates_live_query(self, live_handler, mock_surrealdb):
        """测试启动创建 LIVE SELECT 查询"""
        result = await live_handler.start()

        assert result is True
        assert live_handler.is_running is True
        assert live_handler.live_query_id == "live-query-123"
        mock_surrealdb.query.assert_called_once_with("LIVE SELECT * FROM memory")

        await live_handler.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_merge_task(self, live_handler):
        """测试停止取消合并任务"""
        await live_handler.start()
        assert live_handler.is_running is True

        await live_handler.stop()

        assert live_handler.is_running is False

    @pytest.mark.asyncio
    async def test_handle_change_creates_pending(self, live_handler):
        """测试处理变更创建待处理项"""
        await live_handler.start()

        change = {
            "action": "CREATE",
            "id": "memory:test-1",
            "result": {"id": "memory:test-1", "content": "Hello"},
        }

        await live_handler.handle_change(change)

        assert live_handler.pending_count == 1

        await live_handler.stop()

    @pytest.mark.asyncio
    async def test_handle_create_sends_full_data(self, live_handler, mock_websocket):
        """测试处理 CREATE 发送完整数据"""
        await live_handler.start()

        change = {
            "action": "CREATE",
            "id": "memory:test-1",
            "result": {"id": "memory:test-1", "content": "Hello World"},
        }

        await live_handler.handle_change(change)
        await live_handler._flush_pending_changes()

        assert len(mock_websocket.sent_messages) == 1
        message = mock_websocket.sent_messages[0]
        assert message["type"] == "change"
        assert message["action"] == "CREATE"
        assert message["id"] == "memory:test-1"
        assert message["data"]["content"] == "Hello World"

        await live_handler.stop()

    @pytest.mark.asyncio
    async def test_handle_delete_sends_delete(self, live_handler, mock_websocket):
        """测试处理 DELETE 发送删除消息"""
        await live_handler.start()

        # 先创建记录
        live_handler.update_state_cache("memory:test-1", {"content": "Hello"})

        change = {
            "action": "DELETE",
            "id": "memory:test-1",
        }

        await live_handler.handle_change(change)
        await live_handler._flush_pending_changes()

        assert len(mock_websocket.sent_messages) == 1
        message = mock_websocket.sent_messages[0]
        assert message["type"] == "change"
        assert message["action"] == "DELETE"
        assert message["id"] == "memory:test-1"
        assert "data" not in message

        await live_handler.stop()

    @pytest.mark.asyncio
    async def test_handle_update_generates_patches(self, live_handler, mock_websocket):
        """测试处理 UPDATE 生成 patches"""
        await live_handler.start()

        # 先创建记录
        live_handler.update_state_cache("memory:test-1", {"content": "Hello", "version": 1})

        change = {
            "action": "UPDATE",
            "id": "memory:test-1",
            "result": {"content": "Hello World", "version": 2},
        }

        await live_handler.handle_change(change)
        await live_handler._flush_pending_changes()

        assert len(mock_websocket.sent_messages) == 1
        message = mock_websocket.sent_messages[0]
        assert message["type"] == "change"
        assert message["action"] == "UPDATE"
        assert message["id"] == "memory:test-1"
        assert "patches" in message
        assert len(message["patches"]) > 0

        await live_handler.stop()

    @pytest.mark.asyncio
    async def test_state_cache_updated_on_create(self, live_handler):
        """测试 CREATE 后更新状态缓存"""
        await live_handler.start()

        change = {
            "action": "CREATE",
            "id": "memory:test-1",
            "result": {"content": "Hello"},
        }

        await live_handler.handle_change(change)
        await live_handler._flush_pending_changes()

        assert live_handler.cache_count == 1
        cached = live_handler._state_cache.get("memory:test-1")
        assert cached is not None
        assert cached["content"] == "Hello"

        await live_handler.stop()

    @pytest.mark.asyncio
    async def test_clear_state_cache_single(self, live_handler):
        """测试清除单个状态缓存"""
        await live_handler.start()

        live_handler.update_state_cache("memory:test-1", {"content": "Hello"})
        live_handler.update_state_cache("memory:test-2", {"content": "World"})

        assert live_handler.cache_count == 2

        live_handler.clear_state_cache("memory:test-1")

        assert live_handler.cache_count == 1
        assert "memory:test-1" not in live_handler._state_cache
        assert "memory:test-2" in live_handler._state_cache

        await live_handler.stop()

    @pytest.mark.asyncio
    async def test_clear_state_cache_all(self, live_handler):
        """测试清除所有状态缓存"""
        await live_handler.start()

        live_handler.update_state_cache("memory:test-1", {"content": "Hello"})
        live_handler.update_state_cache("memory:test-2", {"content": "World"})

        assert live_handler.cache_count == 2

        live_handler.clear_state_cache()

        assert live_handler.cache_count == 0

        await live_handler.stop()

    @pytest.mark.asyncio
    async def test_multiple_changes_merged(self, live_handler, mock_websocket):
        """测试多个变更合并发送"""
        await live_handler.start()

        # 发送多个变更
        for i in range(3):
            change = {
                "action": "CREATE",
                "id": f"memory:test-{i}",
                "result": {"id": f"memory:test-{i}", "content": f"Item {i}"},
            }
            await live_handler.handle_change(change)

        # 等待合并间隔
        import asyncio

        await asyncio.sleep(0.02)

        # 手动刷新
        await live_handler._flush_pending_changes()

        assert len(mock_websocket.sent_messages) == 3

        await live_handler.stop()

    @pytest.mark.asyncio
    async def test_not_running_ignores_changes(self, live_handler):
        """测试未运行时忽略变更"""
        change = {
            "action": "CREATE",
            "id": "memory:test-1",
            "result": {"content": "Hello"},
        }

        await live_handler.handle_change(change)

        assert live_handler.pending_count == 0


class TestLiveDiffEdgeCases:
    """LiveDiffHandler 边界情况测试"""

    @pytest.fixture
    def mock_surrealdb(self):
        return MockSurrealDB()

    @pytest.fixture
    def mock_websocket(self):
        return MockWebSocketServer()

    @pytest.mark.asyncio
    async def test_start_already_running(self, mock_surrealdb, mock_websocket):
        """测试重复启动"""
        handler = LiveDiffHandler(mock_surrealdb, mock_websocket)

        result1 = await handler.start()
        assert result1 is True

        result2 = await handler.start()
        assert result2 is True  # 已经运行，返回 True

        await handler.stop()

    @pytest.mark.asyncio
    async def test_handle_change_without_id(self, mock_surrealdb, mock_websocket):
        """测试处理无 ID 的变更"""
        handler = LiveDiffHandler(mock_surrealdb, mock_websocket)
        await handler.start()

        change = {
            "action": "CREATE",
            "result": {"content": "Hello"},
        }

        await handler.handle_change(change)

        assert handler.pending_count == 0

        await handler.stop()

    @pytest.mark.asyncio
    async def test_no_change_skips_message(self, mock_surrealdb, mock_websocket):
        """测试无实际变更跳过消息发送"""
        handler = LiveDiffHandler(mock_surrealdb, mock_websocket)
        await handler.start()

        # 先创建记录
        handler.update_state_cache("memory:test-1", {"content": "Hello"})

        # 发送相同数据
        change = {
            "action": "UPDATE",
            "id": "memory:test-1",
            "result": {"content": "Hello"},
        }

        await handler.handle_change(change)
        await handler._flush_pending_changes()

        assert len(mock_websocket.sent_messages) == 0

        await handler.stop()
