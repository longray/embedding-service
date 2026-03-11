"""WebSocket 实时推送端点测试

测试 /ws/memories/live WebSocket 端点的认证和连接功能。

前置条件：
- SurrealDB 运行在 ws://localhost:18002
- Wrapper 服务运行在 http://localhost:17999

运行方式：
    uv run pytest tests/test_websocket.py -v
"""

import pytest
from websockets import connect
from websockets.exceptions import InvalidStatusCode


WRAPPER_WS_URL = "ws://localhost:17999/ws/memories/live"


class TestWebSocketConnection:
    """WebSocket 连接测试"""

    @pytest.mark.asyncio
    async def test_connect_without_token(self):
        """未配置 token 时允许连接（向后兼容）"""
        try:
            async with connect(f"{WRAPPER_WS_URL}?tenant_id=test") as websocket:
                assert websocket is not None
        except Exception as e:
            pytest.fail(f"连接失败: {e}")

    @pytest.mark.asyncio
    async def test_connect_with_tenant_id(self):
        """指定 tenant_id 连接"""
        try:
            async with connect(f"{WRAPPER_WS_URL}?tenant_id=custom_tenant") as websocket:
                assert websocket is not None
        except Exception as e:
            pytest.fail(f"连接失败: {e}")

    @pytest.mark.asyncio
    async def test_connect_default_tenant(self):
        """使用默认 tenant_id"""
        try:
            async with connect(WRAPPER_WS_URL) as websocket:
                assert websocket is not None
        except Exception as e:
            pytest.fail(f"连接失败: {e}")


class TestWebSocketAuth:
    """WebSocket 认证测试（需要配置 WRAPPER_WEBSOCKET_TOKEN）"""

    @pytest.mark.asyncio
    async def test_connect_with_token(self):
        """提供 token 参数连接"""
        try:
            async with connect(f"{WRAPPER_WS_URL}?tenant_id=test&token=test_token") as websocket:
                assert websocket is not None
        except InvalidStatusCode as e:
            if e.status_code == 1008:
                pytest.skip("服务配置了 token 认证，测试 token 不匹配")
            raise
