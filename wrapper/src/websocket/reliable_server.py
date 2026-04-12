"""可靠 WebSocket 服务器

实现带心跳机制的可靠 WebSocket 连接管理：
- 自动心跳保活
- 连接状态监控
- 优雅断开处理
"""

import asyncio
import logging
from typing import Any, Callable, Optional

from fastapi import WebSocket, WebSocketDisconnect

from .heartbeat import HeartbeatManager

logger = logging.getLogger(__name__)


class ReliableWebSocketServer:
    """可靠 WebSocket 服务器

    包装 FastAPI WebSocket，添加心跳机制和连接状态管理。

    Attributes:
        websocket: FastAPI WebSocket 实例
        heartbeat_manager: 心跳管理器实例
        is_connected: 连接是否活跃
    """

    def __init__(
        self,
        websocket: WebSocket,
        heartbeat_interval: float = 30.0,
        heartbeat_timeout: float = 5.0,
        max_missing_pongs: int = 2,
    ):
        """初始化可靠 WebSocket 服务器

        Args:
            websocket: FastAPI WebSocket 实例
            heartbeat_interval: 心跳间隔（秒），默认 30
            heartbeat_timeout: pong 响应超时（秒），默认 5
            max_missing_pongs: 最大未响应次数，默认 2
        """
        self._websocket = websocket
        self._is_connected = False
        self._heartbeat_manager: Optional[HeartbeatManager] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._disconnect_event = asyncio.Event()

        # 心跳配置
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._max_missing_pongs = max_missing_pongs

        logger.debug(
            "[ReliableWebSocketServer] 初始化: interval=%.1fs, timeout=%.1fs, max_missing=%d",
            heartbeat_interval,
            heartbeat_timeout,
            max_missing_pongs,
        )

    async def accept(self) -> None:
        """接受 WebSocket 连接并启动心跳机制"""
        await self._websocket.accept()
        self._is_connected = True

        self._heartbeat_manager = HeartbeatManager(
            send_ping=self._send_ping,
            on_connection_lost=self._handle_connection_lost,
            interval=self._heartbeat_interval,
            timeout=self._heartbeat_timeout,
            max_missing=self._max_missing_pongs,
        )

        self._receive_task = asyncio.create_task(self._receive_loop(), name="websocket_receive")
        await self._heartbeat_manager.start()

        logger.info("[ReliableWebSocketServer] 连接已接受，心跳机制已启动")

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """关闭 WebSocket 连接"""
        if not self._is_connected:
            return

        self._is_connected = False
        self._disconnect_event.set()

        # 停止心跳
        if self._heartbeat_manager:
            await self._heartbeat_manager.stop()

        # 取消接收任务
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        # 关闭 WebSocket
        try:
            await self._websocket.close(code=code, reason=reason)
            logger.info("[ReliableWebSocketServer] 连接已关闭: code=%d, reason=%s", code, reason)
        except Exception as e:
            logger.debug("[ReliableWebSocketServer] 关闭连接时出错: %s", e)

    async def send_json(self, data: dict[str, Any]) -> None:
        """发送 JSON 数据"""
        if not self._is_connected:
            raise RuntimeError("WebSocket 未连接")
        await self._websocket.send_json(data)

    async def send_text(self, text: str) -> None:
        """发送文本数据"""
        if not self._is_connected:
            raise RuntimeError("WebSocket 未连接")
        await self._websocket.send_text(text)

    async def receive_text(self) -> str:
        """接收文本数据（阻塞直到收到消息或断开）"""
        if not self._is_connected:
            raise RuntimeError("WebSocket 未连接")
        return await self._websocket.receive_text()

    async def receive_json(self) -> dict[str, Any]:
        """接收 JSON 数据（阻塞直到收到消息或断开）"""
        if not self._is_connected:
            raise RuntimeError("WebSocket 未连接")
        return await self._websocket.receive_json()

    async def _send_ping(self) -> None:
        """发送 ping 消息"""
        if self._is_connected:
            await self._websocket.send_json({"type": "ping", "timestamp": asyncio.get_event_loop().time()})
            logger.debug("[ReliableWebSocketServer] ping 已发送")

    async def _handle_connection_lost(self) -> None:
        """处理连接丢失"""
        logger.error("[ReliableWebSocketServer] 连接丢失（心跳超时）")
        await self.close(code=1001, reason="Heartbeat timeout")

    async def _receive_loop(self) -> None:
        """接收循环 - 处理客户端消息和 pong 响应"""
        try:
            while self._is_connected:
                try:
                    message = await self._websocket.receive_json()

                    if isinstance(message, dict) and message.get("type") == "pong":
                        if self._heartbeat_manager:
                            self._heartbeat_manager.on_pong_received()
                        logger.debug("[ReliableWebSocketServer] pong 已接收")
                    else:
                        logger.debug("[ReliableWebSocketServer] 收到消息: %s", message)

                except asyncio.CancelledError:
                    raise
                except WebSocketDisconnect:
                    logger.info("[ReliableWebSocketServer] 客户端断开连接")
                    break
                except Exception as e:
                    logger.error("[ReliableWebSocketServer] 接收消息错误: %s", e)
                    break

        except asyncio.CancelledError:
            logger.debug("[ReliableWebSocketServer] 接收循环已取消")
        finally:
            if self._is_connected:
                try:
                    await self.close(code=1001, reason="Receive loop ended")
                except Exception:
                    pass

    @property
    def is_connected(self) -> bool:
        """连接是否活跃"""
        return self._is_connected

    @property
    def client(self) -> Any:
        """客户端信息"""
        return self._websocket.client

    async def wait_for_disconnect(self) -> None:
        """等待连接断开"""
        await self._disconnect_event.wait()
