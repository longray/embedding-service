"""WebSocket 心跳管理器

实现心跳机制：
- 每 30s 发送 ping 消息
- 5s 内等待 pong 响应
- 连续 2 次未响应触发 on_connection_lost
"""

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


class HeartbeatManager:
    """WebSocket 心跳管理器

    管理 WebSocket 连接的心跳保活机制：
    - 定期发送 ping 消息
    - 检测 pong 响应超时
    - 触发连接丢失回调

    Attributes:
        interval: 心跳间隔（秒），默认 30
        timeout: pong 响应超时（秒），默认 5
        max_missing: 最大未响应次数，默认 2
    """

    def __init__(
        self,
        send_ping: Callable[[], Awaitable[None]],
        on_connection_lost: Callable[[], Awaitable[None]],
        interval: float = 30.0,
        timeout: float = 5.0,
        max_missing: int = 2,
    ):
        """初始化心跳管理器

        Args:
            send_ping: 发送 ping 消息的回调函数
            on_connection_lost: 连接丢失时的回调函数
            interval: 心跳间隔（秒）
            timeout: pong 响应超时（秒）
            max_missing: 最大未响应次数
        """
        self._send_ping = send_ping
        self._on_connection_lost: Callable[[], Awaitable[None]] = on_connection_lost
        self._interval = interval
        self._timeout = timeout
        self._max_missing = max_missing

        self._heartbeat_task: Optional[asyncio.Task] = None
        self._missing_count = 0
        self._last_pong_time = time.time()
        self._is_running = False
        self._lock = asyncio.Lock()

        logger.debug(
            "[HeartbeatManager] 初始化: interval=%.1fs, timeout=%.1fs, max_missing=%d", interval, timeout, max_missing
        )

    async def start(self) -> None:
        """启动心跳机制"""
        async with self._lock:
            if self._is_running:
                logger.warning("[HeartbeatManager] 心跳机制已在运行")
                return

            self._is_running = True
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="websocket_heartbeat")
            logger.info("[HeartbeatManager] 心跳机制已启动")

    async def stop(self) -> None:
        """停止心跳机制"""
        async with self._lock:
            if not self._is_running:
                return

            self._is_running = False

            if self._heartbeat_task and not self._heartbeat_task.done():
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass

            logger.info("[HeartbeatManager] 心跳机制已停止")

    async def _heartbeat_loop(self) -> None:
        """心跳主循环"""
        try:
            while self._is_running:
                try:
                    await self._send_ping()
                    logger.debug("[HeartbeatManager] ping 已发送")
                except Exception as e:
                    logger.error("[HeartbeatManager] 发送 ping 失败: %s", e)
                    await self._on_connection_lost()
                    break

                try:
                    await asyncio.wait_for(self._wait_for_pong(), timeout=self._timeout)
                    self._missing_count = 0
                    logger.debug("[HeartbeatManager] pong 已收到")
                except asyncio.TimeoutError:
                    self._missing_count += 1
                    logger.warning("[HeartbeatManager] pong 超时 (%d/%d)", self._missing_count, self._max_missing)

                    if self._missing_count >= self._max_missing:
                        logger.error("[HeartbeatManager] 连续 %d 次未响应，触发连接丢失", self._max_missing)
                        await self._on_connection_lost()
                        break

                await asyncio.sleep(self._interval)

        except asyncio.CancelledError:
            logger.debug("[HeartbeatManager] 心跳循环已取消")

    async def _wait_for_pong(self) -> None:
        """等待 pong 响应

        这是一个占位方法，实际 pong 检测通过 on_pong_received 方法触发
        """
        # 等待直到 pong_received 被调用
        self._pong_event = asyncio.Event()
        await self._pong_event.wait()

    def on_pong_received(self) -> None:
        """收到 pong 响应时调用"""
        self._last_pong_time = time.time()
        if hasattr(self, "_pong_event") and not self._pong_event.is_set():
            self._pong_event.set()
        logger.debug("[HeartbeatManager] pong 已处理")

    @property
    def is_running(self) -> bool:
        """心跳机制是否正在运行"""
        return self._is_running

    @property
    def missing_count(self) -> int:
        """当前连续未响应次数"""
        return self._missing_count

    @property
    def last_pong_time(self) -> float:
        """最后一次收到 pong 的时间戳"""
        return self._last_pong_time
