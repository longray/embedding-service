"""WebSocket 重连管理器

实现指数退避重连机制：
- 指数退避序列: 1→2→4→8→16→32→64→128→256→300s
- 随机抖动: +random.uniform(0, 1)s
- 最大重试: 10 次
"""

import asyncio
import logging
import random
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


class ReconnectionManager:
    """WebSocket 重连管理器

    实现指数退避重连机制，避免惊群效应。

    Attributes:
        max_retries: 最大重试次数，默认 10
        current_attempt: 当前重试次数
        is_reconnecting: 是否正在重连
    """

    # 指数退避序列（秒）
    BACKOFF_SEQUENCE = [1, 2, 4, 8, 16, 32, 64, 128, 256, 300]

    def __init__(self, max_retries: int = 10):
        """初始化重连管理器

        Args:
            max_retries: 最大重试次数，默认 10
        """
        self._max_retries = max_retries
        self._current_attempt = 0
        self._is_reconnecting = False
        self._reconnect_task: Optional[asyncio.Task] = None

        logger.debug("[ReconnectionManager] 初始化: max_retries=%d", max_retries)

    def calculate_delay(self, attempt: int) -> Optional[float]:
        """计算第 n 次重试的延迟时间

        Args:
            attempt: 重试次数（从 0 开始）

        Returns:
            延迟时间（秒），如果超过最大重试次数返回 None
        """
        if attempt < 0:
            raise ValueError("attempt must be non-negative")

        if attempt >= self._max_retries:
            logger.warning("[ReconnectionManager] 超过最大重试次数: %d", self._max_retries)
            return None

        if attempt >= len(self.BACKOFF_SEQUENCE):
            base_delay = self.BACKOFF_SEQUENCE[-1]
        else:
            base_delay = self.BACKOFF_SEQUENCE[attempt]

        jitter = random.uniform(0, 1)
        total_delay = base_delay + jitter

        logger.debug(
            "[ReconnectionManager] 计算延迟: attempt=%d, base=%ds, jitter=%.2fs, total=%.2fs",
            attempt,
            base_delay,
            jitter,
            total_delay,
        )

        return total_delay

    async def schedule_reconnect(
        self,
        reconnect_callback: Callable[[], Awaitable[None]],
        attempt: Optional[int] = None,
    ) -> bool:
        """调度重连任务

        Args:
            reconnect_callback: 重连回调函数
            attempt: 重试次数（默认使用 current_attempt）

        Returns:
            是否成功调度（False 表示超过最大重试次数）
        """
        if self._is_reconnecting:
            logger.warning("[ReconnectionManager] 重连已在进行中")
            return False

        if attempt is None:
            attempt = self._current_attempt

        delay = self.calculate_delay(attempt)
        if delay is None:
            logger.error("[ReconnectionManager] 重连失败：超过最大重试次数")
            return False

        self._is_reconnecting = True
        self._current_attempt = attempt + 1

        logger.info("[ReconnectionManager] 调度重连: attempt=%d, delay=%.2fs", attempt, delay)

        async def _reconnect_with_delay():
            try:
                await asyncio.sleep(delay)
                await reconnect_callback()
                logger.info("[ReconnectionManager] 重连成功")
                self.reset_counter()
            except Exception as e:
                logger.error("[ReconnectionManager] 重连失败: %s", e)
                raise
            finally:
                self._is_reconnecting = False

        self._reconnect_task = asyncio.create_task(_reconnect_with_delay(), name="websocket_reconnect")
        return True

    def reset_counter(self) -> None:
        """重置重试计数器"""
        self._current_attempt = 0
        self._is_reconnecting = False
        logger.debug("[ReconnectionManager] 计数器已重置")

    async def cancel_reconnect(self) -> None:
        """取消正在进行的重连任务"""
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            logger.debug("[ReconnectionManager] 重连任务已取消")

        self._is_reconnecting = False

    @property
    def max_retries(self) -> int:
        """最大重试次数"""
        return self._max_retries

    @property
    def current_attempt(self) -> int:
        """当前重试次数"""
        return self._current_attempt

    @property
    def is_reconnecting(self) -> bool:
        """是否正在重连"""
        return self._is_reconnecting

    @property
    def can_reconnect(self) -> bool:
        """是否还可以重连"""
        return self._current_attempt < self._max_retries
