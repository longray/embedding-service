"""WebSocket ACK 确认管理器

实现消息确认机制：
- 消息发送后启动 5s 超时计时器
- 收到 ACK 后清除超时
- 超时后自动重试（最多 3 次）
- 达到最大重试次数后 reject

ACK 消息格式: {"type": "ack", "_ackId": "..."}
"""

import asyncio
import logging
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class AckManager:
    """ACK 确认管理器

    实现消息确认机制，确保消息可靠投递。

    Attributes:
        default_timeout: 默认超时时间（秒）
        max_retries: 最大重试次数
    """

    def __init__(self, default_timeout: float = 5.0, max_retries: int = 3):
        """初始化 ACK 管理器

        Args:
            default_timeout: 默认超时时间（秒），默认 5.0
            max_retries: 最大重试次数，默认 3
        """
        self._default_timeout = default_timeout
        self._max_retries = max_retries
        self._pending_acks: Dict[str, asyncio.Event] = {}
        self._retry_counts: Dict[str, int] = {}
        self._lock = asyncio.Lock()

        logger.debug("[AckManager] 初始化: timeout=%.1fs, max_retries=%d", default_timeout, max_retries)

    async def send_with_ack(
        self,
        send_callback: Callable[[dict], Awaitable[None]],
        message: dict,
        timeout: Optional[float] = None,
    ) -> bool:
        """发送消息并等待 ACK

        Args:
            send_callback: 发送消息的回调函数
            message: 要发送的消息
            timeout: 超时时间（秒），默认使用 default_timeout

        Returns:
            是否成功收到 ACK
        """
        if timeout is None:
            timeout = self._default_timeout

        ack_id = str(uuid.uuid4())
        message_with_ack = {**message, "_ackId": ack_id}

        async with self._lock:
            self._pending_acks[ack_id] = asyncio.Event()
            self._retry_counts[ack_id] = 0

        attempt = 0
        while attempt <= self._max_retries:
            try:
                logger.debug("[AckManager] 发送消息: ack_id=%s, attempt=%d", ack_id, attempt)
                await send_callback(message_with_ack)

                wait_task = asyncio.create_task(
                    self._pending_acks[ack_id].wait(),
                    name=f"ack_wait_{ack_id}",
                )

                try:
                    await asyncio.wait_for(wait_task, timeout=timeout)
                    logger.info("[AckManager] 收到 ACK: ack_id=%s", ack_id)
                    await self._cleanup(ack_id)
                    return True
                except asyncio.TimeoutError:
                    attempt += 1
                    if attempt > self._max_retries:
                        logger.warning("[AckManager] 超时，超过最大重试次数: ack_id=%s", ack_id)
                        await self._cleanup(ack_id)
                        return False

                    logger.warning(
                        "[AckManager] 超时，准备重试: ack_id=%s, attempt=%d/%d",
                        ack_id,
                        attempt,
                        self._max_retries,
                    )
                    async with self._lock:
                        self._retry_counts[ack_id] = attempt
                        self._pending_acks[ack_id] = asyncio.Event()

            except Exception as e:
                logger.error("[AckManager] 发送消息失败: ack_id=%s, error=%s", ack_id, e)
                await self._cleanup(ack_id)
                return False

        await self._cleanup(ack_id)
        return False

    def handle_ack(self, ack_id: str) -> bool:
        """处理收到的 ACK

        Args:
            ack_id: ACK ID

        Returns:
            是否成功处理（False 表示 ack_id 不存在或已超时）
        """
        if ack_id not in self._pending_acks:
            logger.debug("[AckManager] 收到未知 ACK: ack_id=%s", ack_id)
            return False

        self._pending_acks[ack_id].set()
        logger.debug("[AckManager] 处理 ACK: ack_id=%s", ack_id)
        return True

    def is_pending(self, ack_id: str) -> bool:
        """检查 ACK ID 是否 pending

        Args:
            ack_id: ACK ID

        Returns:
            是否正在等待 ACK
        """
        return ack_id in self._pending_acks and not self._pending_acks[ack_id].is_set()

    def get_retry_count(self, ack_id: str) -> int:
        """获取重试次数

        Args:
            ack_id: ACK ID

        Returns:
            重试次数（0 表示不存在）
        """
        return self._retry_counts.get(ack_id, 0)

    async def _cleanup(self, ack_id: str) -> None:
        """清理资源"""
        async with self._lock:
            self._pending_acks.pop(ack_id, None)
            self._retry_counts.pop(ack_id, None)

    @property
    def pending_count(self) -> int:
        """当前 pending 的 ACK 数量"""
        return len(self._pending_acks)

    async def clear_all(self) -> None:
        """清除所有 pending 的 ACK"""
        async with self._lock:
            self._pending_acks.clear()
            self._retry_counts.clear()
        logger.debug("[AckManager] 已清除所有 pending ACK")
