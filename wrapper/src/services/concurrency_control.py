"""ConcurrencyControl - 并发控制模块

提供并发控制功能，支持：
- Semaphore 并发限制
- processing Set 去重
- 队列机制
- 超时处理
"""

import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, Optional, Set

logger = logging.getLogger(__name__)


class ConcurrencyControl:
    """并发控制器

    提供并发控制功能，防止同文件重复处理，限制并发数。

    Attributes:
        max_concurrent: 最大并发数
        timeout_seconds: 超时时间（秒）
        max_queue_size: 最大队列大小
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        timeout_seconds: float = 30.0,
        max_queue_size: int = 100,
    ):
        """初始化并发控制器

        Args:
            max_concurrent: 最大并发数，默认 5
            timeout_seconds: 超时时间（秒），默认 30
            max_queue_size: 最大队列大小，默认 100
        """
        self._max_concurrent = max_concurrent
        self._timeout_seconds = timeout_seconds
        self._max_queue_size = max_queue_size

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._processing: Set[str] = set()
        self._queued: Set[str] = set()
        self._queue = asyncio.Queue(maxsize=max_queue_size)
        self._lock = asyncio.Lock()

        self._max_concurrent_reached = 0
        self._total_processed = 0
        self._total_deduplicated = 0
        self._total_timeouts = 0
        self._total_errors = 0

        self._logger = logging.getLogger(__name__)
        self._logger.debug(
            "[ConcurrencyControl] 初始化: max_concurrent=%d, timeout=%.1fs, queue_size=%d",
            max_concurrent,
            timeout_seconds,
            max_queue_size,
        )

    async def process(
        self,
        task_id: str,
        coro: Callable[[], Coroutine[Any, Any, Any]],
    ) -> Any:
        """处理任务

        使用 Semaphore 限制并发，自动去重。

        Args:
            task_id: 任务唯一标识
            coro: 异步任务协程工厂函数

        Returns:
            任务执行结果

        Raises:
            asyncio.TimeoutError: 任务超时
            RuntimeError: 队列已满
        """
        # 检查是否已在处理中
        async with self._lock:
            if task_id in self._processing:
                self._total_deduplicated += 1
                self._logger.debug(
                    "[ConcurrencyControl] 任务已在处理中，跳过: task_id=%s",
                    task_id,
                )
                raise DuplicateTaskError(f"Task {task_id} is already being processed")

            self._processing.add(task_id)

        try:
            # 使用 Semaphore 限制并发
            async with self._semaphore:
                # 记录最大并发数
                current_concurrent = self._max_concurrent - self._semaphore._value
                if current_concurrent > self._max_concurrent_reached:
                    self._max_concurrent_reached = current_concurrent

                self._logger.debug(
                    "[ConcurrencyControl] 开始处理任务: task_id=%s, concurrent=%d/%d",
                    task_id,
                    current_concurrent,
                    self._max_concurrent,
                )

                try:
                    # 执行协程，带超时
                    result = await asyncio.wait_for(
                        coro(),
                        timeout=self._timeout_seconds,
                    )
                    self._total_processed += 1
                    return result

                except asyncio.TimeoutError:
                    self._total_timeouts += 1
                    self._logger.warning(
                        "[ConcurrencyControl] 任务超时: task_id=%s, timeout=%.1fs",
                        task_id,
                        self._timeout_seconds,
                    )
                    raise

                except Exception as e:
                    self._total_errors += 1
                    self._logger.error(
                        "[ConcurrencyControl] 任务执行错误: task_id=%s, error=%s",
                        task_id,
                        e,
                    )
                    raise

        finally:
            # 从 processing 中移除
            async with self._lock:
                self._processing.discard(task_id)

    async def enqueue(
        self,
        task_id: str,
        coro: Callable[[], Coroutine[Any, Any, Any]],
    ) -> None:
        """将任务加入队列

        Args:
            task_id: 任务唯一标识
            coro: 异步任务协程工厂函数

        Raises:
            asyncio.QueueFull: 队列已满
        """
        # 检查是否已在处理中或已在队列中
        async with self._lock:
            if task_id in self._processing or task_id in self._queued:
                self._total_deduplicated += 1
                self._logger.debug(
                    "[ConcurrencyControl] 任务已在处理中或队列中，跳过入队: task_id=%s",
                    task_id,
                )
                return
            self._queued.add(task_id)

        # 将任务放入队列
        await self._queue.put((task_id, coro))
        self._logger.debug(
            "[ConcurrencyControl] 任务已入队: task_id=%s, queue_size=%d/%d",
            task_id,
            self._queue.qsize(),
            self._max_queue_size,
        )

    async def process_queue(self) -> None:
        """处理队列中的任务

        持续从队列中取出任务并处理，直到队列为空。
        """
        while not self._queue.empty():
            task_id, coro = await self._queue.get()
            async with self._lock:
                self._queued.discard(task_id)
            try:
                await self.process(task_id, coro)
            except DuplicateTaskError:
                pass
            except Exception as e:
                self._logger.error(
                    "[ConcurrencyControl] 队列任务处理错误: task_id=%s, error=%s",
                    task_id,
                    e,
                )
            finally:
                self._queue.task_done()

    def is_processing(self, task_id: str) -> bool:
        """检查任务是否正在处理中

        Args:
            task_id: 任务唯一标识

        Returns:
            是否正在处理
        """
        return task_id in self._processing

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        return {
            "max_concurrent": self._max_concurrent,
            "max_concurrent_reached": self._max_concurrent_reached,
            "current_processing": len(self._processing),
            "queue_size": self._queue.qsize(),
            "total_processed": self._total_processed,
            "total_deduplicated": self._total_deduplicated,
            "total_timeouts": self._total_timeouts,
            "total_errors": self._total_errors,
        }

    def clear_stats(self) -> None:
        """清除统计信息"""
        self._max_concurrent_reached = 0
        self._total_processed = 0
        self._total_deduplicated = 0
        self._total_timeouts = 0
        self._total_errors = 0
        self._logger.debug("[ConcurrencyControl] 统计信息已清除")

    async def clear_queue(self) -> None:
        """清空队列"""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        self._logger.debug("[ConcurrencyControl] 队列已清空")

    @property
    def timeout_seconds(self) -> float:
        """超时时间（秒）"""
        return self._timeout_seconds

    @property
    def max_queue_size(self) -> int:
        """最大队列大小"""
        return self._max_queue_size

    @property
    def max_concurrent(self) -> int:
        """最大并发数"""
        return self._max_concurrent

    @property
    def max_concurrent_reached(self) -> int:
        """达到的最大并发数"""
        return self._max_concurrent_reached

    @property
    def current_processing(self) -> int:
        """当前正在处理的任务数"""
        return len(self._processing)

    @property
    def queue_size(self) -> int:
        """当前队列大小"""
        return self._queue.qsize()


class DuplicateTaskError(Exception):
    """重复任务错误"""

    pass
