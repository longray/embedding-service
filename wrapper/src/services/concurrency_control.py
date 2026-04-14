"""ConcurrencyControl - 并发控制模块

提供并发控制功能，支持：
- Semaphore 并发限制
- processing Set 去重
- 队列机制
- 超时处理
- 队列状态持久化
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class DuplicateTaskError(Exception):
    """重复任务错误"""

    pass


@dataclass
class TaskInfo:
    """任务信息

    Attributes:
        task_id: 任务唯一标识
        task_data: 任务数据
        status: 任务状态 (pending, processing, completed, failed)
        priority: 优先级
        retry_count: 重试次数
    """

    task_id: str
    task_data: Dict[str, Any]
    status: str = "pending"
    priority: int = 0
    retry_count: int = 0


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
        db: Any = None,
        tenant_id: str = "default",
    ):
        """初始化并发控制器

        Args:
            max_concurrent: 最大并发数，默认 5
            timeout_seconds: 超时时间（秒），默认 30
            max_queue_size: 最大队列大小，默认 100
            db: 数据库连接（可选），用于队列状态持久化
            tenant_id: 租户 ID，默认 "default"
        """
        self._max_concurrent = max_concurrent
        self._timeout_seconds = timeout_seconds
        self._max_queue_size = max_queue_size
        self._db = db
        self._tenant_id = tenant_id

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

    @property
    def db(self) -> Any:
        """数据库连接"""
        return self._db

    @property
    def tenant_id(self) -> str:
        """租户 ID"""
        return self._tenant_id

    async def save_queue_state(self) -> Dict[str, int]:
        """保存队列状态到数据库

        将当前队列中的任务保存到 task_queue 表。

        Returns:
            统计信息 {"saved": 保存数, "failed": 失败数}
        """
        if self._db is None:
            self._logger.warning("[ConcurrencyControl] 数据库连接未设置，无法保存队列状态")
            return {"saved": 0, "failed": 0}

        saved_count = 0
        failed_count = 0

        # 获取队列中的所有任务
        queue_tasks = []
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                queue_tasks.append(item)
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

        for task_id, coro in queue_tasks:
            try:
                # 保存任务到 DB
                query = """
                    CREATE task_queue CONTENT {
                        tenant_id: $tenant_id,
                        task_id: $task_id,
                        task_data: $task_data,
                        status: 'pending',
                        priority: 0,
                        retry_count: 0,
                        max_retries: 3,
                        created_at: time::now(),
                        updated_at: time::now()
                    }
                """
                await self._db.query(
                    query,
                    {
                        "tenant_id": self._tenant_id,
                        "task_id": task_id,
                        "task_data": {"type": "async_task"},
                    },
                )
                saved_count += 1
            except Exception as e:
                self._logger.error("[ConcurrencyControl] 保存任务失败: %s, error=%s", task_id, e)
                failed_count += 1
            finally:
                # 将任务重新放回内存队列
                try:
                    self._queue.put_nowait((task_id, coro))
                except asyncio.QueueFull:
                    pass

        self._logger.info(
            "[ConcurrencyControl] 队列状态已保存: saved=%d, failed=%d",
            saved_count,
            failed_count,
        )

        return {"saved": saved_count, "failed": failed_count}

    async def restore_queue_state(
        self,
        task_processor: Optional[Callable[[str, Dict[str, Any]], Coroutine[Any, Any, Any]]] = None,
    ) -> int:
        """从数据库恢复队列状态

        从 task_queue 表恢复 pending 状态的任务到内存队列。

        Args:
            task_processor: 任务处理器，用于重建任务协程

        Returns:
            恢复的任务数量
        """
        if self._db is None:
            self._logger.warning("[ConcurrencyControl] 数据库连接未设置，无法恢复队列状态")
            return 0

        try:
            # 查询 pending 状态的任务
            query = """
                SELECT * FROM task_queue
                WHERE tenant_id = $tenant_id AND status = 'pending'
                ORDER BY priority DESC, created_at ASC
            """
            result = await self._db.query(
                query,
                {"tenant_id": self._tenant_id},
            )

            restored_count = 0
            for record in result:
                task_id = record.get("task_id")
                task_data = record.get("task_data", {})

                if task_id and task_id not in self._queued:
                    # 将任务加入内存队列
                    if task_processor:
                        coro_factory = lambda tid=task_id, tdata=task_data: task_processor(tid, tdata)
                        await self.enqueue(task_id, coro_factory)
                        restored_count += 1

            self._logger.info(
                "[ConcurrencyControl] 队列状态已恢复: %d 个任务",
                restored_count,
            )
            return restored_count
        except Exception as e:
            self._logger.error("[ConcurrencyControl] 恢复队列状态失败: %s", e)
            return 0

    async def clear_queue_state_from_db(self) -> int:
        """清除数据库中的队列状态

        删除 task_queue 表中当前租户的所有记录。

        Returns:
            清除的记录数
        """
        if self._db is None:
            return 0

        try:
            query = """
                DELETE FROM task_queue
                WHERE tenant_id = $tenant_id
            """
            result = await self._db.query(
                query,
                {"tenant_id": self._tenant_id},
            )

            deleted_count = len(result) if result else 0
            self._logger.info(
                "[ConcurrencyControl] 队列状态已清除: %d 条记录",
                deleted_count,
            )
            return deleted_count
        except Exception as e:
            self._logger.error("[ConcurrencyControl] 清除队列状态失败: %s", e)
            return 0

    async def update_task_status_in_db(
        self,
        task_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> bool:
        """更新任务状态到数据库

        Args:
            task_id: 任务 ID
            status: 新状态 (pending, processing, completed, failed)
            error_message: 错误信息（可选）

        Returns:
            是否更新成功
        """
        if self._db is None:
            return False

        try:
            query = """
                UPDATE task_queue
                SET status = $status,
                    updated_at = time::now()
            """
            params = {
                "task_id": task_id,
                "status": status,
            }

            if error_message:
                query += ", error_message = $error_message"
                params["error_message"] = error_message

            query += " WHERE task_id = $task_id AND tenant_id = $tenant_id"
            params["tenant_id"] = self._tenant_id

            await self._db.query(query, params)
            return True
        except Exception as e:
            self._logger.error("[ConcurrencyControl] 更新任务状态失败: %s", e)
            return False
