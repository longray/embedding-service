"""LIVE SELECT DIFF 处理器

实现 SurrealDB LIVE SELECT 变更监听和增量同步：
- 监听 SurrealDB 数据变更
- 将变更转换为 JSON Patch
- 发送 diff 消息到客户端
- 支持变更合并（减少消息数量）

使用方式:
    handler = LiveDiffHandler(surrealdb_client, websocket_server)
    await handler.start("memory")
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .diff_manager import DiffManager
from .patch_generator import PatchGenerator

logger = logging.getLogger(__name__)


class LiveDiffHandler:
    """LIVE SELECT DIFF 处理器

    监听 SurrealDB 变更通知，生成 diff 并发送到客户端。

    Attributes:
        surrealdb_client: SurrealDB 客户端
        websocket_server: WebSocket 服务器实例
        diff_manager: DiffManager 实例
        table_name: 监听的表名
    """

    def __init__(
        self,
        surrealdb_client: Any,
        websocket_server: Any,
        diff_manager: Optional[DiffManager] = None,
        table_name: str = "memory",
        merge_interval: float = 0.1,
    ):
        """初始化 LIVE SELECT DIFF 处理器

        Args:
            surrealdb_client: SurrealDB 客户端
            websocket_server: WebSocket 服务器实例
            diff_manager: DiffManager 实例（可选）
            table_name: 监听的表名，默认 memory
            merge_interval: 变更合并间隔（秒），默认 0.1
        """
        self._surrealdb = surrealdb_client
        self._websocket = websocket_server
        self._diff_manager = diff_manager or DiffManager()
        self._table_name = table_name
        self._merge_interval = merge_interval

        self._is_running = False
        self._live_query_id: Optional[str] = None
        self._pending_changes: Dict[str, Dict[str, Any]] = {}
        self._merge_task: Optional[asyncio.Task] = None
        self._subscribe_task: Optional[asyncio.Task] = None
        self._state_cache: Dict[str, Any] = {}

        logger.debug(
            "[LiveDiffHandler] 初始化: table=%s, merge_interval=%.3fs",
            table_name,
            merge_interval,
        )

    async def start(self) -> bool:
        """启动 LIVE SELECT 监听

        Returns:
            是否成功启动
        """
        if self._is_running:
            logger.warning("[LiveDiffHandler] 已经运行中")
            return True

        try:
            # 启动 LIVE SELECT 查询
            self._live_query_id = await self._start_live_select()
            if not self._live_query_id:
                logger.error("[LiveDiffHandler] 启动 LIVE SELECT 失败")
                return False

            self._is_running = True

            # 发送初始快照（首次连接时）
            await self._send_snapshot()

            # 启动订阅循环任务（关键修复：订阅 LIVE SELECT 通知流）
            self._subscribe_task = asyncio.create_task(
                self._subscribe_loop(),
                name="live_diff_subscribe",
            )

            # 启动变更合并任务
            self._merge_task = asyncio.create_task(
                self._merge_changes_loop(),
                name="live_diff_merge",
            )

            logger.info(
                "[LiveDiffHandler] 已启动: table=%s, query_id=%s",
                self._table_name,
                self._live_query_id,
            )
            return True

        except Exception as e:
            logger.error("[LiveDiffHandler] 启动失败: %s", e)
            return False

    async def stop(self) -> None:
        """停止 LIVE SELECT 监听"""
        if not self._is_running:
            return

        self._is_running = False

        # 取消订阅任务
        if self._subscribe_task and not self._subscribe_task.done():
            self._subscribe_task.cancel()
            try:
                await self._subscribe_task
            except asyncio.CancelledError:
                pass

        # 取消合并任务
        if self._merge_task and not self._merge_task.done():
            self._merge_task.cancel()
            try:
                await self._merge_task
            except asyncio.CancelledError:
                pass

        # 停止 LIVE SELECT
        if self._live_query_id and self._surrealdb:
            try:
                await self._surrealdb.query(f"KILL {self._live_query_id}")
                logger.debug("[LiveDiffHandler] LIVE SELECT 已停止: %s", self._live_query_id)
            except Exception as e:
                logger.warning("[LiveDiffHandler] 停止 LIVE SELECT 失败: %s", e)

        # 发送剩余的变更
        await self._flush_pending_changes()

        logger.info("[LiveDiffHandler] 已停止")

    async def _start_live_select(self) -> Optional[str]:
        """启动 LIVE SELECT 查询

        Returns:
            LIVE SELECT 查询 ID
        """
        try:
            # 执行 LIVE SELECT 查询（表名使用 type::table 转换）
            result = await self._surrealdb.query(f"LIVE SELECT * FROM type::table(${self._table_name})")

            # 提取查询 ID
            if result and len(result) > 0:
                query_id = result[0].get("id") if isinstance(result[0], dict) else str(result[0])
                return query_id

            return None

        except Exception as e:
            logger.error("[LiveDiffHandler] 启动 LIVE SELECT 查询失败: %s", e)
            return None

    async def _send_snapshot(self, limit: int = 1000) -> bool:
        """发送当前数据的完整快照

        在首次连接时发送已有数据的完整状态，
        客户端收到后可建立初始状态，后续变更走 diff。

        Args:
            limit: 最大记录数，默认 1000

        Returns:
            是否成功发送
        """
        try:
            # 查询当前数据（带限制防止大数据量）
            result = await self._surrealdb.query(f"SELECT * FROM {self._table_name} LIMIT {limit}")

            if not result:
                logger.debug("[LiveDiffHandler] 无数据可发送快照")
                return True

            # 提取记录并应用过滤器
            records = []
            for item in result:
                if isinstance(item, dict):
                    # 应用过滤器（如果 WebSocket 服务器设置了过滤器）
                    if hasattr(self._websocket, "should_send_to_client"):
                        if self._websocket.should_send_to_client(item):
                            records.append(item)
                    else:
                        records.append(item)
                elif isinstance(item, list):
                    for record in item:
                        if hasattr(self._websocket, "should_send_to_client"):
                            if self._websocket.should_send_to_client(record):
                                records.append(record)
                        else:
                            records.append(record)

            # 初始化状态缓存
            for record in records:
                record_id = record.get("id")
                if record_id:
                    self._state_cache[record_id] = record

            # 发送快照消息
            snapshot_message = {
                "type": "snapshot",
                "data": records,
                "count": len(records),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            await self._websocket.send_json(snapshot_message)

            logger.info("[LiveDiffHandler] 快照已发送: %d 条记录", len(records))
            return True

        except Exception as e:
            logger.error("[LiveDiffHandler] 发送快照失败: %s", e)
            return False

    async def handle_change(self, change: Dict[str, Any]) -> None:
        """处理 SurrealDB 变更通知

        Args:
            change: 变更数据
        """
        if not self._is_running:
            return

        action = change.get("action")
        record_id = change.get("id")
        data = change.get("result")

        if not record_id:
            logger.debug("[LiveDiffHandler] 变更缺少 record_id")
            return

        # 检查过滤器（如果 WebSocket 服务器设置了过滤器）
        if hasattr(self._websocket, "should_send_to_client"):
            if not self._websocket.should_send_to_client(data or {}):
                logger.debug("[LiveDiffHandler] 变更被过滤器拦截: id=%s", record_id)
                return

        # 缓存变更，等待合并
        self._pending_changes[record_id] = {
            "action": action,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.debug(
            "[LiveDiffHandler] 收到变更: action=%s, id=%s",
            action,
            record_id,
        )

    async def _subscribe_loop(self) -> None:
        """订阅 LIVE SELECT 通知循环

        关键修复：订阅 SurrealDB 的 LIVE SELECT 通知流，
        将变更转发到 handle_change() 处理。
        """
        if not self._live_query_id:
            logger.error("[LiveDiffHandler] 无法订阅：query_id 为空")
            return

        retry_count = 0
        max_retries = 3

        while self._is_running and retry_count < max_retries:
            try:
                async for notification in self._surrealdb.subscribe_live(self._live_query_id):
                    if not self._is_running:
                        break

                    # 转发变更到 handle_change
                    await self.handle_change(notification)

                # 流正常结束（非异常）
                logger.debug("[LiveDiffHandler] 订阅流已关闭")
                break

            except asyncio.CancelledError:
                logger.debug("[LiveDiffHandler] 订阅循环已取消")
                raise
            except Exception as e:
                retry_count += 1
                logger.error("[LiveDiffHandler] 订阅循环错误 (重试 %d/%d): %s", retry_count, max_retries, e)
                if retry_count < max_retries:
                    await asyncio.sleep(1.0 * retry_count)  # 指数退避
                else:
                    logger.error("[LiveDiffHandler] 订阅循环达到最大重试次数，停止")
                    break

    async def _merge_changes_loop(self) -> None:
        """变更合并循环"""
        try:
            while self._is_running:
                await asyncio.sleep(self._merge_interval)

                if self._pending_changes:
                    await self._flush_pending_changes()

        except asyncio.CancelledError:
            logger.debug("[LiveDiffHandler] 合并循环已取消")
            raise

    async def _flush_pending_changes(self) -> None:
        """发送所有待处理的变更"""
        if not self._pending_changes:
            return

        changes = dict(self._pending_changes)
        self._pending_changes.clear()

        for record_id, change_info in changes.items():
            try:
                await self._send_change_diff(record_id, change_info)
            except Exception as e:
                logger.error(
                    "[LiveDiffHandler] 发送变更失败: id=%s, error=%s",
                    record_id,
                    e,
                )

    async def _send_change_diff(
        self,
        record_id: str,
        change_info: Dict[str, Any],
    ) -> None:
        """发送变更 diff

        Args:
            record_id: 记录 ID
            change_info: 变更信息
        """
        action = change_info["action"]
        new_data = change_info["data"]

        # 获取旧状态
        old_data = self._state_cache.get(record_id)

        if action == "DELETE":
            # 删除操作
            message = {
                "type": "change",
                "action": "DELETE",
                "id": record_id,
            }
            self._state_cache.pop(record_id, None)

        elif old_data is None or action == "CREATE":
            # 新建操作
            message = {
                "type": "change",
                "action": "CREATE",
                "id": record_id,
                "data": new_data,
            }
            self._state_cache[record_id] = PatchGenerator._deep_copy(new_data)

        else:
            # 更新操作，生成 diff
            patches = PatchGenerator.generate_patch(old_data, new_data)

            if patches:
                message = {
                    "type": "change",
                    "action": "UPDATE",
                    "id": record_id,
                    "patches": patches,
                }
                self._state_cache[record_id] = PatchGenerator._deep_copy(new_data)
            else:
                # 无实际变更
                logger.debug("[LiveDiffHandler] 无实际变更: id=%s", record_id)
                return

        # 发送消息
        if self._websocket and self._websocket.is_connected:
            await self._websocket.send_json(message)
            logger.debug(
                "[LiveDiffHandler] 发送变更: action=%s, id=%s",
                action,
                record_id,
            )

    def update_state_cache(self, record_id: str, data: Any) -> None:
        """手动更新状态缓存

        Args:
            record_id: 记录 ID
            data: 数据
        """
        self._state_cache[record_id] = PatchGenerator._deep_copy(data)
        logger.debug("[LiveDiffHandler] 更新状态缓存: id=%s", record_id)

    def clear_state_cache(self, record_id: Optional[str] = None) -> None:
        """清除状态缓存

        Args:
            record_id: 记录 ID，如果为 None 清除所有
        """
        if record_id is None:
            self._state_cache.clear()
            logger.debug("[LiveDiffHandler] 清除所有状态缓存")
        else:
            self._state_cache.pop(record_id, None)
            logger.debug("[LiveDiffHandler] 清除状态缓存: id=%s", record_id)

    @property
    def is_running(self) -> bool:
        """是否运行中"""
        return self._is_running

    @property
    def live_query_id(self) -> Optional[str]:
        """LIVE SELECT 查询 ID"""
        return self._live_query_id

    @property
    def pending_count(self) -> int:
        """待处理变更数量"""
        return len(self._pending_changes)

    @property
    def cache_count(self) -> int:
        """状态缓存数量"""
        return len(self._state_cache)
