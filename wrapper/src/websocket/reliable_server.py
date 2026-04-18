"""可靠 WebSocket 服务器

实现带心跳机制的可靠 WebSocket 连接管理：
- 自动心跳保活
- 连接状态监控
- 优雅断开处理
"""

import asyncio
import logging
from typing import Any, Callable, Dict, Literal, Optional

from fastapi import WebSocket, WebSocketDisconnect

from .ack_manager import AckManager
from .diff_manager import DiffManager
from .heartbeat import HeartbeatManager
from .message_queue import MessageQueue
from .reconnection import ReconnectionManager
from .state_recovery import StateRecoveryManager

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
        ack_timeout: float = 5.0,
        ack_max_retries: int = 3,
        diff_mode: Literal["diff", "full"] = "diff",
        diff_threshold: float = 50.0,
        diff_min_size: int = 100,
    ):
        """初始化可靠 WebSocket 服务器

        Args:
            websocket: FastAPI WebSocket 实例
            heartbeat_interval: 心跳间隔（秒），默认 30
            heartbeat_timeout: pong 响应超时（秒），默认 5
            max_missing_pongs: 最大未响应次数，默认 2
            ack_timeout: ACK 超时时间（秒），默认 5
            ack_max_retries: ACK 最大重试次数，默认 3
            diff_mode: DIFF 模式（diff/full），默认 diff
            diff_threshold: 带宽节省阈值（百分比），默认 50%
            diff_min_size: 最小 diff 大小（字节），默认 100
        """
        self._websocket = websocket
        self._is_connected = False
        self._heartbeat_manager: Optional[HeartbeatManager] = None
        self._ack_manager: Optional[AckManager] = None
        self._diff_manager: Optional[DiffManager] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._disconnect_event = asyncio.Event()

        # 心跳配置
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._max_missing_pongs = max_missing_pongs

        # ACK 配置
        self._ack_timeout = ack_timeout
        self._ack_max_retries = ack_max_retries

        # DIFF 配置
        self._diff_mode: Literal["diff", "full"] = diff_mode
        self._diff_threshold = diff_threshold
        self._diff_min_size = diff_min_size

        # State Recovery 配置
        self._state_recovery: Optional[StateRecoveryManager] = None
        self._session_id: Optional[str] = None
        self._message_offset: int = 0

        # Message Queue 配置
        self._message_queue: Optional[MessageQueue] = None

        # Reconnection 配置
        self._reconnection_manager: Optional[ReconnectionManager] = None
        self._auto_recovery_enabled: bool = True
        self._recovery_failed: bool = False

        # 订阅过滤器配置
        self._filters: Dict[str, Any] = {}

        logger.debug(
            "[ReliableWebSocketServer] 初始化: interval=%.1fs, timeout=%.1fs, max_missing=%d, "
            "ack_timeout=%.1fs, ack_max_retries=%d, diff_mode=%s, state_recovery=enabled, "
            "auto_recovery=enabled",
            heartbeat_interval,
            heartbeat_timeout,
            max_missing_pongs,
            ack_timeout,
            ack_max_retries,
            diff_mode,
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

        self._ack_manager = AckManager(
            default_timeout=self._ack_timeout,
            max_retries=self._ack_max_retries,
        )

        self._diff_manager = DiffManager(
            mode=self._diff_mode,
            threshold=self._diff_threshold,
            min_diff_size=self._diff_min_size,
        )

        self._state_recovery = StateRecoveryManager()

        self._message_queue = MessageQueue()

        self._reconnection_manager = ReconnectionManager()

        self._receive_task = asyncio.create_task(self._receive_loop(), name="websocket_receive")
        await self._heartbeat_manager.start()

        # 发送 connected 消息
        await self._websocket.send_json(
            {
                "type": "connected",
                "session_id": self._session_id,
                "timestamp": asyncio.get_event_loop().time(),
            }
        )

        logger.info("[ReliableWebSocketServer] 连接已接受，心跳/ACK/DIFF/StateRecovery机制已启动")

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

        # 保存状态
        if self._state_recovery and self._session_id:
            self._state_recovery.save_state(self._session_id, self._message_offset, {"reason": reason, "code": code})
            logger.debug(
                "[ReliableWebSocketServer] 状态已保存: session_id=%s, offset=%d", self._session_id, self._message_offset
            )

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

        # 消息入队（用于 from_offset 恢复）
        if self._message_queue and self._session_id:
            self._message_offset += 1
            self._message_queue.enqueue(
                session_id=self._session_id,
                message_type=data.get("type", "unknown"),
                data=data,
            )

        await self._websocket.send_json(data)

    async def send_json_with_ack(self, data: dict[str, Any], timeout: Optional[float] = None) -> bool:
        """发送 JSON 数据并等待 ACK 确认

        Args:
            data: 要发送的数据
            timeout: ACK 超时时间（秒），默认使用初始化时的 ack_timeout

        Returns:
            是否成功收到 ACK
        """
        if not self._is_connected:
            raise RuntimeError("WebSocket 未连接")

        if self._ack_manager is None:
            raise RuntimeError("ACK 管理器未初始化")

        # 消息入队（用于 from_offset 恢复）
        if self._message_queue and self._session_id:
            self._message_offset += 1
            self._message_queue.enqueue(
                session_id=self._session_id,
                message_type=data.get("type", "unknown"),
                data=data,
            )

        async def send_callback(message: dict) -> None:
            await self._websocket.send_json(message)

        return await self._ack_manager.send_with_ack(send_callback, data, timeout=timeout)

    async def send_data_with_diff(
        self,
        key: str,
        data: Any,
        metadata: Optional[dict] = None,
        use_ack: bool = False,
        ack_timeout: Optional[float] = None,
    ) -> bool:
        """发送数据，自动选择 diff/full 模式

        Args:
            key: 数据标识（用于状态缓存）
            data: 要发送的数据
            metadata: 额外元数据
            use_ack: 是否使用 ACK 确认
            ack_timeout: ACK 超时时间（秒）

        Returns:
            是否成功发送（使用 ACK 时返回 ACK 结果，否则返回 True）
        """
        if not self._is_connected:
            raise RuntimeError("WebSocket 未连接")

        if self._diff_manager is None:
            raise RuntimeError("DIFF 管理器未初始化")

        message = self._diff_manager.create_message(key, data, metadata)

        if use_ack:
            return await self.send_json_with_ack(message, timeout=ack_timeout)
        else:
            # 消息入队（用于 from_offset 恢复）
            if self._message_queue and self._session_id:
                self._message_offset += 1
                self._message_queue.enqueue(
                    session_id=self._session_id,
                    message_type=message.get("type", "unknown"),
                    data=message,
                )
            await self._websocket.send_json(message)
            return True

    def set_diff_mode(self, mode: Literal["diff", "full"]) -> None:
        """设置 DIFF 模式

        Args:
            mode: 模式 (diff/full)
        """
        if self._diff_manager is None:
            raise RuntimeError("DIFF 管理器未初始化")

        self._diff_manager.set_mode(mode)
        self._diff_mode = mode
        logger.info("[ReliableWebSocketServer] DIFF 模式切换: %s", mode)

    def update_diff_state(self, key: str, data: Any) -> None:
        """更新 DIFF 状态缓存

        Args:
            key: 数据标识
            data: 数据
        """
        if self._diff_manager is None:
            raise RuntimeError("DIFF 管理器未初始化")

        self._diff_manager.update_state(key, data)

    def clear_diff_state(self, key: Optional[str] = None) -> None:
        """清除 DIFF 状态缓存

        Args:
            key: 数据标识，如果为 None 清除所有
        """
        if self._diff_manager is None:
            raise RuntimeError("DIFF 管理器未初始化")

        self._diff_manager.clear_state(key)

    @property
    def diff_mode(self) -> str:
        """当前 DIFF 模式"""
        return self._diff_mode if self._diff_manager else "diff"

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
        """接收循环 - 处理客户端消息、pong 响应和 ACK"""
        try:
            while self._is_connected:
                try:
                    message = await self._websocket.receive_json()

                    if not isinstance(message, dict):
                        logger.debug("[ReliableWebSocketServer] 收到非字典消息: %s", message)
                        continue

                    msg_type = message.get("type")

                    if msg_type == "pong":
                        if self._heartbeat_manager:
                            self._heartbeat_manager.on_pong_received()
                        logger.debug("[ReliableWebSocketServer] pong 已接收")
                    elif msg_type == "ack":
                        ack_id = message.get("_ackId")
                        if ack_id and self._ack_manager:
                            self._ack_manager.handle_ack(ack_id)
                            logger.debug("[ReliableWebSocketServer] ACK 已处理: ack_id=%s", ack_id)
                    elif msg_type == "sync_request":
                        from_offset = message.get("from_offset", 0)
                        await self._handle_sync_request(from_offset)
                    elif msg_type == "subscribe":
                        filters = message.get("filters", {})
                        await self._handle_subscribe(filters)
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

    def create_session(self) -> str:
        """创建新 Session

        Returns:
            Session ID
        """
        # 如果 StateRecoveryManager 未初始化，先初始化
        if self._state_recovery is None:
            self._state_recovery = StateRecoveryManager()

        self._session_id = self._state_recovery.generate_session_id()
        self._message_offset = 0
        logger.info("[ReliableWebSocketServer] 创建 Session: %s", self._session_id)
        return self._session_id

    async def restore_session(self, session_id: str) -> bool:
        """恢复 Session 并重放丢失的消息

        Args:
            session_id: Session ID

        Returns:
            是否成功恢复
        """
        if self._state_recovery is None:
            raise RuntimeError("StateRecoveryManager 未初始化")

        state = self._state_recovery.restore_state(session_id)
        if state is not None:
            self._session_id = session_id
            self._message_offset = state.get("offset", 0)

            # 重放丢失的消息
            if self._message_queue and self._message_offset > 0:
                await self._replay_messages(self._message_offset)

            logger.info(
                "[ReliableWebSocketServer] 恢复 Session: %s, offset=%d",
                session_id,
                self._message_offset,
            )
            return True

        # Session 不存在或已过期
        logger.warning("[ReliableWebSocketServer] 恢复 Session 失败: %s", session_id)

        # 发送错误消息给客户端
        try:
            await self._websocket.send_json(
                {
                    "type": "error",
                    "code": "SESSION_EXPIRED",
                    "message": "Session 不存在或已过期",
                }
            )
        except Exception:
            pass

        return False

    async def _replay_messages(self, from_offset: int) -> None:
        """重放从指定 offset 开始的消息

        Args:
            from_offset: 起始 offset
        """
        if not self._message_queue or not self._session_id:
            return

        messages = self._message_queue.get_messages_from_offset(from_offset, self._session_id)

        for msg in messages:
            try:
                await self._websocket.send_json(msg.data)
                self._message_queue.mark_delivered(msg.offset)
                logger.debug("[ReliableWebSocketServer] 重放消息: offset=%d", msg.offset)
            except Exception as e:
                logger.error("[ReliableWebSocketServer] 重放消息失败: offset=%d, error=%s", msg.offset, e)

    async def _handle_sync_request(self, from_offset: int) -> None:
        """处理 sync_request 消息

        从指定 offset 开始同步丢失的消息给客户端。

        Args:
            from_offset: 起始 offset
        """
        if not self._message_queue or not self._session_id:
            logger.warning("[ReliableWebSocketServer] 无法处理 sync_request: queue 或 session 未初始化")
            return

        try:
            messages = self._message_queue.get_messages_from_offset(from_offset, self._session_id)

            for msg in messages:
                try:
                    await self._websocket.send_json(msg.data)
                    self._message_queue.mark_delivered(msg.offset)
                    logger.debug("[ReliableWebSocketServer] sync_request 发送消息: offset=%d", msg.offset)
                except Exception as e:
                    logger.error("[ReliableWebSocketServer] sync_request 发送失败: offset=%d, error=%s", msg.offset, e)

            logger.info(
                "[ReliableWebSocketServer] sync_request 完成: from_offset=%d, sent=%d", from_offset, len(messages)
            )

        except Exception as e:
            logger.error("[ReliableWebSocketServer] sync_request 处理失败: %s", e)

    def update_message_offset(self, offset: int) -> None:
        """更新消息 offset

        Args:
            offset: 新的 offset
        """
        self._message_offset = offset
        if self._state_recovery and self._session_id:
            self._state_recovery.update_offset(self._session_id, offset)
            logger.debug(
                "[ReliableWebSocketServer] 更新 offset: session_id=%s, offset=%d",
                self._session_id,
                offset,
            )

    async def _handle_subscribe(self, filters: Dict[str, Any]) -> None:
        """处理 subscribe 消息

        设置订阅过滤器，只推送符合条件的变更。

        Args:
            filters: 过滤条件，支持 tenant_id、type、tags、project_id
        """
        self._filters = filters
        logger.info("[ReliableWebSocketServer] 订阅过滤器已设置: %s", filters)

        # 发送确认
        await self._websocket.send_json(
            {
                "type": "subscribed",
                "filters": filters,
            }
        )

    def should_send_to_client(self, data: Dict[str, Any]) -> bool:
        """检查数据是否应该发送给客户端

        根据订阅过滤器判断。

        Args:
            data: 变更数据

        Returns:
            是否应该发送
        """
        if not self._filters:
            return True

        # 检查 tenant_id
        if "tenant_id" in self._filters:
            if data.get("tenant_id") != self._filters["tenant_id"]:
                return False

        # 检查 type
        if "type" in self._filters:
            if data.get("type") != self._filters["type"]:
                return False

        # 检查 project_id
        if "project_id" in self._filters:
            if data.get("project_id") != self._filters["project_id"]:
                return False

        # 检查 tags（数据包含任一指定 tag）
        if "tags" in self._filters:
            data_tags = data.get("tags", [])
            filter_tags = self._filters["tags"]
            # 类型安全检查
            if not isinstance(data_tags, list):
                data_tags = [data_tags] if data_tags else []
            if not isinstance(filter_tags, list):
                filter_tags = [filter_tags] if filter_tags else []
            if not any(tag in data_tags for tag in filter_tags):
                return False

        return True

    def increment_message_offset(self) -> int:
        """递增消息 offset

        Returns:
            新的 offset
        """
        self._message_offset += 1
        if self._state_recovery and self._session_id:
            self._state_recovery.update_offset(self._session_id, self._message_offset)
        return self._message_offset

    @property
    def session_id(self) -> Optional[str]:
        """当前 Session ID"""
        return self._session_id

    @property
    def message_offset(self) -> int:
        """当前消息 offset"""
        return self._message_offset
