"""WebSocket 自动恢复功能

实现断线重连后自动恢复状态：
- 恢复 session
- 同步丢失消息（from_offset）
- 恢复后发送 ACK 确认
- 恢复失败进入降级模式

使用方式:
    await server.recover_from_disconnect()
"""

# pyright: reportAttributeAccessIssue=false

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AutoRecoveryMixin:
    """自动恢复 Mixin

    为 ReliableWebSocketServer 提供自动恢复功能。
    """

    async def recover_from_disconnect(self) -> bool:
        """从断线中恢复

        恢复 session 并同步丢失的消息。

        Returns:
            是否成功恢复
        """
        if not hasattr(self, "_auto_recovery_enabled") or not self._auto_recovery_enabled:
            logger.debug("[AutoRecovery] 自动恢复已禁用")
            return False

        if not hasattr(self, "_session_id") or not self._session_id:
            logger.warning("[AutoRecovery] 无法恢复：缺少 session_id")
            return False

        if not hasattr(self, "_state_recovery") or not self._state_recovery:
            logger.warning("[AutoRecovery] 无法恢复：缺少 state_recovery")
            return False

        try:
            # 1. 恢复 session 状态
            state = self._state_recovery.restore_state(self._session_id)
            if state is None:
                logger.warning("[AutoRecovery] 恢复失败：Session 不存在")
                if hasattr(self, "_recovery_failed"):
                    self._recovery_failed = True
                return False

            saved_offset = state.get("offset", 0)
            logger.info(
                "[AutoRecovery] 恢复 Session: %s, saved_offset=%d",
                self._session_id,
                saved_offset,
            )

            # 2. 同步丢失的消息
            if hasattr(self, "_message_queue") and self._message_queue and saved_offset > 0:
                lost_messages = self._message_queue.get_messages_from_offset(
                    saved_offset,
                    session_id=self._session_id,
                    limit=100,
                )

                if lost_messages:
                    logger.info(
                        "[AutoRecovery] 同步丢失消息: %d messages",
                        len(lost_messages),
                    )

                    for msg in lost_messages:
                        if hasattr(self, "_is_connected") and self._is_connected:
                            # 发送丢失的消息
                            await self.send_json(
                                {
                                    "type": "recovery",
                                    "original_offset": msg.offset,
                                    "message_type": msg.message_type,
                                    "data": msg.data,
                                }
                            )
                            # 标记为已送达
                            self._message_queue.mark_delivered(msg.offset)

                    # 3. 发送恢复完成确认
                    await self.send_json(
                        {
                            "type": "recovery_complete",
                            "session_id": self._session_id,
                            "recovered_count": len(lost_messages),
                            "from_offset": saved_offset,
                        }
                    )

                    logger.info("[AutoRecovery] 恢复完成: %d messages", len(lost_messages))
                else:
                    logger.debug("[AutoRecovery] 无丢失消息需要恢复")

            # 4. 更新 offset
            if hasattr(self, "_message_offset"):
                self._message_offset = saved_offset

            # 5. 重置恢复失败标志
            if hasattr(self, "_recovery_failed"):
                self._recovery_failed = False

            return True

        except Exception as e:
            logger.error("[AutoRecovery] 恢复失败: %s", e)
            if hasattr(self, "_recovery_failed"):
                self._recovery_failed = True
            return False

    def enable_auto_recovery(self) -> None:
        """启用自动恢复"""
        self._auto_recovery_enabled = True
        logger.debug("[AutoRecovery] 自动恢复已启用")

    def disable_auto_recovery(self) -> None:
        """禁用自动恢复"""
        self._auto_recovery_enabled = False
        logger.debug("[AutoRecovery] 自动恢复已禁用")

    def is_recovery_failed(self) -> bool:
        """检查恢复是否失败

        Returns:
            是否进入降级模式
        """
        return getattr(self, "_recovery_failed", False)

    async def queue_message(
        self,
        message_type: str,
        data: dict,
    ) -> Optional[int]:
        """将消息加入队列

        Args:
            message_type: 消息类型
            data: 消息数据

        Returns:
            消息 offset，如果失败返回 None
        """
        if not hasattr(self, "_message_queue") or not self._message_queue:
            return None

        if not hasattr(self, "_session_id") or not self._session_id:
            return None

        offset = self._message_queue.enqueue(
            session_id=self._session_id,
            message_type=message_type,
            data=data,
        )

        # 同时递增消息 offset
        if hasattr(self, "increment_message_offset"):
            self.increment_message_offset()

        return offset

    @property
    def auto_recovery_enabled(self) -> bool:
        """自动恢复是否启用"""
        return getattr(self, "_auto_recovery_enabled", False)
