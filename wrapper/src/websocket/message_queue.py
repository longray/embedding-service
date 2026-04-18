"""WebSocket 消息队列

实现消息持久化存储和 from_offset 查询：
- 消息队列持久化到文件
- 支持 from_offset 查询
- 返回指定 offset 之后的所有消息
- 消息过期清理（7天）

存储文件: .opencode/ws-messages.json
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class QueuedMessage:
    """队列消息"""

    offset: int
    session_id: str
    message_type: str
    data: Dict[str, Any]
    timestamp: str
    delivered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "offset": self.offset,
            "session_id": self.session_id,
            "message_type": self.message_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "delivered": self.delivered,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueuedMessage":
        """从字典创建"""
        return cls(
            offset=data["offset"],
            session_id=data["session_id"],
            message_type=data["message_type"],
            data=data["data"],
            timestamp=data["timestamp"],
            delivered=data.get("delivered", False),
        )


class MessageQueue:
    """消息队列

    管理 WebSocket 消息持久化，支持 from_offset 查询。

    Attributes:
        queue_file: 队列文件路径
        ttl_days: 消息过期时间（天）
        max_messages: 最大消息数量
    """

    def __init__(
        self,
        queue_file: str = ".opencode/ws-messages.json",
        ttl_days: int = 7,
        max_messages: int = 10000,
    ):
        """初始化消息队列

        Args:
            queue_file: 队列文件路径，默认 .opencode/ws-messages.json
            ttl_days: 消息过期时间（天），默认 7
            max_messages: 最大消息数量，默认 10000
        """
        self._queue_file = Path(queue_file)
        self._ttl_days = ttl_days
        self._max_messages = max_messages
        self._messages: List[QueuedMessage] = []
        self._current_offset: int = 0

        self._ensure_directory()
        self._load_messages()
        self._cleanup_expired()

        logger.debug(
            "[MessageQueue] 初始化: queue_file=%s, ttl_days=%d, max_messages=%d, loaded=%d",
            queue_file,
            ttl_days,
            max_messages,
            len(self._messages),
        )

    def _ensure_directory(self) -> None:
        """确保目录存在"""
        self._queue_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_messages(self) -> None:
        """从文件加载消息"""
        if not self._queue_file.exists():
            logger.debug("[MessageQueue] 队列文件不存在，创建新队列")
            return

        try:
            with open(self._queue_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                messages_data = data.get("messages", [])
                self._messages = [QueuedMessage.from_dict(m) for m in messages_data]
                self._current_offset = data.get("current_offset", 0)

            logger.debug(
                "[MessageQueue] 加载消息: %d messages, offset=%d",
                len(self._messages),
                self._current_offset,
            )
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("[MessageQueue] 加载消息失败: %s", e)
            self._messages = []
            self._current_offset = 0

    def _save_messages(self) -> None:
        """保存消息到文件"""
        try:
            data = {
                "current_offset": self._current_offset,
                "messages": [m.to_dict() for m in self._messages],
            }
            with open(self._queue_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug("[MessageQueue] 消息已保存: %d messages", len(self._messages))
        except IOError as e:
            logger.error("[MessageQueue] 保存消息失败: %s", e)

    def enqueue(
        self,
        session_id: str,
        message_type: str,
        data: Dict[str, Any],
    ) -> int:
        """添加消息到队列

        Args:
            session_id: Session ID
            message_type: 消息类型
            data: 消息数据

        Returns:
            消息 offset
        """
        self._current_offset += 1

        message = QueuedMessage(
            offset=self._current_offset,
            session_id=session_id,
            message_type=message_type,
            data=data,
            timestamp=datetime.now(timezone.utc).isoformat(),
            delivered=False,
        )

        self._messages.append(message)

        # 限制消息数量
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages :]

        self._save_messages()

        logger.debug(
            "[MessageQueue] 消息入队: offset=%d, session=%s, type=%s",
            message.offset,
            session_id,
            message_type,
        )

        return message.offset

    def get_messages_from_offset(
        self,
        from_offset: int,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[QueuedMessage]:
        """获取从指定 offset 开始的消息

        Args:
            from_offset: 起始 offset（不包含）
            session_id: 可选的 Session ID 过滤
            limit: 最大返回数量，默认 100

        Returns:
            消息列表
        """
        result = []

        for message in self._messages:
            if message.offset > from_offset:
                if session_id is None or message.session_id == session_id:
                    result.append(message)
                    if len(result) >= limit:
                        break

        logger.debug(
            "[MessageQueue] 查询消息: from_offset=%d, session=%s, found=%d",
            from_offset,
            session_id,
            len(result),
        )

        return result

    def mark_delivered(self, offset: int) -> bool:
        """标记消息为已送达

        Args:
            offset: 消息 offset

        Returns:
            是否成功标记
        """
        for message in self._messages:
            if message.offset == offset:
                message.delivered = True
                self._save_messages()
                logger.debug("[MessageQueue] 标记已送达: offset=%d", offset)
                return True

        return False

    def get_undelivered_messages(
        self,
        session_id: str,
        limit: int = 100,
    ) -> List[QueuedMessage]:
        """获取未送达的消息

        Args:
            session_id: Session ID
            limit: 最大返回数量，默认 100

        Returns:
            未送达消息列表
        """
        result = []

        for message in self._messages:
            if message.session_id == session_id and not message.delivered:
                result.append(message)
                if len(result) >= limit:
                    break

        return result

    def get_last_offset(self) -> int:
        """获取最后 offset

        Returns:
            最后 offset
        """
        return self._current_offset

    def get_message_count(self, session_id: Optional[str] = None) -> int:
        """获取消息数量

        Args:
            session_id: 可选的 Session ID 过滤

        Returns:
            消息数量
        """
        if session_id is None:
            return len(self._messages)

        return sum(1 for m in self._messages if m.session_id == session_id)

    def clear_session_messages(self, session_id: str) -> int:
        """清除指定 Session 的消息

        Args:
            session_id: Session ID

        Returns:
            清除的消息数量
        """
        original_count = len(self._messages)
        self._messages = [m for m in self._messages if m.session_id != session_id]
        cleared_count = original_count - len(self._messages)

        if cleared_count > 0:
            self._save_messages()
            logger.info(
                "[MessageQueue] 清除 Session 消息: session=%s, cleared=%d",
                session_id,
                cleared_count,
            )

        return cleared_count

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """解析时间戳字符串，处理带和不带时区的情况"""
        if "Z" in timestamp_str:
            timestamp_str = timestamp_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(timestamp_str)
        # 如果带时区，转换为 naive
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt

    def _cleanup_expired(self) -> int:
        """清理过期消息

        Returns:
            清理的消息数量
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(days=self._ttl_days)

        original_count = len(self._messages)
        self._messages = [m for m in self._messages if self._parse_timestamp(m.timestamp) > cutoff]
        cleared_count = original_count - len(self._messages)

        if cleared_count > 0:
            self._save_messages()
            logger.info(
                "[MessageQueue] 清理过期消息: cleared=%d, remaining=%d",
                cleared_count,
                len(self._messages),
            )

        return cleared_count

    def cleanup(self) -> int:
        """手动清理过期消息

        Returns:
            清理的消息数量
        """
        return self._cleanup_expired()

    @property
    def queue_file(self) -> Path:
        """队列文件路径"""
        return self._queue_file

    @property
    def ttl_days(self) -> int:
        """消息过期时间（天）"""
        return self._ttl_days
