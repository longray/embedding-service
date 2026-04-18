"""WebSocket 状态恢复管理器

实现连接断开后状态恢复：
- Session ID 生成: sess-{timestamp}-{uuid[:9]}
- Offset 持久化到文件
- 状态保存和恢复
- TTL 清理（7天）

存储文件: .opencode/ws-state.json
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class StateRecoveryManager:
    """状态恢复管理器

    管理 WebSocket 连接状态，支持断线重连后恢复。

    Attributes:
        state_file: 状态文件路径
        ttl_days: 状态过期时间（天）
    """

    def __init__(self, state_file: str = ".opencode/ws-state.json", ttl_days: int = 7):
        """初始化状态恢复管理器

        Args:
            state_file: 状态文件路径，默认 .opencode/ws-state.json
            ttl_days: 状态过期时间（天），默认 7
        """
        self._state_file = Path(state_file)
        self._ttl_days = ttl_days
        self._state: Dict[str, Any] = {"sessions": {}}

        self._ensure_directory()
        self._load_state()

        logger.debug(
            "[StateRecoveryManager] 初始化: state_file=%s, ttl_days=%d",
            state_file,
            ttl_days,
        )

    def _ensure_directory(self) -> None:
        """确保目录存在"""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> None:
        """从文件加载状态"""
        if not self._state_file.exists():
            logger.debug("[StateRecoveryManager] 状态文件不存在，创建新状态")
            return

        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                self._state = json.load(f)
            logger.debug(
                "[StateRecoveryManager] 加载状态: %d sessions",
                len(self._state.get("sessions", {})),
            )
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("[StateRecoveryManager] 加载状态失败: %s", e)
            self._state = {"sessions": {}}

    def _save_state(self) -> None:
        """保存状态到文件"""
        try:
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False, indent=2)
            logger.debug("[StateRecoveryManager] 状态已保存")
        except IOError as e:
            logger.error("[StateRecoveryManager] 保存状态失败: %s", e)

    def generate_session_id(self) -> str:
        """生成 Session ID

        Returns:
            Session ID (格式: sess-{timestamp}-{uuid[:9]})
        """
        timestamp = int(time.time())
        short_uuid = uuid.uuid4().hex[:9]
        session_id = f"sess-{timestamp}-{short_uuid}"
        logger.debug("[StateRecoveryManager] 生成 Session ID: %s", session_id)
        return session_id

    def save_state(
        self,
        session_id: str,
        offset: int,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """保存状态

        Args:
            session_id: Session ID
            offset: 消息 offset
            data: 额外数据（可选）
        """
        now = datetime.utcnow().isoformat()

        if session_id not in self._state["sessions"]:
            self._state["sessions"][session_id] = {
                "created_at": now,
            }

        self._state["sessions"][session_id].update(
            {
                "offset": offset,
                "updated_at": now,
            }
        )

        if data is not None:
            self._state["sessions"][session_id]["data"] = data

        self._save_state()
        logger.debug(
            "[StateRecoveryManager] 保存状态: session_id=%s, offset=%d",
            session_id,
            offset,
        )

    def restore_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """恢复状态

        检查 session 是否存在且未过期。

        Args:
            session_id: Session ID

        Returns:
            状态字典，如果不存在或已过期返回 None
        """
        session = self._state["sessions"].get(session_id)
        if session is None:
            logger.debug(
                "[StateRecoveryManager] Session 不存在: %s",
                session_id,
            )
            return None

        # 检查 TTL
        updated_at_str = session.get("updated_at") or session.get("created_at")
        if updated_at_str:
            # 处理带时区的 ISO 格式（+00:00 或 Z）
            updated_at_str = updated_at_str.replace("Z", "+00:00")
            updated_at = datetime.fromisoformat(updated_at_str)
            # 如果带时区，转换为 naive
            if updated_at.tzinfo is not None:
                updated_at = updated_at.replace(tzinfo=None)
            if datetime.utcnow() - updated_at > timedelta(days=self._ttl_days):
                logger.warning("[StateRecoveryManager] Session 已过期: %s", session_id)
                # 删除过期 session
                del self._state["sessions"][session_id]
                self._save_state()
                return None

        logger.debug(
            "[StateRecoveryManager] 恢复状态: session_id=%s, offset=%d",
            session_id,
            session.get("offset", 0),
        )
        return dict(session)

    def delete_state(self, session_id: str) -> bool:
        """删除状态

        Args:
            session_id: Session ID

        Returns:
            是否成功删除
        """
        if session_id in self._state["sessions"]:
            del self._state["sessions"][session_id]
            self._save_state()
            logger.debug(
                "[StateRecoveryManager] 删除状态: %s",
                session_id,
            )
            return True
        return False

    def get_offset(self, session_id: str) -> int:
        """获取当前 offset

        Args:
            session_id: Session ID

        Returns:
            Offset，如果不存在返回 0
        """
        session = self._state["sessions"].get(session_id)
        if session is None:
            return 0
        return session.get("offset", 0)

    def update_offset(self, session_id: str, offset: int) -> None:
        """更新 offset

        Args:
            session_id: Session ID
            offset: 新的 offset
        """
        self.save_state(session_id, offset)

    def cleanup_expired(self) -> int:
        """清理过期状态

        Returns:
            清理的 session 数量
        """
        now = datetime.utcnow()
        expired_sessions = []

        for session_id, session in self._state["sessions"].items():
            updated_at_str = session.get("updated_at") or session.get("created_at")
            if updated_at_str:
                # 处理带时区的 ISO 格式（+00:00 或 Z）
                updated_at_str = updated_at_str.replace("Z", "+00:00")
                updated_at = datetime.fromisoformat(updated_at_str)
                # 如果带时区，转换为 naive
                if updated_at.tzinfo is not None:
                    updated_at = updated_at.replace(tzinfo=None)
                if now - updated_at > timedelta(days=self._ttl_days):
                    expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self._state["sessions"][session_id]

        if expired_sessions:
            self._save_state()
            logger.info(
                "[StateRecoveryManager] 清理过期状态: %d sessions",
                len(expired_sessions),
            )

        return len(expired_sessions)

    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """获取所有 sessions

        Returns:
            Session ID -> 状态字典
        """
        return dict(self._state["sessions"])

    def session_exists(self, session_id: str) -> bool:
        """检查 session 是否存在

        Args:
            session_id: Session ID

        Returns:
            是否存在
        """
        return session_id in self._state["sessions"]

    @property
    def session_count(self) -> int:
        """Session 数量"""
        return len(self._state["sessions"])

    @property
    def state_file(self) -> Path:
        """状态文件路径"""
        return self._state_file
