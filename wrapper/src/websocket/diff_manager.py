"""DIFF 模式管理器

管理 WebSocket 增量同步模式：
- diff 模式: 发送 JSON Patch，减少带宽
- full 模式: 发送完整数据

支持自动模式切换和带宽优化。
"""

import logging
from typing import Any, Dict, List, Literal, Optional

from .patch_generator import PatchGenerator

logger = logging.getLogger(__name__)


class DiffManager:
    """DIFF 模式管理器

    管理 diff/full 模式切换，优化带宽使用。

    Attributes:
        mode: 当前模式 (diff/full)
        threshold: 带宽节省阈值（百分比）
        min_diff_size: 最小 diff 大小（字节）
    """

    def __init__(
        self,
        mode: Literal["diff", "full"] = "diff",
        threshold: float = 50.0,
        min_diff_size: int = 100,
    ):
        """初始化 DIFF 管理器

        Args:
            mode: 默认模式，默认 diff
            threshold: 带宽节省阈值（百分比），默认 50%
            min_diff_size: 最小 diff 大小（字节），默认 100
        """
        self._mode = mode
        self._threshold = threshold
        self._min_diff_size = min_diff_size
        self._last_states: Dict[str, Any] = {}

        logger.debug(
            "[DiffManager] 初始化: mode=%s, threshold=%.1f%%, min_diff_size=%d",
            mode,
            threshold,
            min_diff_size,
        )

    def should_use_diff(self, key: str, new_data: Any) -> bool:
        """判断是否使用 diff 模式

        Args:
            key: 数据标识
            new_data: 新数据

        Returns:
            是否使用 diff 模式
        """
        if self._mode == "full":
            return False

        if key not in self._last_states:
            return False

        old_data = self._last_states[key]
        patches = PatchGenerator.generate_patch(old_data, new_data)

        if not patches:
            return False

        savings = PatchGenerator.calculate_savings(old_data, new_data, patches)
        diff_size = len(str(patches))

        should_diff = savings >= self._threshold and diff_size >= self._min_diff_size

        logger.debug(
            "[DiffManager] 模式判断: key=%s, savings=%.1f%%, diff_size=%d, use_diff=%s",
            key,
            savings,
            diff_size,
            should_diff,
        )

        return should_diff

    def create_message(
        self,
        key: str,
        new_data: Any,
        metadata: Optional[dict] = None,
    ) -> dict:
        """创建消息（diff 或 full）

        Args:
            key: 数据标识
            new_data: 新数据
            metadata: 额外元数据

        Returns:
            消息字典
        """
        use_diff = self.should_use_diff(key, new_data)

        if use_diff:
            old_data = self._last_states[key]
            patches = PatchGenerator.generate_patch(old_data, new_data)

            message = {
                "type": "diff",
                "key": key,
                "patches": patches,
            }

            if metadata:
                message["metadata"] = metadata

            logger.debug("[DiffManager] 创建 diff 消息: key=%s, patches=%d", key, len(patches))
        else:
            message = {
                "type": "full",
                "key": key,
                "data": new_data,
            }

            if metadata:
                message["metadata"] = metadata

            logger.debug("[DiffManager] 创建 full 消息: key=%s", key)

        self._last_states[key] = PatchGenerator._deep_copy(new_data)
        return message

    def update_state(self, key: str, data: Any) -> None:
        """更新状态缓存

        Args:
            key: 数据标识
            data: 数据
        """
        self._last_states[key] = PatchGenerator._deep_copy(data)
        logger.debug("[DiffManager] 更新状态: key=%s", key)

    def get_state(self, key: str) -> Optional[Any]:
        """获取状态缓存

        Args:
            key: 数据标识

        Returns:
            缓存的数据，如果不存在返回 None
        """
        return self._last_states.get(key)

    def clear_state(self, key: Optional[str] = None) -> None:
        """清除状态缓存

        Args:
            key: 数据标识，如果为 None 清除所有
        """
        if key is None:
            self._last_states.clear()
            logger.debug("[DiffManager] 清除所有状态")
        else:
            self._last_states.pop(key, None)
            logger.debug("[DiffManager] 清除状态: key=%s", key)

    def set_mode(self, mode: Literal["diff", "full"]) -> None:
        """设置模式

        Args:
            mode: 模式 (diff/full)
        """
        self._mode = mode
        logger.info("[DiffManager] 模式切换: %s", mode)

    def calculate_savings(self, key: str, new_data: Any) -> float:
        """计算带宽节省百分比

        Args:
            key: 数据标识
            new_data: 新数据

        Returns:
            节省百分比 (0-100)
        """
        if key not in self._last_states:
            return 0.0

        old_data = self._last_states[key]
        patches = PatchGenerator.generate_patch(old_data, new_data)

        return PatchGenerator.calculate_savings(old_data, new_data, patches)

    @property
    def mode(self) -> str:
        """当前模式"""
        return self._mode

    @property
    def threshold(self) -> float:
        """带宽节省阈值"""
        return self._threshold

    @property
    def state_count(self) -> int:
        """状态缓存数量"""
        return len(self._last_states)
