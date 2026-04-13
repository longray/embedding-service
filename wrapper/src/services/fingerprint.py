"""指纹管理器

实现 SHA256 指纹计算和变更检测：
- 计算文件内容指纹
- 检测文件变更
- 支持增量分析
"""

import hashlib
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class FingerprintManager:
    """指纹管理器

    管理文件指纹，支持变更检测和增量分析。

    Attributes:
        fingerprints: 文件路径 -> 指纹的映射
    """

    def __init__(self):
        """初始化指纹管理器"""
        self._fingerprints: Dict[str, str] = {}
        self._logger = logging.getLogger(__name__)

        self._logger.debug("[FingerprintManager] 初始化")

    def calculate_fingerprint(self, content: str) -> str:
        """计算内容指纹

        使用 SHA256 计算内容的哈希值。

        Args:
            content: 文件内容

        Returns:
            SHA256 指纹（64位十六进制字符串）
        """
        if not content:
            return ""

        fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self._logger.debug("[FingerprintManager] 计算指纹: %s...", fingerprint[:8])

        return fingerprint

    def has_changed(self, file_path: str, new_fingerprint: str) -> bool:
        """检查文件是否变更

        对比新旧指纹，判断文件是否变更。

        Args:
            file_path: 文件路径
            new_fingerprint: 新指纹

        Returns:
            是否变更（True = 变更，False = 未变更）
        """
        old_fingerprint = self._fingerprints.get(file_path)

        if old_fingerprint is None:
            self._logger.debug("[FingerprintManager] 新文件: %s", file_path)
            return True

        if old_fingerprint != new_fingerprint:
            self._logger.debug(
                "[FingerprintManager] 文件变更: %s (old=%s..., new=%s...)",
                file_path,
                old_fingerprint[:8],
                new_fingerprint[:8],
            )
            return True

        self._logger.debug("[FingerprintManager] 文件未变更: %s", file_path)
        return False

    def get_fingerprint(self, file_path: str) -> Optional[str]:
        """获取文件指纹

        Args:
            file_path: 文件路径

        Returns:
            指纹，如果不存在返回 None
        """
        return self._fingerprints.get(file_path)

    def save_fingerprint(self, file_path: str, fingerprint: str) -> None:
        """保存文件指纹

        Args:
            file_path: 文件路径
            fingerprint: 指纹
        """
        self._fingerprints[file_path] = fingerprint
        self._logger.debug("[FingerprintManager] 保存指纹: %s", file_path)

    def remove_fingerprint(self, file_path: str) -> bool:
        """删除文件指纹

        Args:
            file_path: 文件路径

        Returns:
            是否成功删除
        """
        if file_path in self._fingerprints:
            del self._fingerprints[file_path]
            self._logger.debug("[FingerprintManager] 删除指纹: %s", file_path)
            return True
        return False

    def clear_cache(self) -> None:
        """清除指纹缓存"""
        self._fingerprints.clear()
        self._logger.debug("[FingerprintManager] 清除缓存")

    def get_all_fingerprints(self) -> Dict[str, str]:
        """获取所有指纹

        Returns:
            文件路径 -> 指纹的映射
        """
        return dict(self._fingerprints)

    @property
    def fingerprint_count(self) -> int:
        """指纹数量"""
        return len(self._fingerprints)
