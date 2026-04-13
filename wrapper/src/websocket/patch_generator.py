"""JSON Patch 生成器 (RFC 6902)

实现标准 JSON Patch 格式：
- replace: 替换值
- add: 添加新值
- remove: 删除值

Patch 格式: [{"op": "replace", "path": "/field", "value": "new"}]
"""

import json
from typing import Any, List


class PatchGenerator:
    """JSON Patch 生成器

    根据新旧数据生成 RFC 6902 标准 JSON Patch。

    支持操作:
    - replace: 替换现有值
    - add: 添加新字段或数组元素
    - remove: 删除字段
    """

    @staticmethod
    def generate_patch(old: Any, new: Any, path: str = "") -> List[dict]:
        """生成 JSON Patch

        Args:
            old: 旧数据
            new: 新数据
            path: 当前路径（递归使用）

        Returns:
            RFC 6902 JSON Patch 列表
        """
        patches = []

        if old == new:
            return patches

        if type(old) != type(new):
            patches.append({"op": "replace", "path": path or "/", "value": new})
            return patches

        if isinstance(old, dict):
            patches.extend(PatchGenerator._diff_dict(old, new, path))
        elif isinstance(old, list):
            patches.extend(PatchGenerator._diff_list(old, new, path))
        else:
            patches.append({"op": "replace", "path": path or "/", "value": new})

        return patches

    @staticmethod
    def _diff_dict(old: dict, new: dict, path: str) -> List[dict]:
        """比较字典差异"""
        patches = []
        old_keys = set(old.keys())
        new_keys = set(new.keys())

        for key in old_keys - new_keys:
            key_path = f"{path}/{key}"
            patches.append({"op": "remove", "path": key_path})

        for key in new_keys - old_keys:
            key_path = f"{path}/{key}"
            patches.append({"op": "add", "path": key_path, "value": new[key]})

        for key in old_keys & new_keys:
            key_path = f"{path}/{key}"
            if old[key] != new[key]:
                if isinstance(old[key], (dict, list)) and isinstance(new[key], (dict, list)):
                    patches.extend(PatchGenerator.generate_patch(old[key], new[key], key_path))
                else:
                    patches.append({"op": "replace", "path": key_path, "value": new[key]})

        return patches

    @staticmethod
    def _diff_list(old: list, new: list, path: str) -> List[dict]:
        """比较列表差异（简化版：如果不同则替换整个列表）"""
        if old != new:
            return [{"op": "replace", "path": path or "/", "value": new}]
        return []

    @staticmethod
    def apply_patch(data: Any, patches: List[dict]) -> Any:
        """应用 JSON Patch 到数据

        Args:
            data: 原始数据
            patches: JSON Patch 列表

        Returns:
            应用 patch 后的数据
        """
        result = PatchGenerator._deep_copy(data)

        for patch in patches:
            op = patch.get("op")
            path = patch.get("path", "/")
            value = patch.get("value")

            if op == "replace":
                PatchGenerator._set_value(result, path, value)
            elif op == "add":
                PatchGenerator._add_value(result, path, value)
            elif op == "remove":
                PatchGenerator._remove_value(result, path)

        return result

    @staticmethod
    def _deep_copy(data: Any) -> Any:
        """深拷贝数据"""
        return json.loads(json.dumps(data))

    @staticmethod
    def _set_value(data: Any, path: str, value: Any) -> None:
        """设置路径上的值"""
        if path == "/":
            return

        parts = path.strip("/").split("/")
        current = data

        for part in parts[:-1]:
            if isinstance(current, dict):
                current = current.get(part, {})
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    current = current[idx] if 0 <= idx < len(current) else {}
                except (ValueError, IndexError):
                    return

        if isinstance(current, dict) and parts:
            current[parts[-1]] = value

    @staticmethod
    def _add_value(data: Any, path: str, value: Any) -> None:
        """添加值"""
        PatchGenerator._set_value(data, path, value)

    @staticmethod
    def _remove_value(data: Any, path: str) -> None:
        """删除值"""
        if path == "/":
            return

        parts = path.strip("/").split("/")
        current = data

        for part in parts[:-1]:
            if isinstance(current, dict):
                current = current.get(part, {})
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    current = current[idx] if 0 <= idx < len(current) else {}
                except (ValueError, IndexError):
                    return

        if isinstance(current, dict) and parts:
            current.pop(parts[-1], None)

    @staticmethod
    def calculate_savings(old: Any, new: Any, patches: List[dict]) -> float:
        """计算带宽节省百分比

        Args:
            old: 旧数据
            new: 新数据
            patches: JSON Patch

        Returns:
            节省百分比 (0-100)
        """
        full_size = len(json.dumps(new, ensure_ascii=False))
        diff_size = len(json.dumps(patches, ensure_ascii=False))

        if full_size == 0:
            return 0.0

        savings = (full_size - diff_size) / full_size * 100
        return max(0.0, savings)
