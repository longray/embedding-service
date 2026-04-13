"""WebSocket DIFF 模式测试

测试范围：
- PatchGenerator JSON Patch 生成
- DiffManager 模式管理
- 带宽节省计算

运行方式：
    uv run pytest tests/test_websocket_diff.py -v
"""

import pytest

from wrapper.src.websocket.patch_generator import PatchGenerator
from wrapper.src.websocket.diff_manager import DiffManager


class TestPatchGenerator:
    """PatchGenerator 单元测试"""

    def test_generate_patch_replace(self):
        """测试 replace 操作"""
        old = {"content": "hello"}
        new = {"content": "world"}

        patches = PatchGenerator.generate_patch(old, new)

        assert len(patches) == 1
        assert patches[0] == {"op": "replace", "path": "/content", "value": "world"}

    def test_generate_patch_add(self):
        """测试 add 操作"""
        old = {"content": "hello"}
        new = {"content": "hello", "tags": ["new"]}

        patches = PatchGenerator.generate_patch(old, new)

        assert len(patches) == 1
        assert patches[0] == {"op": "add", "path": "/tags", "value": ["new"]}

    def test_generate_patch_remove(self):
        """测试 remove 操作"""
        old = {"content": "hello", "tags": ["old"]}
        new = {"content": "hello"}

        patches = PatchGenerator.generate_patch(old, new)

        assert len(patches) == 1
        assert patches[0] == {"op": "remove", "path": "/tags"}

    def test_generate_patch_nested(self):
        """测试嵌套对象"""
        old = {"user": {"name": "Alice", "age": 30}}
        new = {"user": {"name": "Bob", "age": 30}}

        patches = PatchGenerator.generate_patch(old, new)

        assert len(patches) == 1
        assert patches[0] == {"op": "replace", "path": "/user/name", "value": "Bob"}

    def test_generate_patch_no_change(self):
        """测试无变化"""
        old = {"content": "hello"}
        new = {"content": "hello"}

        patches = PatchGenerator.generate_patch(old, new)

        assert len(patches) == 0

    def test_generate_patch_type_change(self):
        """测试类型变化"""
        old = {"value": 123}
        new = {"value": "123"}

        patches = PatchGenerator.generate_patch(old, new)

        assert len(patches) == 1
        assert patches[0] == {"op": "replace", "path": "/value", "value": "123"}

    def test_apply_patch_replace(self):
        """测试应用 replace patch"""
        data = {"content": "hello"}
        patches = [{"op": "replace", "path": "/content", "value": "world"}]

        result = PatchGenerator.apply_patch(data, patches)

        assert result == {"content": "world"}

    def test_apply_patch_add(self):
        """测试应用 add patch"""
        data = {"content": "hello"}
        patches = [{"op": "add", "path": "/tags", "value": ["new"]}]

        result = PatchGenerator.apply_patch(data, patches)

        assert result == {"content": "hello", "tags": ["new"]}

    def test_apply_patch_remove(self):
        """测试应用 remove patch"""
        data = {"content": "hello", "tags": ["old"]}
        patches = [{"op": "remove", "path": "/tags"}]

        result = PatchGenerator.apply_patch(data, patches)

        assert result == {"content": "hello"}

    def test_calculate_savings(self):
        """测试带宽节省计算"""
        old = {"content": "x" * 1000, "metadata": {"author": "Alice"}}
        new = {"content": "y" * 1000, "metadata": {"author": "Alice"}}

        patches = PatchGenerator.generate_patch(old, new)
        savings = PatchGenerator.calculate_savings(old, new, patches)

        assert savings >= 0


class TestDiffManager:
    """DiffManager 单元测试"""

    def test_initialization(self):
        """测试初始化"""
        manager = DiffManager(mode="diff", threshold=50.0)

        assert manager.mode == "diff"
        assert manager.threshold == 50.0
        assert manager.state_count == 0

    def test_should_use_diff_no_previous_state(self):
        """测试无历史状态时使用 diff"""
        manager = DiffManager()

        result = manager.should_use_diff("key1", {"data": "new"})

        assert result is False

    def test_should_use_diff_with_state(self):
        """测试有历史状态时使用 diff"""
        manager = DiffManager(threshold=0.0, min_diff_size=1)
        manager.update_state("key1", {"content": "x" * 1000, "metadata": {"author": "Alice"}})

        result = manager.should_use_diff("key1", {"content": "y" * 1000, "metadata": {"author": "Alice"}})

        assert result is True

    def test_should_use_diff_full_mode(self):
        """测试 full 模式"""
        manager = DiffManager(mode="full")
        manager.update_state("key1", {"content": "old"})

        result = manager.should_use_diff("key1", {"content": "new"})

        assert result is False

    def test_create_message_diff(self):
        """测试创建 diff 消息"""
        manager = DiffManager(threshold=0.0, min_diff_size=1)
        manager.update_state("key1", {"content": "hello", "extra": "data"})

        message = manager.create_message("key1", {"content": "world", "extra": "data"})

        assert message["type"] == "diff"
        assert message["key"] == "key1"
        assert "patches" in message

    def test_create_message_full(self):
        """测试创建 full 消息"""
        manager = DiffManager()

        message = manager.create_message("key1", {"content": "world"})

        assert message["type"] == "full"
        assert message["key"] == "key1"
        assert message["data"] == {"content": "world"}

    def test_update_state(self):
        """测试更新状态"""
        manager = DiffManager()

        manager.update_state("key1", {"content": "data"})

        assert manager.state_count == 1
        assert manager.get_state("key1") == {"content": "data"}

    def test_clear_state(self):
        """测试清除状态"""
        manager = DiffManager()
        manager.update_state("key1", {"content": "data"})

        manager.clear_state("key1")

        assert manager.state_count == 0
        assert manager.get_state("key1") is None

    def test_clear_all_states(self):
        """测试清除所有状态"""
        manager = DiffManager()
        manager.update_state("key1", {"content": "data1"})
        manager.update_state("key2", {"content": "data2"})

        manager.clear_state()

        assert manager.state_count == 0

    def test_set_mode(self):
        """测试设置模式"""
        manager = DiffManager(mode="diff")

        manager.set_mode("full")

        assert manager.mode == "full"

    def test_calculate_savings(self):
        """测试计算节省"""
        manager = DiffManager()
        manager.update_state("key1", {"content": "x" * 1000, "metadata": {"author": "Alice"}})

        savings = manager.calculate_savings("key1", {"content": "y" * 1000, "metadata": {"author": "Alice"}})

        assert savings >= 0


class TestDiffBandwidthSavings:
    """带宽节省测试"""

    def test_large_data_savings(self):
        """测试大数据节省"""
        old = {
            "id": "123",
            "content": "x" * 10000,
            "metadata": {"author": "Alice", "timestamp": "2024-01-01"},
        }
        new = {
            "id": "123",
            "content": "y" * 10000,
            "metadata": {"author": "Alice", "timestamp": "2024-01-01"},
        }

        patches = PatchGenerator.generate_patch(old, new)
        savings = PatchGenerator.calculate_savings(old, new, patches)

        assert savings > 0

    def test_small_change_savings(self):
        """测试小变更节省"""
        old = {"items": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]}
        new = {"items": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Updated"}]}

        patches = PatchGenerator.generate_patch(old, new)
        savings = PatchGenerator.calculate_savings(old, new, patches)

        assert savings >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
