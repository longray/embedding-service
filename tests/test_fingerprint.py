"""指纹管理器测试

测试范围：
- FingerprintManager 基础功能
- SHA256 指纹计算
- 变更检测
- 缓存管理

运行方式：
    uv run pytest tests/test_fingerprint.py -v
"""

import pytest

from wrapper.src.services.fingerprint import FingerprintManager


class TestFingerprintManager:
    """FingerprintManager 单元测试"""

    @pytest.fixture
    def fm(self):
        """创建 FingerprintManager 实例"""
        return FingerprintManager()

    def test_initialization(self, fm):
        """测试初始化"""
        assert fm.fingerprint_count == 0
        assert fm.get_all_fingerprints() == {}

    def test_calculate_fingerprint(self, fm):
        """测试指纹计算"""
        content = "def hello(): pass"
        fingerprint = fm.calculate_fingerprint(content)

        assert len(fingerprint) == 64  # SHA256 hex
        assert all(c in "0123456789abcdef" for c in fingerprint)

    def test_calculate_fingerprint_empty(self, fm):
        """测试空内容指纹"""
        fingerprint = fm.calculate_fingerprint("")

        assert fingerprint == ""

    def test_calculate_fingerprint_consistency(self, fm):
        """测试指纹一致性"""
        content = "def hello(): pass"
        fp1 = fm.calculate_fingerprint(content)
        fp2 = fm.calculate_fingerprint(content)

        assert fp1 == fp2

    def test_calculate_fingerprint_different_content(self, fm):
        """测试不同内容指纹不同"""
        fp1 = fm.calculate_fingerprint("def hello(): pass")
        fp2 = fm.calculate_fingerprint("def world(): pass")

        assert fp1 != fp2

    def test_has_changed_new_file(self, fm):
        """测试新文件变更检测"""
        result = fm.has_changed("/path/to/file.py", "abc123")

        assert result is True

    def test_has_changed_existing_file(self, fm):
        """测试已有文件变更检测"""
        fm.save_fingerprint("/path/to/file.py", "abc123")

        result = fm.has_changed("/path/to/file.py", "abc123")

        assert result is False

    def test_has_changed_modified_file(self, fm):
        """测试修改文件变更检测"""
        fm.save_fingerprint("/path/to/file.py", "abc123")

        result = fm.has_changed("/path/to/file.py", "def456")

        assert result is True

    def test_save_and_get_fingerprint(self, fm):
        """测试保存和获取指纹"""
        fm.save_fingerprint("/path/to/file.py", "abc123")

        result = fm.get_fingerprint("/path/to/file.py")

        assert result == "abc123"

    def test_get_fingerprint_nonexistent(self, fm):
        """测试获取不存在的指纹"""
        result = fm.get_fingerprint("/path/to/nonexistent.py")

        assert result is None

    def test_remove_fingerprint(self, fm):
        """测试删除指纹"""
        fm.save_fingerprint("/path/to/file.py", "abc123")

        result = fm.remove_fingerprint("/path/to/file.py")

        assert result is True
        assert fm.get_fingerprint("/path/to/file.py") is None

    def test_remove_fingerprint_nonexistent(self, fm):
        """测试删除不存在的指纹"""
        result = fm.remove_fingerprint("/path/to/nonexistent.py")

        assert result is False

    def test_clear_cache(self, fm):
        """测试清除缓存"""
        fm.save_fingerprint("/path/to/file1.py", "abc123")
        fm.save_fingerprint("/path/to/file2.py", "def456")

        fm.clear_cache()

        assert fm.fingerprint_count == 0
        assert fm.get_all_fingerprints() == {}

    def test_get_all_fingerprints(self, fm):
        """测试获取所有指纹"""
        fm.save_fingerprint("/path/to/file1.py", "abc123")
        fm.save_fingerprint("/path/to/file2.py", "def456")

        result = fm.get_all_fingerprints()

        assert len(result) == 2
        assert result["/path/to/file1.py"] == "abc123"
        assert result["/path/to/file2.py"] == "def456"

    def test_fingerprint_count(self, fm):
        """测试指纹计数"""
        assert fm.fingerprint_count == 0

        fm.save_fingerprint("/path/to/file1.py", "abc123")
        assert fm.fingerprint_count == 1

        fm.save_fingerprint("/path/to/file2.py", "def456")
        assert fm.fingerprint_count == 2


class TestFingerprintManagerIntegration:
    """集成测试"""

    def test_full_workflow(self):
        """测试完整工作流"""
        fm = FingerprintManager()

        # 初始状态
        content1 = "def hello(): pass"
        fp1 = fm.calculate_fingerprint(content1)

        # 检查变更（新文件）
        assert fm.has_changed("/path/to/file.py", fp1) is True

        # 保存指纹
        fm.save_fingerprint("/path/to/file.py", fp1)

        # 检查变更（未修改）
        assert fm.has_changed("/path/to/file.py", fp1) is False

        # 修改内容
        content2 = "def world(): pass"
        fp2 = fm.calculate_fingerprint(content2)

        # 检查变更（已修改）
        assert fm.has_changed("/path/to/file.py", fp2) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
