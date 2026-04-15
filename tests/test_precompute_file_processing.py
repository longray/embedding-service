"""PrecomputeService 文件处理测试

验证 BL-C-3: PrecomputeService — 文件处理逻辑实现
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from wrapper.src.services.precompute import PrecomputeService


class TestPrecomputeFileProcessing:
    """测试 PrecomputeService 文件处理"""

    @pytest.fixture
    def mock_db(self):
        """模拟数据库连接"""
        return MagicMock()

    @pytest_asyncio.fixture
    async def running_service(self, mock_db):
        """运行中的服务实例"""
        service = PrecomputeService(
            db=mock_db,
            tenant_id="test",
            max_concurrent=3,
        )
        await service.start()
        yield service
        if service._running:
            await service.stop()

    @pytest.mark.asyncio
    async def test_read_file_success(self, running_service):
        """测试读取文件成功"""
        service = running_service

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello(): pass")
            temp_path = f.name

        try:
            content = await service._read_file(temp_path)
            assert content == "def hello(): pass"
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, running_service):
        """测试读取不存在的文件"""
        service = running_service

        content = await service._read_file("/nonexistent/file.py")
        assert content is None

    @pytest.mark.asyncio
    async def test_process_file_success(self, running_service):
        """测试处理文件成功"""
        service = running_service

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello():\n    pass")
            temp_path = f.name

        try:
            result = await service._process_file(temp_path, {})

            assert result is not None
            assert result["file_path"] == temp_path
            assert result["status"] == "processed"
            assert result["language"] == "python"
            assert "fingerprint" in result
            assert "symbols" in result
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_process_file_unchanged(self, running_service):
        """测试处理未变更的文件"""
        service = running_service

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello(): pass")
            temp_path = f.name

        try:
            # 第一次处理
            result1 = await service._process_file(temp_path, {})
            assert result1["status"] == "processed"

            # 第二次处理（应该返回 unchanged）
            result2 = await service._process_file(temp_path, {})
            assert result2["status"] == "unchanged"
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_process_batch(self, running_service):
        """测试批量处理"""
        service = running_service

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f1:
            f1.write("def func1(): pass")
            temp_path1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f2:
            f2.write("def func2(): pass")
            temp_path2 = f2.name

        try:
            batch = [
                {"file_path": temp_path1},
                {"file_path": temp_path2},
            ]

            result = await service.process_batch(batch)

            assert result["processed_count"] == 2
            assert result["error_count"] == 0
            assert result["tenant_id"] == "test"
        finally:
            Path(temp_path1).unlink()
            Path(temp_path2).unlink()

    @pytest.mark.asyncio
    async def test_process_batch_with_errors(self, running_service):
        """测试批量处理（包含错误）"""
        service = running_service

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def func(): pass")
            temp_path = f.name

        try:
            batch = [
                {"file_path": temp_path},
                {"file_path": "/nonexistent/file.py"},
            ]

            result = await service.process_batch(batch)

            assert result["processed_count"] == 1
            assert result["error_count"] == 0
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
