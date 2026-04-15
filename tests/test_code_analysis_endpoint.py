"""代码分析端点测试

验证 BL-C-6: 代码分析端点
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from wrapper.src.routers.code_analysis import analyze_memory_code


class TestCodeAnalysisEndpoint:
    """测试代码分析端点"""

    @pytest.fixture
    def mock_memory_manager(self):
        """模拟 MemoryManager"""
        mm = MagicMock()
        mm.analyze_memory_code = AsyncMock(
            return_value={
                "language": "python",
                "functions": [{"name": "hello", "line": 1}],
                "classes": [],
                "complexity": {"score": 5},
            }
        )
        return mm

    @pytest.mark.asyncio
    async def test_analyze_memory_code_success(self, mock_memory_manager):
        """测试代码分析成功"""
        with patch("wrapper.src.routers.code_analysis.state.memory_manager", mock_memory_manager):
            result = await analyze_memory_code(memory_id="memory:test", tenant_id="default")

            assert result["status"] == "success"
            assert result["memory_id"] == "memory:test"
            assert "result" in result
            assert result["result"]["language"] == "python"
            mock_memory_manager.analyze_memory_code.assert_called_once_with("memory:test", "default")

    @pytest.mark.asyncio
    async def test_analyze_memory_code_empty_result(self, mock_memory_manager):
        """测试代码分析返回空结果（非代码内容）"""
        mock_memory_manager.analyze_memory_code = AsyncMock(return_value={})

        with patch("wrapper.src.routers.code_analysis.state.memory_manager", mock_memory_manager):
            result = await analyze_memory_code(memory_id="memory:test", tenant_id="default")

            assert result["status"] == "skipped"
            assert "message" in result
            assert result["memory_id"] == "memory:test"

    @pytest.mark.asyncio
    async def test_analyze_memory_code_not_found(self, mock_memory_manager):
        """测试分析不存在的记忆"""
        mock_memory_manager.analyze_memory_code = AsyncMock(return_value={})

        with patch("wrapper.src.routers.code_analysis.state.memory_manager", mock_memory_manager):
            result = await analyze_memory_code(memory_id="memory:nonexistent", tenant_id="default")

            assert result["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_memory_manager_not_initialized(self):
        """测试 MemoryManager 未初始化"""
        with patch("wrapper.src.routers.code_analysis.state.memory_manager", None):
            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc_info:
                await analyze_memory_code(memory_id="memory:test")

            assert exc_info.value.status_code == 503
            assert "MemoryManager未初始化" in exc_info.value.detail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
