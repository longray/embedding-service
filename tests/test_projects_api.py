"""Projects API 端点测试

验证 BL-T-2: Projects API 测试
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from wrapper.src.routers.projects import get_project_map, get_project_stats


class TestGetProjectMap:
    """GET /api/v1/projects/{project_id}/map 端点测试"""

    @pytest.fixture
    def mock_memory_manager(self):
        """模拟 MemoryManager"""
        mm = MagicMock()
        mm.get_project_map = AsyncMock(
            return_value={
                "status": "success",
                "project_id": "test-project",
                "file_tree": {
                    "src": {
                        "main.py": {"complexity": 5, "functions": 3},
                        "utils.py": {"complexity": 3, "functions": 2},
                    },
                    "tests": {
                        "test_main.py": {"complexity": 2, "functions": 4},
                    },
                },
                "dependencies": [
                    {"from": "src/main.py", "to": "src/utils.py", "type": "import"},
                ],
                "hot_files": [
                    {"file_path": "src/main.py", "complexity": 5, "function_count": 3},
                ],
                "stats": {
                    "total_files": 3,
                    "total_functions": 9,
                    "avg_complexity": 3.33,
                },
            }
        )
        return mm

    @pytest.mark.asyncio
    async def test_get_project_map_success(self, mock_memory_manager):
        """测试正常获取项目地图"""
        with patch("wrapper.src.routers.projects.state.memory_manager", mock_memory_manager):
            result = await get_project_map(project_id="test-project", tenant_id="default")

        assert result["status"] == "success"
        assert result["project_id"] == "test-project"
        assert "file_tree" in result
        assert "dependencies" in result
        assert "hot_files" in result
        assert "stats" in result
        mock_memory_manager.get_project_map.assert_called_once()
        call_args = mock_memory_manager.get_project_map.call_args[0]
        assert call_args[0] == "test-project"
        assert call_args[1] == "default"

    @pytest.mark.asyncio
    async def test_get_project_map_empty_project(self, mock_memory_manager):
        """测试项目不存在返回空结构"""
        mock_memory_manager.get_project_map = AsyncMock(
            return_value={
                "status": "success",
                "project_id": "empty-project",
                "file_tree": {},
                "dependencies": [],
                "hot_files": [],
                "stats": {
                    "total_files": 0,
                    "total_functions": 0,
                    "avg_complexity": 0,
                },
            }
        )

        with patch("wrapper.src.routers.projects.state.memory_manager", mock_memory_manager):
            result = await get_project_map(project_id="empty-project")

        assert result["status"] == "success"
        assert result["file_tree"] == {}
        assert result["dependencies"] == []
        assert result["stats"]["total_files"] == 0

    @pytest.mark.asyncio
    async def test_get_project_map_field_validation(self, mock_memory_manager):
        """测试返回字段完整性"""
        with patch("wrapper.src.routers.projects.state.memory_manager", mock_memory_manager):
            result = await get_project_map(project_id="test-project")

        # 验证所有必需字段
        assert "file_tree" in result
        assert "dependencies" in result
        assert "hot_files" in result
        assert "stats" in result

        # 验证 stats 字段
        stats = result["stats"]
        assert "total_files" in stats
        assert "total_functions" in stats
        assert "avg_complexity" in stats

    @pytest.mark.asyncio
    async def test_get_project_map_multi_tenant(self, mock_memory_manager):
        """测试多租户隔离"""
        # 租户 A
        mock_memory_manager.get_project_map = AsyncMock(
            return_value={
                "status": "success",
                "project_id": "test-project",
                "file_tree": {"tenant": "A"},
            }
        )

        with patch("wrapper.src.routers.projects.state.memory_manager", mock_memory_manager):
            result_a = await get_project_map(project_id="test-project", tenant_id="tenant-a")

        call_args_a = mock_memory_manager.get_project_map.call_args[0]
        assert call_args_a[0] == "test-project"
        assert call_args_a[1] == "tenant-a"

        # 租户 B
        mock_memory_manager.get_project_map = AsyncMock(
            return_value={
                "status": "success",
                "project_id": "test-project",
                "file_tree": {"tenant": "B"},
            }
        )

        with patch("wrapper.src.routers.projects.state.memory_manager", mock_memory_manager):
            result_b = await get_project_map(project_id="test-project", tenant_id="tenant-b")

        call_args_b = mock_memory_manager.get_project_map.call_args[0]
        assert call_args_b[0] == "test-project"
        assert call_args_b[1] == "tenant-b"

    @pytest.mark.asyncio
    async def test_get_project_map_memory_manager_not_initialized(self):
        """测试 MemoryManager 未初始化"""
        with patch("wrapper.src.routers.projects.state.memory_manager", None):
            with pytest.raises(HTTPException) as exc_info:
                await get_project_map(project_id="test-project")

        assert exc_info.value.status_code == 503
        assert "MemoryManager未初始化" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_project_map_database_error(self, mock_memory_manager):
        """测试数据库查询失败"""
        mock_memory_manager.get_project_map = AsyncMock(
            return_value={
                "status": "error",
                "message": "数据库查询失败",
            }
        )

        with patch("wrapper.src.routers.projects.state.memory_manager", mock_memory_manager):
            with pytest.raises(HTTPException) as exc_info:
                await get_project_map(project_id="test-project")

        assert exc_info.value.status_code == 500
        assert "数据库查询失败" in exc_info.value.detail


class TestGetProjectStats:
    """GET /api/v1/projects/{project_id}/stats 端点测试"""

    @pytest.fixture
    def mock_memory_manager(self):
        """模拟 MemoryManager"""
        mm = MagicMock()
        mm.get_project_stats = AsyncMock(
            return_value={
                "status": "success",
                "project_id": "test-project",
                "total_files": 10,
                "total_functions": 45,
                "total_classes": 8,
                "avg_complexity": 3.5,
                "max_complexity": 12,
            }
        )
        return mm

    @pytest.mark.asyncio
    async def test_get_project_stats_success(self, mock_memory_manager):
        """测试正常获取项目统计"""
        with patch("wrapper.src.routers.projects.state.memory_manager", mock_memory_manager):
            result = await get_project_stats(project_id="test-project", tenant_id="default")

        assert result["status"] == "success"
        assert result["project_id"] == "test-project"
        assert result["total_files"] == 10
        assert result["total_functions"] == 45
        assert result["total_classes"] == 8
        assert result["avg_complexity"] == 3.5
        assert result["max_complexity"] == 12
        mock_memory_manager.get_project_stats.assert_called_once()
        call_args = mock_memory_manager.get_project_stats.call_args[0]
        assert call_args[0] == "test-project"
        assert call_args[1] == "default"

    @pytest.mark.asyncio
    async def test_get_project_stats_field_validation(self, mock_memory_manager):
        """测试统计字段完整性"""
        with patch("wrapper.src.routers.projects.state.memory_manager", mock_memory_manager):
            result = await get_project_stats(project_id="test-project")

        # 验证所有必需字段
        assert "total_files" in result
        assert "total_functions" in result
        assert "total_classes" in result
        assert "avg_complexity" in result
        assert "max_complexity" in result

        # 验证字段类型
        assert isinstance(result["total_files"], int)
        assert isinstance(result["total_functions"], int)
        assert isinstance(result["total_classes"], int)
        assert isinstance(result["avg_complexity"], (int, float))
        assert isinstance(result["max_complexity"], (int, float))

    @pytest.mark.asyncio
    async def test_get_project_stats_empty_project(self, mock_memory_manager):
        """测试空项目统计（全为 0）"""
        mock_memory_manager.get_project_stats = AsyncMock(
            return_value={
                "status": "success",
                "project_id": "empty-project",
                "total_files": 0,
                "total_functions": 0,
                "total_classes": 0,
                "avg_complexity": 0,
                "max_complexity": 0,
            }
        )

        with patch("wrapper.src.routers.projects.state.memory_manager", mock_memory_manager):
            result = await get_project_stats(project_id="empty-project")

        assert result["status"] == "success"
        assert result["total_files"] == 0
        assert result["total_functions"] == 0
        assert result["total_classes"] == 0
        assert result["avg_complexity"] == 0
        assert result["max_complexity"] == 0

    @pytest.mark.asyncio
    async def test_get_project_stats_multi_tenant(self, mock_memory_manager):
        """测试多租户隔离"""
        mock_memory_manager.get_project_stats = AsyncMock(
            return_value={
                "status": "success",
                "project_id": "test-project",
                "total_files": 5,
            }
        )

        with patch("wrapper.src.routers.projects.state.memory_manager", mock_memory_manager):
            result = await get_project_stats(project_id="test-project", tenant_id="tenant-a")

        call_args = mock_memory_manager.get_project_stats.call_args[0]
        assert call_args[0] == "test-project"
        assert call_args[1] == "tenant-a"

    @pytest.mark.asyncio
    async def test_get_project_stats_complexity_accuracy(self, mock_memory_manager):
        """测试复杂项目统计准确性"""
        mock_memory_manager.get_project_stats = AsyncMock(
            return_value={
                "status": "success",
                "project_id": "complex-project",
                "total_files": 100,
                "total_functions": 500,
                "total_classes": 50,
                "avg_complexity": 5.75,
                "max_complexity": 25,
            }
        )

        with patch("wrapper.src.routers.projects.state.memory_manager", mock_memory_manager):
            result = await get_project_stats(project_id="complex-project")

        assert result["total_files"] == 100
        assert result["total_functions"] == 500
        assert result["avg_complexity"] == 5.75
        assert result["max_complexity"] == 25

    @pytest.mark.asyncio
    async def test_get_project_stats_memory_manager_not_initialized(self):
        """测试 MemoryManager 未初始化"""
        with patch("wrapper.src.routers.projects.state.memory_manager", None):
            with pytest.raises(HTTPException) as exc_info:
                await get_project_stats(project_id="test-project")

        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_get_project_stats_database_error(self, mock_memory_manager):
        """测试数据库查询失败"""
        mock_memory_manager.get_project_stats = AsyncMock(
            return_value={
                "status": "error",
                "message": "统计查询失败",
            }
        )

        with patch("wrapper.src.routers.projects.state.memory_manager", mock_memory_manager):
            with pytest.raises(HTTPException) as exc_info:
                await get_project_stats(project_id="test-project")

        assert exc_info.value.status_code == 500
        assert "统计查询失败" in exc_info.value.detail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
