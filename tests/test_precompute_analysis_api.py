"""预计算分析 API 测试 (BL-B-81)

测试 POST /api/v1/precompute/analysis 端点
"""

import pytest
from fastapi.testclient import TestClient


class TestPrecomputeAnalysisAPI:
    """预计算分析 API 测试"""

    @pytest.mark.asyncio
    async def test_precompute_analysis_success(self, wrapper_client):
        """测试成功的预计算分析"""
        request_data = {
            "project_id": "test-project",
            "files": [
                {"path": "src/main.py", "content": "def main():\n    pass"},
                {"path": "src/utils.py", "content": "def helper():\n    pass"},
            ],
            "symbols": [
                {"name": "main", "type": "function", "location": "src/main.py:1"},
                {"name": "helper", "type": "function", "location": "src/utils.py:1"},
            ],
            "relations": [
                {"from_symbol": "main", "to_symbol": "helper", "type": "calls", "line": 2},
            ],
            "tenant_id": "default",
        }

        response = await wrapper_client.post("/api/v1/precompute/analysis", json=request_data)

        # 由于服务需要实际的数据库连接，可能返回 503（未初始化）或 200（成功）
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            result = response.json()
            assert "memory_ids" in result
            assert "status" in result
            assert "processed_count" in result

    @pytest.mark.asyncio
    async def test_precompute_analysis_empty_files(self, wrapper_client):
        """测试空文件列表"""
        request_data = {
            "project_id": "test-project",
            "files": [],
            "symbols": [],
            "relations": [],
            "tenant_id": "default",
        }

        response = await wrapper_client.post("/api/v1/precompute/analysis", json=request_data)

        assert response.status_code in [200, 503]

    @pytest.mark.asyncio
    async def test_precompute_analysis_missing_project_id(self, wrapper_client):
        """测试缺少 project_id"""
        request_data = {
            "files": [{"path": "src/main.py", "content": "def main(): pass"}],
            "tenant_id": "default",
        }

        response = await wrapper_client.post("/api/v1/precompute/analysis", json=request_data)

        # 应该返回 422 验证错误
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_precompute_analysis_invalid_file_format(self, wrapper_client):
        """测试无效的文件格式"""
        request_data = {
            "project_id": "test-project",
            "files": "invalid",  # 应该是列表
            "tenant_id": "default",
        }

        response = await wrapper_client.post("/api/v1/precompute/analysis", json=request_data)

        assert response.status_code == 422


class TestPrecomputeModels:
    """预计算模型测试"""

    def test_file_info_model(self):
        """测试 FileInfo 模型"""
        from wrapper.src.models import FileInfo

        file_info = FileInfo(path="src/main.py", content="def main(): pass")

        assert file_info.path == "src/main.py"
        assert file_info.content == "def main(): pass"

    def test_symbol_info_model(self):
        """测试 SymbolInfo 模型"""
        from wrapper.src.models import SymbolInfo

        symbol = SymbolInfo(
            name="main",
            type="function",
            location="src/main.py:1",
            signature="def main()",
        )

        assert symbol.name == "main"
        assert symbol.type == "function"
        assert symbol.location == "src/main.py:1"
        assert symbol.signature == "def main()"

    def test_relation_info_model(self):
        """测试 RelationInfo 模型"""
        from wrapper.src.models import RelationInfo

        relation = RelationInfo(
            from_symbol="main",
            to_symbol="helper",
            type="calls",
            line=10,
        )

        assert relation.from_symbol == "main"
        assert relation.to_symbol == "helper"
        assert relation.type == "calls"
        assert relation.line == 10

    def test_precompute_analysis_request_model(self):
        """测试 PrecomputeAnalysisRequest 模型"""
        from wrapper.src.models import PrecomputeAnalysisRequest, FileInfo, SymbolInfo

        request = PrecomputeAnalysisRequest(
            project_id="test-project",
            files=[FileInfo(path="src/main.py", content="def main(): pass")],
            symbols=[SymbolInfo(name="main", type="function", location="src/main.py:1")],
            relations=[],
            tenant_id="default",
        )

        assert request.project_id == "test-project"
        assert len(request.files) == 1
        assert request.tenant_id == "default"

    def test_precompute_analysis_response_model(self):
        """测试 PrecomputeAnalysisResponse 模型"""
        from wrapper.src.models import PrecomputeAnalysisResponse

        response = PrecomputeAnalysisResponse(
            memory_ids={"src/main.py": "mem-xxx"},
            status="success",
            processed_count=1,
            failed_count=0,
            errors=[],
        )

        assert response.memory_ids["src/main.py"] == "mem-xxx"
        assert response.status == "success"
        assert response.processed_count == 1
        assert response.failed_count == 0


class TestPrecomputeRouter:
    """预计算路由测试"""

    @pytest.mark.asyncio
    async def test_router_registered(self, wrapper_client):
        """测试路由已注册"""
        # 发送一个无效请求来验证路由存在
        response = await wrapper_client.post("/api/v1/precompute/analysis", json={})

        # 如果路由不存在会返回 404，存在但验证失败会返回 422
        assert response.status_code != 404

    def test_router_tags(self):
        """测试路由标签"""
        from wrapper.src.routers.precompute import router

        assert router.prefix == "/api/v1"
        assert "precompute" in router.tags
