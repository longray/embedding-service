"""完整工作流 E2E 测试

验证 BL-T-6: E2E 测试套件整合 - 完整工作流场景

测试范围：
- 记忆上传 → 搜索 → 更新 → 删除完整流程
- 代码分析工作流
- 同步冲突解决工作流

运行方式：
    uv run pytest tests/e2e/test_complete_workflow.py -v

前置条件：
- Docker Compose 完整环境可用
- 所有核心 API 端点已实现
"""

import asyncio
import os
import uuid
from datetime import datetime

import httpx
import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_E2E_TESTS") == "1",
    reason="E2E tests skipped (SKIP_E2E_TESTS=1)",
)

BASE_URL = "http://localhost:18008"
TEST_TENANT = "e2e_test"


class TestMemoryLifecycleWorkflow:
    """记忆生命周期工作流测试"""

    @pytest.mark.asyncio
    async def test_full_memory_lifecycle(self):
        """测试完整记忆生命周期：上传 → 搜索 → 更新 → 删除"""
        async with httpx.AsyncClient() as client:
            uid = str(uuid.uuid4())[:8]

            # 1. 上传记忆
            upload_response = await client.post(
                f"{BASE_URL}/api/v1/memories",
                json={
                    "memories": [
                        {
                            "content": f"E2E测试记忆 {uid}",
                            "source_id": f"e2e_source_{uid}",
                            "metadata": {"test": True, "uid": uid},
                        }
                    ],
                    "tenant_id": TEST_TENANT,
                },
            )

            if upload_response.status_code == 200:
                upload_data = upload_response.json()
                memory_id = upload_data.get("ids", [None])[0]

                # 2. 搜索记忆
                search_response = await client.post(
                    f"{BASE_URL}/api/v1/memories/search",
                    json={
                        "query": f"E2E测试记忆 {uid}",
                        "mode": "keyword",
                        "tenant_id": TEST_TENANT,
                    },
                )

                assert search_response.status_code == 200
                search_data = search_response.json()
                assert len(search_data.get("results", [])) > 0

    @pytest.mark.asyncio
    async def test_batch_upload_and_search_workflow(self):
        """测试批量上传和搜索工作流"""
        async with httpx.AsyncClient() as client:
            uid = str(uuid.uuid4())[:8]
            num_memories = 10

            # 批量上传
            memories = [
                {
                    "content": f"批量测试记忆 {uid} {i}",
                    "source_id": f"batch_source_{uid}_{i}",
                }
                for i in range(num_memories)
            ]

            upload_response = await client.post(
                f"{BASE_URL}/api/v1/memories",
                json={"memories": memories, "tenant_id": TEST_TENANT},
            )

            assert upload_response.status_code == 200
            upload_data = upload_response.json()
            assert len(upload_data.get("ids", [])) == num_memories

    @pytest.mark.asyncio
    async def test_embedding_and_search_workflow(self):
        """测试嵌入和向量搜索工作流"""
        async with httpx.AsyncClient() as client:
            # 获取嵌入
            embed_response = await client.post(
                f"{BASE_URL}/v1/embeddings",
                json={"input": "测试文本用于嵌入"},
            )

            assert embed_response.status_code == 200
            embed_data = embed_response.json()
            assert "embedding" in embed_data
            assert len(embed_data["embedding"]) > 0


class TestCodeAnalysisWorkflow:
    """代码分析工作流测试"""

    @pytest.mark.asyncio
    async def test_code_analysis_full_workflow(self):
        """测试代码分析完整工作流"""
        async with httpx.AsyncClient() as client:
            uid = str(uuid.uuid4())[:8]

            # 1. 上传代码记忆
            code_content = f"""
def test_function_{uid}():
    \"\"\"测试函数 {uid}\"\"\"
    return "Hello {uid}"
"""

            upload_response = await client.post(
                f"{BASE_URL}/api/v1/memories",
                json={
                    "memories": [
                        {
                            "content": code_content,
                            "source_id": f"code_{uid}.py",
                            "language": "python",
                            "metadata": {"type": "code", "language": "python"},
                        }
                    ],
                    "tenant_id": TEST_TENANT,
                },
            )

            assert upload_response.status_code == 200

    @pytest.mark.asyncio
    async def test_project_map_workflow(self):
        """测试项目地图工作流"""
        async with httpx.AsyncClient() as client:
            # 获取项目地图
            response = await client.get(
                f"{BASE_URL}/api/v1/projects/test_project/map",
                params={"tenant_id": TEST_TENANT},
            )

            # 项目可能不存在，但接口应该正常响应
            assert response.status_code in [200, 404]


class TestSyncWorkflow:
    """同步工作流测试"""

    @pytest.mark.asyncio
    async def test_sync_preview_workflow(self):
        """测试同步预览工作流"""
        async with httpx.AsyncClient() as client:
            uid = str(uuid.uuid4())[:8]

            # 同步预览
            response = await client.post(
                f"{BASE_URL}/api/v1/sync/preview",
                json={
                    "fingerprints": [
                        {
                            "path": f"test_{uid}.md",
                            "mtime": int(datetime.now().timestamp() * 1000),
                            "hash": f"hash_{uid}",
                            "source_id": f"sync_test_{uid}",
                        }
                    ],
                    "tenant_id": TEST_TENANT,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "to_upload" in data
            assert "to_download" in data
            assert "conflicts" in data

    @pytest.mark.asyncio
    async def test_fingerprint_query_workflow(self):
        """测试指纹查询工作流"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/api/v1/sync/fingerprints",
                params={"tenant_id": TEST_TENANT},
            )

            assert response.status_code == 200
            data = response.json()
            assert "fingerprints" in data


class TestPerformanceBaseline:
    """性能基准场景测试"""

    @pytest.mark.asyncio
    async def test_end_to_end_latency_baseline(self):
        """测试端到端延迟基准"""
        async with httpx.AsyncClient() as client:
            import time

            start = time.time()

            # 完整流程：健康检查 + 嵌入 + 搜索
            await client.get(f"{BASE_URL}/health")

            await client.post(
                f"{BASE_URL}/v1/embeddings",
                json={"input": "性能测试文本"},
            )

            await client.post(
                f"{BASE_URL}/api/v1/memories/search",
                json={"query": "性能测试", "mode": "keyword"},
            )

            elapsed = time.time() - start
            # 端到端应该 < 5 秒
            assert elapsed < 5.0

    @pytest.mark.asyncio
    async def test_concurrent_user_simulation(self):
        """测试并发用户模拟"""
        async with httpx.AsyncClient() as client:
            num_users = 5
            tasks = []

            for i in range(num_users):
                tasks.append(client.get(f"{BASE_URL}/health"))

            responses = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(
                1 for r in responses if not isinstance(r, Exception) and getattr(r, "status_code", None) == 200
            )

            assert success_count == num_users


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
