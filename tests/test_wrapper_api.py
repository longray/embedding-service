"""
核心API端点完善测试套件

测试三个核心API端点的真实端到端功能：
1. POST /v1/embeddings - 文本嵌入（带缓存）
2. POST /api/v1/memories - 批量上传记忆
3. POST /api/v1/memories/search - 搜索记忆

前置条件：
- Embedding服务运行在 http://localhost:18000
- SurrealDB运行在 ws://localhost:8000
- Wrapper服务运行在 http://localhost:17999

运行方式：
    uv run pytest tests/test_wrapper_api.py -v
"""

import pytest
import pytest_asyncio
import httpx
import asyncio
import time
import uuid
from typing import Any

WRAPER_MINIMAL_URL = "http://localhost:17999"
EMBEDDING_SERVICE_URL = "http://localhost:18000"
DEFAULT_TIMEOUT = 60.0


@pytest_asyncio.fixture
async def client():
    """HTTP客户端fixture"""
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        yield c


@pytest_asyncio.fixture
async def unique_memories():
    """生成唯一的测试记忆数据（避免重复）"""
    uid = str(uuid.uuid4())[:8]
    return [
        {
            "content": f"Python是一门流行的编程语言，广泛用于Web开发和数据科学。[{uid}]",
            "metadata": {"category": "programming", "test_id": uid},
        },
        {
            "content": f"FastAPI是一个现代、高性能的Python Web框架。[{uid}]",
            "metadata": {"category": "web", "test_id": uid},
        },
        {
            "content": f"SurrealDB是一个支持向量搜索的多模型数据库。[{uid}]",
            "metadata": {"category": "database", "test_id": uid},
        },
    ]


# ============================================================================
# 测试类：健康检查端点
# ============================================================================


class TestHealthEndpoint:
    """GET /health 端点测试"""

    @pytest.mark.asyncio
    async def test_health_check_returns_200(self, client):
        """测试健康检查返回200"""
        response = await client.get(f"{WRAPER_MINIMAL_URL}/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_check_response_structure(self, client):
        """测试健康检查响应结构"""
        response = await client.get(f"{WRAPER_MINIMAL_URL}/health")
        data = response.json()

        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert "port" in data
        assert data["service"] == "minimal-wrapper"

    @pytest.mark.asyncio
    async def test_health_check_includes_embedding_service_status(self, client):
        """测试健康检查包含Embedding服务状态"""
        response = await client.get(f"{WRAPER_MINIMAL_URL}/health")
        data = response.json()

        assert "embedding_service" in data
        embedding_status = data["embedding_service"]
        assert "status" in embedding_status

    @pytest.mark.asyncio
    async def test_health_check_includes_surrealdb_status(self, client):
        """测试健康检查包含SurrealDB状态"""
        response = await client.get(f"{WRAPER_MINIMAL_URL}/health")
        data = response.json()

        assert "surrealdb" in data
        db_status = data["surrealdb"]
        assert "status" in db_status

    @pytest.mark.asyncio
    async def test_health_check_includes_cache_stats(self, client):
        """测试健康检查包含缓存统计"""
        response = await client.get(f"{WRAPER_MINIMAL_URL}/health")
        data = response.json()

        assert "cache_stats" in data
        stats = data["cache_stats"]
        assert "max_size" in stats
        assert "current_size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats


# ============================================================================
# 测试类：Embeddings端点基础功能
# ============================================================================


class TestEmbeddingsBasic:
    """POST /v1/embeddings 基础功能测试"""

    @pytest.mark.asyncio
    async def test_single_text_embedding(self, client):
        """测试单个文本嵌入"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"input": "这是一个测试文本", "model": "Qwen3-Embedding-0.6B"}
        )
        assert response.status_code == 200

        data = response.json()
        assert "data" in data
        assert len(data["data"]) == 1

        embedding = data["data"][0]
        assert "embedding" in embedding
        assert "index" in embedding
        assert len(embedding["embedding"]) == 1024

    @pytest.mark.asyncio
    async def test_embedding_with_default_model(self, client):
        """测试使用默认模型的嵌入"""
        response = await client.post(f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"input": "测试默认模型"})
        assert response.status_code == 200

        data = response.json()
        assert len(data["data"]) == 1
        assert len(data["data"][0]["embedding"]) == 1024

    @pytest.mark.asyncio
    async def test_embedding_response_has_usage(self, client):
        """测试嵌入响应包含usage信息"""
        response = await client.post(f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"input": "测试usage字段"})
        assert response.status_code == 200

        data = response.json()
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]


# ============================================================================
# 测试类：Embeddings缓存功能
# ============================================================================


class TestEmbeddingsCache:
    """POST /v1/embeddings 缓存功能测试"""

    @pytest.mark.asyncio
    async def test_cache_miss_on_first_request(self, client):
        """测试首次请求缓存未命中"""
        unique_text = f"唯一测试文本 {uuid.uuid4()}"

        response = await client.post(f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"input": unique_text})
        assert response.status_code == 200

        health = await client.get(f"{WRAPER_MINIMAL_URL}/health")
        stats = health.json().get("cache_stats", {})
        assert stats.get("misses", 0) >= 1

    @pytest.mark.asyncio
    async def test_cache_hit_on_second_request(self, client):
        """测试第二次请求缓存命中"""
        unique_text = f"缓存命中测试 {uuid.uuid4()}"

        await client.post(f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"input": unique_text})

        response = await client.post(f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"input": unique_text})
        assert response.status_code == 200

        health = await client.get(f"{WRAPER_MINIMAL_URL}/health")
        stats = health.json().get("cache_stats", {})
        assert stats.get("hits", 0) >= 1

    @pytest.mark.asyncio
    async def test_cached_result_matches_original(self, client):
        """测试缓存结果与原始结果一致"""
        unique_text = f"一致性测试 {uuid.uuid4()}"

        response1 = await client.post(f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"input": unique_text})
        embedding1 = response1.json()["data"][0]["embedding"]

        response2 = await client.post(f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"input": unique_text})
        embedding2 = response2.json()["data"][0]["embedding"]

        assert embedding1 == embedding2

    @pytest.mark.asyncio
    async def test_cache_hit_is_faster(self, client):
        """测试缓存命中响应更快"""
        unique_text = f"性能测试 {uuid.uuid4()}"

        start = time.time()
        await client.post(f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"input": unique_text})
        first_duration = time.time() - start

        start = time.time()
        await client.post(f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"input": unique_text})
        second_duration = time.time() - start

        assert second_duration < first_duration

    @pytest.mark.asyncio
    async def test_different_inputs_different_cache_keys(self, client):
        """测试不同输入使用不同缓存键"""
        text1 = f"文本一 {uuid.uuid4()}"
        text2 = f"文本二 {uuid.uuid4()}"

        r1 = await client.post(f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"input": text1})
        e1 = r1.json()["data"][0]["embedding"]

        r2 = await client.post(f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"input": text2})
        e2 = r2.json()["data"][0]["embedding"]

        assert e1 != e2


# ============================================================================
# 测试类：Embeddings边界条件
# ============================================================================


class TestEmbeddingsBoundary:
    """POST /v1/embeddings 边界条件测试"""

    @pytest.mark.asyncio
    async def test_empty_string_input(self, client):
        """测试空字符串输入"""
        response = await client.post(f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"input": ""})
        assert response.status_code in [200, 400, 422]

    @pytest.mark.asyncio
    async def test_whitespace_only_input(self, client):
        """测试仅空白字符输入"""
        response = await client.post(f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"input": "   \t\n   "})
        assert response.status_code in [200, 400, 422]

    @pytest.mark.asyncio
    async def test_special_characters_input(self, client):
        """测试特殊字符输入"""
        special_texts = [
            "Hello 世界！🌍🎉",
            "测试\n换行\t制表符",
            "特殊字符: <>&\"'",
            "数学符号: α β γ δ ∑ ∫",
        ]

        for text in special_texts:
            response = await client.post(f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"input": text})
            assert response.status_code == 200
            assert len(response.json()["data"][0]["embedding"]) == 1024

    @pytest.mark.asyncio
    async def test_long_text_input(self, client):
        """测试超长文本输入"""
        long_text = "这是一段很长的文本。" * 500

        response = await client.post(f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"input": long_text})
        assert response.status_code == 200
        assert len(response.json()["data"][0]["embedding"]) == 1024


# ============================================================================
# 测试类：Embeddings错误处理
# ============================================================================


class TestEmbeddingsErrors:
    """POST /v1/embeddings 错误处理测试"""

    @pytest.mark.asyncio
    async def test_missing_input_field(self, client):
        """测试缺失input字段"""
        response = await client.post(f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"model": "Qwen3-Embedding-0.6B"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_input_type(self, client):
        """测试无效的input类型"""
        response = await client.post(f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"input": 12345})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_null_input(self, client):
        """测试null输入"""
        response = await client.post(f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"input": None})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_malformed_json(self, client):
        """测试格式错误的JSON"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/v1/embeddings",
            content="{invalid json}",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


# ============================================================================
# 测试类：Memories上传基础功能
# ============================================================================


class TestMemoriesUploadBasic:
    """POST /api/v1/memories 基础功能测试"""

    @pytest.mark.asyncio
    async def test_upload_single_memory(self, client):
        """测试上传单个记忆"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories",
            json={"memories": [{"content": f"测试记忆 {uuid.uuid4()}", "metadata": {"test": True}}]},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 1
        assert data["success"] == 1
        assert data["failed"] == 0
        assert len(data["memory_ids"]) == 1

    @pytest.mark.asyncio
    async def test_upload_multiple_memories(self, client, unique_memories):
        """测试批量上传记忆"""
        response = await client.post(f"{WRAPER_MINIMAL_URL}/api/v1/memories", json={"memories": unique_memories})
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == len(unique_memories)
        assert data["success"] == len(unique_memories)
        assert data["failed"] == 0
        assert len(data["memory_ids"]) == len(unique_memories)

    @pytest.mark.asyncio
    async def test_upload_memories_returns_valid_ids(self, client, unique_memories):
        """测试上传返回有效的记忆ID"""
        response = await client.post(f"{WRAPER_MINIMAL_URL}/api/v1/memories", json={"memories": unique_memories})
        assert response.status_code == 200

        data = response.json()
        for mid in data["memory_ids"]:
            assert mid.startswith("memory:")

    @pytest.mark.asyncio
    async def test_upload_memory_with_metadata(self, client):
        """测试上传带元数据的记忆"""
        metadata = {
            "source": "test",
            "category": "programming",
            "priority": 1,
            "tags": ["python", "testing"],
        }

        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories",
            json={"memories": [{"content": f"带元数据的记忆 {uuid.uuid4()}", "metadata": metadata}]},
        )
        assert response.status_code == 200
        assert response.json()["success"] == 1


# ============================================================================
# 测试类：Memories上传边界条件
# ============================================================================


class TestMemoriesUploadBoundary:
    """POST /api/v1/memories 边界条件测试"""

    @pytest.mark.asyncio
    async def test_empty_memories_list(self, client):
        """测试空记忆列表"""
        response = await client.post(f"{WRAPER_MINIMAL_URL}/api/v1/memories", json={"memories": []})
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_memory_without_content(self, client):
        """测试没有content的记忆"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories", json={"memories": [{"metadata": {"test": True}}]}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == 1

    @pytest.mark.asyncio
    async def test_memory_with_empty_content(self, client):
        """测试空内容的记忆"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories", json={"memories": [{"content": "", "metadata": {}}]}
        )
        assert response.status_code in [200, 400]

    @pytest.mark.asyncio
    async def test_large_batch_upload(self, client):
        """测试大批量上传"""
        batch_size = 20
        memories = [
            {"content": f"批量测试记忆 {i} [{uuid.uuid4()}]", "metadata": {"batch": True}} for i in range(batch_size)
        ]

        response = await client.post(f"{WRAPER_MINIMAL_URL}/api/v1/memories", json={"memories": memories})
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == batch_size
        assert data["success"] == batch_size

    @pytest.mark.asyncio
    async def test_memory_with_special_characters(self, client):
        """测试特殊字符内容"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories",
            json={"memories": [{"content": "特殊字符测试：🎉🌍中文<>&\"'", "metadata": {}}]},
        )
        assert response.status_code == 200
        assert response.json()["success"] == 1


# ============================================================================
# 测试类：Memories上传错误处理
# ============================================================================


class TestMemoriesUploadErrors:
    """POST /api/v1/memories 错误处理测试"""

    @pytest.mark.asyncio
    async def test_missing_memories_field(self, client):
        """测试缺失memories字段"""
        response = await client.post(f"{WRAPER_MINIMAL_URL}/api/v1/memories", json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_memories_type(self, client):
        """测试无效的memories类型"""
        response = await client.post(f"{WRAPER_MINIMAL_URL}/api/v1/memories", json={"memories": "not a list"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_null_memories(self, client):
        """测试null memories"""
        response = await client.post(f"{WRAPER_MINIMAL_URL}/api/v1/memories", json={"memories": None})
        assert response.status_code == 422


# ============================================================================
# 测试类：Memories搜索基础功能
# ============================================================================


class TestMemoriesSearchBasic:
    """POST /api/v1/memories/search 基础功能测试"""

    @pytest_asyncio.fixture
    async def setup_search_data(self, client):
        """准备搜索测试数据"""
        uid = str(uuid.uuid4())[:8]
        memories = [
            {"content": f"Python是一门流行的编程语言[{uid}]", "metadata": {"type": "programming"}},
            {"content": f"JavaScript是Web开发的核心语言[{uid}]", "metadata": {"type": "programming"}},
            {"content": f"今天天气很好，适合外出[{uid}]", "metadata": {"type": "life"}},
        ]

        await client.post(f"{WRAPER_MINIMAL_URL}/api/v1/memories", json={"memories": memories})
        await asyncio.sleep(0.5)
        return uid

    @pytest.mark.asyncio
    async def test_keyword_search(self, client, setup_search_data):
        """测试关键词搜索"""
        _ = setup_search_data

        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search", json={"query": "Python", "mode": "keyword", "limit": 10}
        )
        assert response.status_code == 200

        data = response.json()
        assert data["mode"] == "keyword"
        assert data["query"] == "Python"

    @pytest.mark.asyncio
    async def test_vector_search(self, client, setup_search_data):
        """测试向量搜索"""
        _ = setup_search_data

        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search",
            json={"query": "编程语言", "mode": "vector", "limit": 10, "threshold": 0.3},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["mode"] == "vector"

    @pytest.mark.asyncio
    async def test_hybrid_search(self, client, setup_search_data):
        """测试混合搜索"""
        _ = setup_search_data

        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search", json={"query": "Web开发", "mode": "hybrid", "limit": 10}
        )
        assert response.status_code == 200

        data = response.json()
        assert data["mode"] == "hybrid"

    @pytest.mark.asyncio
    async def test_search_returns_results_structure(self, client, setup_search_data):
        """测试搜索结果结构"""
        _ = setup_search_data

        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search", json={"query": "编程", "mode": "keyword", "limit": 5}
        )
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert "total" in data
        assert "mode" in data
        assert "query" in data

    @pytest.mark.asyncio
    async def test_search_result_has_required_fields(self, client, setup_search_data):
        """测试搜索结果包含必需字段"""
        _ = setup_search_data

        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search", json={"query": "Python", "mode": "keyword", "limit": 5}
        )
        data = response.json()

        if data["results"]:
            result = data["results"][0]
            assert "id" in result
            assert "content" in result
            assert "metadata" in result


# ============================================================================
# 测试类：Memories搜索参数测试
# ============================================================================


class TestMemoriesSearchParams:
    """POST /api/v1/memories/search 参数测试"""

    @pytest.mark.asyncio
    async def test_limit_parameter(self, client):
        """测试limit参数"""
        for limit in [1, 5, 10, 50]:
            response = await client.post(
                f"{WRAPER_MINIMAL_URL}/api/v1/memories/search",
                json={"query": "测试", "mode": "keyword", "limit": limit},
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) <= limit

    @pytest.mark.asyncio
    async def test_threshold_parameter(self, client):
        """测试threshold参数"""
        for threshold in [0.1, 0.5, 0.9]:
            response = await client.post(
                f"{WRAPER_MINIMAL_URL}/api/v1/memories/search",
                json={"query": "编程", "mode": "vector", "limit": 10, "threshold": threshold},
            )
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_default_search_mode(self, client):
        """测试默认搜索模式"""
        response = await client.post(f"{WRAPER_MINIMAL_URL}/api/v1/memories/search", json={"query": "测试查询"})
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "hybrid"

    @pytest.mark.asyncio
    async def test_default_limit(self, client):
        """测试默认limit"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search", json={"query": "测试", "mode": "keyword"}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 10


# ============================================================================
# 测试类：Memories搜索边界条件
# ============================================================================


class TestMemoriesSearchBoundary:
    """POST /api/v1/memories/search 边界条件测试"""

    @pytest.mark.asyncio
    async def test_empty_query(self, client):
        """测试空查询"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search", json={"query": "", "mode": "keyword"}
        )
        assert response.status_code in [200, 400, 422]

    @pytest.mark.asyncio
    async def test_query_with_special_characters(self, client):
        """测试特殊字符查询"""
        queries = [
            "查询🌍🎉",
            "测试\n换行",
            "特殊<>&\"'",
        ]

        for query in queries:
            response = await client.post(
                f"{WRAPER_MINIMAL_URL}/api/v1/memories/search", json={"query": query, "mode": "keyword"}
            )
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_limit_boundary_values(self, client):
        """测试limit边界值"""
        for limit in [1, 100]:
            response = await client.post(
                f"{WRAPER_MINIMAL_URL}/api/v1/memories/search",
                json={"query": "测试", "mode": "keyword", "limit": limit},
            )
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_threshold_boundary_values(self, client):
        """测试threshold边界值"""
        for threshold in [0.0, 1.0]:
            response = await client.post(
                f"{WRAPER_MINIMAL_URL}/api/v1/memories/search",
                json={"query": "测试", "mode": "vector", "threshold": threshold},
            )
            assert response.status_code == 200


# ============================================================================
# 测试类：Memories搜索错误处理
# ============================================================================


class TestMemoriesSearchErrors:
    """POST /api/v1/memories/search 错误处理测试"""

    @pytest.mark.asyncio
    async def test_missing_query_field(self, client):
        """测试缺失query字段"""
        response = await client.post(f"{WRAPER_MINIMAL_URL}/api/v1/memories/search", json={"mode": "keyword"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_mode(self, client):
        """测试无效的搜索模式"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search", json={"query": "测试", "mode": "invalid_mode"}
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_limit_out_of_range(self, client):
        """测试limit超出范围"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search", json={"query": "测试", "mode": "keyword", "limit": 101}
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_threshold_out_of_range(self, client):
        """测试threshold超出范围"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search", json={"query": "测试", "mode": "vector", "threshold": 1.5}
        )
        assert response.status_code == 422


# ============================================================================
# 测试类：端到端集成测试
# ============================================================================


class TestEndToEndIntegration:
    """端到端集成测试：上传→搜索验证"""

    @pytest.mark.asyncio
    async def test_upload_then_vector_search(self, client):
        """测试上传后向量搜索可以找到"""
        uid = str(uuid.uuid4())[:8]
        unique_content = f"端到端测试唯一内容 {uid}"

        upload_response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories",
            json={"memories": [{"content": unique_content, "metadata": {"test_id": uid}}]},
        )
        assert upload_response.status_code == 200
        assert upload_response.json()["success"] == 1

        await asyncio.sleep(1.0)

        search_response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search",
            json={"query": unique_content, "mode": "vector", "limit": 10, "threshold": 0.5},
        )
        assert search_response.status_code == 200

        data = search_response.json()
        found = any(uid in r.get("content", "") for r in data["results"])
        assert found, f"未找到刚上传的内容，搜索结果: {data['results']}"

    @pytest.mark.asyncio
    async def test_multiple_uploads_then_hybrid_search(self, client):
        """测试多次上传后混合搜索"""
        uid = str(uuid.uuid4())[:8]
        contents = [f"多次上传测试 {i} {uid}" for i in range(3)]

        for content in contents:
            await client.post(
                f"{WRAPER_MINIMAL_URL}/api/v1/memories",
                json={"memories": [{"content": content, "metadata": {"batch": True}}]},
            )

        await asyncio.sleep(1.0)

        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search",
            json={"query": uid, "mode": "hybrid", "limit": 10, "threshold": 0.3},
        )
        assert response.status_code == 200
        assert response.json()["total"] >= 1


# ============================================================================
# 测试类：性能测试
# ============================================================================


class TestPerformance:
    """性能测试"""

    @pytest.mark.asyncio
    async def test_embedding_response_time(self, client):
        """测试嵌入响应时间 < 5秒"""
        start = time.time()
        response = await client.post(f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"input": f"性能测试 {uuid.uuid4()}"})
        duration = time.time() - start

        assert response.status_code == 200
        assert duration < 5.0, f"响应时间过长: {duration:.2f}秒"

    @pytest.mark.asyncio
    async def test_search_response_time(self, client):
        """测试搜索响应时间 < 3秒"""
        start = time.time()
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search", json={"query": "性能测试", "mode": "hybrid"}
        )
        duration = time.time() - start

        assert response.status_code == 200
        assert duration < 3.0, f"搜索响应时间过长: {duration:.2f}秒"

    @pytest.mark.asyncio
    async def test_concurrent_embeddings(self, client):
        """测试并发嵌入请求"""
        texts = [f"并发测试 {i} {uuid.uuid4()}" for i in range(5)]

        start = time.time()
        tasks = [client.post(f"{WRAPER_MINIMAL_URL}/v1/embeddings", json={"input": text}) for text in texts]
        responses = await asyncio.gather(*tasks)
        duration = time.time() - start

        for r in responses:
            assert r.status_code == 200

        assert duration < 15.0, f"并发处理时间过长: {duration:.2f}秒"

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, client):
        """测试并发搜索请求"""
        queries = ["Python", "JavaScript", "Web", "编程", "测试"]

        start = time.time()
        tasks = [
            client.post(f"{WRAPER_MINIMAL_URL}/api/v1/memories/search", json={"query": q, "mode": "keyword"})
            for q in queries
        ]
        responses = await asyncio.gather(*tasks)
        duration = time.time() - start

        for r in responses:
            assert r.status_code == 200

        assert duration < 10.0, f"并发搜索时间过长: {duration:.2f}秒"
