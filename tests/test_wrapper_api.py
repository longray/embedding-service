"""
核心API端点完善测试套件

测试三个核心API端点的真实端到端功能：
1. POST /v1/embeddings - 文本嵌入（带缓存）
2. POST /api/v1/memories - 批量上传记忆
3. POST /api/v1/memories/search - 搜索记忆

前置条件：
- Embedding服务运行在 http://localhost:18000
- SurrealDB运行在 ws://localhost:18002
- Wrapper服务运行在 http://localhost:17999

运行方式：
    uv run pytest tests/test_wrapper_api.py -v
"""

import asyncio
import time
import uuid

import httpx
import pytest
import pytest_asyncio

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


# ============================================================================
# 测试类：多租户隔离
# ============================================================================


class TestMultiTenancy:
    """多租户功能测试"""

    @pytest.mark.asyncio
    async def test_upload_with_default_tenant(self, client):
        """测试默认租户上传（向后兼容：不传 tenant_id）"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories",
            json={"memories": [{"content": f"默认租户测试 {uuid.uuid4()}"}]},
        )
        assert response.status_code == 200
        assert response.json()["success"] == 1

    @pytest.mark.asyncio
    async def test_upload_with_custom_tenant(self, client):
        """测试自定义租户上传"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories",
            json={
                "memories": [{"content": f"自定义租户测试 {uuid.uuid4()}"}],
                "tenant_id": "test_tenant_A",
            },
        )
        assert response.status_code == 200
        assert response.json()["success"] == 1

    @pytest.mark.asyncio
    async def test_tenant_isolation(self, client):
        """测试租户数据隔离：A 的数据 B 搜不到"""
        uid = str(uuid.uuid4())[:8]

        # 上传到租户 A
        await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories",
            json={
                "memories": [{"content": f"租户A隔离数据 {uid}"}],
                "tenant_id": "isolation_test_A",
            },
        )

        await asyncio.sleep(0.5)

        # 用租户 B 搜索，应该找不到
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search",
            json={"query": uid, "mode": "keyword", "tenant_id": "isolation_test_B"},
        )
        assert response.status_code == 200
        assert response.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_search_within_same_tenant(self, client):
        """测试相同租户内可以搜到数据"""
        uid = str(uuid.uuid4())[:8]
        tenant = f"same_tenant_{uid}"

        await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories",
            json={
                "memories": [{"content": f"租户搜索验证 {uid}"}],
                "tenant_id": tenant,
            },
        )

        await asyncio.sleep(1.0)

        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search",
            json={"query": uid, "mode": "vector", "threshold": 0.3, "tenant_id": tenant},
        )
        assert response.status_code == 200
        data = response.json()
        found = any(uid in r.get("content", "") for r in data["results"])
        assert found, f"相同租户内应能搜到数据，结果: {data['results']}"


# ============================================================================
# 测试类：新字段映射
# ============================================================================


class TestNewFieldMapping:
    """新字段映射测试（type, tags, project_id, source_id）"""

    @pytest.mark.asyncio
    async def test_upload_with_new_fields(self, client):
        """测试上传包含新字段的记忆"""
        uid = str(uuid.uuid4())[:8]
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories",
            json={
                "memories": [
                    {
                        "content": f"新字段测试 {uid}",
                        "type": "decision",
                        "tags": ["test", "upgrade"],
                        "project_id": "embedding_service",
                        "source_id": f"mem_{uid}",
                        "metadata": {"extra": "data"},
                    }
                ]
            },
        )
        assert response.status_code == 200
        assert response.json()["success"] == 1

    @pytest.mark.asyncio
    async def test_search_results_include_new_fields(self, client):
        """测试搜索结果包含新字段（type, tags, project_id, score）"""
        uid = str(uuid.uuid4())[:8]

        await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories",
            json={
                "memories": [
                    {
                        "content": f"字段验证测试 {uid}",
                        "type": "analysis",
                        "tags": ["field", "verify"],
                        "project_id": "test_project",
                    }
                ]
            },
        )

        await asyncio.sleep(1.0)

        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search",
            json={"query": f"字段验证测试 {uid}", "mode": "vector", "threshold": 0.3},
        )
        assert response.status_code == 200
        data = response.json()

        if data["results"]:
            result = data["results"][0]
            assert "type" in result
            assert "tags" in result
            assert "project_id" in result
            assert "score" in result

    @pytest.mark.asyncio
    async def test_source_id_deduplication(self, client):
        """测试 source_id UNIQUE 索引去重"""
        uid = str(uuid.uuid4())[:8]
        source_id = f"dedup_test_{uid}"

        # 第一次上传
        r1 = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories",
            json={"memories": [{"content": f"去重测试 {uid}", "source_id": source_id}]},
        )
        assert r1.json()["success"] == 1

        # 第二次上传相同 source_id，应该因 UNIQUE 冲突失败
        r2 = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories",
            json={"memories": [{"content": f"去重测试重复 {uid}", "source_id": source_id}]},
        )
        assert r2.json()["failed"] == 1


# ============================================================================
# 测试类：RRF 混合搜索 + 向量搜索分数验证
# ============================================================================


class TestRRFHybridSearch:
    """RRF 混合搜索 + 向量搜索分数验证"""

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_rrf_score(self, client):
        """测试混合搜索结果包含 RRF 分数"""
        uid = str(uuid.uuid4())[:8]

        await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories",
            json={
                "memories": [
                    {"content": f"Python编程语言RRF测试 {uid}"},
                    {"content": f"JavaScript前端框架RRF测试 {uid}"},
                ]
            },
        )

        await asyncio.sleep(1.0)

        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search",
            json={"query": f"Python {uid}", "mode": "hybrid", "threshold": 0.1},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "hybrid"

        if data["results"]:
            for r in data["results"]:
                assert "score" in r
                assert isinstance(r["score"], (int, float))

    @pytest.mark.asyncio
    async def test_vector_search_similarity_score(self, client):
        """测试向量搜索返回相似度分数（distance→similarity 转换，范围 [0,1]）"""
        uid = str(uuid.uuid4())[:8]

        await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories",
            json={"memories": [{"content": f"向量分数测试 {uid}"}]},
        )

        await asyncio.sleep(1.0)

        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search",
            json={"query": f"向量分数测试 {uid}", "mode": "vector", "threshold": 0.1},
        )
        assert response.status_code == 200
        data = response.json()

        if data["results"]:
            score = data["results"][0]["score"]
            # 相似度分数应在 [0, 1] 范围内
            assert 0.0 <= score <= 1.0, f"分数超出范围: {score}"

    @pytest.mark.asyncio
    async def test_keyword_search_with_bm25(self, client):
        """测试 BM25 关键词搜索（@1@ 操作符替代 CONTAINS）"""
        uid = str(uuid.uuid4())[:8]

        await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories",
            json={"memories": [{"content": f"BM25关键词搜索测试 {uid}"}]},
        )

        await asyncio.sleep(1.0)

        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search",
            json={"query": f"BM25 {uid}", "mode": "keyword"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "keyword"


# ============================================================================
# 测试类：中文分词搜索质量验证（Phase 2 - Task D）
# ============================================================================


class TestChineseSearch:
    """验证 ngram(2,8) 分析器对中文/英文 BM25 关键词搜索的支持

    测试场景覆盖：
    - 中文 2 字词搜索（如"数据"）
    - 中文 4 字词搜索（如"向量搜索"）
    - 英文短词搜索（如"Python"）
    - 中英混合搜索
    - 搜索结果相关性验证
    """

    @pytest_asyncio.fixture(autouse=True)
    async def setup_chinese_data(self, client):
        """上传中文测试数据"""
        self.uid = str(uuid.uuid4())[:8]
        self.tenant = f"cn_test_{self.uid}"
        self.memories = [
            {
                "content": "SurrealDB是一个多模型数据库支持向量搜索和图查询功能",
                "metadata": {"lang": "zh"},
            },
            {
                "content": "Python的FastAPI框架可以快速构建高性能推理服务",
                "metadata": {"lang": "zh"},
            },
            {
                "content": "深度学习模型需要大量训练数据和GPU计算资源",
                "metadata": {"lang": "zh"},
            },
            {
                "content": "知识图谱通过实体关系构建语义网络实现智能问答",
                "metadata": {"lang": "zh"},
            },
            {
                "content": "Embedding嵌入模型将文本映射到高维向量空间",
                "metadata": {"lang": "zh"},
            },
        ]
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories",
            json={"memories": self.memories, "tenant_id": self.tenant},
        )
        assert response.status_code == 200
        data = response.json()
        self.memory_ids = data.get("memory_ids", [])
        # 等待 BM25 索引更新
        await asyncio.sleep(1.5)

    @pytest.mark.asyncio
    async def test_chinese_2char_keyword(self, client):
        """测试中文 2 字词搜索（'数据' 出现在训练数据、数据库）"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search",
            json={"query": "数据", "mode": "keyword", "threshold": 0.0, "tenant_id": self.tenant},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "keyword"
        # ngram(2,8) 应能匹配 "数据" 二字词
        assert len(data["results"]) >= 1, f"中文2字词搜索应返回结果, got {data}"

    @pytest.mark.asyncio
    async def test_chinese_4char_keyword(self, client):
        """测试中文 4 字词搜索（'向量搜索'）"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search",
            json={"query": "向量搜索", "mode": "keyword", "threshold": 0.0, "tenant_id": self.tenant},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) >= 1, f"中文4字词搜索应返回结果, got {data}"

    @pytest.mark.asyncio
    async def test_english_keyword_within_ngram_range(self, client):
        """测试英文短词搜索（'Python' = 6字符，在ngram(2,8)范围内）"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search",
            json={"query": "Python", "mode": "keyword", "threshold": 0.0, "tenant_id": self.tenant},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) >= 1, f"英文短词搜索应返回结果, got {data}"

    @pytest.mark.asyncio
    async def test_chinese_english_mixed_search(self, client):
        """测试中英混合搜索（'深度学习' + 'GPU'）"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search",
            json={"query": "深度学习", "mode": "keyword", "threshold": 0.0, "tenant_id": self.tenant},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) >= 1, f"中英混合内容搜索应返回结果, got {data}"

    @pytest.mark.asyncio
    async def test_keyword_search_relevance(self, client):
        """测试搜索结果相关性：查"知识图谱"应返回含该词的记忆"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search",
            json={"query": "知识图谱", "mode": "keyword", "threshold": 0.0, "tenant_id": self.tenant},
        )
        assert response.status_code == 200
        data = response.json()
        results = data["results"]
        assert len(results) >= 1, f"应至少返回含'知识图谱'的记忆, got {data}"
        # 验证最相关的结果包含搜索词
        found = any("知识图谱" in r.get("content", "") for r in results)
        assert found, f"搜索结果中应包含'知识图谱', results={[r.get('content', '')[:30] for r in results]}"

    @pytest.mark.asyncio
    async def test_keyword_no_false_positive(self, client):
        """测试不存在的词不会崩溃"""
        fake_query = "量子纠缠超导"
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/search",
            json={"query": fake_query, "mode": "keyword", "threshold": 0.0, "tenant_id": self.tenant},
        )
        assert response.status_code == 200
        data = response.json()
        # 不存在的词应返回空结果或极少结果
        assert isinstance(data["results"], list)


# ============================================================================
# 测试类：RELATE 图关系 API（Phase 2 - Task E）
# ============================================================================


class TestRelationsAPI:
    """测试 SurrealDB RELATE 图关系的 CRUD 操作和图遍历

    覆盖场景：
    - 创建关系（POST /api/v1/memories/relations）
    - 查询关系（POST /api/v1/memories/{id}/relations）
    - 删除关系（DELETE /api/v1/memories/relations/{id}）
    - 图遍历（POST /api/v1/memories/{id}/graph）
    - 参数验证和错误处理
    - 租户隔离
    """

    @pytest_asyncio.fixture(autouse=True)
    async def setup_relation_data(self, client):
        """上传测试记忆数据，获取记忆 ID 供关系测试使用"""
        self.uid = str(uuid.uuid4())[:8]
        self.tenant = f"rel_test_{self.uid}"
        memories = [
            {
                "content": f"关系测试-源节点A-机器学习基础概念 [{self.uid}]",
                "metadata": {"test_id": self.uid, "node": "A"},
            },
            {
                "content": f"关系测试-目标节点B-深度学习进阶 [{self.uid}]",
                "metadata": {"test_id": self.uid, "node": "B"},
            },
            {
                "content": f"关系测试-节点C-自然语言处理 [{self.uid}]",
                "metadata": {"test_id": self.uid, "node": "C"},
            },
        ]
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories",
            json={"memories": memories, "tenant_id": self.tenant},
        )
        assert response.status_code == 200
        data = response.json()
        self.ids = data.get("memory_ids", [])
        assert len(self.ids) >= 3, f"应成功上传3条记忆, got {data}"
        # 等待数据持久化
        await asyncio.sleep(1.0)

    @pytest.mark.asyncio
    async def test_create_relation(self, client):
        """测试创建关系：A→B (follow_up)"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/relations",
            json={
                "from_id": self.ids[0],
                "to_id": self.ids[1],
                "relationship_type": "follow_up",
                "weight": 0.8,
                "tenant_id": self.tenant,
                "description": "从基础到进阶",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data, f"创建关系应返回 id, got {data}"
        assert data["relationship_type"] == "follow_up"
        assert data["weight"] == 0.8

    @pytest.mark.asyncio
    async def test_create_relation_default_values(self, client):
        """测试创建关系使用默认值（type=related, weight=0.5）"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/relations",
            json={
                "from_id": self.ids[0],
                "to_id": self.ids[2],
                "tenant_id": self.tenant,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["relationship_type"] == "related"
        assert data["weight"] == 0.5

    @pytest.mark.asyncio
    async def test_create_relation_invalid_type(self, client):
        """测试无效关系类型应返回 400"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/relations",
            json={
                "from_id": self.ids[0],
                "to_id": self.ids[1],
                "relationship_type": "invalid_type",
                "tenant_id": self.tenant,
            },
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_create_relation_invalid_weight(self, client):
        """测试无效权重应返回 422（Pydantic 验证）"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/relations",
            json={
                "from_id": self.ids[0],
                "to_id": self.ids[1],
                "weight": 1.5,
                "tenant_id": self.tenant,
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_relations_outgoing(self, client):
        """测试查询出向关系"""
        # 先创建关系 A→B
        await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/relations",
            json={
                "from_id": self.ids[0],
                "to_id": self.ids[1],
                "relationship_type": "follow_up",
                "tenant_id": self.tenant,
            },
        )

        # 查询 A 的出向关系
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/{self.ids[0]}/relations",
            json={"direction": "outgoing", "tenant_id": self.tenant},
        )
        assert response.status_code == 200
        data = response.json()
        assert "relations" in data
        assert data["total"] >= 1
        for rel in data["relations"]:
            assert rel["direction"] == "outgoing"

    @pytest.mark.asyncio
    async def test_query_relations_incoming(self, client):
        """测试查询入向关系"""
        # 创建关系 A→B
        await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/relations",
            json={
                "from_id": self.ids[0],
                "to_id": self.ids[1],
                "relationship_type": "elaboration",
                "tenant_id": self.tenant,
            },
        )

        # 查询 B 的入向关系
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/{self.ids[1]}/relations",
            json={"direction": "incoming", "tenant_id": self.tenant},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        for rel in data["relations"]:
            assert rel["direction"] == "incoming"

    @pytest.mark.asyncio
    async def test_query_relations_both_directions(self, client):
        """测试查询双向关系"""
        # 创建 A→B 和 C→A 两条关系
        await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/relations",
            json={
                "from_id": self.ids[0],
                "to_id": self.ids[1],
                "relationship_type": "related",
                "tenant_id": self.tenant,
            },
        )
        await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/relations",
            json={
                "from_id": self.ids[2],
                "to_id": self.ids[0],
                "relationship_type": "reference",
                "tenant_id": self.tenant,
            },
        )

        # 查询 A 的双向关系
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/{self.ids[0]}/relations",
            json={"direction": "both", "tenant_id": self.tenant},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 2, f"A 应有至少2条关系(出+入), got {data}"

    @pytest.mark.asyncio
    async def test_query_relations_filter_by_type(self, client):
        """测试按关系类型过滤"""
        # 创建不同类型关系
        await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/relations",
            json={
                "from_id": self.ids[0],
                "to_id": self.ids[1],
                "relationship_type": "follow_up",
                "tenant_id": self.tenant,
            },
        )
        await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/relations",
            json={
                "from_id": self.ids[0],
                "to_id": self.ids[2],
                "relationship_type": "reference",
                "tenant_id": self.tenant,
            },
        )

        # 只查询 follow_up 类型
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/{self.ids[0]}/relations",
            json={
                "direction": "outgoing",
                "relationship_type": "follow_up",
                "tenant_id": self.tenant,
            },
        )
        assert response.status_code == 200
        data = response.json()
        for rel in data["relations"]:
            assert rel["relationship_type"] == "follow_up"

    @pytest.mark.asyncio
    async def test_delete_relation(self, client):
        """测试删除关系"""
        # 创建关系
        create_resp = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/relations",
            json={
                "from_id": self.ids[0],
                "to_id": self.ids[1],
                "relationship_type": "related",
                "tenant_id": self.tenant,
            },
        )
        assert create_resp.status_code == 200
        relation_id = create_resp.json()["id"]

        # 删除关系
        delete_resp = await client.delete(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/relations/{relation_id}",
            params={"tenant_id": self.tenant},
        )
        assert delete_resp.status_code == 200
        assert delete_resp.json()["deleted"] is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent_relation(self, client):
        """测试删除不存在的关系应返回 404"""
        fake_id = f"memory_relation:nonexist_{self.uid}"
        response = await client.delete(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/relations/{fake_id}",
            params={"tenant_id": self.tenant},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_graph_traversal_depth_1(self, client):
        """测试图遍历：深度1（A→B）"""
        # 创建关系 A→B
        await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/relations",
            json={
                "from_id": self.ids[0],
                "to_id": self.ids[1],
                "relationship_type": "follow_up",
                "tenant_id": self.tenant,
            },
        )

        # 从 A 出发遍历深度1
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/{self.ids[0]}/graph",
            json={"depth": 1, "tenant_id": self.tenant},
        )
        assert response.status_code == 200
        data = response.json()
        assert "memories" in data
        assert data["depth"] == 1
        assert data["source"] == self.ids[0]
        assert data["total"] >= 1, f"深度1遍历应至少找到节点B, got {data}"

    @pytest.mark.asyncio
    async def test_graph_traversal_depth_2(self, client):
        """测试图遍历：深度2（A→B→C 链式路径）"""
        # 创建 A→B, B→C 链
        await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/relations",
            json={
                "from_id": self.ids[0],
                "to_id": self.ids[1],
                "relationship_type": "follow_up",
                "tenant_id": self.tenant,
            },
        )
        await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/relations",
            json={
                "from_id": self.ids[1],
                "to_id": self.ids[2],
                "relationship_type": "elaboration",
                "tenant_id": self.tenant,
            },
        )

        # 从 A 出发深度2
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/{self.ids[0]}/graph",
            json={"depth": 2, "tenant_id": self.tenant},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["depth"] == 2
        assert data["total"] >= 1, f"深度2遍历应找到节点, got {data}"

    @pytest.mark.asyncio
    async def test_graph_traversal_no_relations(self, client):
        """测试对没有关系的节点进行图遍历"""
        response = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/{self.ids[2]}/graph",
            json={"depth": 1, "tenant_id": self.tenant},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["memories"] == []

    @pytest.mark.asyncio
    async def test_relation_all_types(self, client):
        """测试所有6种合法关系类型"""
        valid_types = ["related", "follow_up", "elaboration", "contradiction", "reference", "derived_from"]
        for rel_type in valid_types:
            response = await client.post(
                f"{WRAPER_MINIMAL_URL}/api/v1/memories/relations",
                json={
                    "from_id": self.ids[0],
                    "to_id": self.ids[1],
                    "relationship_type": rel_type,
                    "tenant_id": self.tenant,
                },
            )
            assert response.status_code == 200, (
                f"关系类型 '{rel_type}' 应成功创建, got {response.status_code}"
            )

    @pytest.mark.asyncio
    async def test_relation_tenant_isolation(self, client):
        """测试关系的租户隔离"""
        other_tenant = f"other_{self.uid}"

        # 在 self.tenant 下创建关系
        create_resp = await client.post(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/relations",
            json={
                "from_id": self.ids[0],
                "to_id": self.ids[1],
                "relationship_type": "related",
                "tenant_id": self.tenant,
            },
        )
        assert create_resp.status_code == 200
        relation_id = create_resp.json()["id"]

        # 用其他租户 ID 尝试删除 → 应失败 (404)
        delete_resp = await client.delete(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/relations/{relation_id}",
            params={"tenant_id": other_tenant},
        )
        assert delete_resp.status_code == 404, "不同租户不应能删除其他租户的关系"

        # 用正确的租户 ID 应能删除
        delete_resp2 = await client.delete(
            f"{WRAPER_MINIMAL_URL}/api/v1/memories/relations/{relation_id}",
            params={"tenant_id": self.tenant},
        )
        assert delete_resp2.status_code == 200
