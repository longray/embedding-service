# Embedding Service - 测试计划

**版本**: v2.2-v3.0  
**创建日期**: 2026-03-18  
**适用范围**: Phase A/B/C 全阶段

---

## 1. 测试策略

### 1.1 核心原则

1. **测试优先级**：
   - P0（关键）：数据完整性、向量搜索准确性、安全性
   - P1（重要）：性能、API可靠性、并发处理
   - P2（增强）：监控、日志、优化

2. **测试金字塔**：

   ```
   E2E测试 (10%)      ← 完整API流程
   集成测试 (30%)     ← 数据库交互
   单元测试 (60%)     ← 函数级别
   ```

3. **数据库测试策略**：
   - 使用独立测试数据库（memory_test）
   - 每个测试前清空数据
   - 使用真实SurrealDB和Meilisearch

### 1.2 测试环境

| 环境 | 用途 | 配置 |
|------|------|------|
| 本地开发 | 单元测试 | Python 3.11+, pytest |
| CI环境 | 自动化测试 | GitHub Actions, Docker |
| 集成环境 | 集成测试 | SurrealDB + Meilisearch |
| 性能测试 | 压力测试 | Locust, 模拟1000+并发 |

---

## 2. 测试工具和框架

### 2.1 核心依赖

```toml
[tool.uv.dev-dependencies]
pytest = "^8.0.0"
pytest-asyncio = "^0.23.0"
pytest-cov = "^4.1.0"
httpx = "^0.27.0"
faker = "^24.0.0"
```

### 2.2 测试辅助工具

```python
# tests/conftest.py
import pytest
import asyncio
from surrealdb import Surreal
from meilisearch import Client as MeiliClient

@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def test_db():
    """测试数据库连接"""
    # 开发/测试环境使用ws://，生产环境使用wss://
    db_url = os.getenv("SURREALDB_URL", "ws://localhost:18002")
    db = Surreal(db_url)
    await db.connect()
    await db.signin({"user": "root", "pass": "root"})
    await db.use("memory_ns", "memory_test")
    
    # 清空测试数据
    await db.query("DELETE FROM memory")
    await db.query("DELETE FROM memory_relation")
    
    yield db
    
    await db.close()

@pytest.fixture
async def test_meili():
    """测试Meilisearch客户端"""
    client = MeiliClient("http://localhost:17700", "test_key")
    index = client.index("memory_test")
    
    # 清空测试数据
    index.delete_all_documents()
    
    yield index
    
    index.delete()

@pytest.fixture
def mock_embedding():
    """Mock embedding向量"""
    return [0.1] * 1024
```

---

## 3. Phase A 测试计划

### 3.1 单元测试（60%）

#### 3.1.1 向量搜索优化测试

**测试文件**: `tests/unit/test_vector_search.py`

```python
import pytest
from memory_manager import MemoryManager

@pytest.mark.asyncio
async def test_vector_search_no_duplicate_calculation(test_db, mock_embedding):
    """P0: 消除重复向量计算"""
    manager = MemoryManager(test_db)
    
    # 插入测试数据
    await test_db.query(
        "CREATE memory SET content = $content, embedding = $embedding, tenant_id = $tenant",
        {"content": "测试内容", "embedding": mock_embedding, "tenant": "test"}
    )
    
    # 使用KNN搜索（不应重复计算）
    results = await manager._search_by_vector(
        embedding=mock_embedding,
        limit=10,
        threshold=0.7,
        tenant_id="test"
    )
    
    assert len(results) > 0
    assert "score" in results[0]
    assert results[0]["score"] >= 0.7

@pytest.mark.asyncio
async def test_hnsw_index_exists(test_db):
    """验证HNSW索引已创建"""
    result = await test_db.query("INFO FOR TABLE memory")
    indexes = result[0]["result"]["indexes"]
    
    assert "memory_embedding_hnsw" in indexes
    assert indexes["memory_embedding_hnsw"]["fields"] == ["embedding"]
```

#### 3.1.2 智能去重决策测试

**测试文件**: `tests/unit/test_deduplication.py`

```python
from datetime import datetime, timedelta
from deduplication import decide_duplicate_action

def test_discard_duplicate_within_5min():
    """相似度>0.98 且 时间差<5分钟 → DISCARD"""
    new_mem = {
        "content": "测试内容",
        "created_at": datetime.now(),
        "type": "general"
    }
    old_mem = {
        "content": "测试内容",
        "created_at": datetime.now() - timedelta(minutes=3),
        "type": "general"
    }
    
    action = decide_duplicate_action(new_mem, old_mem, similarity=0.99)
    assert action == "DISCARD"

def test_update_preference():
    """type=preference → UPDATE"""
    new_mem = {"type": "preference", "content": "新偏好", "created_at": datetime.now()}
    old_mem = {"type": "preference", "content": "旧偏好", "created_at": datetime.now() - timedelta(days=1)}
    
    action = decide_duplicate_action(new_mem, old_mem, similarity=0.90)
    assert action == "UPDATE"

def test_keep_both_decision():
    """type=decision → KEEP_BOTH"""
    new_mem = {"type": "decision", "content": "新决策", "created_at": datetime.now()}
    old_mem = {"type": "decision", "content": "旧决策", "created_at": datetime.now() - timedelta(days=7)}
    
    action = decide_duplicate_action(new_mem, old_mem, similarity=0.88)
    assert action == "KEEP_BOTH"

def test_update_longer_content():
    """新内容长度>旧内容1.5倍 → UPDATE"""
    new_mem = {"content": "a" * 150, "created_at": datetime.now(), "type": "general"}
    old_mem = {"content": "a" * 100, "created_at": datetime.now() - timedelta(days=1), "type": "general"}
    
    action = decide_duplicate_action(new_mem, old_mem, similarity=0.92)
    assert action == "UPDATE"
```

#### 3.1.3 批量操作测试

**测试文件**: `tests/unit/test_batch_operations.py`

```python
@pytest.mark.asyncio
async def test_batch_insert_with_transaction(test_db):
    """批量插入使用事务"""
    memories = [
        {"content": f"测试{i}", "embedding": [0.1] * 1024, "tenant_id": "test"}
        for i in range(100)
    ]
    
    # 使用事务批量插入
    await test_db.query("BEGIN TRANSACTION")
    for mem in memories:
        await test_db.query("CREATE memory SET $data", {"data": mem})
    await test_db.query("COMMIT TRANSACTION")
    
    # 验证
    result = await test_db.query("SELECT count() FROM memory WHERE tenant_id = 'test' GROUP ALL")
    assert result[0]["result"][0]["count"] == 100

@pytest.mark.asyncio
async def test_batch_embedding_generation(mock_embedding_service):
    """批量生成embedding（5-10x提升）"""
    texts = [f"文本{i}" for i in range(100)]
    
    import time
    start = time.time()
    embeddings = await mock_embedding_service.get_embeddings_batch(texts, batch_size=32)
    duration = time.time() - start
    
    assert len(embeddings) == 100
    assert duration < 2.0  # 应该在2秒内完成
```

#### 3.1.4 安全性测试

**测试文件**: `tests/unit/test_security.py`

```python
@pytest.mark.asyncio
async def test_tls_encryption_enabled():
    """P0: 验证TLS加密已启用（wss://）"""
    from surrealdb_client import get_connection_url
    
    url = get_connection_url()
    assert url.startswith("wss://"), f"应使用wss://，实际：{url}"

def test_tenant_id_validation():
    """租户ID验证"""
    from security import validate_tenant_id
    
    # 有效租户ID
    assert validate_tenant_id("tenant-123")
    assert validate_tenant_id("default")
    
    # 无效租户ID
    assert not validate_tenant_id("tenant'; DROP TABLE memory;--")
    assert not validate_tenant_id("")
    assert not validate_tenant_id(None)

def test_query_sanitization():
    """查询参数清理"""
    from security import sanitize_query
    
    # 移除SQL注入字符
    assert sanitize_query("test'; DROP TABLE") == "test DROP TABLE"
    
    # 保留中文和字母数字
    assert sanitize_query("测试TypeScript") == "测试TypeScript"
    
    # 限制长度
    long_text = "a" * 1000
    assert len(sanitize_query(long_text)) <= 500
```

---

### 3.2 集成测试（30%）

#### 3.2.1 完整API流程测试

**测试文件**: `tests/integration/test_api_flow.py`

```python
@pytest.mark.asyncio
async def test_upload_search_flow(test_db, test_meili):
    """上传 → 索引 → 搜索 → 验证"""
    manager = MemoryManager(test_db, test_meili)
    
    # 1. 上传记忆
    memories = [{
        "content": "用户喜欢使用TypeScript进行开发",
        "type": "preference",
        "tags": ["typescript", "preference"],
        "tenant_id": "test"
    }]
    
    result = await manager.upload_memories(memories)
    assert result["success_count"] == 1
    memory_id = result["success"][0]["id"]
    
    # 2. 向量搜索
    vector_results = await manager.search(
        query="TypeScript偏好",
        mode="vector",
        tenant_id="test"
    )
    assert len(vector_results) > 0
    assert memory_id in [r["id"] for r in vector_results]
    
    # 3. 关键词搜索
    keyword_results = await manager.search(
        query="TypeScript",
        mode="keyword",
        tenant_id="test"
    )
    assert len(keyword_results) > 0

@pytest.mark.asyncio
async def test_duplicate_handling(test_db):
    """去重处理：相似记忆 → 智能决策"""
    manager = MemoryManager(test_db)
    
    # 第一次上传
    mem1 = {
        "content": "用户偏好TypeScript",
        "type": "preference",
        "tenant_id": "test"
    }
    result1 = await manager.upload_memories([mem1])
    assert result1["success_count"] == 1
    
    # 第二次上传（相似内容）
    mem2 = {
        "content": "用户喜欢TypeScript",
        "type": "preference",
        "tenant_id": "test"
    }
    result2 = await manager.upload_memories([mem2])
    
    # 应该更新而非重复
    assert result2["duplicate_count"] == 0
    assert result2["updated_count"] == 1
```

#### 3.2.2 混合搜索测试

**测试文件**: `tests/integration/test_hybrid_search.py`

```python
@pytest.mark.asyncio
async def test_rrf_merge(test_db, test_meili):
    """RRF混合搜索：向量70% + 关键词30%"""
    manager = MemoryManager(test_db, test_meili)
    
    # 插入测试数据
    memories = [
        {"content": "Python是一门优秀的编程语言", "tenant_id": "test"},
        {"content": "TypeScript提供类型安全", "tenant_id": "test"},
        {"content": "JavaScript是Web开发的基础", "tenant_id": "test"}
    ]
    await manager.upload_memories(memories)
    
    # 混合搜索
    results = await manager.search(
        query="编程语言",
        mode="hybrid",
        tenant_id="test"
    )
    
    assert len(results) > 0
    # Python应该排在前面（语义+关键词都匹配）
    assert "Python" in results[0]["content"]
```

---

### 3.3 性能测试

**测试文件**: `tests/performance/test_performance.py`

```python
@pytest.mark.asyncio
async def test_vector_search_latency(test_db, benchmark_data):
    """向量搜索延迟：<50ms"""
    manager = MemoryManager(test_db)
    
    # 插入1000条测试数据
    await manager.upload_memories(benchmark_data)
    
    import time
    latencies = []
    
    for _ in range(100):
        start = time.time()
        await manager._search_by_vector(
            embedding=[0.1] * 1024,
            limit=10,
            threshold=0.7,
            tenant_id="test"
        )
        latencies.append((time.time() - start) * 1000)
    
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[94]
    
    assert avg_latency < 50, f"平均延迟{avg_latency}ms，超过50ms"
    assert p95_latency < 100, f"P95延迟{p95_latency}ms，超过100ms"

@pytest.mark.asyncio
async def test_batch_insert_throughput(test_db):
    """批量插入吞吐量：>100条/秒"""
    manager = MemoryManager(test_db)
    
    memories = [
        {"content": f"测试{i}", "embedding": [0.1] * 1024, "tenant_id": "test"}
        for i in range(1000)
    ]
    
    import time
    start = time.time()
    await manager.batch_insert(memories, batch_size=50)
    duration = time.time() - start
    
    throughput = 1000 / duration
    assert throughput > 100, f"吞吐量{throughput}条/秒，低于100条/秒"
```

---

## 4. Phase B 测试计划

### 4.1 变更检测API测试

**测试文件**: `tests/integration/test_sync_api.py`

```python
@pytest.mark.asyncio
async def test_detect_changes_api(test_client):
    """POST /sync/detect-changes：返回to_upload/to_delete列表"""
    local_fingerprints = [
        {"source_id": "entry-001", "content_hash": "abc123", "mtime": 1710000000},
        {"source_id": "entry-002", "content_hash": "def456", "mtime": 1710000100}
    ]
    
    response = await test_client.post(
        "/sync/detect-changes",
        json={"fingerprints": local_fingerprints, "tenant_id": "test"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "to_upload" in data
    assert "to_delete" in data
    assert "conflicts" in data

@pytest.mark.asyncio
async def test_batch_upload_api(test_client):
    """POST /sync/upload-batch：批量上传50条/批"""
    memories = [
        {"content": f"测试{i}", "type": "general", "tenant_id": "test"}
        for i in range(100)
    ]
    
    response = await test_client.post(
        "/sync/upload-batch",
        json={"memories": memories, "batch_size": 50}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success_count"] == 100
    assert data["batches"] == 2
```

### 4.2 冲突处理测试

```python
@pytest.mark.asyncio
async def test_conflict_resolution_timestamp(test_db):
    """时间戳裁决：新>旧 → 更新"""
    manager = MemoryManager(test_db)
    
    # 旧记忆
    old_mem = {
        "source_id": "conflict-test",
        "content": "旧内容",
        "created_at": datetime.now() - timedelta(days=1),
        "tenant_id": "test"
    }
    await manager.upload_memories([old_mem])
    
    # 新记忆（冲突）
    new_mem = {
        "source_id": "conflict-test",
        "content": "新内容",
        "created_at": datetime.now(),
        "tenant_id": "test"
    }
    result = await manager.upload_memories([new_mem])
    
    assert result["updated_count"] == 1
    
    # 验证内容已更新
    mem = await test_db.query(
        "SELECT * FROM memory WHERE source_id = 'conflict-test'"
    )
    assert mem[0]["result"][0]["content"] == "新内容"

@pytest.mark.asyncio
async def test_conflict_resolution_vector_similarity(test_db):
    """向量相似度>0.95 → 自动合并"""
    manager = MemoryManager(test_db)
    
    mem1 = {"content": "用户喜欢TypeScript", "tenant_id": "test"}
    mem2 = {"content": "用户偏好TypeScript", "tenant_id": "test"}
    
    await manager.upload_memories([mem1])
    result = await manager.upload_memories([mem2])
    
    # 应该合并而非创建新记忆
    assert result["merged_count"] == 1
```

---

## 5. Phase C 测试计划

### 5.1 查询性能分析测试

```python
@pytest.mark.asyncio
async def test_explain_query_plan(test_db):
    """使用EXPLAIN分析查询计划"""
    query = """
    EXPLAIN SELECT * FROM memory 
    WHERE tenant_id = 'test' 
    AND embedding <|10,COSINE|> $embedding
    """
    
    result = await test_db.query(query, {"embedding": [0.1] * 1024})
    plan = result[0]["result"]
    
    # 验证使用了HNSW索引
    assert "memory_embedding_hnsw" in str(plan)
```

### 5.2 连接池优化测试

```python
@pytest.mark.asyncio
async def test_connection_pool_reuse():
    """连接池复用：避免频繁连接"""
    from surrealdb_client import get_connection_pool
    
    pool = get_connection_pool()
    
    # 获取10个连接
    connections = []
    for _ in range(10):
        conn = await pool.acquire()
        connections.append(conn)
    
    # 释放连接
    for conn in connections:
        await pool.release(conn)
    
    # 验证连接被复用
    assert pool.size() == 10
    assert pool.available() == 10
```

### 5.3 监控和日志测试

```python
def test_structured_logging():
    """结构化日志：JSON格式"""
    import logging
    import json
    
    logger = logging.getLogger("memory_service")
    
    with LogCapture() as logs:
        logger.info("Memory uploaded", extra={
            "memory_id": "memory:123",
            "tenant_id": "test",
            "duration_ms": 45
        })
    
    log_entry = json.loads(logs[0])
    assert log_entry["message"] == "Memory uploaded"
    assert log_entry["memory_id"] == "memory:123"
    assert log_entry["duration_ms"] == 45

@pytest.mark.asyncio
async def test_prometheus_metrics():
    """Prometheus指标导出"""
    from prometheus_client import REGISTRY
    
    # 触发一些操作
    await manager.upload_memories([{"content": "test"}])
    await manager.search(query="test")
    
    # 验证指标
    metrics = REGISTRY.collect()
    metric_names = [m.name for m in metrics]
    
    assert "memory_upload_total" in metric_names
    assert "memory_search_duration_seconds" in metric_names
```

---

## 6. 验收标准

### 6.1 代码覆盖率

| 类型 | 目标 | 最低要求 |
|------|------|----------|
| 语句覆盖率 | 85% | 75% |
| 分支覆盖率 | 80% | 70% |
| 函数覆盖率 | 90% | 80% |
| 行覆盖率 | 85% | 75% |

### 6.2 性能指标

| 操作 | 目标 | 最大值 |
|------|------|--------|
| 向量搜索（1K数据） | <50ms | 100ms |
| 关键词搜索 | <20ms | 50ms |
| 批量插入（100条） | <500ms | 1000ms |
| 去重检测 | <10ms | 20ms |

### 6.3 功能完整性

**Phase A**：

- ✅ 消除重复向量计算（使用KNN搜索）
- ✅ 启用传输加密（wss://）
- ✅ 智能去重决策框架（6个核心规则）
- ✅ 批量事务支持
- ✅ 查询结果缓存

**Phase B**：

- ✅ 变更检测API
- ✅ 批量上传API
- ✅ 全量同步API
- ✅ 冲突处理机制

**Phase C**：

- ✅ 查询性能分析（EXPLAIN）
- ✅ 连接池优化
- ✅ 结构化日志
- ✅ Prometheus指标

---

## 7. 测试执行计划

### 7.1 开发阶段

```bash
# 单元测试（每次提交）
uv run pytest tests/unit/

# 监听模式（开发时）
uv run pytest-watch tests/

# 覆盖率报告
uv run pytest --cov=. --cov-report=html
```

### 7.2 集成阶段

```bash
# 启动测试服务
docker-compose -f docker-compose.test.yml up -d

# 集成测试
uv run pytest tests/integration/

# 性能测试
uv run pytest tests/performance/ -v
```

### 7.3 发布前

```bash
# 完整测试套件
uv run pytest tests/ --cov=. --cov-report=term-missing

# 压力测试
locust -f tests/load/locustfile.py --headless -u 1000 -r 100

# 安全扫描
bandit -r . -ll
```

---

## 8. 测试数据管理

### 8.1 测试夹具

```
tests/
├── fixtures/
│   ├── memories/
│   │   ├── 1k-samples.json      # 1000条测试数据
│   │   ├── 10k-samples.json     # 10000条测试数据
│   │   └── edge-cases.json      # 边界情况
│   ├── embeddings/
│   │   └── mock-embeddings.npy  # 预生成的embedding
│   └── responses/
│       ├── search-results.json
│       └── upload-responses.json
└── temp/  # 测试运行时临时目录
```

### 8.2 Mock数据生成

```python
# tests/helpers/mock_data.py
from faker import Faker
import numpy as np

fake = Faker('zh_CN')

def generate_mock_memory(overrides=None):
    """生成Mock记忆"""
    memory = {
        "content": fake.text(max_nb_chars=200),
        "type": fake.random_element(["preference", "decision", "long-term", "general"]),
        "tags": [fake.word() for _ in range(3)],
        "tenant_id": "test",
        "embedding": np.random.rand(1024).tolist()
    }
    if overrides:
        memory.update(overrides)
    return memory

def generate_mock_memories(count=100):
    """批量生成Mock记忆"""
    return [generate_mock_memory() for _ in range(count)]
```

---

## 9. 持续集成配置

### 9.1 GitHub Actions

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      surrealdb:
        image: surrealdb/surrealdb:latest
        ports:
          - 18002:8000
        options: >-
          --health-cmd "curl -f http://localhost:8000/health || exit 1"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      meilisearch:
        image: getmeili/meilisearch:latest
        ports:
          - 17700:7700
        env:
          MEILI_MASTER_KEY: test_key
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      
      - name: Install dependencies
        run: uv sync
      
      - name: Run tests
        run: uv run pytest tests/ --cov=. --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

---

## 10. 总结

### 10.1 测试优先级

**P0（必须通过）**：

- 向量搜索准确性
- 数据完整性
- 安全性（TLS加密、租户隔离）
- 去重逻辑正确性

**P1（强烈建议）**：

- 性能指标达标
- API可靠性
- 并发处理
- 错误处理

**P2（可选）**：

- 压力测试
- 监控指标
- 日志完整性

### 10.2 测试时间分配

| 阶段 | 单元测试 | 集成测试 | 性能测试 | 总计 |
|------|----------|----------|----------|------|
| Phase A | 1.5h | 0.5h | 0.5h | 2.5h |
| Phase B | 2h | 1h | 0.5h | 3.5h |
| Phase C | 1.5h | 0.5h | 1h | 3h |
| **总计** | **5h** | **2h** | **2h** | **9h** |

### 10.3 关键测试场景

1. **数据完整性**：上传 → 存储 → 检索 → 验证
2. **去重准确性**：相似记忆 → 智能决策 → 正确处理
3. **性能稳定性**：1000+并发 → 延迟<100ms → 无错误
4. **安全性**：租户隔离 → 注入防护 → TLS加密
5. **可靠性**：网络故障 → 自动重试 → 数据不丢失
