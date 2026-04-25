# Tests - Agent 指南

> **Scope**: 测试套件 (96个测试文件)  
> **Framework**: pytest + pytest-asyncio  
> **Markers**: unit | integration | e2e | slow

---

## Structure

```text
tests/
├── conftest.py              # 共享fixtures + 自动跳过逻辑
├── test_wrapper_api.py      # 核心API测试 (56个)
├── test_meili_integration.py  # Meilisearch集成 (23个)
├── test_phase_b_sync.py     # 同步冲突测试
├── test_code_analysis_integration.py  # 代码分析 (738行)
├── integration/             # 集成测试子目录
├── e2e/                     # E2E测试子目录
├── performance/             # 性能测试
│   └── benchmark.py         # 基准测试 (902行)
└── resilience/              # 弹性/容错测试
```

---

## Where to Look

| Task | Location | Notes |
|------|----------|-------|
| 添加单元测试 | `tests/test_*.py` | 使用`@pytest.mark.unit` |
| 集成测试 | `tests/integration/` | 部分mock |
| E2E测试 | `tests/e2e/` | 真实服务HTTP调用 |
| 性能测试 | `tests/performance/` | 标记`slow` |
| 共享fixtures | `conftest.py` | session/function级 |

---

## Conventions

### 测试标记 (四层体系)

```python
# pytest.ini / pyproject.toml 双重定义
markers = [
    "unit: pure logic or mock tests",
    "integration: partial mocking",
    "e2e: real services via HTTP",
    "slow: slow tests",
]
```

**使用**:

```bash
# 只运行单元测试 (pre-commit默认)
uv run pytest tests/ -m unit

# 运行所有
uv run pytest tests/ -v

# 跳过慢测试
uv run pytest tests/ -m "not slow"
```

### Fixtures (conftest.py)

```python
# Session级 - 复用连接
@pytest_asyncio.fixture(scope="session")
async def http_client():
    async with httpx.AsyncClient() as client:
        yield client

# Function级 - 每个测试独立
@pytest_asyncio.fixture
async def embedding_client():
    return EmbeddingClient()

# 自动跳过LLM依赖测试
@pytest_asyncio.fixture(scope="session")
async def llm_client():
    try:
        client = LLMClient()
        await client.health_check()
        return client
    except Exception:
        pytest.skip("LLM service unavailable")
```

### 异步测试

```python
# 必须显式使用pytest-asyncio
@pytest_asyncio.fixture
async def memory_manager():
    return MemoryManager()

# 测试函数
@pytest.mark.unit
async def test_create_memory(memory_manager):
    result = await memory_manager.create(...)
    assert result.id is not None
```

---

## Anti-Patterns (THIS PROJECT)

| 问题 | 位置 | 说明 |
|------|------|------|
| 裸`except:` | benchmark.py L211,217,282,342 | 吞掉所有异常 |
| 硬编码`sleep` | 68处 | 测试不稳定因素 |
| 长测试文件 | benchmark.py 902行 | 应拆分 |
| 重复代码 | 多个test_*.py | 应提取fixtures |

**硬编码sleep示例** (应改用polling/event):

```python
# ❌ 当前做法
await asyncio.sleep(0.5)  # 等待服务就绪
await asyncio.sleep(2.5)  # WebSocket重连测试

# ✅ 推荐做法
await wait_for_service(url, timeout=5.0)
```

---

## Commands

```bash
# 单元测试 (快速)
uv run pytest tests/ -m unit -v

# 集成测试 (需要SurrealDB)
uv run pytest tests/ -m integration -v

# E2E测试 (需要全服务)
uv run pytest tests/ -m e2e -v

# 性能基准
uv run python tests/performance/benchmark.py --iterations 5

# 覆盖率
uv run pytest tests/ --cov=wrapper.src --cov-report=html
```

---

## Notes

- **conftest.py自动跳过**: LLM服务不可用时自动跳过相关测试
- **CI配置**: `.github/workflows/ci.yml`运行`pytest tests/ -v --tb=short`
- **pre-commit**: 只运行`-m unit`测试
- **测试数据**: 使用`tests/fixtures/`目录存放测试数据
- **Mock策略**: 外部HTTP调用必须mock，数据库可选真实/内存
