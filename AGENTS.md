# EMBEDDING SERVICE - PROJECT KNOWLEDGE BASE

**Generated**: 2026-03-09  
**Commit**: 9963398  
**Branch**: master

## OVERVIEW

AI服务栈：Qwen3 Embedding + MiniCPM4 LLM + FastAPI包装层，支持向量搜索和对话生成。
技术栈：Python 3.11 + FastAPI + SurrealDB + uv包管理。

## STRUCTURE

```
embedding_service/
├── src/qwen3_embedding_service/    # Embedding + LLM 核心服务
│   ├── embedding_service.py        # Qwen3-Embedding-0.6B (413行)
│   ├── llm_service.py              # MiniCPM4-0.5B (546行, 复杂度热点)
│   └── models/                     # 模型文件 (不提交)
├── wrapper-service/                # 包装层 (⚠️ 非标准hyphen命名)
│   ├── src/
│   │   ├── main.py                 # FastAPI主程序 (287行)
│   │   └── utils/                  # 熔断器、缓存、连接池 (10个工具)
│   ├── tests/                      # 包装层专用测试
│   └── requirements.txt            # ⚠️ 独立依赖 (应统一到pyproject.toml)
├── tests/                          # 主服务测试 (15个文件)
├── scripts/                        # 工具脚本 (⚠️ 硬编码Windows路径)
├── start_services.py               # 统一启动入口
└── pyproject.toml                  # 主项目配置 + uv.lock
```

## WHERE TO LOOK

| 任务 | 位置 | 说明 |
|------|------|------|
| 启动所有服务 | `start_services.py --with-llm` | 必须用 `uv run` |
| Embedding API | `src/qwen3_embedding_service/embedding_service.py` | 端口 18000 |
| LLM API | `src/qwen3_embedding_service/llm_service.py` | 端口 18001, 546行需重构 |
| 包装层API | `wrapper-service/src/main.py` | 端口 17999, 熔断器+缓存 |
| 熔断器逻辑 | `wrapper-service/src/utils/circuit_breaker.py` | 状态机实现 |
| 缓存实现 | `wrapper-service/src/utils/cache.py` | 线程安全LRU |
| SurrealDB集成 | `wrapper-service/src/utils/surrealdb_client.py` | 向量搜索 (无HNSW索引) |
| 测试配置 | `tests/conftest.py` | Async fixtures |
| CI配置 | `.github/workflows/ci.yml` | Ruff + Pyright + Bandit |
| 预提交钩子 | `.pre-commit-config.yaml` | P0安全 → P1类型 → P2格式 |

## CODE MAP

| 符号 | 类型 | 位置 | 职责 |
|------|------|------|------|
| `start_service()` | Function | start_services.py:92 | 服务启动主函数 |
| `lifespan()` | AsyncContext | wrapper-service/src/main.py:44 | FastAPI生命周期 |
| `create_embeddings()` | Endpoint | wrapper-service/src/main.py:124 | 嵌入API (带缓存) |
| `CircuitBreaker` | Class | wrapper-service/src/utils/circuit_breaker.py | 熔断器状态机 |
| `ThreadSafeLRUCache` | Class | wrapper-service/src/utils/cache.py | 线程安全缓存 |
| `MemoryManager` | Class | wrapper-service/src/utils/memory_manager.py | SurrealDB记忆管理 |
| `generate_response()` | Function | src/qwen3_embedding_service/llm_service.py:118 | LLM生成核心 |

## CONVENTIONS (偏离标准)

### 包管理 - 强制使用 uv
```bash
# ✅ 正确
uv run python script.py
uv run pytest tests/

# ❌ 禁止 (会导致模块导入错误)
python script.py
pytest tests/
```

### 项目结构 - 非标准双src/
- **问题**: `wrapper-service/src/` 嵌套结构 + hyphen命名
- **影响**: 无法作为Python包导入
- **临时方案**: 保持现状，待P3重构

### 测试路径 - 双测试目录
- `tests/` - 主服务测试 (默认pytest路径: `wrapper-service/tests`)
- `wrapper-service/tests/` - 包装层测试
- **运行**: `uv run pytest tests/` (主) 或 `uv run pytest wrapper-service/tests/` (包装层)

### 依赖管理 - 分裂配置
- 主项目: `pyproject.toml` + `uv.lock`
- Wrapper: 独立 `requirements.txt`
- **风险**: 版本不一致

### CORS - 生产环境过于宽松
```python
# ⚠️ 当前配置
allow_origins=["*"]  # embedding_service.py:191, llm_service.py

# 建议: 使用环境变量配置白名单
```

## ANTI-PATTERNS (此项目禁止)

### 🔴 P0 - 裸异常捕获 (60处)
```python
# ❌ 禁止
except Exception as e:
    print(f"错误: {e}")

# ✅ 使用
except ServiceStartupError as e:
    logger.error("service_failed", error=str(e))
    raise
```

**违规文件**: `test_api_integration.py` (19处), `verify_p1_features.py` (10处), `scripts/sync-standards.py` (4处)

### 🟡 P1 - 全局可变状态 (2处)
```python
# ❌ 禁止
global surrealdb_pool, memory_manager
surrealdb_pool = SurrealDBConnectionPool(...)

# ✅ 使用
app.state.surrealdb_pool = SurrealDBConnectionPool(...)
```

**违规位置**: `wrapper-service/src/main.py:50`, `http_pool.py:74`

### 🟡 P2 - 生产代码中的 print (272处)
- **允许**: `scripts/`, `tests/`, `start_services.py`
- **禁止**: `wrapper-service/src/` 生产代码

### 🟠 P3 - 硬编码路径
```python
# ❌ 禁止
STANDARDS_REPO = Path(r"D:\github\code-quality-standard")

# ✅ 使用
STANDARDS_REPO = Path(os.getenv("STANDARDS_REPO", "./code-quality-standard"))
```

## COMMANDS

```bash
# 环境管理 (⚠️ 不要删除venv - PyTorch很大)
uv pip install -e ".[dev]"

# 启动服务
uv run python start_services.py              # Embedding + Wrapper
uv run python start_services.py --with-llm   # 全部服务
uv run python start_services.py --no-wrapper # 仅后端

# 代码检查 (pre-commit顺序)
uv run ruff format --check .    # P2: 格式化
uv run ruff check . --fix       # P2: Linting
uv run pyright                  # P1: 类型检查
uv run bandit -r src/           # P0: 安全扫描

# 测试
uv run pytest tests/ -v                          # 主服务测试
uv run pytest wrapper-service/tests/ -v          # 包装层测试
uv run pytest tests/test_performance.py -v       # 性能测试 (SLA断言)

# Docker
docker-compose up                                # 生产环境
docker-compose -f docker-compose.dev.yml up      # 开发环境 (热重载)

# SurrealDB初始化
surreal sql --conn ws://localhost:8000/rpc --user root --pass root \
  --ns memory_ns --db memory_db \
  --file wrapper-service/scripts/init_surrealdb_v3.surql
```

## NOTES

### ⚠️ 关键注意事项

1. **不要删除 Python 虚拟环境**
   - PyTorch 2.4.0+cu121 体积很大 (~2GB)
   - 重新下载浪费流量和时间
   - 如果包有问题，用 `uv pip install --force-reinstall package_name` 修复

2. **必须使用 uv 运行**
   - 项目依赖 uv 管理，直接用 `python` 会导致模块导入错误
   - 所有命令前缀: `uv run`

3. **端口占用**
   - Embedding: 18000
   - LLM: 18001
   - Wrapper: 17999 (可通过 `WRAPPER_PORT` 环境变量修改)
   - SurrealDB: 8000

4. **显存要求**
   - MiniCPM4-0.5B: 最低 1GB (自适应配置)
   - Qwen3-Embedding-0.6B: 约 600MB
   - 同时运行: 建议 ≥2GB

5. **测试覆盖率要求**
   - 最低: 70% (配置在 `pyproject.toml`)
   - 新代码必须达标

6. **已移除功能**
   - Prometheus 监控 (commit 9963398)
   - 使用 structlog 替代

### 🐛 已知问题

- `llm_service.py` 546行，复杂度高 (P3待重构)
- SurrealDB 向量搜索无 HNSW 索引 (全表扫描)
- 60处裸异常捕获需修复
- `wrapper-service/` hyphen命名无法作为包导入

### 📚 参考文档

- API规范: `API_SPECIFICATION.md`
- 包装层设计: `wrapper-service/WRAPPER_SERVICE_DESIGN.md`
- 测试报告: `tests/TEST_REPORT.md`
- 质量标准: `quality-standards/` (同步自 `longray/code-quality-standard`)
