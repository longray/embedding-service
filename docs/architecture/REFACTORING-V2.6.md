# 技术重构方案 (v2.6.0)

**日期**: 2026-04-02
**关联**: BACKLOG.md (BL-33 ~ BL-37)
**前置阅读**: `docs/product-decisions.md` (PD-2)

---

## 1. memory_manager.py 拆分方案

### 1.1 现状

`wrapper/src/utils/memory_manager.py` 共 **1660 行**，承担 5 类职责：

| 职责 | 估算行数 | 关键方法 |
|------|----------|----------|
| CRUD（增删改查） | ~400 | `upload_memories`, `_upsert_single`, `_update_memory`, `delete_memory` |
| 搜索（向量/全文/混合/RRF） | ~350 | `search_memories`, `_vector_search`, `_keyword_search`, `_hybrid_search` |
| 同步（指纹/预览/全量/冲突） | ~350 | `get_fingerprints`, `sync_preview`, `sync_full`, `resolve_conflict` |
| 图关系 | ~150 | `create_relation`, `get_relations`, `delete_relation`, `graph_traversal` |
| 去重 + 代码分析 + 辅助 | ~410 | `_decide_duplicate_action`, `analyze_memory_code`, `_build_meili_doc` 等 |

最严重问题：`upload_memories()` **246 行、36 分支**。

### 1.2 目标结构

```
wrapper/src/utils/memory_manager/
├── __init__.py              # 导出 MemoryManager
├── manager.py               # MemoryManager 主类（编排层，~200 行）
├── crud.py                  # 上传、更新、删除
├── search.py                # 搜索路由、RRF 融合
├── sync.py                  # 同步、指纹、冲突解决
├── relations.py             # 图关系、遍历
├── dedup.py                 # 去重决策、content_hash
├── meili_sync.py            # Meilisearch 双写/同步（从 meili_client 交互逻辑抽取）
└── code_analysis.py         # 代码分析调用（wrapper 调用 code_analyzer.py 的桥接）
```

### 1.3 拆分原则

1. **MemoryManager 主类保留为编排层**：对外接口不变（`upload_memories`、`search_memories` 等），内部委托给子模块。
2. **共享状态通过构造函数注入**：`db_manager`、`meili_client`、`embedding_service_url` 等通过 `__init__` 传入各子模块。
3. **保持所有现有测试通过**：`from .utils.memory_manager import MemoryManager` 导入路径不变（`__init__.py` 重新导出）。
4. **不改变任何 API 行为**：纯重构，零功能变更。

### 1.4 upload_memories() 拆分策略

当前 246 行的单函数拆分为：

```
upload_memories()           # 主编排（~40 行）
├── _validate_and_preprocess()   # 校验 + 预处理（~30 行）
├── _batch_get_embeddings()      # 批量向量化（~30 行）
├── _dedup_check_batch()         # 批量去重检查（~40 行）
├── _upsert_batch()              # 批量写入（~40 行）
└── _sync_to_meili_batch()       # 批量 Meilisearch 同步（~30 行）
```

---

## 2. main.py 路由拆分方案

### 2.1 现状

`wrapper/src/main.py` 共 **1173 行**，所有路由定义在一个文件中。

### 2.2 目标结构

```
wrapper/src/
├── main.py                  # 应用创建 + lifespan（~100 行）
├── models.py                # 所有 Pydantic 模型（从 main.py 提取）
└── routers/
    ├── __init__.py
    ├── health.py            # /health
    ├── embeddings.py        # /v1/embeddings
    ├── memories.py          # /api/v1/memories (CRUD)
    ├── search.py            # /api/v1/memories/search
    ├── sync.py              # /api/v1/sync/*
    ├── relations.py         # /api/v1/memories/relations, /graph
    └── websocket.py         # /ws/memories/live
```

### 2.3 拆分原则

1. **使用 FastAPI `APIRouter`**：每个路由文件定义独立的 `router = APIRouter()`，在 `main.py` 中 `app.include_router(router)`。
2. **共享依赖通过 FastAPI `Depends`**：`memory_manager`、`meili_client` 等通过依赖注入。
3. **模型统一放 `models.py`**：所有 `BaseModel` 定义集中管理，避免循环导入。
4. **导入路径不变**：现有测试中 `from wrapper.src.main import app` 仍然有效。

---

## 3. pyproject.toml 修复方案

### 3.1 问题清单

| 问题 | 位置 | 修复 |
|------|------|------|
| 重复忽略规则（4 处） | `[tool.ruff.lint.per-file-ignores]` | 删除重复的 `E501`, `W293`, `RUF001`, `RUF002`, `RUF003` |
| testpaths 过时 | `[tool.pytest.ini_options]` | `"wrapper-service/tests"` → `"tests"` |
| coverage source 过时 | `[tool.coverage.run]` | `"src", "wrapper-service/src"` → `"src", "wrapper/src"` |

### 3.2 注意事项

- `pytest.ini` 存在时会覆盖 `pyproject.toml` 中的 pytest 配置。当前项目根目录有 `pytest.ini`，需确认其内容是否也需要更新。
- 删除重复规则后运行 `uv run ruff check .` 确认无回归。

---

## 4. meilisearch_code/ 类型修复方案

### 4.1 根因

`meilisearch.IndexStats` 是一个 dataclass（非 dict），代码中错误地使用 `.get()` 方法。

### 4.2 修复映射

| 文件 | 行 | 当前代码 | 修复为 |
|------|-----|---------|--------|
| `init_index.py` | 34 | `except: ...` (缺少 body) | `except Exception: pass` |
| `monitor_index.py` | 29 | `stats.get("number_of_documents")` | `stats.number_of_documents` |
| `monitor_index.py` | 30 | `stats.get("is_indexing")` | `stats.is_indexing` |
| `optimize_index.py` | 26 | `stats.get("number_of_documents")` | `stats.number_of_documents` |
| `optimize_index.py` | 27 | `stats.get("is_indexing")` | `stats.is_indexing` |
| `optimize_index.py` | 36 | `stats.get("number_of_documents")` | `stats.number_of_documents` |
| `optimize_index.py` | 37 | `stats.get("is_indexing")` | `stats.is_indexing` |
| `optimize_index.py` | 40 | `stats.get("number_of_documents")`, `stats.get("number_of_documents")` | 同上 |

### 4.3 验证

```bash
uv run pyright meilisearch_code/
# 期望: 0 errors, 0 warnings
```

---

## 5. 安全加固方案

### 5.1 meili_client.py 默认 API Key

**位置**: `wrapper/src/utils/meili_client.py:34`

**当前**: `api_key="masterKey"` 作为默认参数值。

**方案**: 移除默认值，改为 `api_key: str | None = None`，在 `__init__` 中检查：

```python
if api_key is None:
    logger.warning("No API key provided for Meilisearch, write operations may fail")
```

**注意**: 这不影响功能——Docker Compose 中通过环境变量 `WRAPPER_MEILI_API_KEY` 注入，不依赖代码默认值。但移除硬编码默认值符合安全最佳实践。

---

## 6. 测试补充方案

### 6.1 缺失的测试模块

| 模块 | 文件 | 行数 | 测试重点 |
|------|------|------|----------|
| LRU 缓存 | `cache.py` | 85 | 命中/未命中/TTL/线程安全/容量淘汰 |
| HTTP 连接池 | `http_pool.py` | 82 | 连接复用/超时/关闭清理 |
| 认证 | `auth.py` | 24 | token 验证成功/失败/缺失 |
| 自定义异常 | `exceptions.py` | 47 | 异常层级/消息格式 |

### 6.2 测试策略

- 使用 `unittest.mock` 模拟外部依赖（不连接真实服务）
- 每个模块一个测试文件：`tests/test_cache.py`、`tests/test_http_pool.py` 等
- 目标覆盖率：每个模块 ≥ 80%

---

*本文档归档位置: `docs/architecture/REFACTORING-V2.6.md`*
