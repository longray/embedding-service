# Backlog

> 后端任务追踪文档，按真实场景驱动排序。已完成任务归档至 `archive/docs/`。

**更新时间**: 2026-04-03

---

## 场景 1: 记忆上传与搜索

> **用户流程**: AI 对话 → 插件识别重要信息 → `POST /api/v1/memories` → 后端存储 → 用户提问 → `POST /api/v1/memories/search` → 返回相关记忆
>
> **当前状态**: 核心流程可用，61/61 核心 API 测试通过。

### 已完成

| 编号 | 目标 | 状态 |
|------|------|------|
| BL-18 | 修复测试用例适配 abstract/overview 必填 | ✅ |
| BL-19 | 修复双写测试验证上传→Meilisearch 流程 | ✅ |
| BL-20 | 端到端验证上传→搜索全链路（87/87） | ✅ |
| BL-24 | 修复 get_memory_summary 连接泄露 | ✅ |
| BL-25 | 清理调试日志 | ✅ |
| BL-26 | 实现智能去重决策（替代硬编码 KEEP_BOTH） | ✅ |
| BL-27 | 实现 _update_memory（SurrealDB UPDATE + Meilisearch 同步） | ✅ |

---

## 场景 2: 代码分析

> **用户流程**: 用户编辑代码 → 插件 Tree-sitter 解析 → `POST /api/v1/memories` (type="code") → 后端存储 + 自动分析 → 用户搜索函数名 → 返回匹配代码
>
> **当前状态**: 上传、LLM 摘要、自动代码分析均可用。

### 已完成

| 编号 | 目标 | 状态 |
|------|------|------|
| BL-4 | 代码分析结果持久化（Phase A） | ✅ |
| BL-6 | LLM 代码摘要生成（Phase C） | ✅ |
| BL-28 | analyze_memory_code 实现（CodeAnalyzer 集成） | ✅ v2.6.0 |
| BL-CA-07 | 代码文件指纹同步 API | ✅ |
| BL-CA-08 | 代码文件 Upsert | ✅ |

### 待修复

#### BL-CA-05 [P1] code_filter 添加 max_complexity 支持

**目标**: 搜索 API 的 code_filter 参数新增 max_complexity 过滤，转换为 `code_complexity <= N` Meilisearch 过滤条件。

**涉及范围**:

- `wrapper/src/routers/search.py`:
  - `search_memories()` 函数（当前第 19-26 行）
  - 在现有 filter_parts 逻辑中添加 `max_complexity` 分支

**前置依赖**: 无

**完成标准**:

- [ ] code_filter 含 max_complexity 时生成 `code_complexity <= N` 过滤条件
- [ ] 与 language / min_complexity 组合使用时用 AND 连接
- [ ] 不含 max_complexity 时行为不变（向后兼容）
- [ ] 编写 pytest 用例验证过滤字符串生成正确

**验证方式**:

```bash
# 运行代码分析测试
uv run pytest tests/test_code_analysis.py -v

# 手动验证搜索过滤
curl -X POST http://localhost:17999/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication",
    "code_filter": {
      "language": "python",
      "min_complexity": 5,
      "max_complexity": 30
    }
  }'
```

**状态**: 📋 待开始（1 行代码，高价值）

---

#### BL-CA-06 [P1] 修复 v1.2 设计文档 4 个小问题

**目标**: 修正 CODE-ANALYSIS-DESIGN-v1.2.md 中发现的 4 个文档错误，确保文档与实现一致。

**涉及范围**:

- `archive/docs/CODE-ANALYSIS-DESIGN-v1.2.md`（4 处修改）

**前置依赖**: 无

**完成标准**:

- [ ] 第 304 行 "内存 < 100MB" → "system available memory < 100MB"
- [ ] Section 4.3 batch upload 示例补充 `tenant_id: "default"`
- [ ] Section 9.2 search 响应 `hits` → `results`（与后端实际返回一致）
- [ ] 通过 markdownlint 检查

**验证方式**:

```bash
# Markdown 格式检查
uv run task lint-md

# 人工 review 4 处修改
```

**状态**: 📋 待开始（文档债务）

---

#### BL-CA-09 [P2] 代码分析集成测试补充

**目标**: 补充代码分析功能的端到端集成测试，验证上传→字段提取→搜索过滤完整流程。

**涉及范围**:

- `tests/test_code_analysis_integration.py`（新建测试文件）
- 测试覆盖：上传代码记忆、Meilisearch字段提取、code_filter搜索、Upsert逻辑

**前置依赖**: BL-CA-05（max_complexity 支持）

**完成标准**:

- [ ] 测试上传代码记忆时 Meilisearch 正确接收 code_language/code_complexity/code_symbols 等字段
- [ ] 测试 code_filter 所有参数（language, min_complexity, max_complexity）组合过滤
- [ ] 测试 Upsert 逻辑：同一 file_path + project_id 更新而非新建
- [ ] 测试搜索返回结果包含 code_analysis 元数据
- [ ] 测试 code_symbols 可被全文搜索匹配

**验证方式**:

```bash
# 运行集成测试
uv run pytest tests/test_code_analysis_integration.py -v

# 测试覆盖率检查
uv run pytest tests/test_code_analysis_integration.py --cov=wrapper.src.utils.code_analyzer --cov-report=term-missing
```

**状态**: 📋 待开始（技术债务，建议与 BL-CA-05 同步进行）

---

#### BL-CA-10 [P2] 代码分析 API 文档更新

**目标**: 更新 API_SPECIFICATION.md，添加代码分析相关端点的完整文档。

**涉及范围**:

- `docs/API_SPECIFICATION.md`:
  - 添加 `POST /api/v1/memories` code 类型请求示例
  - 添加 `code_filter` 参数详细说明（language, min_complexity, max_complexity）
  - 添加代码分析响应字段说明

**前置依赖**: BL-CA-05, BL-CA-06

**完成标准**:

- [ ] 添加代码记忆上传的完整请求/响应示例
- [ ] 添加 code_filter 参数表格（字段、类型、必填、说明）
- [ ] 添加 CodeAnalysisResult 字段说明
- [ ] 文档通过 markdownlint 检查

**验证方式**:

```bash
# Markdown 格式检查
uv run task lint-md

# 人工 review 确保示例可运行
```

**状态**: 📋 待开始（文档债务）

---

## 场景 3: 多设备同步

> **用户流程**: 插件端有本地记忆文件 → `GET /api/v1/sync/fingerprints` 获取服务端指纹 → `POST /api/v1/sync/preview` 比对差异 → 用户确认 → `POST /api/v1/sync/full` 执行同步 → 冲突时解决
>
> **当前状态**: 端点已注册，API 模型已定义，测试已编写（32 个 mock 测试）。但 sync_preview/sync_full 仍是 stub，19 个测试失败。
>
> **产品文档**: `docs/product-sync-v2.7.md`
> **开发文档**: `docs/dev-sync-v2.7.md`

### 待修复

#### BL-29 [P2] 实现指纹查询 #scene3

**目标**: `GET /api/v1/sync/fingerprints` 返回服务端所有记忆的指纹列表。

**涉及范围**:

- `wrapper/src/utils/memory_manager/sync.py`:
  - `get_fingerprints()` (当前第 12-17 行，返回空列表)

**前置依赖**: 无

**完成标准**:

- [ ] 查询 SurrealDB `SELECT source_id, content_hash, updated_at FROM memory WHERE source_id != NONE AND tenant_id = $tenant_id`
- [ ] 字段映射: `content_hash` → `hash`, `updated_at` → `mtime`
- [ ] `tests/test_phase_b_sync.py::TestSyncFingerprints` 3 个测试通过
- [ ] 数据库有数据时返回非空列表

**验证方式**:

```bash
uv run pytest tests/test_phase_b_sync.py::TestSyncFingerprints -v --tb=short
```

**状态**: 📋 待开始

---

#### BL-30 [P2] 实现同步预览 #scene3

**目标**: `POST /api/v1/sync/preview` 比对本地与服务端指纹，返回 to_upload/to_delete/conflicts 三分类。

**涉及范围**:

- `wrapper/src/utils/memory_manager/sync.py`:
  - `sync_preview()` (当前第 19-33 行，返回空结果)
  - 新增 `_record_conflict()` 辅助方法（写入 conflict 表）
- `wrapper/src/utils/memory_manager/sync.py` 或新文件:
  - 新增 `get_conflicts()`, `get_conflict_detail()` 查询方法

**前置依赖**: BL-29

**完成标准**:

- [ ] 新记录 → `to_upload`（reason: "new"）
- [ ] 服务端有但本地无 → `to_delete`
- [ ] hash 不同 → `conflicts`（含 local_hash/server_hash）
- [ ] hash 相同 → 不出现在任何列表
- [ ] 冲突记录写入 `conflict` 表（status="pending"）
- [ ] `tests/test_phase_b_sync.py::TestSyncPreview` 4 个测试通过

**验证方式**:

```bash
uv run pytest tests/test_phase_b_sync.py::TestSyncPreview -v --tb=short
```

**状态**: 📋 待开始（依赖 BL-29）

---

#### BL-31 [P2] 实现全量同步 #scene3

**目标**: `POST /api/v1/sync/full` 批量上传/更新记忆。

**涉及范围**:

- `wrapper/src/utils/memory_manager/sync.py`:
  - `sync_full()` (当前第 35-51 行，返回 success=0)

**前置依赖**: 无（与 BL-29 并行，直接调用 `upload_memories`）

**完成标准**:

- [ ] 透传调用 `self.upload_memories(memories, tenant_id=tenant_id)`
- [ ] 去重跳过 → `skipped` 列表
- [ ] 部分失败不中断 → `errors` 列表
- [ ] `tests/test_phase_b_sync.py::TestSyncFull` 3 个测试通过

**验证方式**:

```bash
uv run pytest tests/test_phase_b_sync.py::TestSyncFull -v --tb=short
```

**状态**: 📋 待开始

---

#### BL-32 [P2] 实现冲突解决 #scene3

**目标**: `POST /api/v1/sync/conflicts/{id}/resolve` 支持 use_local/use_remote/keep_both 三种策略。

**涉及范围**:

- `wrapper/src/utils/memory_manager/sync.py`:
  - `resolve_conflict()` (当前第 53-63 行，返回 not implemented)
- SurrealDB `conflict` 表（需创建 schema）

**前置依赖**: BL-30（需要 conflict 表有数据）

**完成标准**:

- [ ] `use_local`: UPDATE memory 内容为本地版本，重新生成 embedding
- [ ] `use_remote`: 保留服务端，仅标记 conflict 为 resolved
- [ ] `keep_both`: CREATE 新记忆（复制服务端 + 修改 source_id 加后缀），重新生成 embedding
- [ ] 不存在的 conflict_id → 返回错误
- [ ] `tests/test_phase_b_sync.py::TestResolveConflict` 3 个测试通过
- [ ] `tests/test_phase_b_sync.py::TestConflictPersistence` 3 个测试通过

**验证方式**:

```bash
uv run pytest tests/test_phase_b_sync.py::TestResolveConflict tests/test_phase_b_sync.py::TestConflictPersistence -v --tb=short
```

**状态**: 📋 待开始（依赖 BL-30）

| 端点 | 说明 |
|------|------|
| `/api/v1/hnsw/stats` | HNSW 索引统计 |
| `/api/v1/hnsw/optimize` | HNSW 优化 |
| `/api/v1/hnsw/rebuild` | HNSW 重建 |
| `/api/v1/cache/stats` | 缓存统计 |
| `/api/v1/cache/clear` | 缓存清空 |
| `/api/v1/cache/warmup` | 缓存预热 |
| `/api/v1/prefetch/related` | 关系预取 |
| `/api/v1/prefetch/popular` | 热门查询预取 |
| `/api/v1/memories/{id}/analyze/code` | 代码分析（独立端点） |
| `/api/v1/memories/cluster/leiden` | Leiden 聚类 |

> 不删除（保持 API 契约完整性），README 中标注"计划中"。

---

## 场景 4: 测试架构优化

> **用户流程**: 开发者 `git commit` → pre-commit 秒级跑 unit 测试 → 推送前手动跑 unit+integration → CI 跑全量（含 e2e）
>
> **当前状态**: 27 个文件 / 299 个用例，12 文件 OK / 15 文件失败。全量 ~395s，pre-commit 经常超 120s。无分层标记，无 fixture scope 优化。
>
> **根因诊断**: 架构问题，非局部问题 — 详见 `handoffs/handoff-20260403-test-architecture-diagnosis.md`
>
> **失败文件分类**:
>
> | 原因 | 文件数 | 失败数 | 说明 |
> |------|--------|--------|------|
> | SERVICE_DOWN | 7 | 44F | 需要真实服务（embedding/llm/wrapper 3001 端口未启动） |
> | ATTR/TYPE_ERROR | 4 | 44F | BL-35 Mixin 拆分后 mock 断言不匹配新结构 |
> | ASSERT_FAIL | 2 | 5F | 接口返回字段变更（embedding health、去重逻辑） |
> | UNKNOWN | 2 | 1F | 待进一步确认 |

### 待实现

#### BL-T1 [P0] 定义测试分层标记 (pytest.mark) #scene4

**目标**: 给全部 27 个测试文件添加 `unit` / `integration` / `e2e` 标记，使 pre-commit 可按标记过滤。

**涉及范围**:

- `tests/conftest.py`:
  - 注册自定义 marks: `pytest.mark.unit`, `pytest.mark.integration`, `pytest.mark.e2e`
- 27 个 `tests/test_*.py` 文件:
  - 按诊断结果分类添加对应 mark（文件级或类级）
- `pyproject.toml`:
  - 配置 markers（可选，避免 pytest 警告）

**分层标准**:

| 层级 | 标记 | 判断标准 | 预期用例数 |
|------|------|---------|-----------|
| unit | `@pytest.mark.unit` | 不调用任何真实服务，纯 mock/纯逻辑 | ~66P (8 文件) |
| integration | `@pytest.mark.integration` | 部分 mock + 部分真实调用，或 mock 复杂场景 | ~96P (8 文件) |
| e2e | `@pytest.mark.e2e` | 全部调用真实服务（localhost:17999/18000/18001/3001） | ~137P (11 文件) |

**前置依赖**: 无

**完成标准**:

- [ ] `uv run pytest -m unit --collect-only` 收集到 ~66 个用例
- [ ] `uv run pytest -m integration --collect-only` 收集到 ~96 个用例
- [ ] `uv run pytest -m e2e --collect-only` 收集到 ~137 个用例
- [ ] `uv run pytest -m "unit or integration or e2e" --collect-only` 收集到 299 个用例（全覆盖）
- [ ] pytest 不输出 "unknown marker" 警告

**验证方式**:

```bash
uv run pytest -m unit --collect-only -q
uv run pytest -m integration --collect-only -q
uv run pytest -m e2e --collect-only -q
uv run pytest -m "unit or integration or e2e" --collect-only -q
```

**状态**: 📋 待开始

---

#### BL-T2 [P1] 优化 conftest fixture scope #scene4

**目标**: 将 `conftest.py` 中 httpx 客户端 fixture 从 `function` 改为 `session` 级别，消除重复建连开销。

**涉及范围**:

- `tests/conftest.py`:
  - `http_client` → `scope="session"`
  - `embedding_client` → `scope="session"`
  - `llm_client` → `scope="session"`
  - `wrapper_client` → `scope="session"`
  - `wrapper_minimal_client` → `scope="session"`

**前置依赖**: 无（可与 BL-T1 并行）

**完成标准**:

- [ ] e2e 组耗时减少 ≥ 60s（当前 ~340s → 目标 ~280s 以内）
- [ ] `uv run pytest tests/test_wrapper_api.py -v` 全部通过（65P）
- [ ] 无 fixture 相关的 teardown 失败

**验证方式**:

```bash
# 优化前
uv run pytest tests/test_wrapper_api.py -v --durations=0

# 优化后对比
uv run pytest tests/test_wrapper_api.py -v --durations=0
```

**状态**: 📋 待开始（可与 BL-T1 并行）

---

#### BL-T3 [P0] 修复 Mixin 模式导致的 mock 断言失败 #scene4

**目标**: 修复 BL-35 memory_manager Mixin 拆分后，4 个测试文件中 44 个 mock 断言失败。

**涉及范围**:

| 文件 | 失败数 | 根因 |
|------|--------|------|
| `test_meili_integration.py` | 11F | `_to_meili_id` / `_from_meili_id` 方法签名变更 |
| `test_phase_b_sync.py` | 19F | MemoryManager 构造参数 / mock 属性不匹配 |
| `test_sync_conflicts.py` | 7F | 同上 |
| `test_db_connection.py` | 1F | 属性访问错误 |

**前置依赖**: 无（可与 BL-T1 并行）

**完成标准**:

- [ ] 4 个文件全部 0F
- [ ] `uv run pytest tests/test_meili_integration.py tests/test_phase_b_sync.py tests/test_sync_conflicts.py tests/test_db_connection.py -v` 全部通过
- [ ] 不修改被测业务代码，只修改测试代码

**验证方式**:

```bash
uv run pytest tests/test_meili_integration.py tests/test_phase_b_sync.py tests/test_sync_conflicts.py tests/test_db_connection.py -v --tb=short
```

**状态**: ✅ 已完成（2026-04-04）

---

#### BL-T4 [P1] 修复接口变更导致的测试失败 #scene4

**目标**: 修复 3 个文件中因接口返回字段变更导致的 6 个断言失败。

**涉及范围**:

| 文件 | 失败数 | 根因 | 修复 |
|------|--------|------|------|
| `test_embedding_service.py` | 2F | health 不再返回 `model_loaded`；stats 返回 `cache` 而非 `cache_stats` | 删除/更新断言 |
| `test_semantic_deduplication.py` | 3F | 去重返回 `skipped` 而非 `errors`；语义阈值未触发 | 更新断言 + 2 个 skip |
| `test_embedding_service_extended.py` | 1F | 缺失 model 时服务返回 200（使用默认模型）而非 422 | 422→200 |

**前置依赖**: 无（可与 BL-T1、BL-T3 并行）

**完成标准**:

- [x] 3 个文件中接口变更导致的断言失败为 0F
- [x] `uv run pytest tests/test_semantic_deduplication.py -v` 全部通过（3P 2S）

**验证方式**:

```bash
uv run pytest tests/test_semantic_deduplication.py -v --tb=short
```

**状态**: ✅ 已完成（2026-04-04）

---

#### BL-T8 [P0] 修复 conftest 配置错误和 session scope 回归 #scene4

**目标**: 消除我们引入的两类测试回归：(1) `wrapper_client` 端口配置错误 (3001→17999)，(2) session scope async fixture 的 Event loop 生命周期问题。

**真实场景**: 开发者启动了 embedding 服务 (18000) 和 wrapper 服务 (17999)，运行 `uv run pytest tests/ -m e2e`，期望相关测试通过，但实际 31 个测试因配置错误和 Event loop 回归而失败。

**涉及范围**:

| 根因 | 文件 | F数 | 修复方式 |
|------|------|-----|----------|
| `WRAPPER_SERVICE_URL` 错误 (3001→17999) | `test_wrapper_service.py`(4F), `test_wrapper_service_extended.py`(10F), `test_integration.py`(2F), `test_performance.py`(2F), `test_security.py`(1F) | **19F** | 修改 conftest.py 端口 |
| session scope Event loop closed | `test_embedding_service.py`(3F), `test_embedding_service_extended.py`(6F), `test_security.py`(3F) | **12F** | 调整 fixture scope 或升级 pytest-asyncio |

**根因分析**:

1. **端口错误**: conftest.py 第 21 行 `WRAPPER_SERVICE_URL = "http://localhost:3001"` 与实际 wrapper 端口 (17999) 不一致。`test_wrapper_api.py` 不受影响因为它有自带的 client fixture。
2. **Event loop**: pytest-asyncio 1.3.0 在 Windows ProactorEventLoop 下，session scope async fixture 的 teardown 与 function scope event loop 不匹配。表现为 PASSED/FAILED 交替出现。

**前置依赖**: 无

**完成标准**:

- [ ] conftest.py `WRAPPER_SERVICE_URL` 改为 `http://localhost:17999`
- [ ] `test_wrapper_service.py` 和 `test_wrapper_service_extended.py` 在服务启动时全部 PASSED
- [ ] `test_embedding_service.py` 和 `test_embedding_service_extended.py` 中 Event loop 错误为 0
- [ ] unit/integration 测试无回归
- [ ] 总 e2e 失败从 ~53F 降至 ~22F（仅剩 LLM SERVICE_DOWN + db_connection）

**验证方式**:

```bash
# 1. 端口修复后 wrapper 相关测试通过
uv run pytest tests/test_wrapper_service.py tests/test_wrapper_service_extended.py tests/test_integration.py -v --tb=short

# 2. Event loop 修复
uv run pytest tests/test_embedding_service.py tests/test_embedding_service_extended.py tests/test_security.py -v --tb=short 2>&1 | Select-String "Event loop"

# 3. 无回归
uv run pytest tests/ -m "unit or integration" -q
```

**状态**: 📋 待开始

---

#### BL-T9 [P1] LLM 服务和 SDK 变更测试条件跳过 #scene4

**目标**: 为依赖 LLM 服务 (18001) 和 SurrealDB SDK 的 e2e 测试添加条件跳过，服务未启动时自动 skip。

**真实场景**: 开发者本地只有 embedding (18000) + wrapper (17999)，没有 LLM 服务 (18001)。运行全量测试时 21 个 LLM 失败是噪音，掩盖真正的代码问题。

**涉及范围**:

| 文件 | F数 | 依赖 |
|------|-----|------|
| `test_llm_service.py` | 5F | LLM 服务 (18001) |
| `test_llm_service_extended.py` | 13F | LLM 服务 (18001) |
| `test_performance.py` | 2F | LLM 服务 (18001) |
| `test_security.py` | 1F | LLM 服务 (18001) |
| `test_db_connection.py` | 1F | SurrealDB SDK API 变更 |

**前置依赖**: **BL-T8**（端口修复后才能准确区分 SERVICE_DOWN 和配置错误）

**完成标准**:

- [ ] LLM 服务未启动时，4 个 LLM 文件全部显示 SKIPPED
- [ ] LLM 服务启动后，skip 自动解除
- [ ] `test_db_connection.py` 修复 SDK API 变更或标记 skip
- [ ] 全量 e2e 在 embedding+wrapper 启动时 0F

**验证方式**:

```bash
# LLM 未启动时应全部 skipped
uv run pytest tests/test_llm_service.py tests/test_llm_service_extended.py -v
```

**状态**: 📋 待开始（依赖 BL-T8）

---

#### BL-T10 [P3] 语义去重阈值测试修复 #scene4

**目标**: 修复或重新设计 2 个被 skip 的语义去重测试，使其在当前 embedding 模型下能稳定触发。

**涉及范围**:

| 测试 | 状态 | 问题 |
|------|------|------|
| `test_semantic_deduplication_high_similarity` | SKIP | 中文短句 embedding 相似度未达 0.95 阈值 |
| `test_batch_deduplication` | SKIP | 同上 |

**前置依赖**: 无

**完成标准**:

- [ ] 2 个 skip 移除，测试稳定通过
- [ ] 断言阈值与业务配置一致

**验证方式**:

```bash
uv run pytest tests/test_semantic_deduplication.py -v
```

**状态**: 📋 待开始

---

#### BL-T5 [P2] 清理无效测试文件 #scene4

**目标**: 处理 `test_memory_search_gate.py`（0 个用例）。

**涉及范围**:

| 文件 | 问题 | 处理 |
|------|------|------|
| `test_memory_search_gate.py` | 0 个用例（空 fixture 定义） | 删除文件 |

> 注：原 BL-T5 中的 SERVICE_DOWN 和端口问题已分别归入 BL-T8（端口修复）和 BL-T9（条件跳过）。

**前置依赖**: 无

**完成标准**:

- [ ] `test_memory_search_gate.py` 已删除
- [ ] `uv run pytest tests/ --collect-only -q` 总数正确
- [ ] unit 测试无回归

**验证方式**:

```bash
uv run pytest tests/ -m unit -q
```

**状态**: 📋 待开始（依赖 BL-T8 替代原 BL-T1）

---

#### BL-T6 [P0] pre-commit 配置调整 #scene4

**目标**: pre-commit 中 pytest 只跑 `unit` 组，60s 内完成。

**涉及范围**:

- `.pre-commit-config.yaml`:
  - pytest entry 改为: `uv run pytest tests/ -m unit -v --tb=short`

**前置依赖**: **BL-T1**（需要 mark 才能过滤）

**完成标准**:

- [x] `git commit` 时 pytest < 60s 完成（实测 9.66s）
- [x] pre-commit 其他 hook（gitleaks, bandit, ruff, pyright）不受影响
- [x] 全量测试仍可手动运行: `uv run pytest tests/ -v`

**验证方式**:

```bash
uv run pytest tests/ -m unit -v --tb=short
```

**状态**: ✅ 已完成（2026-04-04）

---

#### BL-T7 [P2] 合并小型测试文件 #scene4

**目标**: 将 10 个 ≤6 用例的小文件合并到同主题的大文件中，减少 pytest 收集开销。

**涉及范围**:

| 被合并文件 | 用例数 | 合并目标 |
|-----------|--------|---------|
| `test_code_filter_max_complexity.py` | 4 | → `test_code_analysis.py` |
| `test_memory_search_gate.py` | 0 | → `test_wrapper_api.py` |
| `test_db_connection.py` | 1 | → `test_wrapper_service.py` |
| `test_integration.py` | 2 | → `test_api_integration.py` |
| `test_websocket.py` | 4 | → `test_wrapper_service_extended.py` |
| `test_wrapper_service.py` | 4 | → `test_wrapper_service_extended.py` |
| `test_http_pool.py` | 5 | → `test_wrapper_api.py` |
| `test_llm_service.py` | 5 | → `test_llm_service_extended.py` |
| `test_auth.py` | 6 | → `test_security.py` |
| `test_embedding_service.py` | 6 | → `test_embedding_service_extended.py` |

**前置依赖**: **BL-T8**（Event loop 修复后合并更安全）

**完成标准**:

- [ ] 文件数从 27 减少到 ~17
- [ ] 总用例数 299 不减少
- [ ] 合并后所有测试通过
- [ ] git history 可追溯（不 force push）

**验证方式**:

```bash
uv run pytest tests/ -v --tb=short
uv run pytest tests/ --collect-only -q | tail -1  # 确认 299 collected
```

**状态**: 📋 待开始（依赖 BL-T3 + BL-T4）

---

## 执行路线图

```text
v2.6.0 质量治理 — 全部完成 ✅
├── BL-33/34 快速修复 ───────────────────────► ✅
├── BL-28 analyze_memory_code ───────────────► ✅
├── BL-35 memory_manager 拆分 ────────────────► ✅
├── BL-36 main.py 路由拆分 ──────────────────► ✅
├── BL-37 工具模块单元测试 ──────────────────► ✅
├── BL-38/39 清理 ───────────────────────────► ✅
├── BL-D1 归档 ─────────────────────────────► ✅
└── BL-D2 文档对齐 ─────────────────────────► ✅

代码分析完善 — 全部完成 ✅
├── BL-CA-05 max_complexity 支持 ────────────► ✅ 已完成（2026-04-03）
├── BL-CA-06 v1.2 文档修复 ─────────────────► ✅ 已完成（2026-04-03）
├── BL-CA-09 集成测试补充 ──────────────────► ✅ 已完成（2026-04-03）
└── BL-CA-10 API 文档更新 ──────────────────► ✅ 已完成（2026-04-03）

 当前阶段 — 测试架构优化
 ├── BL-T1 测试分层标记 ─────────────────────► ✅ 已完成（2026-04-04）
 ├── BL-T2 fixture scope 优化 ────────────────► ✅ 已完成（2026-04-04）
 ├── BL-T3 修复 Mixin mock 断言 ─────────────► ✅ 已完成（2026-04-04）
 ├── BL-T4 修复接口变更断言 ─────────────────► ✅ 已完成（2026-04-04）
 ├── BL-T8 conftest 端口 + Event loop 回归 ──► 📋 待开始（P0，影响 31F）
 ├── BL-T9 LLM/SERVICE_DOWN 条件跳过 ────────► 📋 待开始（P1，依赖 BL-T8，影响 22F）
 ├── BL-T5 清理无效文件 ─────────────────────► 📋 待开始（P2）
 ├── BL-T7 合并小型文件 ─────────────────────► 📋 待开始（P2，依赖 BL-T8）
 ├── BL-T6 pre-commit 配置 ──────────────────► ✅ 已完成（2026-04-04）
 └── BL-T10 语义去重阈值修复 ───────────────► 📋 待开始（P3）

下一阶段 — 多设备同步 v2.7.0（P2，约 4-6 小时）
├── BL-29 指纹查询 ─────────────────────────► 📋 待开始
├── BL-30 同步预览 ─────────────────────────► 📋 待开始（依赖 BL-29）
├── BL-31 全量同步 ─────────────────────────► 📋 待开始（可与 BL-29 并行）
└── BL-32 冲突解决 ─────────────────────────► 📋 待开始（依赖 BL-30）
```

> **产品文档**: `docs/product-sync-v2.7.md`
> **开发文档**: `docs/dev-sync-v2.7.md`

---

*最后更新: 2026-04-04（更新场景 4: BL-T1~T4/T6 已完成，新增 BL-T8~T10）*
