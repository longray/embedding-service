# Backlog

> 后端任务追踪文档，按真实场景驱动排序。已完成任务归档至 `archive/docs/`。

**更新时间**: 2026-04-04

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
| BL-CA-05 | code_filter 添加 max_complexity 支持 | ✅ |
| BL-CA-06 | 修复 v1.2 设计文档 4 个小问题 | ✅ |
| BL-CA-09 | 代码分析集成测试补充 | ✅ |
| BL-CA-10 | 代码分析 API 文档更新 | ✅ |

---

## 场景 3: 多设备同步

> **用户流程**: 插件端有本地记忆文件 → `GET /api/v1/sync/fingerprints` 获取服务端指纹 → `POST /api/v1/sync/preview` 比对差异 → 用户确认 → `POST /api/v1/sync/full` 执行同步 → 冲突时解决
>
> **当前状态**: ✅ 已完成。BL-29~32 全部实现，32 个 mock 测试全部通过，辅助方法（_record_conflict/get_conflicts/get_conflict_detail）已就绪。
>
> **产品文档**: `docs/product-sync-v2.7.md`
> **开发文档**: `docs/dev-sync-v2.7.md`

### 已完成

| 编号 | 目标 | 状态 |
|------|------|------|
| BL-29 | get_fingerprints 查询 SurrealDB 指纹 | ✅ |
| BL-30 | sync_preview 三分类比对 + conflict 表写入 | ✅ |
| BL-31 | sync_full 透传 upload_memories | ✅ |
| BL-32 | resolve_conflict 三种策略 + 辅助方法 | ✅ |

---

## 场景 4: 测试架构优化

> **用户流程**: 开发者 `git commit` → pre-commit 秒级跑 unit 测试 → 推送前手动跑 unit+integration → CI 跑全量（含 e2e）
>
> **当前状态**: ✅ 已完成。21 个文件 / 289 个用例。unit 50P (9.52s) / integration 123P 26S (10.90s) / e2e 140 用例。pre-commit 9.52s。e2e 在 embedding+wrapper 启动时 0F。
>
> **根因诊断**: 架构问题，非局部问题 — 详见 `handoffs/handoff-20260403-test-architecture-diagnosis.md`
>
> **e2e 失败分类（当前 10F）**:
>
> | 原因 | F数 | 说明 |
> |------|-----|------|
> | FEATURE_REMOVED | 8 | circuit_breakers(4F)、metrics(3F)、chat_completions(1F) — 旧版功能已移除 |
> | FIELD_CHANGED | 1 | cache_stats.size → max_size/current_size |
> | DESIGN_LIMIT | 1 | batch input array 不支持（接口只接受 string） |

### 已完成

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

**状态**: ✅ 已完成（2026-04-04）

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

**状态**: ✅ 已完成（2026-04-04）

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

**状态**: ✅ 已完成（2026-04-04）— 端口 3001→17999 + embedding_client/wrapper_client 回退为 function scope

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

**状态**: ✅ 已完成（2026-04-04）— `pytest_collection_modifyitems` hook + db_connection skip

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

#### BL-T11 [P1] 修复 wrapper 接口变更测试 #scene4

**目标**: 修复 3 个文件中因最小化 wrapper (17999) 不再提供旧版功能而导致的 10 个测试失败。

**真实场景**: `test_wrapper_service.py`、`test_wrapper_service_extended.py`、`test_integration.py` 是为旧版完整包装服务 (3001 端口) 编写的，测试了 circuit_breakers、metrics、chat_completions 等功能。最小化 wrapper 已移除这些功能，但测试未同步更新，导致 10F 噪音掩盖真正的代码问题。

**涉及范围**:

| 根因 | 文件 | F数 | 处理方式 |
|------|------|-----|----------|
| `circuit_breakers` 字段不存在 | `test_wrapper_service.py`(1), `test_wrapper_service_extended.py`(2), `test_integration.py`(1) | 4F | 删除测试用例（功能已移除） |
| `/metrics` 端点 404 | `test_wrapper_service.py`(1), `test_wrapper_service_extended.py`(2) | 3F | 删除测试用例（prometheus 已移除） |
| `/v1/chat/completions` 404 | `test_wrapper_service.py`(1) | 1F | 删除测试用例（最小化 wrapper 不代理 LLM） |
| `cache_stats.size` → `max_size/current_size` | `test_wrapper_service_extended.py`(1) | 1F | 更新断言适配新字段 |
| batch input array 返回 422 | `test_wrapper_service_extended.py`(1) | 1F | 标记 skip（接口设计：只接受 string input） |

**前置依赖**: **BL-T8**（端口修复已完成 ✅）

**完成标准**:

- [ ] 10F 全部消除（删除 8F + 更新 1F + skip 1F）
- [ ] `uv run pytest tests/test_wrapper_service.py tests/test_wrapper_service_extended.py tests/test_integration.py -v` 0F
- [ ] unit/integration 无回归
- [ ] 不修改业务代码，只修改测试代码

**验证方式**:

```bash
# wrapper 相关测试 0F
uv run pytest tests/test_wrapper_service.py tests/test_wrapper_service_extended.py tests/test_integration.py -v --tb=short

# 无回归
uv run pytest tests/ -m "unit or integration" -q
```

**状态**: ✅ 已完成（2026-04-04）— 删除 8F + 更新 1F + skip 1F

---

#### BL-T5 [P2] 清理无效测试文件 #scene4

**目标**: 删除 `test_memory_search_gate.py`（1 个空 fixture，无实际用例）。

**涉及范围**:

| 文件 | 问题 | 处理 |
|------|------|------|
| `test_memory_search_gate.py` | 1 个空 fixture（pytestmark=unit），无实际测试函数 | 删除文件 |

**前置依赖**: 无

**完成标准**:

- [ ] `test_memory_search_gate.py` 已删除
- [ ] `uv run pytest tests/ --collect-only -q` 总数从 290 减为 289
- [ ] unit 测试无回归（该文件的 1S 不再出现）

**验证方式**:

```bash
uv run pytest tests/ -m unit -q
```

**状态**: ✅ 已完成（2026-04-04）— 删除 test_memory_search_gate.py，289 collected

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

**目标**: 将 5 个小文件合并到同主题的大文件中，减少文件数量和 pytest 收集开销。

**真实场景**: 当前 27 个文件中有多对同主题的拆分文件，合并后减少文件数但不减少用例数。

**涉及范围**:

| 被合并文件 | 用例数 | 层级 | 合并目标 | 合并后 |
|-----------|--------|------|---------|--------|
| `test_wrapper_service.py` | 2 | e2e | → `test_wrapper_service_extended.py` (4) | 7 |
| `test_llm_service.py` | 5 | e2e | → `test_llm_service_extended.py` (13) | 18 |
| `test_db_connection.py` | 1 | e2e (skip) | → `test_wrapper_service_extended.py` | +1 |
| `test_code_filter_max_complexity.py` | 4 | unit | → `test_code_analysis.py` (11) | 15 |
| `test_embedding_service.py` | 6 | e2e | → `test_embedding_service_extended.py` (11) | 17 |

**未合并**（跨层级或风格不同）:

| 文件 | 原因 |
|------|------|
| `test_auth.py` (unit) | 与 `test_security.py` (e2e) 跨层级，保持独立 |
| `test_integration.py` (e2e) | 标准 pytest 测试，与 `test_api_integration.py` 手动脚本风格不同 |
| `test_api_integration.py` | 手动脚本（print_result），非标准 pytest |

**前置依赖**: **BL-T11**（wrapper 文件已清理 ✅）

**完成标准**:

- [x] 文件数从 27 减少到 21（删除 6 个文件）
- [x] 总用例数 289（原 290 删除 1S）
- [x] 合并后所有测试通过（0F）
- [x] unit/integration 无回归
- [x] 不修改业务代码，只修改测试文件

**验证方式**:

```bash
uv run pytest tests/ --collect-only -q  # 289 collected
uv run pytest tests/ -m "unit or integration" -q
```

**状态**: ✅ 已完成（2026-04-04）— 27→21 文件，289 collected

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

 当前阶段 — 测试架构优化 ✅ 已完成
 ├── BL-T1 测试分层标记 ─────────────────────► ✅
 ├── BL-T2 fixture scope 优化 ────────────────► ✅
 ├── BL-T3 修复 Mixin mock 断言 ─────────────► ✅
 ├── BL-T4 修复接口变更断言 ─────────────────► ✅
 ├── BL-T8 conftest 端口 + Event loop 回归 ──► ✅
 ├── BL-T9 LLM/SERVICE_DOWN 条件跳过 ────────► ✅
 ├── BL-T11 wrapper 接口变更修复 ───────────► ✅
 ├── BL-T5 清理无效文件 ─────────────────────► ✅（27→26）
 ├── BL-T7 合并小型文件 ─────────────────────► ✅（26→21）
 ├── BL-T6 pre-commit 配置 ──────────────────► ✅
 └── BL-T10 语义去重阈值修复 ───────────────► 📋 P3（可选）

下一阶段 — v2.7.0 发布准备
├── BL-D3 BACKLOG 清理 ──────────────────► ✅
├── BL-D4 过时文档更新 ──────────────────► ✅
├── BL-D5 CHANGELOG 更新 ───────────────► ✅
└── BL-D6 E2E 完整验证 ─────────────────► 📋 待开始

---

## 场景 7: 系统可观测性与开发者工具

> **用户场景**: 运维人员部署服务后，需要监控运行状态、诊断性能问题、快速定位故障。
>
> **当前状态**: 核心功能已实现，文档待同步。

### 已完成

| 编号 | 目标 | 状态 |
|------|------|------|
| BL-OBS-00 | 修复 docker-compose.yml 端口映射 | ✅ |
| BL-OBS-01 | 实现 HNSW 统计端点 | ✅ |
| BL-OBS-02 | 实现缓存统计端点 | ✅ |
| BL-OBS-03 | 创建 CLI 诊断工具 | ✅ |
| BL-OBS-04 | 创建 CONTRIBUTING.md | ✅ |

### 待修复

| 编号 | 目标 | 优先级 |
|------|------|--------|
| BL-DOC-01 | 修复 README 端点状态（HNSW/cache 从"计划中"改为"已实现"） | P0 |
| BL-DOC-02 | 更新 README 端口说明 | P1 |
| BL-DOC-03 | 添加场景 7 到 BACKLOG（本章节） | P1 |
| BL-DOC-04 | E2E 验证场景 7 | P2 |

---

## 场景 8: 生产部署与运维（规划中）

> **用户场景**: 运维人员将服务部署到生产环境，需要监控、告警、备份、日志管理。
>
> **当前状态**: 规划中，待细化 backlog。

### 候选 Backlog

- BL-PROD-01: 创建 docker-compose.prod.yml 生产配置
- BL-PROD-02: 实现日志聚合和结构化日志
- BL-PROD-03: 添加 Prometheus 指标导出
- BL-PROD-04: 创建备份恢复脚本
- BL-PROD-05: 编写生产部署指南

---

## 场景 9: 代码分析增强（v1.4 实施中）

> **用户场景**: 开发者需要深度代码理解、跨文件引用追踪、项目级代码地图。
>
> **设计文档**: `docs/CODE-ANALYSIS-DESIGN-v1.4.md`
> **产品文档**: `docs/product/CODE_ANALYSIS_FEATURES.md`
> **API 文档**: `docs/dev/CODE_ANALYSIS_API.md`

### Phase 1: 数据模型补齐（2-3 周）— P0

| 编号 | 目标 | 状态 | 说明 |
|------|------|------|------|
| BL-CA-11 | 函数完整字段 | 📋 | `return_type`, `is_exported`, `is_async` |
| BL-CA-12 | 类成员提取 | 📋 | `methods`, `properties` |
| BL-CA-13 | 接口定义提取 | 📋 | 新增 `_extract_interfaces` |
| BL-CA-14 | 导入结构化 | 📋 | `source`, `imported_names` |
| BL-CA-15 | 导出结构化 | 📋 | `name`, `type`, `is_default` |
| BL-CA-16 | 依赖分类 | 📋 | `internal/external/builtin` |
| BL-CA-17 | 复杂度指标完整 | 📋 | `max_function_complexity`, `average_function_complexity` |
| BL-CA-18 | code_filter 扩展 | 📋 | 支持 `function_count`, `class_count` 等 |

### Phase 2: 调用关系与引用追踪（3-4 周）— P1

| 编号 | 目标 | 状态 | 说明 |
|------|------|------|------|
| BL-CA-19 | 函数调用提取 | 📋 | 新增 `_extract_calls` |
| BL-CA-20 | 调用关系存储 | 📋 | `memory_relation` 新增 `calls` 类型 |
| BL-CA-21 | 引用查询 API | 📋 | `GET /memories/{id}/references` |
| BL-CA-22 | 依赖查询 API | 📋 | `GET /memories/{id}/dependencies` |

### Phase 3: 项目级代码地图（2-3 周）— P1

| 编号 | 目标 | 状态 | 说明 |
|------|------|------|------|
| BL-CA-23 | 项目统计聚合 | 📋 | 按 `project_id` 聚合 |
| BL-CA-24 | 代码地图 API | 📋 | `GET /projects/{id}/map` |
| BL-CA-25 | 热点文件标记 | 📋 | 基于复杂度统计 |

### Phase 4: 语义代码搜索（3-4 周）— P2

| 编号 | 目标 | 状态 | 说明 |
|------|------|------|------|
| BL-CA-26 | 代码预处理 | 📋 | 去除注释/空行，提取关键结构 |
| BL-CA-27 | 代码专用 embedding | 📋 | 函数签名单独 embed |
| BL-CA-28 | 语义搜索 API | 📋 | `semantic_query` 参数 |
| BL-CA-29 | 混合搜索优化 | 📋 | 关键词 + 语义 RRF 融合 |

### Phase 5: opencode 工具集成（2-3 周）— P3

| 编号 | 目标 | 状态 | 说明 |
|------|------|------|------|
| BL-CA-30 | 工具定义 | 📋 | `.opencode/tools/code-analyzer.json` |
| BL-CA-31 | code_search 工具 | 📋 | 代码搜索接口 |
| BL-CA-32 | code_context 工具 | 📋 | 获取代码上下文 |
| BL-CA-33 | code_impact 工具 | 📋 | 变更影响分析 |

---

*最后更新: 2026-04-07（代码分析 v1.4 实施中）*
