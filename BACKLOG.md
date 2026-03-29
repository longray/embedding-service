# Backlog

> 后端任务追踪文档，按优先级排序

**更新时间**: 2026-03-28

---

## v2.4.1 - sync_preview conflict 检测修复

### B-005: upload_memories 上传后 get_fingerprints 返回空 ✅ 已完成

**真实场景**: 用户备份恢复后，本地有多条记忆需要同步。调用 `full_sync` 上传完毕，再用 `sync_preview`，期望检测到本地有修改的条目（hash 变了）产生 conflict。但实际 sync_preview 总是返回所有条目为 `new`，永远检测不到 conflict。

**根因**: 分三步

#### B-005-A: SCHEMAFULL 字段未定义 ✅

- schema 文件未定义 `content_hash` 字段，`TYPE NORMAL SCHEMAFULL` 模式下 INSERT 时字段被静默忽略
- 已通过直接 SQL 修复

#### B-005-B: SurrealDB 3.0 SDK 结果解析逻辑错误 ✅

- **涉及范围**: `wrapper/src/utils/memory_manager.py` 第 1130-1150 行
- **修复**: 复用已有的 `_extract_records()` 方法

#### B-005-C: `get_conflict_detail` 参数化表名语法错误 ✅

- **涉及范围**: `wrapper/src/utils/memory_manager.py` 第 1363-1372 行
- **修复**: 用 `WHERE type::string(id) = $conflict_id` 替代 `FROM $conflict_id`

### 代码质量修复 ✅ 已完成

#### B-006: SCHEMA_TARGET_VERSION 版本号未更新 ✅

- **问题**: `SCHEMA_TARGET_VERSION = "2.3.0"` 未随版本更新
- **修复**: 更新为 `"2.4.1"`
- **文件**: `wrapper/src/main.py:213`

#### B-007: FastAPI app 定义位置错误 ✅

- **问题**: `app = FastAPI(...)` 缩进在 lifespan 函数内，导致模块级别无法访问
- **修复**: 移到 lifespan 函数外部模块级别
- **文件**: `wrapper/src/main.py:411`
- **影响**: Pyright 34 个 "app is not defined" 错误，测试 11 个失败

#### B-008: 重复 API 端点定义 ✅

- **问题**: `analyze_memory_code` 和 `cluster_memories_leiden` 定义了两次
- **修复**: 删除重复的代码块（28 行）
- **文件**: `wrapper/src/main.py:941-970`

#### B-009: tree_sitter 导入类型错误 ✅

- **问题**: 可选导入 `tree_sitter` 导致 Pyright 报错
- **修复**: 添加 `# type: ignore` 标记
- **文件**: `wrapper/src/utils/code_analyzer.py:15-16`

**E2E 验证**:

1. 上传 memory → 获取 fingerprints ✅
2. 修改本地 hash → sync_preview 检测到 conflict ✅
3. 调用 resolve → conflict 解决成功 ✅

**质量验证**:

- Pyright 类型检查: 34 errors → 0 errors ✅
- 同步测试: 32/32 passed ✅

---

## v2.4.2 - API 稳定性修复 + 文档整理 + 性能基线

### B-010: sync_preview 返回 500（to_delete 含 None）✅ 已完成

**真实场景**: 用户在多设备间同步记忆。插件调用 `sync_preview` 分析本地与服务端差异，期望拿到 `to_upload`、`to_delete`、`conflicts` 三个列表。当服务端存在 `source_id` 为 None 的记忆记录时，`to_delete` 列表包含 None 值，Pydantic 验证 `list[str]` 失败，API 返回 500。

**目标**: sync_preview 在任何数据状态下都返回合法响应，不因脏数据崩溃。

**涉及范围**:

- `wrapper/src/utils/memory_manager.py` — `get_fingerprints()` 第 1145 行 `record.get("source_id")` 可返回 None
- `wrapper/src/utils/memory_manager.py` — `sync_preview()` 第 1209 行 `to_delete.append(source_id)` 追加 None
- `wrapper/src/main.py` — `SyncPreviewResponse.to_delete: list[str]` 第 124 行

**根因**:

1. `get_fingerprints()` 未过滤 `source_id` 为 None 的记录
2. `sync_preview()` 的步骤 3（服务端→本地比对）直接 append source_id，未做空值过滤
3. Pydantic `list[str]` 严格拒绝 None → 500

**前置依赖**: 无

**修复方案**（两处防御）:

1. `get_fingerprints()`: 过滤 `source_id` 为 None 的记录（源头防御）
2. `sync_preview()`: `to_delete.append` 前检查 source_id 非 None（二次防御）

**完成标准**:

- `sync_preview` 在存在 source_id=None 记录时返回 200
- `to_delete` 列表不含 None 值
- `to_upload`、`conflicts` 不受影响

**验证方式**:

1. 构造 source_id=None 的记忆记录，调用 `sync_preview`，断言返回 200
2. 运行 `uv run pytest tests/test_phase_b_sync.py -v`，32/32 通过
3. 运行 `uv run python scripts/comprehensive_api_test.py`，Sync Preview 从 500 → 200

---

### B-011: 项目文档三分类整理 ✅ 已完成

**真实场景**: 项目有 60 个 Markdown 文件，散落在根目录、docs/、.opencode/plans/、archive/ 等位置。新开发者加入或自己回头找文档时，不知道哪个是用户指南、哪个是架构设计、哪个是过时计划。文档噪音太多，影响效率。

**目标**: 将活跃文档从 60 个精简到 25 个，三类分明、交叉引用正确。

**涉及范围**:

- 归档 28 个历史文件 → `archive/` 对应子目录（`archive/docs/`、`archive/plans/`、`archive/scripts/`、`archive/reports/`）
- 归档 2 个设计审查报告（原计划合并，实际内容已体现在代码中，直接归档）
  - `docs/DESIGN_REVIEW_REPORT.md` → `archive/docs/`
  - `docs/architecture/WRAPPER_SERVICE_REVIEW.md` → `archive/docs/`
- 三分类结构：
  - 📖 产品文档（8 个）：README, API_SPECIFICATION, START_GUIDE, STARTUP_SCRIPTS_GUIDE, SYNC_CONFLICT_RESOLUTION, Meilisearch方案, scripts/README, meilisearch_code/README
  - 🔧 开发文档（16 个）：AGENTS, CHANGELOG, ROADMAP, VERIFICATION-GUIDE, MEMORY_SEARCH_GATE, SURREALDB_3_UPGRADE_DESIGN, surrealdb-persistent-connection, testing-plan, WRAPPER_SERVICE_DESIGN, BACKUP_GUIDE, DEDUPLICATION_GUIDE, tests/README, TEST_PLAN, quality-standards/×4
  - 📋 Backlog（1 个）：BACKLOG.md

**前置依赖**: 无

**完成标准**:

- 活跃文档 25 个，三分类清晰 ✅
- 归档文件在 `archive/` 子目录下可查 ✅
- 归档文件完整可访问 ✅

**验证结果**:

1. 活跃 md 文件 29 个（含 3 个模型 README + 1 个 pytest 缓存，实际项目文档 25 个）✅
2. 归档文件分布：`archive/docs/` 8 个、`archive/plans/` 16 个、`archive/scripts/` 3 个、`archive/reports/` 2 个 ✅

---

### B-012: Cache/HNSW 500 错误 ✅ 已完成

**真实场景**: 用户调用 `/api/v1/cache/warmup`、`/api/v1/hnsw/stats`、`/api/v1/hnsw/optimize` 返回 500。这三个是非核心功能，不影响主流程（上传、搜索、同步）。

**目标**: 修复 Cache 和 HNSW 相关 API 的 500 错误。

**涉及范围**:

- `wrapper/src/utils/memory_manager.py` — `get_memory_stats()`、`warmup_embedding_cache()`

**根因**:

1. `get_memory_stats()` 使用 PostgreSQL 函数 `date_trunc`，SurrealDB 不支持 → 改用 `time::group(created_at, 'day')`
2. `warmup_embedding_cache()` 用 f-string 拼 SQL LIMIT（SQL 注入风险）→ 改参数化 `$limit`
3. `warmup_embedding_cache()` 未处理 `_vector_cache` 为 None 的情况 → 提前返回降级响应

**前置依赖**: 无

**完成标准**:

- `/api/v1/hnsw/stats` 不再因 SQL 语法错误返回 500
- `/api/v1/cache/warmup` 不再因缓存未启用返回 500
- 32 个同步测试无回归

**验证结果**:

1. `uv run pytest tests/test_phase_b_sync.py -v` → 32/32 passed ✅
2. `uv run pyright wrapper/src/utils/memory_manager.py` → 0 errors ✅

**注意**: 完整的 E2E 验证（hnsw/stats 返回 200）需要后端服务在线 + SurrealDB 运行中。

---

### B-013: Tenant ID 不匹配 ⚪ 低优先级（暂缓）

**真实场景**: 用户配置的 tenant_id 是 `longray`，但插件默认使用 `default`。导致插件上传的记忆和用户实际数据不在同一个租户下。

**目标**: 确保插件和后端使用一致的 tenant_id。

**涉及范围**:

- 插件侧：`opencode-memory-plugin` 配置
- 后端侧：无（已支持参数化 tenant_id）

**前置依赖**: 无

**完成标准**:

- 插件使用用户配置的 tenant_id 调用所有 API
- 上传和查询使用同一 tenant_id

**验证方式**:

1. 插件上传一条记忆 → 后端用 `longray` tenant_id 查询 → 能找到
2. 后端用 `default` tenant_id 查询 → 找不到

---

### B-014: LLM 服务并发请求导致 OOM 崩溃 ✅ 已完成

**真实场景**: LLM 服务部署后，两个用户同时调用 `/v1/chat/completions`，两个请求同时执行 `model.generate()`。MiniCPM4-0.5B 在 GPU 上推理时占 ~1.8GB 显存，两个并发推理瞬间翻倍到 ~3.6GB，超过低显存 GPU（如 2GB 显卡的笔记本）直接 OOM 崩溃。即使 GPU 显存够，并发推理也会互相争抢 CUDA 核心，延迟飙升。

**目标**: LLM 服务在同一时刻只执行一次推理，并发请求排队等待，不因并发而崩溃。

**涉及范围**:

- `src/qwen3_embedding_service/llm_service.py` — `generate_response()` 函数（第 119-165 行）是同步阻塞调用，无并发保护
- `src/qwen3_embedding_service/llm_service.py` — 两个 API 端点 `create_chat_completion` 和 `simple_generate` 都直接调用 `generate_response`

**前置依赖**: 无

**修复方案**: 在 `generate_response` 外层加 `asyncio.Lock`，确保同一时刻只有一个推理任务在 GPU 上执行。FastAPI 的 async handler 需要将同步的 `model.generate()` 包装到线程池执行，锁控制并发。

**完成标准**:

- 两个并发请求不会同时执行 `model.generate()`
- 第二个请求等待第一个完成后正常返回（不超时、不崩溃）
- 单请求延迟不显著增加（锁等待开销 <10ms）

**验证方式**:

1. 单元测试：mock `model.generate`，并发 5 个请求，断言全部 200 且实际调用次数 = 5（排队执行不丢弃）
2. 手动验证：同时发 2 个 curl 请求到 `/v1/chat/completions`，两个都正常返回

---

### B-015: LLM 服务 Pydantic `@validator` 弃用警告 ✅ 已完成

**真实场景**: 项目依赖 Pydantic v2（`pyproject.toml` 中已锁定）。`llm_service.py` 使用了 Pydantic v1 的 `@validator` 装饰器，当前版本能工作但会触发弃用警告。Pydantic v3 将完全移除 `@validator`，届时会直接报错服务无法启动。

**目标**: 迁移到 Pydantic v2 的 `@field_validator`，消除弃用警告，保持向前兼容。

**涉及范围**:

- `src/qwen3_embedding_service/llm_service.py` — `ChatCompletionRequest.validate_messages`（第 201 行）、`SimpleGenerateRequest.validate_prompt`（第 238 行）

**前置依赖**: 无

**完成标准**:

- `@validator` 全部替换为 `@field_validator`
- 无 Pydantic 弃用警告
- API 行为不变（空消息列表、过长 prompt 仍然被拒绝）

**验证方式**:

1. `uv run python -c "from qwen3_embedding_service.llm_service import app"` 无 DeprecationWarning
2. 请求空消息列表 → 422，请求超长 prompt → 422

---

### B-016: LLM 服务版本号硬编码未更新 ✅ 已完成

**真实场景**: 开发者查看 LLM 服务的 `/health` 或 `/docs` 页面，看到版本号 `2.3.0`，但项目实际已经到 v2.4.2。版本不一致导致排查问题时误判代码版本。

**目标**: LLM 服务的版本号与项目版本保持同步。

**涉及范围**:

- `src/qwen3_embedding_service/llm_service.py` — `FastAPI(version="2.3.0")`（第 273 行）

**前置依赖**: 无

**修复方案**: 从 `pyproject.toml` 读取版本号，或定义为常量统一管理。

**完成标准**:

- `/health` 和 `/docs` 页面显示的版本号与 `pyproject.toml` 一致
- 后续版本更新只需改一处

**验证方式**:

1. `curl http://localhost:18001/health` → `version` 字段与 `pyproject.toml` 的 `version` 一致

---

### B-017: wrapper 层 `llm_service_url` 配而不用 ✅ 已完成

**真实场景**: 开发者看到 `config.py` 中有 `llm_service_url = "http://localhost:18001"`，以为 wrapper 会调用 LLM 服务。实际搜索代码发现没有任何地方使用这个配置。属于死代码，误导开发者。

**目标**: 明确 LLM 服务与 wrapper 的关系，消除误导。

**涉及范围**:

- `wrapper/src/config.py` — `ServiceConfig.llm_service_url`（第 52 行）

**前置依赖**: B-014（线程安全修复完成后再决定是否集成）

**完成标准**（二选一）:

- 方案 A：删除 `llm_service_url` 配置，在注释中说明 LLM 服务独立运行
- 方案 B：保留配置，在 wrapper 中实现至少一个 LLM 调用场景

**验证方式**:

1. 代码搜索 `llm_service_url` 不出现"配而不用"的状态

---

### B-018: SurrealDB `count(*)` 语法不兼容 ✅ 已完成

**真实场景**: 用户调用 `/api/v1/hnsw/stats`，服务端执行 `SELECT count(*) as total FROM memory`，SurrealDB 3.0 不支持 `count(*)` 语法（返回 Parse error），API 返回 500。

**目标**: HNSW Stats API 在任何数据状态下返回 200。

**涉及范围**:

- `wrapper/src/utils/memory_manager.py` — `get_memory_stats()` 第 1630、1636 行

**根因**: SurrealDB 3.0 使用 `count()` 而非 SQL 标准的 `count(*)`

**修复**: `count(*)` → `count()`（两处）

**验证**: 需要**重启 wrapper 服务**后 E2E 验证

---

### B-025: B608 record_id SQL 注入修复 ✅ 已完成

**真实场景**: Bandit 安全扫描报告 B608（硬编码 SQL 表达式）警告，发现多处 `WHERE in = {mem_ref}`、`FROM {mem_ref}`、`UPDATE {record_id}` 使用 f-string 拼接 record ID，存在 SQL 注入风险。

**目标**: 使用 SurrealDB 3.0+ 的 `type::record($table, $id)` 函数安全构建 record ID，消除 f-string 拼接。

**涉及范围**:

- `wrapper/src/utils/memory_manager.py`:
  - L842, L856: `get_relations()` - `WHERE in/out = {mem_ref}`
  - L887, L893: `delete_relation()` - `SELECT FROM {rel_ref}`, `DELETE {rel_ref}`
  - L921-922: `get_related_memories()` - `FROM {mem_ref}`
  - L1106: `_update_memory()` - `UPDATE {record_id}`

**修复策略**:

1. 拆分 record ID（`"memory:abc123"` → `table="memory"`, `id="abc123"`）
2. 使用 `type::record($table, $id)` 安全构建 record ID
3. 通过参数绑定传递 table 和 id

**修复示例**:

```python
# Before
await self._db.query(f"UPDATE {record_id} SET {set_str}", params)

# After  
rid_parts = record_id.split(":")
rid_table, rid_id = rid_parts[0], rid_parts[1]
params["rid_table"] = rid_table
params["rid_id"] = rid_id
await self._db.query("UPDATE type::record($rid_table, $rid_id) SET {set_str}", params)
```

**验证**:

- `uv run pytest tests/test_phase_b_sync.py -v` → 32/32 passed ✅
- `uv run pyright wrapper/src/utils/memory_manager.py` → 0 errors ✅
- Bandit B608 剩余警告：仅 `where_clause`（白名单构建，实际安全）

**SurrealDB 参考**: `type::record()` 函数官方文档 —— 从参数安全构建 record ID（替代 v1.x 的 `type::thing()`）

---

### B-026: Bandit 安全扫描标记 # nosec ✅ 已完成

**背景**: Bandit 预提交 hook 拦截了 `git commit`，需要标记已知安全代码以通过扫描。

**标记统计**:

- **B608**: 11处已标记
  - path 常量（`"->memory_relation->memory"`）
  - set_str 内部构建（字段列表，非用户输入）
  - where_clause 白名单构建
- **B110**: 6处已标记（try_except_pass）
  - 缓存读取失败静默处理
  - 缓存写入失败静默处理
  - 缓存统计/清理失败静默处理
- **B112**: 3处已标记（try_except_continue）
  - 批量预热跳过失败项
  - 批量预取跳过失败项

**剩余 1 处**:

- L1352 `where_clause` - `.format()` 无法被 `# nosec` 识别，需 pyproject.toml 配置排除

---

## v2.4.3 - 性能优化 + LLM 集成（规划中）

### B-020: 性能基线建立 📊 基线数据

**脚本**: `scripts/benchmark.py`（已创建）

**测试环境**:

- GPU: NVIDIA GeForce GTX 1060 6GB
- 模型: Qwen3-Embedding-0.6B
- 数据库: SurrealDB 3.0 + Meilisearch 1.4
- 日期: 2026-03-28

**基线数据**（3 次迭代）:

| 操作 | 平均延迟 | P50 | P95 |
|------|----------|-----|-----|
| 单文本 Embedding | 156ms | 160ms | 201ms |
| 批量 Embedding (10条) | 1643ms | 1604ms | 2009ms |
| 长文本 Embedding (~10K字符) | 4875ms | 10ms | 14606ms |
| 关键词搜索 | 112ms | 19ms | 299ms |
| 向量搜索 | 157ms | 156ms | 180ms |
| 混合搜索 | 21ms | 22ms | 24ms |
| 单条上传 | 542ms | 699ms | 724ms |
| 批量上传 (5条) | 905ms | 885ms | 956ms |
| 获取指纹 | 9ms | 10ms | 10ms |
| 同步预览 (10条) | 11ms | 11ms | 12ms |
| 同步预览 (100条) | 12ms | 13ms | 14ms |
| E2E 完整流程 | 883ms | 884ms | 919ms |

**发现的问题**:

1. 长文本 Embedding 延迟方差极大（8ms~14606ms），可能存在缓存命中 vs 冷启动差异
2. 关键词搜索首次延迟高（299ms），后续 19ms，Meilisearch 冷启动
3. 批量 Embedding 是逐条调用（wrapper 不支持批量输入），可优化
4. 单条上传 542ms，其中 Embedding 生成 ~160ms，其余为 DB 写入

**优化方向**:

- B-021: wrapper 支持批量 Embedding 输入（减少 HTTP 往返）
- B-022: 上传流水线优化（并行 embedding + DB 写入）

---

### B-023: LLM 服务集成场景规划 📋 规划中

**当前状态**: LLM 服务（MiniCPM4-0.5B，端口 18001）完全独立运行，wrapper 从未调用。

**场景优先级矩阵**:

| 场景 | 优先级 | 复杂度 | ROI | 说明 |
|------|--------|--------|-----|------|
| 自动生成 abstract/overview | P0 | 低 | 高 | 上传时自动补充 L0/L1 层级内容 |
| 自动标签 | P0 | 低 | 高 | 从内容提取 3-5 个标签 |
| 搜索查询扩展 | P1 | 中 | 中 | 同义词/相关词扩展搜索范围 |
| 语义去重增强 | P2 | 中 | 中 | embedding 阈值模糊时用 LLM 二次判断 |
| 冲突合并建议 | P3 | 高 | 低 | 多设备同步时建议合并策略 |

**P0 场景详细设计**:

#### 自动 abstract/overview

- **触发点**: `upload_memories()` 上传时，若 `content_abstract` 或 `content_overview` 为空
- **调用方式**: HTTP 调用 LLM `/generate` 端点（带缓存）
- **Prompt**: `"用≤100字符总结：{content[:500]}"` / `"用≤500字符概括：{content[:1000]}"`
- **降级策略**: LLM 不可用时 fallback 到截取前 N 字符
- **预估延迟**: 单次 ~50-100ms（MiniCPM4-0.5B 短输出推理）

#### 自动标签

- **触发点**: 同上
- **Prompt**: `"从以下内容提取3-5个标签，逗号分隔：{content[:300]}"`
- **合并策略**: LLM 标签 + 用户标签合并去重

**前置依赖**:

- 新建 `wrapper/src/utils/llm_client.py`（异步 HTTP 客户端）
- `wrapper/src/config.py` 恢复 `llm_service_url` 配置（之前 B-017 删除了）
- 上传流程增加 LLM 增强步骤

**风险**:

- LLM 服务不可用时不影响主流程（降级）
- MiniCPM4-0.5B 中文摘要质量待验证（0.5B 模型能力有限）
- 并发保护已实现（B-014 threading.Lock）

---

### B-013: Tenant ID 不匹配 ⚪ 低优先级（暂缓）

### B-001: relationship_type 错误提示优化 ✅

- **完成时间**: 2026-03-28
- **结论**: 当前行为已正确，返回 `400 + 合法值列表`，无需改动
- **验证**: `curl -X POST /api/v1/memories/relations -d '{"relationship_type":"bad"}'` → `{"detail":"Invalid relationship_type: bad. Must be one of {...}}"}`

### B-002: conflict resolution 大小写兼容 ✅

- **完成时间**: 2026-03-28
- **文件**: `wrapper/src/utils/memory_manager.py` line 1433
- **改动**: `resolution = resolution.lower().strip()` 归一化
- **验证**: `curl -X POST /api/v1/sync/conflicts/test/resolve -d '{"resolution":"USE_LOCAL"}'` → 不再报"无效的解决策略"，正确返回"冲突不存在"

### B-003: full_sync 返回 skipped 列表 ✅

- **完成时间**: 2026-03-28
- **文件**: `wrapper/src/utils/memory_manager.py`, `wrapper/src/main.py`, `tests/test_phase_b_sync.py`
- **改动**:
  - `upload_memories`: 新增 `skipped` 列表，hash/语义去重时追加 `{local_id, existing_id, reason, similarity}`
  - `sync_full`: 从 `upload_memories` 收集 skipped，汇总返回
  - `SyncFullResponse`: 新增 `skipped: list[dict]` 和 `updated: int` 字段
  - 去重信息从 `errors` 移到 `skipped`，`errors` 只保留真正的异常
  - 新增测试 `test_sync_full_with_skipped`
- **测试**: TestSyncFull 3/3 通过
- **curl 验证**: 上传 2 条相同内容 → `{"total":2,"success":1,"skipped":[{"local_id":"b003-dup","existing_id":"memory:xxx","reason":"hash","similarity":null}],"errors":[]}`

### B-004: sync_incremental → sync_preview 重命名 ✅

- **完成时间**: 2026-03-28
- **文件**: `wrapper/src/main.py`, `wrapper/src/utils/memory_manager.py`, `tests/test_phase_b_sync.py`
- **改动**:
  - 新增 `/api/v1/sync/preview` 路由 + `SyncPreviewRequest/Response` schema
  - `/api/v1/sync/incremental` 保留为别名（指向同一 handler）
  - `sync_incremental()` → `sync_preview()`
  - 测试类 `TestSyncIncremental` → `TestSyncPreview`，保留旧 model 兼容测试
  - 修复 `test_sync_preview_conflicts` mock 缺少 `create` 的已有 bug
- **测试**: TestSyncPreview 4/4 通过，路由/模型/兼容性测试全部通过
- **curl 验证**: `/sync/preview` 和 `/sync/incremental` 均返回 200 + 差异列表

---

## 历史归档

> v2.4.0 之前的已完成任务已归档至 CHANGELOG.md

---

## v2.5.0 - 代码分析增强

>
> 基于 GitNexus (代码智能引擎) 的功能分析，增强 Memory Stack 的代码分析能力。
> 核心定位： **记忆级代码分析**（非仓库级），输入是逐条上传的记忆片段。
>
> **依赖关系图**:
>
> ```
> B-027 (修复设计文档) ── B-028 (代码分析持久化) ── B-029 (Meilisearch 搜索增强)
>                                                      └── B-030 (LLM 代码摘要)
>                                                               B-031 (跨文件关系，远期考虑)
> ```
>
> **本项目不做什么**（与 GitNexus 的边界）：
>
> - 不做 CLI 命令行工具（本项目是服务端 API）
> - 不做仓库级扫描（输入是记忆片段， 鲞完整仓库)
> - 不做 Web UI（本项目通过 OpenCode 插件交互)
> - 不做 `detect_changes`（后端不访问 git 仓库）
> - 不做 `rename`（记忆条目是不可变的历史记录）
> - 不做增量索引（记忆更新后可手动触发重新分析）
>
> **许可证说明**: GitNexus 使用 PolyForm Noncommercial License， 本项目仅参考其设计理念, 不使用其代码。

### B-027: 修复代码分析增强设计文档

**真实场景**: 设计文档 `docs/code-analysis-enhancement.md` 存在大量格式问题（`>` blockquote 泛滥）、事实性错误（语言数量、图数据库描述、LLM 描述）、和内容缺失（回滚策略、 测试策略、 性能评估, 数据源差异分析)。 在开始编码之前, 先确保设计文档准确完整。

团队成员或未来的开发者需要一份可靠的设计文档作为实施参考.

**目标**: 修复设计文档中的格式问题、事实错误、和缺失内容,使其达到可实施级质量.

**涉及范围**:

- `docs/code-analysis-enhancement.md` — 全面重写

**前置依赖**: 无

**完成标准**:

- 无 `>` blockquote 格式问题（正文不显示为引用块）
- 术语统一："插件端"（非"前端"）, "代码分析增强"（非"代码智能增强"）
- 事实性数据全部修正: 语言数量(11)、图数据库(SurrealDB)、LLM(外部 LLM API)、工具数量(7)
版本号(当前版本)
- 补充缺失章节: 回滚策略、 测试策略、 性能评估、 数据迁移、 增量索引、 Meilisearch 同步细节、 完整数据流、 "本项目不做什么"清单
- 补充数据源差异影响分析: 讯忆级 vs 仓库级代码分析的区别和限制
- Phase 依赖关系图(用 Mermaid 表示)
- Phase 顺序调整为: Phase 2(持久化) → Phase 1(搜索增强) → Phase 3(LLM 摘要) → Phase 4(跨文件关系,远期考虑)
- 插件端代码示例基于 `@opencode-ai/plugin` v1.3.3 的 `Hooks.tool` 类型重写
- 参考文献标注"仅供参考,非规范要求"
- 版本号不预设(删除 `v2.5+`, 使用当前版本)

- 错别字修正

- 本地 Markdown 渲染无格式问题
- 文档结构清晰、逻辑连贯

**验证方式**:

- 本地 Markdown 渲染检查
- 团队成员审阅确认内容准确

- 对照审核报告逐项确认修复

---

### B-028: 代码分析结果持久化（Phase A: 基础持久化）

**真实场景**: 用户通过插件上传一段 Python 代码。后端现有的 `POST /api/v1/memories/{memory_id}/analyze/code` 能分析代码,但分析结果**只返回给调用者,不存储**。每次查看分析结果都需要重新计算。用户希望上传代码后自动分析,分析结果随记忆一起持久化,后续搜索时能利用这些结果.

**目标**: 将 `analyze_memory_code()` 的返回结果持久化到记忆的 `metadata.code_analysis` 字段,上传代码记忆时自动触发分析.

**涉及范围**:

- `wrapper/src/utils/memory_manager.py`:
  - `analyze_memory_code()` 方法: 分析后将结果写入 `metadata.code_analysis`
  - `create_memory()` 方法: 上传时如果检测到代码,自动触发分析（可选）
  - 新增 `_is_code_content()` 方法: 判断内容是否为代码（复用 `_detect_programming_language()`）
- `wrapper/src/main.py`:
  - `POST /api/v1/memories/{memory_id}/analyze/code`: 返回持久化后的结果
  - `POST /api/v1/memories`: 请求体添加 `auto_analyze_code` 选项（默认 `true`）
- `wrapper/src/utils/code_analyzer.py`:
  - `CodeAnalysisResult` 数据类: 新增 `analyzed_at` 和 `analyzer_version` 字段
- `wrapper/src/config.py`:
  - `AppConfig` 新增 `code_analysis` 配置节（`auto_analyze: bool = True`）

**前置依赖**: B-027（设计文档修复后实施更准确）

**完成标准**:

- `POST /api/v1/memories/{memory_id}/analyze/code` 返回分析结果**并写入** `metadata.code_analysis`
- `POST /api/v1/memories` 上传代码记忆时,`auto_analyze_code=true` 自动触发分析并持久化
- 分析结果包含: `language`, `functions`, `classes`, `imports`, `complexity`, `analyzed_at`, `analyzer_version`
- 分析失败**不影响上传**（只记录警告,`metadata.code_analysis` 为 `null`）
- 现有 32 个同步测试全部通过
- Pyright 类型检查 0 errors

- 新增单元测试覆盖: 自动分析触发、手动分析触发、分析失败降级

**验证方式**:

```bash
# 单元测试
uv run pytest tests/test_code_analysis_persistence.py -v

# 回归测试
uv run pytest tests/test_phase_b_sync.py -v  # 32/32 通过

# E2E 测试
uv run python scripts/comprehensive_api_test.py

# 类型检查
uv run pyright
```

---

### B-029: Meilisearch 代码分析字段索引（Phase B: 搜索增强）

**真实场景**: 用户搜索 "authentication" 相关代码.当前 Meilisearch 只索引记忆的 content/embedding,搜索结果无法区分代码和普通文本.增强后,用户可以按语言、函数名、复杂度等过滤搜索结果.例如: "找出所有 Python 函数中复杂度大于 5 的代码".

**目标**: 将代码分析结果同步到 Meilisearch 索引,支持按代码属性过滤搜索.
**涉及范围**:

- `wrapper/src/utils/memory_manager.py`:
  - `_build_meili_doc()`: 构建 Meilisearch 文档时,添加 `code_analysis` 字段
  - `create_memory()` / `_update_memory()`: 同步时携带代码分析字段
- `wrapper/src/utils/meili_client.py`:
  - Meilisearch 索引设置: 添加 `code_language`、`code_functions`、`code_complexity` 可过滤/可搜索字段
- Meilisearch 索引 schema 更新（需重新初始化索引或添加新字段）
**前置依赖**: B-028（需要先有代码分析数据）
**完成标准**:
- Meilisearch 文档包含 `code_analysis` 字段（`language`, `function_names`, `class_names`, `max_complexity`）
- 支持按 `code_language` 过滤: `filter = "code_language = python"`
- 支持按 `code_functions` 搜索: 搜索函数名
- 支持按 `code_complexity` 排序: 按复杂度降序
- 搜索 API 添加 `code_filter` 参数: `POST /api/v1/memories/search` 支持 `code_filter: {language: "python", min_complexity: 5}`
- 现有搜索功能不受影响（向后兼容）
**验证方式**:

```bash
# Meilisearch 字段验证
curl -X GET "http://localhost:18003/indexes/memories/settings" \
  -H "Authorization: Bearer masterKey"

# 代码搜索测试
curl -X POST http://localhost:17999/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication", "code_filter": {"language": "python"}}'

# 单元测试
uv run pytest tests/test_code_search.py -v
```

---

### B-030: LLM 代码摘要生成（Phase C: LLM 增强）

**真实场景**: 用户上传了一段 200 行的 Python 代码.分析结果显示有 5 个函数、2 个类,但没有自然语言描述.用户希望能看到"这个模块实现了用户认证功能,包含登录、登出、令牌验证等核心逻辑"这样的摘要,而不是一堆函数名.
**目标**: 调用外部 LLM API 为代码记忆生成自然语言摘要,存入 `metadata.code_summary`.
**涉及范围**:

- `wrapper/src/config.py`:
  - 新增 `LLMConfig` 数据类: `endpoint`, `api_key`, `model_name`, `max_tokens`
  - 环境变量: `WRAPPER_LLM_ENDPOINT`, `WRAPPER_LLM_API_KEY`, `WRAPPER_LLM_MODEL`
- `wrapper/src/utils/memory_manager.py`:
  - 新增 `async _generate_code_summary()` 方法: 调用 LLM API 生成摘要
  - `create_memory()`: 上传代码后异步触发摘要生成（不阻塞上传）
  - `metadata.code_summary` 字段: `{summary, key_functions, purpose, generated_at, model}`
- `wrapper/src/main.py`:
  - 新增 `POST /api/v1/memories/{memory_id}/enrich/llm`: 手动触发 LLM 摘要生成
  - 新增 `GET /api/v1/memories/{memory_id}/summary`: 获取代码摘要
**前置依赖**: B-028（需要先有代码分析结果作为 LLM 输入）
**完成标准**:
- `POST /api/v1/memories` 上传代码后异步触发 LLM 摘要（不阻塞上传响应）
- LLM 调用失败不影响上传（只记录警告）
- 摘要结果存入 `metadata.code_summary`: `{summary, key_functions, purpose, generated_at, model}`
- 支持 Kaggle LLM 或其他 OpenAI 兼容 API（通过配置切换）
- `POST /api/v1/memories/{memory_id}/enrich/llm` 支持手动触发
- `GET /api/v1/memories/{memory_id}/summary` 返回摘要
**验证方式**:

```bash
# 手动触发 LLM 摘要
curl -X POST http://localhost:17999/api/v1/memories/{memory_id}/enrich/llm \
  -H "Content-Type: application/json" \
  -d '{"type": "summary"}'

# 获取摘要
curl http://localhost:17999/api/v1/memories/{memory_id}/summary

# 单元测试（mock LLM 调用）
uv run pytest tests/test_llm_summary.py -v
```

---

### B-031: 跨文件关系解析（Phase D: 远期考虑）

**真实场景**: 用户上传了一个项目中的多个文件.文件 A 导入了文件 B 的函数,文件 C 调用了文件 A 的类.当前系统无法感知这些跨文件关系.用户希望看到"谁调用了 `validate_user`"或"这个函数被哪些文件依赖".
**⚠️ 可行性评估**: 记忆级代码分析的核心限制是**上传顺序不确定**——文件 A 上传时文件 B 可能还不存在,关系无法立即建立.需要**延迟解析**策略:上传时标记 import,后续批量关联.
**目标**: 解析代码记忆间的 import/call 关系,存入 SurrealDB `relation` 表,带置信度评分.
**涉及范围**:

- `wrapper/src/utils/code_analyzer.py`:
  - 新增 `resolve_imports()` 方法: 解析 import 语句,提取目标文件路径
  - 新增 `resolve_calls()` 方法: 解析函数调用,提取被调用函数名
- `wrapper/src/utils/memory_manager.py`:
  - 新增 `async _resolve_code_relations()` 方法: 在已有记忆中搜索匹配的 import/call 目标
  - 延迟解析: 上传时标记 import,定时或手动触发批量关联
  - `create_relation()` 复用: 关系类型新增 `IMPORTS`, `CALLS`, `EXTENDS`, `IMPLEMENTS`
  - 所有关系带 `confidence` (0.0-1.0) 和 `reason` 字段（复用 `weight` 字段）
- `wrapper/src/main.py`:
  - 新增 `POST /api/v1/memories/{memory_id}/resolve-relations`: 手动触发关系解析
  - 新增 `GET /api/v1/memories/{memory_id}/knowledge-graph`: 获取代码知识图谱
**前置依赖**: B-028（需要先有代码分析结果）, B-029（需要 Meilisearch 索引来搜索目标文件）
**完成标准**:
- 上传代码后自动提取 import 语句和函数调用
- `POST /api/v1/memories/{memory_id}/resolve-relations` 在已有记忆中搜索匹配目标,建立关系
- 关系带置信度: 同文件 1.0, import 解析 0.85, 模糊匹配 0.5
- `GET /api/v1/memories/{memory_id}/knowledge-graph` 返回 N 跳关系图
- 延迟解析策略: 上传时标记,批量关联时建立关系
**验证方式**:

```bash
# 上传两个有依赖关系的文件,然后解析关系
curl -X POST http://localhost:17999/api/v1/memories/{memory_id}/resolve-relations

# 查看知识图谱
curl http://localhost:17999/api/v1/memories/{memory_id}/knowledge-graph?depth=2

# 单元测试
uv run pytest tests/test_code_relations.py -v
```

---

### B-032: 插件端代码分析工具（Phase E: 插件集成）

**真实场景**: AI agent 通过 OpenCode 插件与后端交互.当前插件有 `memory_write`/`memory_read`/`memory_search` 等工具,但没有代码分析相关的工具.Agent 想要"分析这段代码"或"查看这个函数的依赖关系",但没有对应的工具可以调用.
**目标**: 在 OpenCode 插件中注册代码分析相关工具,与后端新增 API 对齐.
**涉及范围**:

- 插件端（TypeScript）:
  - 新增 `memory_code_analyze` 工具: 调用 `POST /api/v1/memories/{id}/analyze/code`
  - 新增 `memory_code_graph` 工具: 调用 `GET /api/v1/memories/{id}/knowledge-graph`
  - 新增 `memory_code_impact` 工具: 调用 `POST /api/v1/memories/{id}/impact`（远期）
  - 新增 `memory_code_processes` 工具: 调用 `GET /api/v1/memories/{id}/processes`（远期）
  - 新增 `memory_code_enrich` 工具: 调用 `POST /api/v1/memories/{id}/enrich/llm`
  - 新增 `memory_code_wiki` 工具: 调用 `POST /api/v1/memories/{id}/wiki`（远期）
  - 新增 `memory_code_search` 工具: 调用 `POST /api/v1/memories/search` 带 `code_filter`
- `memory-config.json`:
  - 新增 `code_analysis` 配置节: `auto_analyze`, `auto_enrich_llm`
**前置依赖**: B-028 ~ B-031（后端 API 就绪后才能注册工具）
**完成标准**:
- 7 个新工具注册成功,LLM 可通过工具名调用
- 每个工具有完整的 `description` 和 `parameters` JSON Schema
- 工具注册代码基于 `@opencode-ai/plugin` v1.3.3 的 `Hooks.tool` 类型
- `memory-config.json` 包含 `code_analysis` 配置
**验证方式**:
- 在 OpenCode 中调用 `memory_code_analyze` 工具,验证返回分析结果
- 在 OpenCode 中调用 `memory_code_search` 工具带 `code_filter`,验证过滤生效
- 检查 `memory-config.json` 格式正确

## v2.5.0 - Markdown 质量门禁 (Markdownlint)

> 目标：引入 markdownlint-cli2 保证文档质量，使用 pre-commit 纯净管理（不引入 package.json）。

### BL-MD-01: 建立 Markdown 检查基础配置 ✅ 已完成

**目标**: 在根目录引入 `.markdownlint-cli2.jsonc` 配置文件。

| 项目 | 内容 |
|------|------|
| **涉及范围** | `.markdownlint-cli2.jsonc` |
| **完成标准** | 规则对齐插件端（禁用 MD013/024/033/034/036/040/041/051），配置 `ignores` 忽略三方和生成目录。 |
| **验证方式** | 配置文件存在且格式正确。 |

### BL-MD-02: 集成 Pre-commit 质量门禁 ✅ 已完成

**目标**: 将 markdownlint-cli2 挂载到 Git 提交流程中，由 pre-commit 环境管理。

| 项目 | 内容 |
|------|------|
| **涉及范围** | `.pre-commit-config.yaml` |
| **完成标准** | 添加 DavidAnson/markdownlint-cli2 的 hook 配置，作为后续流程的守护者。 |
| **验证方式** | 运行 `uvx pre-commit run markdownlint-cli2 --all-files` 能够正确拉取镜像并执行。 |

### BL-MD-03: 全局存量 Markdown 错误修复 ✅ 已完成

**目标**: 使现有文档通过新的质量门禁。

| 项目 | 内容 |
|------|------|
| **涉及范围** | 项目中现存的所有 `.md` 文件（如 README.md, AGENTS.md, ROADMAP.md 等）。 |
| **完成标准** | 执行全量扫描无任何报错。 |
| **验证方式** | `uvx pre-commit run markdownlint-cli2 --all-files` 结果为 Passed (exit code 0)。 |

### BL-MD-04: 文档体系同步更新 ✅ 已完成

**目标**: 将产品文档、开发文档和 Backlog 状态进行最终更新。

| 项目 | 内容 |
|------|------|
| **涉及范围** | `README.md`, `AGENTS.md`, `BACKLOG.md` |
| **完成标准** | 1. `README.md` 包含"文档规范"及执行命令（`uvx pre-commit run markdownlint-cli2 --all-files`）。<br>2. `AGENTS.md` 包含底层逻辑、工具链选择（Marksman vs CLI）及被忽略规则原因。<br>3. `BACKLOG.md` 标记完成。 |
| **验证方式** | 文档更新内容符合预期并完成提交。 |
