# Backlog

> 后端任务追踪文档，按真实场景驱动排序。已完成任务归档至 `backlog_archive.md`。

**更新时间**: 2026-04-08

---

## 场景 1: 记忆上传与搜索

> **用户流程**: AI 对话 → 插件识别重要信息 → `POST /api/v1/memories` → 后端存储 → 用户提问 → `POST /api/v1/memories/search` → 返回相关记忆
>
> **当前状态**: ✅ 核心流程完成。已完成任务归档至 `backlog_archive.md`。

---

## 场景 2: 代码分析

> **用户流程**: 用户编辑代码 → 插件 Tree-sitter 解析 → `POST /api/v1/memories` (type="code") → 后端存储 + 自动分析 → 用户搜索函数名 → 返回匹配代码
>
> **当前状态**: ✅ 上传、LLM 摘要、自动代码分析均可用。已完成任务归档至 `backlog_archive.md`。

---

## 场景 3: 多设备同步

> **用户流程**: 插件端有本地记忆文件 → `GET /api/v1/sync/fingerprints` 获取服务端指纹 → `POST /api/v1/sync/preview` 比对差异 → 用户确认 → `POST /api/v1/sync/full` 执行同步 → 冲突时解决
>
> **当前状态**: ✅ 已完成。BL-29~32 全部实现，32 个 mock 测试全部通过。已完成任务归档至 `backlog_archive.md`。
>
> **产品文档**: `docs/product-sync-v2.7.md`
> **开发文档**: `docs/dev-sync-v2.7.md`

---

## 场景 4: 测试架构优化

> **用户流程**: 开发者 `git commit` → pre-commit 秒级跑 unit 测试 → 推送前手动跑 unit+integration → CI 跑全量（含 e2e）
>
> **当前状态**: ✅ 已完成（BL-T1~T9, T11）。已完成任务归档至 `backlog_archive.md`。
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

### 待开始

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
├── BL-CA-05 max_complexity 支持 ────────────► ✅
├── BL-CA-06 v1.2 文档修复 ─────────────────► ✅
├── BL-CA-09 集成测试补充 ──────────────────► ✅
└── BL-CA-10 API 文档更新 ──────────────────► ✅

测试架构优化 — 全部完成 ✅
├── BL-T1 测试分层标记 ─────────────────────► ✅
├── BL-T2 fixture scope 优化 ────────────────► ✅
├── BL-T3 修复 Mixin mock 断言 ─────────────► ✅
├── BL-T4 修复接口变更断言 ─────────────────► ✅
├── BL-T8 conftest 端口 + Event loop 回归 ──► ✅
├── BL-T9 LLM/SERVICE_DOWN 条件跳过 ────────► ✅
├── BL-T11 wrapper 接口变更修复 ───────────► ✅
├── BL-T5 清理无效文件 ─────────────────────► ✅
├── BL-T7 合并小型文件 ─────────────────────► ✅
├── BL-T6 pre-commit 配置 ──────────────────► ✅
└── BL-T10 语义去重阈值修复 ───────────────► 📋 P3（可选）

v2.7.0 发布准备
├── BL-D3 BACKLOG 清理 ──────────────────► ✅
├── BL-D4 过时文档更新 ──────────────────► ✅
├── BL-D5 CHANGELOG 更新 ───────────────► ✅
└── BL-D6 E2E 完整验证 ─────────────────► 📋 待开始

代码分析上传修复 — 全部完成 ✅（2026-04-08）
├── BL-CA-FIX-01 跳过去重 ───────────────► ✅
├── BL-CA-FIX-02 新文件插入 ─────────────► ✅
├── BL-CA-FIX-03 file_path 查询语法 ─────► ✅
├── BL-CA-FIX-04 字段名统一 ─────────────► ✅
├── BL-CA-FIX-05 Schema 合并 ────────────► ✅
└── BL-CA-FIX-06 验证脚本修复 ───────────► ✅

---

## 场景 7: 系统可观测性与开发者工具

> **用户场景**: 运维人员部署服务后，需要监控运行状态、诊断性能问题、快速定位故障。
>
> **当前状态**: 核心功能已实现。已完成任务归档至 `backlog_archive.md`。

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

## 场景 9: 代码分析增强（后端任务）

> **用户场景**: 开发者需要深度代码理解、跨文件引用追踪、项目级代码地图。
>
> **设计文档**: `docs/CODE-ANALYSIS-DESIGN-v1.4.md`
>
> **插件端任务** (BL-CA-11~17, 19): AST 解析增强（函数完整字段、类成员、接口、导入导出、依赖分类、复杂度、调用提取）— 已移至插件仓库。
>
> **本场景仅含后端任务**。

### Phase 1: 存储与 Schema 扩展（1-2 周）— P0

| 编号 | 目标 | 状态 | 依赖 | 说明 |
|------|------|------|------|------|
| BL-CA-18 | 后端存储 schema 扩展 | 🔄 | 无 | Meilisearch 新增 `code_has_exports` filterable 字段；SurrealDB `memory_relation` 表已支持 `calls` 类型 |

### Phase 2: 调用关系与引用追踪 API（2-3 周）— P1

| 编号 | 目标 | 状态 | 依赖 | 说明 |
|------|------|------|------|------|
| BL-CA-20 | 实现调用关系存储 API | 🔄 | BL-CA-18 | `POST /api/v1/calls/batch` — 批量创建调用关系，最大 100 条/批次，支持错误列表返回。✅ 已修复：支持 `relationship_type: "calls"` |
| BL-CA-21 | 实现引用查询 API | 🔄 | BL-CA-20 | `GET /api/v1/memories/{id}/references` — 查询谁调用了该符号，返回 `memory_id`, `file_path`, `line`, `caller_function`, `confidence` |
| BL-CA-22 | 实现依赖分析 API | 🔄 | BL-CA-20 | `GET /api/v1/memories/{id}/dependencies` — 查询该符号依赖了谁，返回 `memory_id`, `file_path`, `line`, `callee_function`, `type` (internal/external/builtin) |

### Phase 3: 代码地图与搜索增强（2-3 周）— P1

| 编号 | 目标 | 状态 | 依赖 | 说明 |
|------|------|------|------|------|
| BL-CA-23 | 实现代码地图 API | 🔄 | BL-CA-18 | `GET /api/v1/projects/{id}/map` — 返回 `file_tree`, `module_dependencies`, `hot_files`, `statistics` |
| BL-CA-24 | 实现代码搜索 API（增强） | 🔄 | BL-CA-18 | `code_filter` 扩展：新增 `min/max_function_count`, `min/max_class_count`, `has_exports`, `analyzer` 过滤条件 |
| BL-CA-25 | 实现代码统计 API | 🔄 | BL-CA-18 | `GET /api/v1/projects/{id}/stats` — 按 `project_id` 聚合 `total_files`, `total_functions`, `total_classes`, `avg_complexity`, `max_complexity` |

### Phase 4: 批量与增量分析（2-3 周）— P2

| 编号 | 目标 | 状态 | 依赖 | 说明 |
|------|------|------|------|------|
| BL-CA-26 | 实现批量分析 API | 📋 | BL-CA-18 | `POST /api/v1/memories/analyze/batch` — 接收多文件路径列表，批量触发代码分析，返回每文件分析结果 |
| BL-CA-27 | 实现增量分析 API | 📋 | BL-CA-26 | `POST /api/v1/memories/analyze/incremental` — 基于 `file_path` + `mtime` 判断是否需要重新分析，跳过未变更文件 |

### Phase 5: 分析基础设施（2-3 周）— P2

| 编号 | 目标 | 状态 | 依赖 | 说明 |
|------|------|------|------|------|
| BL-CA-28 | 实现分析结果缓存 | 🔄 | BL-CA-18 | `CodeAnalyzer` 添加 LRU 缓存（默认 100 条，TTL 3600s），避免重复解析相同代码 |
| BL-CA-29 | 实现分析任务队列 | 📋 | BL-CA-26 | 内存队列（max=10），并发控制（max=2），防抖 300ms，避免大项目批量分析压垮服务（v1.4 §1.1 AnalysisManager） |
| BL-CA-30 | 实现分析进度查询 | 📋 | BL-CA-29 | `GET /api/v1/memories/analyze/progress` — 返回队列状态：pending/running/done/failed，支持按 `project_id` 过滤 |

### Phase 6: 数据交换与验证（1-2 周）— P3

| 编号 | 目标 | 状态 | 依赖 | 说明 |
|------|------|------|------|------|
| BL-CA-31 | 实现分析结果导出 | 📋 | BL-CA-25 | `GET /api/v1/projects/{id}/export` — 导出项目全量分析数据为 JSON，含 `file_tree`, `statistics`, `call_graph` |
| BL-CA-32 | 实现分析结果导入 | 📋 | BL-CA-31 | `POST /api/v1/projects/{id}/import` — 批量导入外部分析结果，Upsert 已有记录 |
| BL-CA-33 | 集成测试与性能优化 | 📋 | 全部 | 端到端测试：上传→分析→引用查询→代码地图；性能基线：单文件分析 < 500ms，项目地图 < 2s |

### 依赖关系图

```text
BL-CA-18 (schema)
    ├── BL-CA-20 (调用关系存储)
    │       ├── BL-CA-21 (引用查询)
    │       └── BL-CA-22 (依赖分析)
    ├── BL-CA-23 (代码地图)
    ├── BL-CA-24 (搜索增强)
    ├── BL-CA-25 (代码统计)
    ├── BL-CA-26 (批量分析)
    │       ├── BL-CA-27 (增量分析)
    │       └── BL-CA-29 (任务队列)
    │               └── BL-CA-30 (进度查询)
    └── BL-CA-28 (结果缓存)

BL-CA-25 ──► BL-CA-31 (导出) ──► BL-CA-32 (导入)

全部 ──► BL-CA-33 (集成测试)
```

---

## 场景 10: 代码分析数据上传修复（2026-04-08 紧急修复）

> **用户场景**: 插件端上传代码分析数据，后端返回成功但数据无法查询。
>
> **当前状态**: ✅ 已完成

### 已修复问题

| 编号 | 问题 | 修复内容 | 状态 |
|------|------|---------|------|
| BL-CA-FIX-01 | 代码数据被 hash 去重跳过 | 添加 `type: "code"` 跳过去重逻辑 | ✅ |
| BL-CA-FIX-02 | 新代码文件未写入 | 修复 BL-CA-08 逻辑，新文件进入批量插入 | ✅ |
| BL-CA-FIX-03 | 相同 file_path 不更新 | 修复 SurrealDB 语法 `metadata->file_path` → `metadata.file_path` | ✅ |
| BL-CA-FIX-04 | 字段名不一致 | 统一使用 `abstract` 和 `overview`（代码、模型、数据库一致） | ✅ |
| BL-CA-FIX-05 | Schema 未初始化 | 将 `abstract`/`overview` 字段定义合并到主 schema | ✅ |
| BL-CA-FIX-06 | 验证脚本错误 | 修复 SurrealDB 3.0 兼容性（`INFO FOR DB` 替代 `SELECT name FROM tables`） | ✅ |
| BL-CA-FIX-07 | 项目地图 module_dependencies 为空 | 添加 `_extract_call_dependencies` 方法，使用 `type::record()` 转换 | ✅ |
| BL-CA-FIX-08 | 模块依赖重复 | 添加去重逻辑，使用 `(from_path, to_path)` 作为 key | ✅ |
| BL-CA-FIX-09 | 项目地图查询返回空 | 移除 `metadata.code_analysis IS NOT NONE` 查询条件 | ✅ |

### 修复文件

- `wrapper/src/utils/memory_manager/crud.py` - 去重逻辑、字段名、UPDATE 语句
- `wrapper/src/utils/memory_manager/search.py` - 查询字段名
- `wrapper/src/utils/memory_manager/relations.py` - 查询字段名
- `wrapper/src/utils/memory_manager/meili_sync.py` - 字段名
- `scripts/init_surrealdb.surql` - 添加 `abstract`/`overview` 字段定义
- `scripts/init_database.py` - 修复验证逻辑

### 验证结果

```bash
# 测试命令
uv run python test_plugin_format.py

# 结果
✅ 上传成功
✅ 查询成功
✅ 字段名统一
```

### 待优化项（非阻塞）

#### BL-CA-OPT-01: RELATE 语句 SQL 注入风险防护

| 属性 | 内容 |
|------|------|
| **目标** | 消除 `relations.py:create_relation` 中的 SQL 注入风险，同时保持 RELATE 语句的功能完整性 |
| **涉及范围** | `wrapper/src/utils/memory_manager/relations.py` 的 `create_relation` 方法 |
| **前置依赖** | 无（独立任务） |
| **完成标准** | 1. 所有动态值使用参数绑定<br>2. RecordID 格式经过验证<br>3. 通过安全测试（尝试注入失败） |
| **验证方式** | 1. 代码审查：确认无字符串拼接<br>2. 单元测试：验证异常输入被拦截<br>3. 集成测试：关系创建功能正常 |
| **优先级** | P1 |
| **预计工作量** | 2h |
| **方案** | 部分参数化 + 输入验证（非完全重构） |

---

#### BL-CA-OPT-02: RecordID 格式统一

| 属性 | 内容 |
|------|------|
| **目标** | 统一项目中所有 RecordID 的处理方式，消除格式不一致导致的查询失败 |
| **涉及范围** | 全项目 SQL 查询，重点文件：<br>- `wrapper/src/utils/memory_manager/stubs.py`<br>- `wrapper/src/utils/memory_manager/relations.py`<br>- `wrapper/src/utils/memory_manager/crud.py`<br>- `wrapper/src/utils/memory_manager/search.py` |
| **前置依赖** | 无（独立任务） |
| **完成标准** | 1. 所有查询使用 `type::record()` 函数<br>2. 数组查询使用 `array::map()` 转换<br>3. 无直接字符串拼接 RecordID |
| **验证方式** | 1. 全局搜索：确认无 `f"...{id}"` 模式<br>2. 运行测试：`test_project_map.py` 通过<br>3. 运行测试：`test_data_persistence.py` 通过 |
| **优先级** | P0 |
| **预计工作量** | 4h |
| **方案** | 统一使用 `type::record()`，添加格式验证辅助函数 |

---

#### BL-CA-OPT-03: 嵌套字段查询性能优化

| 属性 | 内容 |
|------|------|
| **目标** | 优化 `metadata.file_path` 等嵌套字段的查询性能，避免全表扫描 |
| **涉及范围** | `wrapper/src/utils/memory_manager/stubs.py` 的项目地图查询<br>`scripts/init_surrealdb.surql` 的索引定义 |
| **前置依赖** | BL-CA-OPT-02（RecordID 统一） |
| **完成标准** | 1. 添加复合索引覆盖常用查询<br>2. 查询性能提升 > 50%<br>3. 不修改现有 schema（避免数据迁移） |
| **验证方式** | 1. EXPLAIN 分析：确认使用索引<br>2. 性能测试：项目地图查询 < 500ms<br>3. 功能测试：查询结果正确 |
| **优先级** | P2 |
| **预计工作量** | 2h |
| **方案** | 添加复合索引，暂缓 schema 变更 |

---

#### BL-CA-OPT-04: 批量插入分批处理

| 属性 | 内容 |
|------|------|
| **目标** | 优化大批量记忆上传的性能和稳定性，避免单次插入数据过大导致超时或内存问题 |
| **涉及范围** | `wrapper/src/utils/memory_manager/crud.py` 的 `upload_memories` 方法中的批量插入逻辑 |
| **前置依赖** | 无（独立任务） |
| **完成标准** | 1. 批量数据分批处理（50条/批）<br>2. 单批失败不影响其他批次<br>3. 总性能不下降（允许轻微 overhead） |
| **验证方式** | 1. 单元测试：120条数据上传成功<br>2. 性能测试：总时间 < 单批插入 × 批次<br>3. 错误测试：中间批次失败，前后批次成功 |
| **优先级** | P1 |
| **预计工作量** | 2h |
| **方案** | 分批处理（50条/批），保持现有失败回退机制 |
| **状态** | ✅ 已完成 |

---

#### BL-CA-OPT-05: SQL 查询规范文档

| 属性 | 内容 |
|------|------|
| **目标** | 建立 SurrealDB SQL 开发规范，防止未来出现类似问题，统一团队开发标准 |
| **涉及范围** | 文档：`docs/dev/SURREALDB_SQL_BEST_PRACTICES.md` |
| **前置依赖** | BL-CA-OPT-01, BL-CA-OPT-02, BL-CA-OPT-04, BL-CA-OPT-06 完成（基于实践经验） |
| **完成标准** | 1. 文档包含 RecordID 处理规范<br>2. 文档包含参数绑定最佳实践<br>3. 文档包含性能优化建议<br>4. 文档包含常见陷阱和解决方案<br>5. 团队评审通过 |
| **验证方式** | 1. 文档评审：开发团队确认<br>2. 实践验证：后续开发遵循规范 |
| **优先级** | P2 |
| **预计工作量** | 2h |
| **方案** | 编写开发规范文档，包含示例代码 |
| **状态** | ✅ 已完成 |

---

#### BL-CA-OPT-06: Meilisearch 同步分批

| 属性 | 内容 |
|------|------|
| **目标** | 解决大批量数据上传时 Meilisearch 同步导致的超时问题，将同步操作改为分批处理 |
| **涉及范围** | `wrapper/src/utils/memory_manager/crud.py` 的 `upload_memories` 方法中的 Meilisearch 同步部分 |
| **前置依赖** | BL-CA-OPT-04（SurrealDB 分批插入已完成，验证分批逻辑可行） |
| **完成标准** | 1. Meilisearch 同步改为 50条/批处理<br>2. 单批失败记录错误但不影响其他批次<br>3. 所有批次完成后统一返回结果<br>4. 保持现有 API 响应格式不变 |
| **验证方式** | 1. 120条数据上传测试：总耗时 < 30s，不超时<br>2. 检查日志：确认分批同步日志输出<br>3. Meilisearch 查询：确认所有文档已同步 |
| **优先级** | P1 |
| **预计工作量** | 2h |
| **方案** | 参照 SurrealDB 分批逻辑，将 `meili_docs` 按 50条/批分批同步 |
| **状态** | ✅ 已完成 |

---

#### BL-CA-OPT-07: 大批量上传异步化（架构优化）

| 属性 | 内容 |
|------|------|
| **目标** | 支持任意规模的数据上传（10000+ 条），彻底解决同步阻塞问题，提供任务进度查询和断点续传能力 |
| **涉及范围** | **API 层**：<br>- `wrapper/src/routers/memories.py`：新增异步上传接口<br>- 新增 `wrapper/src/routers/tasks.py`：任务状态查询接口<br><br>**任务层**：<br>- `wrapper/src/utils/task_manager.py`：任务队列管理（新建文件）<br>- 任务状态：pending/running/completed/failed<br><br>**存储层**：<br>- 临时数据存储（内存或 Redis）<br>- 任务元数据存储<br><br>**处理层**：<br>- 后台 worker 处理任务<br>- 分批处理：embedding → SurrealDB → Meilisearch |
| **前置依赖** | BL-CA-OPT-06（Meilisearch 分批完成，验证分批逻辑稳定） |
| **完成标准** | 1. 新增 `POST /api/v1/memories/async` 接口，返回 task_id<br>2. 新增 `GET /api/v1/tasks/{task_id}` 接口，返回进度和状态<br>3. 支持 10000+ 条数据上传不超时<br>4. 支持断点续传（任务可恢复）<br>5. 支持失败重试（最多 3 次） |
| **验证方式** | 1. 10000条数据上传测试：创建任务成功，后台处理完成<br>2. 任务状态查询测试：各阶段状态正确<br>3. 断点续传测试：中断后恢复，数据完整<br>4. 失败重试测试：模拟失败，验证重试机制 |
| **优先级** | P2（架构优化，非紧急） |
| **预计工作量** | 16h+ |
| **方案** | 异步任务队列 + 后台 worker + 状态管理 + 断点续传 |
| **状态** | 📋 规划中 |

---

#### BL-CA-OPT-08: 查询接口 embedding 字段优化

| 属性 | 内容 |
|------|------|
| **目标** | 减少查询响应体积，默认不返回 embedding 向量，需要时显式指定，提升查询性能和用户体验 |
| **涉及范围** | `wrapper/src/routers/memories.py`：<br>- `GET /api/v1/memories/{id}` 接口<br>- 添加 `include_embedding` 查询参数<br><br>`wrapper/src/utils/memory_manager/memories.py`：<br>- `get_memory` 方法修改<br>- 根据参数决定是否查询 embedding 字段 |
| **前置依赖** | 无（独立任务） |
| **完成标准** | 1. 添加 `include_embedding` 查询参数（默认 false）<br>2. 默认不返回 embedding 字段<br>3. 显式设置 `include_embedding=true` 时返回完整 embedding<br>4. 向后兼容，不影响现有调用<br>5. 响应体积减少 90%+ |
| **验证方式** | 1. 默认查询测试：`GET /api/v1/memories/{id}` 不返回 embedding<br>2. 显式查询测试：`GET /api/v1/memories/{id}?include_embedding=true` 返回 embedding<br>3. 响应体积测试：对比优化前后响应大小<br>4. 性能测试：查询耗时减少<br>5. 向后兼容测试：现有客户端调用正常 |
| **优先级** | P1 |
| **预计工作量** | 1h |
| **方案** | 添加可选参数 `include_embedding`，默认 false，修改 SQL 查询语句 |
| **状态** | 📋 待执行 |

---

### 优化项执行状态汇总

| 编号 | 名称 | 优先级 | 状态 | 实际工作量 |
|------|------|--------|------|-----------|
| BL-CA-OPT-01 | RELATE SQL 注入防护 | P1 | ✅ 已完成 | 2h |
| BL-CA-OPT-02 | RecordID 格式统一 | P0 | ✅ 已完成 | 4h |
| BL-CA-OPT-03 | 嵌套字段查询优化 | P2 | ✅ 已完成 | 2h |
| BL-CA-OPT-04 | 批量插入分批处理 | P1 | ✅ 已完成 | 2h |
| BL-CA-OPT-05 | SQL 查询规范文档 | P2 | 📋 待执行 | 2h |
| BL-CA-OPT-06 | Meilisearch 同步分批 | P1 | ✅ 已完成 | 2h |
| BL-CA-OPT-07 | 大批量上传异步化 | P2 | 📋 规划中 | 16h+ |
| BL-CA-OPT-08 | embedding 字段优化 | P1 | 📋 待执行 | 1h |

**已完成**: 5/8 项（BL-CA-OPT-01~04, 06）  
**待执行**: 2/8 项（BL-CA-OPT-05, 08）  
**规划中**: 1/8 项（BL-CA-OPT-07）

---

### 批次大小统一配置

| 组件 | 批次大小 | 配置位置 |
|------|---------|---------|
| SurrealDB 插入 | **50条** | `crud.py:BATCH_SIZE` |
| Meilisearch 同步 | **50条** | `crud.py:MEILI_BATCH_SIZE` |

**统一批次大小的好处**：

1. 简化运维和调优
2. 更均匀的负载分布
3. 更好的内存管理
4. 便于后续异步化改造

|------|------|
| **目标** | 优化大批量记忆上传的性能和稳定性，避免单次插入数据过大 |
| **涉及范围** | `wrapper/src/utils/memory_manager/crud.py` 的 `upload_memories` 方法 |
| **前置依赖** | 无（独立任务） |
| **完成标准** | 1. 批量数据分批处理（100条/批）<br>2. 单批失败不影响其他批次<br>3. 总性能不下降（允许轻微 overhead） |
| **验证方式** | 1. 单元测试：1000条数据上传成功<br>2. 性能测试：总时间 < 单批插入 × 批次<br>3. 错误测试：中间批次失败，前后批次成功 |
| **优先级** | P2 |
| **预计工作量** | 2h |
| **方案** | 分批处理（100条/批） |

---

#### BL-CA-OPT-05: SQL 查询规范文档

| 属性 | 内容 |
|------|------|
| **目标** | 建立 SurrealDB SQL 开发规范，防止未来出现类似问题 |
| **涉及范围** | 文档：`docs/dev/SURREALDB_SQL_BEST_PRACTICES.md` |
| **前置依赖** | BL-CA-OPT-01, BL-CA-OPT-02 完成（基于实践经验） |
| **完成标准** | 1. 文档包含 RecordID 处理规范<br>2. 文档包含参数绑定最佳实践<br>3. 文档包含性能优化建议<br>4. 团队评审通过 |
| **验证方式** | 1. 文档评审：开发团队确认<br>2. 实践验证：后续开发遵循规范 |
| **优先级** | P2 |
| **预计工作量** | 2h |
| **方案** | 编写开发规范文档 |

---

### 优化项依赖关系

```text
BL-CA-OPT-02 (RecordID 统一) ──┬──► BL-CA-OPT-03 (嵌套字段优化)
                              │
BL-CA-OPT-01 (SQL 注入防护) ───┤
                              │
BL-CA-OPT-04 (批量插入) ───────┤
                              │
                              └──► BL-CA-OPT-05 (SQL 规范文档)
```

### 执行建议顺序

1. **P0 - 立即执行**: BL-CA-OPT-02 (RecordID 统一)
2. **P1 - 本周执行**: BL-CA-OPT-01 (SQL 注入防护)
3. **P2 - 下周执行**: BL-CA-OPT-03 (嵌套字段优化) → BL-CA-OPT-04 (批量插入)
4. **P2 - 文档**: BL-CA-OPT-05 (SQL 规范文档，基于前述经验)

---

*最后更新: 2026-04-08（Scene 10 更新：添加待优化项详细规范）*
