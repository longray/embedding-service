# Backlog

> 后端任务追踪文档，按真实场景驱动排序。已完成任务归档至 `backlog_archive.md`。

**更新时间**: 2026-04-07

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
| BL-CA-20 | 实现调用关系存储 API | 📋 | BL-CA-18 | `POST /api/v1/memories/relations` 扩展 `relationship_type="calls"`，metadata 含 `line`, `column`, `file_path`（v1.4 §4.1 CallRelation） |
| BL-CA-21 | 实现引用查询 API | 📋 | BL-CA-20 | `GET /api/v1/memories/{id}/references` — 查询谁调用了该符号，返回 `memory_id`, `file_path`, `line`, `caller_function`, `confidence`（v1.4 §4.2） |
| BL-CA-22 | 实现依赖分析 API | 📋 | BL-CA-20 | `GET /api/v1/memories/{id}/dependencies` — 查询该符号依赖了谁，返回 `memory_id`, `file_path`, `line`, `callee_function`, `type` (internal/external/builtin)（v1.4 §4.2） |

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

*最后更新: 2026-04-07（Scene 9 清理：仅保留后端任务）*
