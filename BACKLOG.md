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

---

## 场景 3: 多设备同步

> **用户流程**: 插件端有本地记忆文件 → `GET /api/v1/sync/fingerprints` 获取服务端指纹 → `POST /api/v1/sync/preview` 比对差异 → 用户确认 → `POST /api/v1/sync/full` 执行同步 → 冲突时解决
>
> **当前状态**: 端点已注册，但 sync_preview/sync_full 仍是 stub（返回空结果）。19 个 sync 测试失败。

### 待修复

#### BL-29 [P2] 实现指纹查询 #scene3

**目标**: `GET /api/v1/sync/fingerprints` 返回服务端所有记忆的指纹列表。

**涉及范围**:

- `wrapper/src/utils/memory_manager/sync.py`:
  - `get_fingerprints()` (当前返回空列表 `[]`)

**前置依赖**: 无

**完成标准**:

- [ ] 查询 SurrealDB `SELECT id, content_hash, local_id, source_id, mtime FROM memory WHERE tenant_id = $tenant_id`
- [ ] 数据库有数据时返回非空列表
- [ ] `tests/test_phase_b_sync.py::TestSyncFingerprints` 相关测试通过

**验证方式**:

```bash
uv run pytest tests/test_phase_b_sync.py::TestSyncFingerprints -v --tb=short
```

**状态**: 📋 待开始

---

#### BL-30 [P2] 实现同步预览 #scene3

**目标**: `POST /api/v1/sync/preview` 比对本地与服务端指纹，返回 to_upload/to_delete/conflicts。

**前置依赖**: BL-29

**状态**: 📋 待开始

---

#### BL-31 [P2] 实现全量同步 #scene3

**目标**: `POST /api/v1/sync/full` 批量上传/更新/删除记忆。

**前置依赖**: BL-29

**状态**: 📋 待开始

---

#### BL-32 [P2] 实现冲突解决 #scene3

**目标**: `POST /api/v1/sync/conflicts/{id}/resolve` 支持 USE_LOCAL/USE_BACKEND/KEEP_BOTH。

**前置依赖**: BL-30

**状态**: 📋 待开始

---

## 场景 4: 开发者体验（v2.6.0 质量治理）

> **目标**: 代码质量、模块化、文档一致性。
>
> **当前状态**: 全部完成。

### 已完成

| 编号 | 目标 | Commit |
|------|------|--------|
| BL-28 | analyze_memory_code 实现 | afdf896 |
| BL-33 | pyproject.toml 修复 | afdf896 |
| BL-34 | meilisearch_code/ Pyright 类型修复 | afdf896 |
| BL-35 | memory_manager.py 1715行 → Mixin 10 子模块 | 44423c6 |
| BL-36 | main.py 1063行 → routers/ 12 模块 | 2fec8ff |
| BL-37 | utils 单元测试 (35 cases) | 214567c |
| BL-38 | 移除硬编码 API Key | afdf896 |
| BL-39 | scripts/ 裸 except 清理 | afdf896 |
| BL-D1 | 归档 29 个过时文档 + 23 个 JSON 报告 | 9b15585 |
| BL-D2 | CHANGELOG/README/AGENTS.md 对齐 | 2b5c69d |

---

## 暂缓任务

| 编号 | 目标 | 原因 |
|------|------|------|
| BL-5 [P2] Meilisearch 代码分析字段索引 | 依赖 BL-28（已完成，可启动） |
| BL-7 [P3] 跨文件关系解析 | 记忆级输入顺序不确定 |
| BL-8 [P3] 插件端代码分析工具 | 需插件端配合 |
| BL-1 [P2] Tenant ID 不匹配 | 需插件端配合 |

## 可无限期推迟（无用户场景）

以下 11 个端点已注册但无调用方，当前返回 500（NotImplementedError 被 exception handler 统一处理）：

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

## 执行路线图

```
v2.6.0 质量治理 — 全部完成 ✅
├── BL-33/34 快速修复 ───────────────────────► ✅
├── BL-28 analyze_memory_code ───────────────► ✅
├── BL-35 memory_manager 拆分 ────────────────► ✅
├── BL-36 main.py 路由拆分 ──────────────────► ✅
├── BL-37 工具模块单元测试 ──────────────────► ✅
├── BL-38/39 清理 ───────────────────────────► ✅
├── BL-D1 归档 ─────────────────────────────► ✅
└── BL-D2 文档对齐 ─────────────────────────► ✅

下一阶段 — 多设备同步（P2，约 4-6 小时）
├── BL-29 指纹查询 ─────────────────────────► 📋 待开始
├── BL-30 同步预览 ─────────────────────────► 📋 待开始（依赖 BL-29）
├── BL-31 全量同步 ─────────────────────────► 📋 待开始（依赖 BL-29）
└── BL-32 冲突解决 ─────────────────────────► 📋 待开始（依赖 BL-30）
```

---

*最后更新: 2026-04-03*
