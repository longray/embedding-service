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

下一阶段 — 多设备同步 v2.7.0（P2，约 4-6 小时）
├── BL-29 指纹查询 ─────────────────────────► 📋 待开始
├── BL-30 同步预览 ─────────────────────────► 📋 待开始（依赖 BL-29）
├── BL-31 全量同步 ─────────────────────────► 📋 待开始（可与 BL-29 并行）
└── BL-32 冲突解决 ─────────────────────────► 📋 待开始（依赖 BL-30）
```

> **产品文档**: `docs/product-sync-v2.7.md`
> **开发文档**: `docs/dev-sync-v2.7.md`

---

*最后更新: 2026-04-03*
