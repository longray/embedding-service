# Backlog

> 后端任务追踪文档，按真实场景驱动排序。已完成任务归档至 `archive/docs/`。

**更新时间**: 2026-04-02

---

## 场景 1: 记忆上传与搜索

> **用户流程**: AI 对话 → 插件识别重要信息 → `POST /api/v1/memories` → 后端存储 → 用户提问 → `POST /api/v1/memories/search` → 返回相关记忆
>
> **当前状态**: 核心流程可用，87/87 核心 API 测试通过。

### 待修复

#### BL-25 [P1] 清理调试日志 #scene1

**目标**: 移除 `memory_manager.py` 中残留的调试日志，避免生产环境日志刷屏。

**涉及范围**:

- `wrapper/src/utils/memory_manager.py` 中 `[Meili Debug]` / `[_update_memory]` 等调试前缀日志

**前置依赖**: 无

**完成标准**:

- [ ] `rg "Meili Debug" wrapper/src/` 结果为 0
- [ ] `rg "\[_update_memory\]" wrapper/src/` 结果为 0（或改为正常业务日志）
- [ ] 核心测试通过

**验证方式**:

```bash
rg "Meili Debug" wrapper/src/
uv run pytest tests/test_wrapper_api.py -v --tb=short
```

**状态**: ✅ 已完成（代码中已无 Meili Debug 日志）

---

### 已完成

| 编号 | 目标 | 状态 |
|------|------|------|
| BL-18 | 修复测试用例适配 abstract/overview 必填 | ✅ 已完成 |
| BL-19 | 修复双写测试验证上传→Meilisearch 流程 | ✅ 已完成 |
| BL-20 | 端到端验证上传→搜索全链路（87/87） | ✅ 已完成 |
| BL-24 | 修复 get_memory_summary 连接泄露 | ✅ 已完成 |
| BL-25 | 清理调试日志 | ✅ 已完成 |
| BL-26 | 实现智能去重决策（替代硬编码 KEEP_BOTH） | ✅ 已完成 |
| BL-27 | 实现 _update_memory（SurrealDB UPDATE + Meilisearch 同步） | ✅ 已完成 |

---

## 场景 2: 代码分析

> **用户流程**: 用户编辑代码 → 插件 Tree-sitter 解析 → `POST /api/v1/memories` (type="code") → 后端存储 + 自动分析 → 用户搜索函数名 → 返回匹配代码
>
> **当前状态**: 上传和 LLM 摘要可用。`analyze_memory_code` 抛 NotImplementedError（被 try/except 吞掉，不影响上传但无分析结果）。

### 待修复

#### BL-28 [P1] 实现 analyze_memory_code #scene2

**目标**: 上传 type="code" 的记忆时，自动调用 CodeAnalyzer 获取分析结果，写入 `metadata.code_analysis`。

**涉及范围**:

- `wrapper/src/utils/memory_manager.py`:
  - `analyze_memory_code()` (当前第 1499 行，抛 NotImplementedError)
- `wrapper/src/utils/code_analyzer.py` (已有 `CodeAnalyzer.analyze()`，可直接调用)

**前置依赖**: 无（CodeAnalyzer 已就绪）

**完成标准**:

- [ ] `analyze_memory_code` 调用 `CodeAnalyzer.analyze()` 获取结果
- [ ] 分析结果写入 `metadata.code_analysis`（language, functions, classes, complexity 等）
- [ ] 分析失败不影响上传（记录 warning，`metadata.code_analysis` 为 null）
- [ ] 核心测试通过

**验证方式**:

```bash
uv run pytest tests/test_code_analysis.py -v --tb=short
uv run pytest tests/test_wrapper_api.py -v --tb=short
```

**状态**: ✅ 已完成（v2.6.0, commit afdf896）

---

### 已完成

| 编号 | 目标 | 状态 |
|------|------|------|
| BL-4 | 代码分析结果持久化（Phase A） | ✅ 已实现 |
| BL-6 | LLM 代码摘要生成（Phase C） | ✅ 已实现 |
| BL-CA-07 | 代码文件指纹同步 API | ✅ 已实现 |
| BL-CA-08 | 代码文件 Upsert | ✅ 已实现 |

---

## 场景 3: 多设备同步

> **用户流程**: 插件端有本地记忆文件 → `GET /api/v1/sync/fingerprints` 获取服务端指纹 → `POST /api/v1/sync/preview` 比对差异 → 用户确认 → `POST /api/v1/sync/full` 执行同步 → 冲突时解决
>
> **当前状态**: 4 个核心端点全部是 stub，返回空结果。测试用例已编写但全部失败（测试的是期望行为）。

### 待修复

#### BL-29 [P2] 实现指纹查询 #scene3

**目标**: `GET /api/v1/sync/fingerprints` 返回服务端所有记忆的指纹列表。

**涉及范围**:

- `wrapper/src/utils/memory_manager.py`:
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

**状态**: ✅ 已完成（v2.6.0, commit afdf896）

---

#### BL-D1 [P0] 归档历史文档 #docs

**目标**: 将过时的 md 文件和 JSON 报告移入 `archive/`，保持 `docs/` 目录干净。

**涉及范围**:

- 归档 22 个 md → `archive/docs/`
- 归档 11 个 JSON → `archive/reports/`

**前置依赖**: 无

**完成标准**:

- [ ] `docs/` 目录仅保留活跃文档（约 13 个 md）
- [ ] 归档文件在 `archive/` 中可找到
- [ ] 无死链

**验证方式**:

```bash
ls docs/*.md | wc -l
```

**状态**: 📋 待开始

---

#### BL-D2 [P0] 更新设计与 README 反映真实状态 #docs

**目标**: 消除 WRAPPER_SERVICE_DESIGN.md / README.md 与实际代码的偏差。

**涉及范围**:

- `docs/architecture/WRAPPER_SERVICE_DESIGN.md`: Stub 表更新
- `README.md`: stub 端点标注、CHANGELOG 补充 v2.5.0 条目
- `CHANGELOG.md`: 补充 v2.5.0 变更记录

**前置依赖**: BL-D1

**完成标准**:

- [ ] WRAPPER_SERVICE_DESIGN.md Stub 表与实际代码一致
- [ ] README.md stub 端点有"计划中"标注
- [ ] CHANGELOG.md 包含 v2.5.0 条目

**验证方式**:

```bash
rg "计划中" README.md
```

**状态**: 📋 待开始

---

## 暂缓任务

| 编号 | 目标 | 原因 |
|------|------|------|
| BL-5 [P2] Meilisearch 代码分析字段索引 | 依赖 BL-28 |
| BL-7 [P3] 跨文件关系解析 | 记忆级输入顺序不确定 |
| BL-8 [P3] 插件端代码分析工具 | 需插件端配合 |
| BL-1 [P2] Tenant ID 不匹配 | 需插件端配合 |

## 可无限期推迟（无用户场景）

以下 9 个端点已注册但无调用方，当前返回 NotImplementedError：

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
| `/api/v1/memories/cluster/leiden` | Leiden 聚类 |

> 不删除（保持 API 契约完整性），README 中标注"计划中"。

---

## 执行路线图

```
场景 4: 开发者体验 — 快速修复（P1，约 30 分钟）
├── BL-33 [P1] pyproject.toml 修复 ───────────► ✅ 已完成
└── BL-34 [P1] meilisearch_code/ 类型修复 ────► ✅ 已完成

场景 2: 代码分析（P1，约 1-2 小时）
└── BL-28 [P1] analyze_memory_code 实现 ──────► ✅ 已完成

场景 4: 开发者体验 — 中等重构（P1-P2，约 6-9 小时）
├── BL-35 [P1] memory_manager.py 拆分 ────────► ✅ 已完成（Mixin 模式 10 子模块）
├── BL-36 [P2] main.py 路由拆分 ──────────────► ✅ 已完成
├── BL-37 [P2] 工具模块单元测试 ──────────────► ✅ 已完成
├── BL-38 [P2] 移除硬编码 API Key ────────────► ✅ 已完成
└── BL-39 [P3] scripts/ 裸 except 清理 ───────► ✅ 已完成

场景 3: 多设备同步（P2，约 4-6 小时）
├── BL-29 [P2] 指纹查询 ─────────────────────► 📋 待开始
├── BL-30 [P2] 同步预览 ─────────────────────► 📋 待开始（依赖 BL-29）
├── BL-31 [P2] 全量同步 ─────────────────────► 📋 待开始（依赖 BL-29）
└── BL-32 [P2] 冲突解决 ─────────────────────► 📋 待开始（依赖 BL-30）

文档治理（P0，约 30 分钟）
├── BL-D1 [P0] 归档历史文档 ─────────────────► ✅ 已完成
└── BL-D2 [P0] 更新设计与 README ─────────────► 🔄 进行中（CHANGELOG 已更新）
```

---

## Backlog 规范

**格式**: `BL-{N} [{Priority}] 描述 #标签`

**优先级**: P0 = 紧急, P1 = 重要, P2 = 普通, P3 = 低优先级

**5 要素**:

1. **目标**: 解决什么问题，达成什么效果
2. **涉及范围**: 修改哪些文件/模块
3. **前置依赖**: 依赖哪些任务/条件
4. **完成标准**: 具体的验收 checklist
5. **验证方式**: 如何测试/验证完成

---

*最后更新: 2026-04-02*
