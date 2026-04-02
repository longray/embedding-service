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

**状态**: 📋 待开始

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

**状态**: 📋 待开始

---

#### BL-30 [P2] 实现同步预览 #scene3

**目标**: `POST /api/v1/sync/preview` 比对本地与服务端指纹，返回差异（新增/跳过/冲突）。

**涉及范围**:

- `wrapper/src/utils/memory_manager.py`:
  - `sync_preview()` (当前返回空结构)

**前置依赖**: BL-29

**完成标准**:

- [ ] 接受 `SyncFingerprint` 列表（path, mtime, hash, source_id）
- [ ] 比对 content_hash → 匹配则跳过（skipped）
- [ ] 比对 source_id → 匹配但 hash 不同则冲突（conflicts）
- [ ] 无匹配 → 需上传（to_upload）
- [ ] `tests/test_phase_b_sync.py::TestSyncPreview` 相关测试通过

**验证方式**:

```bash
uv run pytest tests/test_phase_b_sync.py::TestSyncPreview -v --tb=short
```

**状态**: 📋 待开始

---

#### BL-31 [P2] 实现全量同步 #scene3

**目标**: `POST /api/v1/sync/full` 批量上传所有记忆，自动去重 + 双写 Meilisearch。

**涉及范围**:

- `wrapper/src/utils/memory_manager.py`:
  - `sync_full()` (当前返回 `errors: ["Not implemented"]`)

**前置依赖**: BL-29

**完成标准**:

- [ ] 接受 memories 列表，复用 `upload_memories` 的去重 + 双写逻辑
- [ ] 返回 `{total, success, failed, updated, skipped, errors}`
- [ ] `tests/test_phase_b_sync.py::TestSyncFull` 相关测试通过

**验证方式**:

```bash
uv run pytest tests/test_phase_b_sync.py::TestSyncFull -v --tb=short
```

**状态**: 📋 待开始

---

#### BL-32 [P2] 实现冲突解决 #scene3

**目标**: `POST /api/v1/sync/conflicts/{id}/resolve` 支持三种解决策略。

**涉及范围**:

- `wrapper/src/utils/memory_manager.py`:
  - `resolve_conflict()` (当前返回 `{"resolved": False, "error": "Not implemented"}`)

**前置依赖**: BL-30（需要先有冲突记录）

**完成标准**:

- [ ] `use_local`: 用本地版本覆盖服务端（UPDATE SurrealDB + Meilisearch）
- [ ] `use_backend`: 丢弃本地版本（删除冲突记录）
- [ ] `keep_both`: 两版本都保留（标记冲突为已解决）
- [ ] `tests/test_phase_b_sync.py::TestResolveConflict` 相关测试通过

**验证方式**:

```bash
uv run pytest tests/test_phase_b_sync.py::TestResolveConflict -v --tb=short
```

**状态**: 📋 待开始

---

## 场景 4: 开发者体验

> **开发者流程**: 开发者拉取代码 → 运行 lint/typecheck/测试 → 修改功能 → 添加测试 → 提交
>
> **当前状态**: 核心代码 ruff/pyright 通过，但 pyproject.toml 有过时配置导致覆盖率/测试路径错误，meilisearch_code/ 有 9 个类型错误，memory_manager.py 1660 行难以维护。

### 待修复

#### BL-33 [P1] 修复 pyproject.toml 过时配置 #dx

**目标**: 消除重复忽略规则和过时路径，使开发者工具配置反映真实项目结构。

**涉及范围**:

- `pyproject.toml`:
  - 第 85-92 行: `[tool.ruff.lint.per-file-ignores]` 中 `RUF001`, `RUF002`, `RUF003` 各重复一次，`E501` 缺失第二次但 `W293` 也有重复 → 删除第 90-92 行的重复项
  - 第 98 行: `testpaths = ["wrapper-service/tests"]` → `"tests"`（注: `pytest.ini` 已正确配置为 `tests`，pyproject.toml 中此配置被覆盖不生效，但仍应修正避免误导）
  - 第 107 行: `source = ["src", "wrapper-service/src"]` → `["src", "wrapper/src"]`

**前置依赖**: 无

**完成标准**:

- [ ] `rg "wrapper-service" pyproject.toml` 结果为 0
- [ ] `uv run ruff check .` 通过
- [ ] `uv run pytest tests/ --collect-only -q` 能正确收集测试（确认 pyproject.toml 不会干扰 pytest.ini）

**验证方式**:

```bash
rg "wrapper-service" pyproject.toml
uv run ruff check .
uv run pytest tests/ --collect-only -q
```

**状态**: 📋 待开始

---

#### BL-34 [P1] 修复 meilisearch_code/ Pyright 错误 #dx

**目标**: 消除 `meilisearch_code/` 目录下的 9 个类型错误，使 `uv run pyright .` 全量通过。

**涉及范围**:

- `meilisearch_code/init_index.py:34`: `except:` 缺少 body → `except Exception: pass`
- `meilisearch_code/monitor_index.py:29-30`: `stats.get("number_of_documents")` → `stats.number_of_documents`
- `meilisearch_code/optimize_index.py:26-27,36-37,40`: 同上（共 8 处 `.get()` → 属性访问）

**前置依赖**: 无

**完成标准**:

- [ ] `uv run pyright .` 返回 0 errors, 0 warnings

**验证方式**:

```bash
uv run pyright .
```

**状态**: 📋 待开始

---

#### BL-35 [P1] 拆分 memory_manager.py #dx #refactor

**目标**: 将 1660 行的上帝文件拆分为职责单一的子模块，降低维护难度。拆分后 `upload_memories()` 从 246 行/36 分支降至 ≤ 50 行编排层。

**涉及范围**:

- 新建 `wrapper/src/utils/memory_manager/` 目录:
  - `__init__.py` — 导出 `MemoryManager`（保持 `from .utils.memory_manager import MemoryManager` 导入路径不变）
  - `manager.py` — 主编排层（~200 行）
  - `crud.py` — 上传、更新、删除
  - `search.py` — 搜索路由、RRF 融合
  - `sync.py` — 同步、指纹、冲突
  - `relations.py` — 图关系、遍历
  - `dedup.py` — 去重决策、content_hash
  - `meili_sync.py` — Meilisearch 双写/同步
  - `code_analysis.py` — 代码分析桥接
- 删除 `wrapper/src/utils/memory_manager.py`（替换为目录）

**前置依赖**: 无

**完成标准**:

- [ ] `upload_memories()` ≤ 50 行（当前 246 行）
- [ ] 每个子模块 ≤ 300 行
- [ ] `from .utils.memory_manager import MemoryManager` 导入路径不变
- [ ] `uv run pytest tests/` 全部通过
- [ ] `uv run ruff check .` 通过
- [ ] `uv run pyright wrapper/src/` 通过

**验证方式**:

```bash
uv run pytest tests/ -v
uv run ruff check .
uv run pyright wrapper/src/
```

**状态**: 📋 待开始

---

#### BL-36 [P2] 拆分 main.py 路由 #dx #refactor

**目标**: 将 1173 行的 `main.py` 拆分为 FastAPI Router 模块，每个路由文件 ≤ 200 行。

**涉及范围**:

- 新建 `wrapper/src/routers/` 目录:
  - `health.py`, `embeddings.py`, `memories.py`, `search.py`
  - `sync.py`, `relations.py`, `websocket.py`
- 新建 `wrapper/src/models.py`: 所有 Pydantic 模型集中管理
- 精简 `wrapper/src/main.py` 为应用创建 + lifespan + include_router（~100 行）

**前置依赖**: 无（可与 BL-35 并行）

**完成标准**:

- [ ] `main.py` ≤ 150 行
- [ ] 每个 router 文件 ≤ 200 行
- [ ] `from wrapper.src.main import app` 导入路径不变
- [ ] 所有测试通过

**验证方式**:

```bash
uv run pytest tests/ -v
uv run ruff check .
```

**状态**: 📋 待开始

---

#### BL-37 [P2] 补充工具模块单元测试 #dx #testing

**目标**: 为 cache、http_pool、auth、exceptions 四个工具模块补充单元测试，确保重构时有安全网。

**涉及范围**:

- 新建 `tests/test_cache.py`: 命中/未命中/TTL/线程安全/容量淘汰
- 新建 `tests/test_http_pool.py`: 连接复用/超时/关闭清理
- 新建 `tests/test_auth.py`: token 验证成功/失败/缺失
- 新建 `tests/test_exceptions.py`: 异常层级/消息格式

**前置依赖**: 无

**完成标准**:

- [ ] 每个模块 ≥ 3 个测试用例
- [ ] 全部新测试通过

**验证方式**:

```bash
uv run pytest tests/test_cache.py tests/test_http_pool.py tests/test_auth.py tests/test_exceptions.py -v
```

**状态**: 📋 待开始

---

#### BL-38 [P2] 移除硬编码默认 API Key #dx #security

**目标**: 消除 `meili_client.py` 中的硬编码默认值 `"masterKey"`。

**涉及范围**:

- `wrapper/src/utils/meili_client.py`:
  - 构造函数参数 `api_key="masterKey"` → `api_key: str | None = None`
  - `api_key is None` 时记录 warning 而非使用硬编码值

**前置依赖**: 无

**完成标准**:

- [ ] `rg 'masterKey' wrapper/src/` 无匹配
- [ ] Docker Compose 通过环境变量正常工作
- [ ] 测试通过

**验证方式**:

```bash
rg "masterKey" wrapper/src/
uv run pytest tests/test_meili_integration.py -v --tb=short
```

**状态**: 📋 待开始

---

#### BL-39 [P3] 清理 scripts/ 裸 except 块 #dx

**目标**: 将 scripts/ 和 tests/ 中的裸 `except:` 改为 `except Exception:`，避免静默吞掉 KeyboardInterrupt 等系统异常。

**涉及范围**:

- `tests/test_phase_a_backend.py:57`
- `scripts/collect-metrics.py:40,49,60`
- `scripts/generate-report.py:24`

**前置依赖**: 无

**完成标准**:

- [ ] `rg "except\s*:" --type py scripts/ tests/` 仅匹配 `except Exception:`（无裸 except）

**验证方式**:

```bash
rg "except\s*:" --type py scripts/ tests/
```

**状态**: 📋 待开始

---

## 文档治理

> 不直接服务用户场景，但消除文档与代码的偏差可避免误导后续开发。

### 待修复

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
├── BL-33 [P1] pyproject.toml 修复 ───────────► 📋 待开始（~15 分钟）
└── BL-34 [P1] meilisearch_code/ 类型修复 ────► 📋 待开始（~15 分钟）

场景 2: 代码分析（P1，约 1-2 小时）
└── BL-28 [P1] analyze_memory_code 实现 ──────► 📋 待开始

场景 4: 开发者体验 — 中等重构（P1-P2，约 6-9 小时）
├── BL-35 [P1] memory_manager.py 拆分 ────────► 📋 待开始（~4-6 小时）
├── BL-36 [P2] main.py 路由拆分 ──────────────► 📋 待开始（~2-3 小时，可与 BL-35 并行）
├── BL-37 [P2] 工具模块单元测试 ──────────────► 📋 待开始（~1-2 小时）
├── BL-38 [P2] 移除硬编码 API Key ────────────► 📋 待开始（~15 分钟）
└── BL-39 [P3] scripts/ 裸 except 清理 ───────► 📋 待开始（~15 分钟）

场景 3: 多设备同步（P2，约 4-6 小时）
├── BL-29 [P2] 指纹查询 ─────────────────────► 📋 待开始
├── BL-30 [P2] 同步预览 ─────────────────────► 📋 待开始（依赖 BL-29）
├── BL-31 [P2] 全量同步 ─────────────────────► 📋 待开始（依赖 BL-29）
└── BL-32 [P2] 冲突解决 ─────────────────────► 📋 待开始（依赖 BL-30）

文档治理（P0，约 30 分钟）
├── BL-D1 [P0] 归档历史文档 ─────────────────► 📋 待开始
└── BL-D2 [P0] 更新设计与 README ─────────────► 📋 待开始（依赖 BL-D1）
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
