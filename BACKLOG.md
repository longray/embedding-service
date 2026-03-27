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

## 待定

（暂无）

---

## 已完成

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
