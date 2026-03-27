# Backlog

> 后端任务追踪文档，按优先级排序

**更新时间**: 2026-03-28

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
