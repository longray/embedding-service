# 多设备同步 — 产品文档 (v2.7.0)

**日期**: 2026-04-03
**关联**: BACKLOG.md (BL-29 ~ BL-32)
**前置阅读**: `docs/product-decisions.md`

---

## 1. 用户场景

### 谁在使用

OpenCode Memory 插件端。用户在多台设备上使用 OpenCode，每台设备本地维护一份记忆文件（Markdown）。插件需要将这些本地记忆与服务端保持同步。

### 用户流程

```
插件启动
  │
  ├─ 1. 获取服务端指纹列表
  │     GET /api/v1/sync/fingerprints?tenant_id=default
  │
  ├─ 2. 比对本地与服务端差异
  │     POST /api/v1/sync/preview
  │     请求体: { fingerprints: [{path, mtime, hash, source_id}], tenant_id }
  │     响应: { to_upload: [...], to_delete: [...], conflicts: [...] }
  │
  ├─ 3a. 无冲突 → 全量同步
  │     POST /api/v1/sync/full
  │     请求体: { memories: [...], tenant_id }
  │
  ├─ 3b. 有冲突 → 展示给用户选择
  │     用户选择: use_local | use_remote | keep_both
  │     POST /api/v1/sync/conflicts/{id}/resolve
  │
  └─ 4. 同步完成
```

---

## 2. 功能需求

### FR-1: 指纹查询 (BL-29)

**输入**: `tenant_id`

**输出**: 服务端所有记忆的指纹列表

```json
{
  "fingerprints": [
    { "source_id": "entry-001", "hash": "abc123", "mtime": 1234567890 }
  ],
  "count": 1
}
```

**行为规则**:
- 仅返回 `source_id` 非空的记忆
- `source_id` 是插件端的唯一标识（对应本地文件路径）
- 按 `tenant_id` 隔离
- 空数据库返回 `{"fingerprints": [], "count": 0}`

### FR-2: 同步预览 (BL-30)

**输入**: 本地指纹列表

**输出**: 三分类结果

| 分类 | 含义 | 条件 |
|------|------|------|
| `to_upload` | 服务端没有，需要上传 | source_id 不在服务端 |
| `to_delete` | 服务端有但本地没有 | source_id 在服务端但不在本地列表 |
| `conflicts` | 两边都有但 hash 不同 | source_id 匹配但 hash 不匹配 |
| 无输出 | 两边都有且 hash 相同 | 不出现在任何列表中 |

**行为规则**:
- 预览操作**不修改任何数据**（只读）
- 检测到冲突时，在 `conflict` 表中创建 `status=pending` 的记录
- `conflicts[].local_hash` 和 `conflicts[].server_hash` 用于 UI 展示差异

### FR-3: 全量同步 (BL-31)

**输入**: 完整的记忆列表

**输出**: 每条记忆的处理结果

```json
{
  "total": 3,
  "success": 2,
  "failed": 0,
  "updated": 1,
  "skipped": [
    { "local_id": "entry-001", "existing_id": "memory:abc", "reason": "hash", "similarity": null }
  ],
  "errors": []
}
```

**行为规则**:
- 复用已有的 `upload_memories()` 方法（含 embedding、去重、Meilisearch 双写）
- `skipped` 包含被语义去重跳过的条目
- 部分失败不中断：继续处理剩余条目

### FR-4: 冲突解决 (BL-32)

**输入**: `conflict_id` + `resolution`

**策略**:

| 策略 | 行为 |
|------|------|
| `use_local` | 用本地内容覆盖服务端（UPDATE） |
| `use_remote` | 保留服务端内容，丢弃本地（DELETE 本地 / 不做操作） |
| `keep_both` | 保留服务端，额外创建一条本地版本（CREATE，修改 source_id 加后缀） |

**行为规则**:
- 解决后更新 `conflict` 记录的 `status` 为 `resolved`
- `use_local` 和 `keep_both` 需要重新生成 embedding
- 不存在的 conflict_id 返回 404

---

## 3. 验收标准

### AC-1: 基本流程

```bash
# 1. 上传一条记忆
POST /api/v1/memories { memories: [{ content: "test", abstract: "t", overview: "t", source_id: "a.md" }] }

# 2. 获取指纹，确认有数据
GET /api/v1/sync/fingerprints → count >= 1

# 3. 预览，确认 unchanged（hash 相同）
POST /api/v1/sync/preview → to_upload=0, to_delete=0, conflicts=0

# 4. 修改内容后预览，确认 conflict
POST /api/v1/memories { memories: [{ content: "modified", abstract: "t", overview: "t", source_id: "a.md" }] }
POST /api/v1/sync/preview → conflicts=1
```

### AC-2: 测试通过

```bash
uv run pytest tests/test_phase_b_sync.py -v
# 期望: 全部通过（当前 13 passed, 19 failed）
```

### AC-3: 向后兼容

- `/api/v1/sync/incremental` 仍可用（转发到 `/sync/preview`）
- 所有 Pydantic 模型（SyncIncrementalRequest/Response）仍可导入

---

## 4. 不做什么

- **不做增量同步**：`sync_full` 每次上传完整列表，由 `upload_memories` 内部处理去重和更新
- **不做冲突自动解决**：所有冲突必须由用户手动选择策略
- **不做 WebSocket 推送同步状态**：同步是请求-响应模式
