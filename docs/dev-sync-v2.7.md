# 多设备同步 — 开发文档 (v2.7.0)

**日期**: 2026-04-03
**关联**: `docs/product-sync-v2.7.md` (产品文档), BACKLOG.md (BL-29 ~ BL-32)

---

## 1. 现有代码状态

### 已就绪

| 组件 | 状态 | 位置 |
|------|------|------|
| API 端点（5 个） | ✅ 已注册 | `wrapper/src/routers/sync.py` |
| Pydantic 模型（6 个） | ✅ 已定义 | `wrapper/src/models.py` |
| `sync_code_fingerprints()` | ✅ 已实现（代码文件专用） | `wrapper/src/utils/memory_manager/sync.py` |
| 测试用例（32 个） | ✅ 已编写（含基本测试 + 真实策略测试 + 冲突持久化测试） | `tests/test_phase_b_sync.py` |
| `upload_memories()` | ✅ 已实现（含 embedding + 去重 + Meilisearch 双写） | `wrapper/src/utils/memory_manager/crud.py` |

### 已实现

| 方法 | 位置 | 说明 |
|------|------|------|
| `get_fingerprints()` | sync.py | 查询 SurrealDB 指纹，字段映射 content_hash→hash, updated_at→mtime |
| `sync_preview()` | sync.py | 三分类比对 (to_upload/to_delete/conflicts) + conflict 表写入 |
| `sync_full()` | sync.py | 透传 `upload_memories()`，含 embedding + 去重 + 双写 |
| `resolve_conflict()` | sync.py | 三种策略 (use_local/use_remote/keep_both) + Meilisearch 同步 |
| `_record_conflict()` | sync.py | 创建 conflict 记录 |
| `get_conflicts()` | sync.py | 查询 conflict 列表（支持 status 过滤） |
| `get_conflict_detail()` | sync.py | 查询单条 conflict 详情（自动补全 conflict: 前缀） |

---

## 2. SurrealDB 数据模型

### memory 表（已有）

```sql
SELECT id, source_id, content_hash, content, metadata, tenant_id, updated_at
FROM memory
WHERE tenant_id = $tenant_id
```

关键字段：
- `source_id`: 插件端唯一标识（对应本地文件路径），可为空
- `content_hash`: 内容 MD5 哈希，用于指纹比对
- `updated_at`: 最后更新时间戳

### conflict 表（需创建）

```sql
DEFINE TABLE conflict SCHEMAFULL;
DEFINE FIELD source_id ON TYPE conflict TYPE string;
DEFINE FIELD local_hash ON TYPE conflict TYPE string;
DEFINE FIELD server_hash ON TYPE conflict TYPE string;
DEFINE FIELD tenant_id ON TYPE conflict TYPE string;
DEFINE FIELD status ON TYPE conflict TYPE string DEFAULT "pending";
DEFINE FIELD local_content ON TYPE conflict TYPE string;
DEFINE FIELD server_content ON TYPE conflict TYPE string;
DEFINE FIELD created_at ON TYPE conflict TYPE datetime DEFAULT time::now();
DEFINE FIELD resolved_at ON TYPE conflict TYPE datetime;
DEFINE FIELD resolution ON TYPE conflict TYPE string;
```

---

## 3. 各方法实现方案

### 3.1 get_fingerprints (BL-29)

```
查询: SELECT source_id, content_hash, updated_at FROM memory
      WHERE source_id != NONE AND tenant_id = $tenant_id
映射: content_hash → hash, updated_at → mtime
返回: [{"source_id": ..., "hash": ..., "mtime": ...}, ...]
```

### 3.2 sync_preview (BL-30)

```
1. 获取服务端指纹（调用 get_fingerprints）
2. 构建 server_map: {source_id: {hash, mtime, id}}
3. 构建 local_set: {source_id for fp in fingerprints}
4. 遍历 local:
   - source_id 不在 server_map → to_upload (reason: "new")
   - source_id 在 server_map 但 hash 不同 → conflict
     - 调用 _record_conflict() 写入 conflict 表
   - hash 相同 → 跳过
5. 遍历 server:
   - source_id 不在 local_set → to_delete
6. 返回 {synced: 0, to_upload, to_delete, conflicts}
```

### 3.3 sync_full (BL-31)

```
直接调用 self.upload_memories(memories, tenant_id=tenant_id)
透传返回值（upload_memories 已处理去重、embedding、双写）
```

### 3.4 resolve_conflict (BL-32)

```
1. 查询 conflict 记录（WHERE id = $conflict_id）
2. 不存在 → 返回 404
3. 已 resolved → 返回已解决
4. 根据 resolution 策略:
   - use_local: UPDATE memory SET content=local_content, content_hash=local_hash
     WHERE source_id = conflict.source_id
   - use_remote: 不操作（保留服务端），仅标记 resolved
   - keep_both: CREATE memory (复制服务端记录，修改 source_id 加后缀)
     需要重新生成 embedding
5. UPDATE conflict SET status="resolved", resolution=..., resolved_at=time::now()
```

---

## 4. 测试期望（从 test_phase_b_sync.py 提取）

### TestSyncFingerprints (3 tests)
- `test_get_fingerprints_returns_list`: mock DB 返回 2 条记录，期望 `len==2`，字段包含 `source_id`, `hash`, `mtime`
- `test_get_fingerprints_empty_result`: mock DB 返回空，期望 `==[]`
- `test_get_fingerprints_tenant_isolation`: 验证 `tenant_id` 传入 DB 查询

### TestSyncPreview (4 tests)
- `test_sync_preview_new_entries`: 服务端无数据，本地 1 条 → `to_upload==1`
- `test_sync_preview_deleted_entries`: 服务端 1 条，本地空 → `to_delete==1`
- `test_sync_preview_conflicts`: 服务端 1 条 hash 不同 → `conflicts==1`，含 `local_hash`/`server_hash`
- `test_sync_preview_unchanged_entries`: hash 相同 → 三个列表都为空

### TestSyncFull (3 tests)
- `test_sync_full_success`: mock upload_memories 返回 success=2 → `result["success"]==2`
- `test_sync_full_with_skipped`: mock 返回 skipped → 验证 skipped 结构
- `test_sync_full_with_failures`: mock 第三次调用失败 → `failed==1, errors==1`

### TestResolveConflict (3 tests)
- `test_resolve_conflict_use_local`: mock DB 返回 conflict → `status=="resolved"`
- `test_resolve_conflict_use_remote`: 同上 → `resolution=="use_remote"`
- `test_resolve_conflict_keep_both`: mock DB + mock `_get_embeddings` → `resolution=="keep_both"`

### TestConflictPersistence (3 tests)
- 需要 `_record_conflict()`, `get_conflicts()`, `get_conflict_detail()`

### TestResolveConflictRealStrategies (3 tests)
- 实际 DB 操作验证（需 SurrealDB 连接或 mock）

### TestConflictIsolation (1 test)
- tenant_id 隔离验证

---

## 5. 实现顺序

```
BL-29 (指纹查询) ──── 最简单，无依赖
    │
    ▼
BL-30 (同步预览) ──── 依赖 BL-29 + _record_conflict
    │
    ├─► BL-31 (全量同步) ──── 独立，直接调用 upload_memories
    │
    └─► BL-32 (冲突解决) ──── 依赖 BL-30 创建的 conflict 记录
```

BL-29 和 BL-31 可并行开发（无依赖关系）。
