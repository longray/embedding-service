# Backlog - 代码分析集成（Phase 2 - 增量同步 API）

> **关联设计**: `docs/CODE-ANALYSIS-UNIFIED-DESIGN.md` (增量同步 API 定义)  
> **前置 Phase**: Phase 1 (BL-CA-01~06) 已完成  
> **状态**: 待实现

---

## 使用场景

### 场景 1：插件端启动时增量同步

```text
用户打开 VSCode 项目
  → 插件端读取本地 .code_fingerprints.json
  → 扫描当前文件系统，计算新指纹
  → POST /api/v1/sync/code-fingerprints {fingerprints: [...]}
  → 后端比对，返回:
      - changed: 服务端有更新，需要下载
      - unchanged: 无需操作
      - missing: 本地新增，需要上传
  → 插件端根据结果:
      - 下载 changed 文件的分析结果
      - 上传 missing 文件到后端
```

**后端需要做的**：接收指纹列表，按 project_id + file_path 查询现有代码记忆，比较 hash 和 symbols_hash，分类返回。

### 场景 2：多设备同步冲突检测

```text
用户在设备 A 修改 src/utils.ts
  → 设备 A 上传新分析结果
  → 用户切换到设备 B
  → 设备 B 启动增量同步
  → POST /api/v1/sync/code-fingerprints
  → 后端检测到 mtime 冲突（服务端比本地新）
  → 返回 conflicts 列表
  → 设备 B 提示用户选择：使用本地 / 使用服务端 / 合并
```

**后端需要做的**：检测到 mtime 冲突时，将文件加入 conflicts 而非 changed。

### 场景 3：符号变更检测

```text
用户重命名函数 foo → bar
  → 内容 hash 可能不变（如果只是函数名）
  → symbols_hash 必然改变
  → 后端通过 symbols_hash 检测到符号变更
  → 返回 changed (reason: "symbols_modified")
  → 触发插件端重新分析依赖关系
```

**后端需要做的**：同时比较 hash 和 symbols_hash，任一变更都视为 changed。

---

## Backlog 项

### BL-CA-07：实现 POST /api/v1/sync/code-fingerprints API

| 字段 | 内容 |
|------|------|
| **目标** | 实现代码文件专用的增量同步 API，支持基于 path + hash + symbols_hash 的变更检测，返回 changed/unchanged/missing 三类结果 |
| **涉及范围** | `wrapper/src/main.py`（新增端点 + Pydantic 模型）、`wrapper/src/utils/memory_manager.py`（新增 sync_code_fingerprints 方法）、可能需要 SurrealDB schema 确认 |
| **前置依赖** | BL-CA-01~06 已完成（Phase 1 全部完成）、通用 sync 基础设施已存在（/api/v1/sync/preview）、SurrealDB session 重试已修复 |
| **完成标准** | ① 新增 `CodeFingerprint`, `CodeSyncRequest`, `CodeSyncResponse` Pydantic 模型 ② 实现 `POST /api/v1/sync/code-fingerprints` 端点 ③ MemoryManager 实现 `sync_code_fingerprints()` 方法 ④ 能按 `type="code"` + `project_id` + `metadata->file_path` 查询 ⑤ 正确比较 hash 和 symbols_hash，分类返回 changed/unchanged/missing ⑥ 检测到 mtime 冲突时归入 conflicts |
| **验证方式** | 编写 pytest 用例：mock 不同场景（全新文件、内容变更、符号变更、未变更、mtime 冲突），验证返回结果分类正确；手动 curl 测试端到端流程 |

---

### BL-CA-08：代码文件 Upsert 逻辑（如未实现）

| 字段 | 内容 |
|------|------|
| **目标** | 确保代码文件上传时，同一 file_path + project_id 只保留最新版本（Upsert 而非 Insert） |
| **涉及范围** | `wrapper/src/utils/memory_manager.py` 的 `upload_memories` 方法 |
| **前置依赖** | 需先确认当前 upload_memories 是否已支持 code 类型的 Upsert |
| **完成标准** | ① 上传 type="code" 且 metadata.file_path 已存在时，更新而非新建 ② 旧版本被软删除或标记为历史 ③ 返回正确的 memory_id（新创建或更新的） |
| **验证方式** | pytest：两次上传同一 file_path，验证只保留一条记录；验证 memory_id 是否一致（更新）或不同（新建后删除旧） |

---

### BL-CA-09：端到端联调测试

| 字段 | 内容 |
|------|------|
| **目标** | 与插件端完成端到端联调，验证增量同步全流程 |
| **涉及范围** | 双方测试环境、API 对接、文档同步 |
| **前置依赖** | BL-CA-07 和 BL-CA-08 完成 |
| **完成标准** | ① 插件端能成功调用 code-fingerprints API ② 变更检测准确 ③ 上传/下载流程通畅 ④ 多设备同步无冲突 |
| **验证方式** | 双方共同执行测试用例，记录联调日志，确认无阻塞问题 |

---

## 依赖关系

```text
BL-CA-07 (code-fingerprints API)
        ↓
BL-CA-08 (Upsert 逻辑确认/实现)
        ↓
BL-CA-09 (端到端联调)
```

## 执行顺序建议

1. **先确认 BL-CA-08 状态**：检查当前 upload_memories 是否已支持 code Upsert
2. **并行处理**：
   - 若 BL-CA-08 已实现 → 直接开始 BL-CA-07
   - 若未实现 → BL-CA-07 和 BL-CA-08 可并行
3. **最后联调**：BL-CA-09

---

## 技术要点备忘

### SurrealDB 查询示例

```sql
-- 查询项目下所有代码文件
SELECT id, metadata, content_hash, mtime 
FROM memory 
WHERE type = "code" 
  AND project_id = $project_id
  AND tenant_id = $tenant_id;

-- 查询特定文件路径
SELECT id, metadata->file_path, content_hash, metadata->symbols_hash, mtime
FROM memory 
WHERE type = "code" 
  AND project_id = $project_id
  AND metadata->file_path = $file_path
  AND tenant_id = $tenant_id;
```

### 变更检测逻辑

```python
def detect_change(local, server):
    if server is None:
        return "missing"  # 服务端没有，需要上传
    
    if local["hash"] == server["content_hash"]:
        if local["symbols_hash"] == server["metadata"]["symbols_hash"]:
            return "unchanged"  # 完全一致
        else:
            return "changed"  # 仅符号变更
    else:
        # 内容变更，检查 mtime 冲突
        if local["mtime"] < server["mtime"]:
            return "conflict"  # 服务端更新，可能冲突
        return "changed"
```

### 响应格式

```yaml
# changed: 需要下载服务端版本（服务端新）或上传本地版本（本地新）
changed:
  - path: "src/index.js"
    reason: "content_modified" | "symbols_modified"
    server_mtime: 1712340000
    direction: "download" | "upload"  # 谁更新

# unchanged: 完全一致
unchanged:
  - path: "src/utils.js"

# missing: 服务端没有，需要上传
missing:
  - path: "src/new-file.ts"

# conflicts: mtime 冲突，需要人工解决
conflicts:
  - path: "src/auth.js"
    local_mtime: 1712345678
    server_mtime: 1712345600
```

---

*最后更新: 2026-03-31*
