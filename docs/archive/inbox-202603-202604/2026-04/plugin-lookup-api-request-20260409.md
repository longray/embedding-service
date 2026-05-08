# 后端 API 需求：Memory Lookup 接口

**发件人**: OpenCode Memory Plugin 前端团队  
**收件人**: rs-memory-service 后端团队  
**日期**: 2026-04-09  
**优先级**: P1  
**关联任务**: BL-CA-33

---

## 背景

前端已实现 memory_id 缓存机制（`file_path` → `source_id` → `memory_id` 三层映射），用于支持代码分析 v1.4 的调用关系功能。

**问题**：缓存可能丢失（换电脑、重装系统、清理磁盘等），需要后端提供查询接口来重建缓存。

---

## 需求描述

### API 端点

```http
GET /api/v1/memories/lookup
```

### 请求参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `source_id` | string | 可选 | 本地生成的 ULID |
| `file_path` | string | 可选 | 文件相对路径 |
| `project_id` | string | 可选 | 项目 ID（使用 file_path 查询时必需）|
| `hash` | string | 可选 | 内容哈希（MD5）|

**查询优先级**：
1. 如果提供 `source_id`，按 source_id 精确匹配
2. 如果提供 `file_path` + `project_id`，按路径匹配
3. 如果提供 `hash`，按内容哈希匹配

### 响应格式

```json
{
  "found": true,
  "memory_id": "memory:xyz...",
  "source_id": "01H1ABC...",
  "file_path": "src/utils.ts",
  "project_id": "my-project",
  "content_hash": "md5:abc123...",
  "created_at": "2026-04-09T10:30:00Z"
}
```

**未找到时**：
```json
{
  "found": false,
  "message": "Memory not found"
}
```

### 错误响应

```json
{
  "error": "Invalid query parameters",
  "details": "At least one of source_id, file_path, or hash must be provided"
}
```

---

## 使用场景

### 场景 1：缓存重建

```javascript
// 缓存丢失后，通过 file_path 重建
const result = await client.lookupMemory({
  file_path: 'src/utils.ts',
  project_id: 'my-project'
});

if (result.found) {
  // 重建缓存
  cache.set(result.file_path, result.source_id, result.memory_id);
}
```

### 场景 2：多设备同步

```javascript
// 在新设备上通过 source_id 查询
const result = await client.lookupMemory({
  source_id: '01H1ABC...'
});
```

### 场景 3：内容去重

```javascript
// 通过 hash 检查是否已存在
const result = await client.lookupMemory({
  hash: 'md5:abc123...'
});
```

---

## 技术实现建议

### 数据库索引

建议添加以下索引以优化查询性能：

```sql
-- source_id 索引（唯一）
CREATE INDEX idx_memories_source_id ON memories(source_id);

-- file_path + project_id 复合索引
CREATE INDEX idx_memories_file_project ON memories(
  (metadata->>'file_path'),
  project_id
);

-- content_hash 索引
CREATE INDEX idx_memories_hash ON memories(
  (metadata->>'content_hash')
);
```

### 查询逻辑

```python
def lookup_memory(source_id=None, file_path=None, project_id=None, hash=None):
    if source_id:
        return db.query("SELECT * FROM memories WHERE source_id = ?", source_id)
    
    if file_path and project_id:
        return db.query("""
            SELECT * FROM memories 
            WHERE metadata->>'file_path' = ? AND project_id = ?
            ORDER BY created_at DESC LIMIT 1
        """, file_path, project_id)
    
    if hash:
        return db.query("""
            SELECT * FROM memories 
            WHERE metadata->>'content_hash' = ?
            ORDER BY created_at DESC LIMIT 1
        """, hash)
    
    raise ValueError("Invalid query parameters")
```

---

## 替代方案

如果实现 lookup API 有困难，可以考虑以下替代方案：

### 方案 A：扩展 search API

在现有 `/api/v1/memories/search` 中添加精确匹配模式：

```json
{
  "query": "src/utils.ts",
  "mode": "exact",
  "code_filter": {
    "file_path": "src/utils.ts",
    "project_id": "my-project"
  }
}
```

### 方案 B：支持 source_id 创建调用关系

在 `POST /api/v1/calls/batch` 中支持使用 `source_id`：

```json
{
  "calls": [
    {
      "caller_source_id": "01H1ABC...",
      "callee_source_id": "01H2DEF...",
      "line": 42
    }
  ]
}
```

后端根据 `source_id` 查找对应的 `memory_id`。

---

## 优先级

| 功能 | 优先级 | 说明 |
|------|--------|------|
| source_id 查询 | P1 | 最常用，精确匹配 |
| file_path 查询 | P1 | 缓存重建必需 |
| hash 查询 | P2 | 内容去重，可选 |

---

## 时间线

- **期望完成时间**: 2026-04-16（1 周）
- **集成测试时间**: 2026-04-17
- **文档更新时间**: 2026-04-18

---

## 联系方式

如有问题或需要讨论实现细节，请随时联系。

**前端团队**  
OpenCode Memory Plugin
