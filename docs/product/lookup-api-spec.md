# Memory Lookup API 产品规格书

**文档版本**: 1.0  
**最后更新**: 2026-04-09  
**关联任务**: BL-CA-33  
**状态**: ✅ 需求确认，待开发

---

## 1. 产品概述

### 1.1 背景

前端已实现 memory_id 缓存机制（`file_path` → `source_id` → `memory_id` 三层映射），用于支持代码分析 v1.4 的调用关系功能。

**问题**: 缓存可能丢失（换电脑、重装系统、清理磁盘等），需要后端提供查询接口来重建缓存。

### 1.2 目标

提供 Memory Lookup API，支持通过多种标识符查询记忆，实现缓存重建和多设备同步。

### 1.3 使用场景

| 场景 | 描述 | 查询方式 |
|------|------|----------|
| 缓存重建 | 缓存丢失后通过 file_path 重建 | file_path + project_id |
| 多设备同步 | 在新设备上通过 source_id 查询 | source_id |
| 内容去重 | 通过 hash 检查是否已存在 | hash |

---

## 2. 功能规格

### 2.1 API 端点

```
GET /api/v1/memories/lookup
```

### 2.2 请求参数

| 参数 | 类型 | 必需 | 说明 | 优先级 |
|------|------|------|------|--------|
| `source_id` | string | 可选 | 本地生成的 ULID | 1（最高） |
| `hash` | string | 可选 | 内容哈希（32位十六进制） | 2 |
| `hash_algorithm` | string | 可选 | 哈希算法，默认 `md5` | - |
| `file_path` | string | 可选 | 文件相对路径 | 3 |
| `project_id` | string | 可选* | 项目 ID | 3 |
| `type` | string | 可选 | 记忆类型过滤，如 `code` | - |
| `tenant_id` | string | 可选 | 租户 ID，默认 `default` | - |
| `limit` | integer | 可选 | 返回数量，默认 `1` | - |
| `all` | boolean | 可选 | 返回全部历史版本，默认 `false` | - |

**注意**: 
- `file_path` 使用时必须配合 `project_id` 或 `tenant_id`
- 至少提供 `source_id`、`hash` 或 `file_path` 之一

### 2.3 查询优先级

当提供多个参数时，按以下优先级查询：

1. `source_id` - 最精确，直接匹配
2. `hash` - 内容唯一标识
3. `file_path` + `project_id` - 路径唯一标识

### 2.4 响应格式

#### 2.4.1 单条响应（默认 limit=1）

**成功找到**:
```json
{
  "found": true,
  "memory_id": "memory:xyz...",
  "source_id": "01H1ABC...",
  "file_path": "src/utils.ts",
  "project_id": "my-project",
  "type": "code",
  "content_hash": "d41d8cd98f00b204...",
  "created_at": "2026-04-09T10:30:00Z",
  "updated_at": "2026-04-09T10:30:00Z"
}
```

**未找到**:
```json
{
  "found": false,
  "message": "No memory found matching the query criteria"
}
```

#### 2.4.2 多条响应（limit > 1 或 all=true）

```json
{
  "found": true,
  "count": 3,
  "memories": [
    {
      "memory_id": "memory:xyz...",
      "source_id": "01H1ABC...",
      "file_path": "src/utils.ts",
      "created_at": "2026-04-09T10:30:00Z"
    }
  ]
}
```

### 2.5 错误响应

**参数不足**:
```json
{
  "error": "Invalid query parameters",
  "message": "Provide at least one of: source_id, hash, or (file_path + project_id)"
}
```

**其他错误**:
```json
{
  "error": "Internal server error",
  "message": "..."
}
```

---

## 3. 使用示例

### 3.1 场景 1：缓存重建

```javascript
// 缓存丢失后，通过 file_path 重建
const result = await client.lookupMemory({
  file_path: 'src/utils.ts',
  project_id: 'my-project',
  type: 'code'
});

if (result.found) {
  // 重建缓存
  cache.set(result.file_path, result.source_id, result.memory_id);
}
```

### 3.2 场景 2：多设备同步

```javascript
// 在新设备上通过 source_id 查询
const result = await client.lookupMemory({
  source_id: '01H1ABC...'
});
```

### 3.3 场景 3：内容去重

```javascript
// 通过 hash 检查是否已存在
const result = await client.lookupMemory({
  hash: 'd41d8cd98f00b204...',
  hash_algorithm: 'md5'
});
```

### 3.4 场景 4：获取历史版本

```javascript
// 获取文件的所有历史版本
const result = await client.lookupMemory({
  file_path: 'src/utils.ts',
  project_id: 'my-project',
  all: true
});
```

---

## 4. 非功能需求

### 4.1 性能要求

- 响应时间: P95 < 100ms
- 支持并发: 100 QPS

### 4.2 安全要求

- 必须按 `tenant_id` 隔离数据
- 不返回敏感信息（如 embedding）

### 4.3 兼容性

- 向后兼容: 新增参数不得破坏现有调用
- 版本控制: URL 版本号 `/api/v1/`

---

## 5. 验收标准

### 5.1 功能验收

- [ ] API 端点可访问
- [ ] source_id 查询正常工作
- [ ] hash 查询正常工作
- [ ] file_path + project_id 查询正常工作
- [ ] 查询优先级正确
- [ ] 单条/多条响应格式正确
- [ ] 错误处理正确

### 5.2 性能验收

- [ ] P95 响应时间 < 100ms
- [ ] 支持 100 QPS 并发

### 5.3 集成验收

- [ ] 插件端可成功重建缓存
- [ ] 多设备同步正常工作

---

## 6. 时间线

| 阶段 | 时间 | 任务 | 负责人 |
|------|------|------|--------|
| 开发 | 2026-04-09 ~ 2026-04-16 | 实现 lookup API | 后端团队 |
| 测试 | 2026-04-17 | 集成测试 | 双方 |
| 文档 | 2026-04-18 | 更新 API 文档 | 后端团队 |
| 上线 | 2026-04-19 | 部署到生产环境 | 后端团队 |

---

## 7. 相关文档

- [需求文档](../inbox/plugin-lookup-api-request-20260409.md)
- [需求澄清](../inbox/plugin-lookup-api-clarification-20260409.md)
- [需求确认](../inbox/plugin-lookup-api-confirmation-20260409.md)
- [技术设计](./lookup-api-design.md)（待创建）

---

**文档维护**: 后端团队  
**审核**: 前端团队
