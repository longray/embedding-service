# 致后端团队：API 测试结果反馈

**发件人**: OpenCode Memory Plugin (插件端) 团队  
**日期**: 2026-04-08  
**主题**: API 测试结果 - calls/batch 仍有问题

---

## 1. 测试结果

### ✅ 已验证成功的 API

| API | 端点 | 状态 | 结果 |
|-----|------|------|------|
| Health Check | GET /health | ✅ | 正常 |
| Upload Memory | POST /api/v1/memories | ✅ | 正常 |
| References Query | GET /api/v1/memories/{id}/references | ✅ | 正常 |
| Dependencies Query | GET /api/v1/memories/{id}/dependencies | ✅ | 正常 |

### ❌ 仍有问题的 API

| API | 端点 | 问题 |
|-----|------|------|
| Calls Batch | POST /api/v1/calls/batch | ❌ 返回 relationship_type 错误 |

---

## 2. 详细测试记录

### 测试 1: calls/batch（失败）

**请求**:
```bash
POST http://localhost:17999/api/v1/calls/batch
Content-Type: application/json

{
  "calls": [{
    "caller_memory_id": "memory:rdm1dtmxs23ca2f5vqv6",
    "callee_memory_id": "memory:ihvclhn43qeqkg3f3twt",
    "line": 6,
    "column": 25,
    "file_path": "src/auth.ts"
  }],
  "tenant_id": "default"
}
```

**响应**:
```json
{
  "status": "partial_success",
  "created": 0,
  "total": 3,
  "errors": [{
    "index": 0,
    "error": "Invalid relationship_type: calls. Must be one of {'related', 'elaboration', 'reference', 'derived_from', 'follow_up', 'contradiction'}"
  }]
}
```

**问题**: 后端仍然检查 relationship_type，但请求中没有提供该字段。

---

### 测试 2: references（成功）

**请求**:
```bash
GET http://localhost:17999/api/v1/memories/memory:ihvclhn43qeqkg3f3twt/references
```

**响应**:
```json
{
  "status": "success",
  "memory_id": "memory:ihvclhn43qeqkg3f3twt",
  "references": [],
  "total": 0
}
```

**说明**: API 存在，但返回空（因为我们之前用 relations 创建的，不是 calls）。

---

### 测试 3: dependencies（成功）

**请求**:
```bash
GET http://localhost:17999/api/v1/memories/memory:rdm1dtmxs23ca2f5vqv6/dependencies
```

**响应**:
```json
{
  "status": "success",
  "memory_id": "memory:rdm1dtmxs23ca2f5vqv6",
  "dependencies": [],
  "total": 0
}
```

**说明**: API 存在，但返回空。

---

## 3. 问题分析

### 根本原因

`POST /api/v1/calls/batch` 内部实现可能调用了 `memory_relation` 表的创建逻辑，该逻辑强制检查 `relationship_type` 字段，且不支持 `"calls"` 类型。

### 可能的解决方案

**方案 1: 后端修复（推荐）**
- 修改 `calls/batch` 实现，内部使用 `"reference"` 类型，或
- 在 `memory_relation` 表中添加 `"calls"` 类型支持

**方案 2: 插件端适配（临时）**
- 继续使用 `POST /api/v1/memories/relations`
- 在 description 中标注调用信息
- 等待后端修复

---

## 4. 建议下一步

### 立即执行

1. **后端检查**: 确认 `calls/batch` 内部实现是否使用了 `memory_relation` 表
2. **快速修复**: 在 `memory_relation` 类型枚举中添加 `"calls"`
3. **验证**: 修复后插件端重新测试

### 如果今天无法修复

1. 插件端使用 `relations` 端点继续联调
2. 在 description 中记录调用信息
3. 后续迭代中修复 `calls/batch`

---

## 5. 当前联调状态

| 场景 | 状态 | 说明 |
|------|------|------|
| 场景 1: 基础调用关系 | ⚠️ 部分完成 | 使用 relations 替代 calls |
| 场景 2: 代码地图 | ⏳ 待测试 | API 已就绪 |
| 场景 3: 错误处理 | ⏳ 待测试 | API 已就绪 |

---

## 6. 需要后端确认

1. **修复时间**: 今天能否修复 `calls/batch` 的 relationship_type 问题？
2. **替代方案**: 如果今天无法修复，是否接受使用 `relations` 端点继续联调？
3. **数据迁移**: 之前用 `relations` 创建的关系，能否迁移到 `calls` 类型？

---

**测试时间**: 2026-04-08 15:45  
**测试环境**: localhost:17999  
**后端版本**: 2.4.1

期待回复！

---

*文档版本: v1.0*  
*日期: 2026-04-08*  
*状态: 等待后端修复*
