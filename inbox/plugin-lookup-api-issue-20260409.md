# Memory Lookup API 索引问题报告

**发件人**: OpenCode Memory Plugin 前端团队  
**日期**: 2026-04-09  
**优先级**: P1 - 高优先级  
**状态**: 等待后端确认

---

## 问题概述

前端代码已完成 Memory Lookup API 的集成（BL-CA-33 和 BL-CA-34），但集成测试显示后端 lookup API 无法返回已上传的记忆数据。

## 测试环境

- **后端地址**: `http://localhost:17999`
- **API 端点**: `GET /api/v1/memories/lookup`
- **测试项目**: `test-lookup-project`
- **测试时间**: 2026-04-09

## 复现步骤

### 1. 上传代码（成功）

```javascript
const result = await wrapperClient.uploadMemories([
  {
    type: 'code',
    content: JSON.stringify(analysis),
    abstract: 'Lookup test function',
    overview: 'Test file for lookup API',
    source_id: 'test-source-1775746841629',
    local_id: 'test-source-1775746841629',
    project_id: 'test-lookup-project',
    metadata: {
      file_path: 'src/lookup-test.ts',
      content_hash: 'abc123',
    },
  },
]);

// 返回结果
{
  "success": 1,
  "memory_ids": ["memory:0cprgytkpldx3vsbjq80"],
  "failed": [],
  "errors": []
}
```

✅ **上传成功**，返回了 memory_id

### 2. 通过 source_id 查询（失败）

```bash
curl -X GET "http://localhost:17999/api/v1/memories/lookup?source_id=test-source-1775746841629" \
  -H "WRAPPER_MEILI_API_KEY: test-api-key"
```

**返回结果**:
```json
{
  "found": false,
  "message": "未找到匹配的记忆"
}
```

❌ **查询失败**，但数据刚刚上传成功

### 3. 通过 file_path 查询（失败）

```bash
curl -X GET "http://localhost:17999/api/v1/memories/lookup?file_path=src/lookup-test.ts&project_id=test-lookup-project" \
  -H "WRAPPER_MEILI_API_KEY: test-api-key"
```

**返回结果**:
```json
{
  "found": false,
  "message": "未找到匹配的记忆"
}
```

❌ **同样失败**

## 后端健康状态

```bash
curl http://localhost:17999/health
```

**返回结果**:
```json
{
  "status": "healthy",
  "service": "minimal-wrapper",
  "version": "2.4.1",
  "port": 17999,
  "embedding_service": {
    "status": "healthy",
    "service": "embedding",
    "version": "2.0.1",
    "device": "cuda"
  }
}
```

✅ **后端服务健康**

## 可能的原因

1. **索引延迟**: 数据上传后需要异步索引时间，但测试等待了足够时间仍失败
2. **字段映射问题**: `source_id` 和 `file_path` 字段可能未正确映射到 SurrealDB 索引
3. **索引未建立**: lookup API 依赖的索引可能尚未创建
4. **数据过滤**: 可能有过滤条件导致查询不到数据（如 tenant_id、project_id 匹配问题）

## 需要后端确认的问题

1. **Lookup API 是否已实现完成？**
   - 根据之前的确认，API 已实现，但是否需要重新索引数据？

2. **索引字段配置**
   - SurrealDB 中 `source_id` 和 `file_path` 是否已建立索引？
   - 索引类型是什么（唯一索引、普通索引）？

3. **数据一致性**
   - 上传的数据是否确实保存到了 SurrealDB？
   - 能否提供一个已知的 source_id 供我们测试查询？

4. **查询参数要求**
   - 是否必须提供 `tenant_id`？
   - `project_id` 是否区分大小写？

## 测试文件位置

前端集成测试文件：
- `opencode-memory-plugin/tests/integration/lookup-api.integration.test.js`
- `opencode-memory-plugin/tests/integration/code-analysis-v14-e2e.test.js`

## 临时解决方案

在前端代码中，我们已经实现了 MemoryIdCache 作为本地缓存层：
- 上传成功后立即缓存到本地
- 优先从本地缓存查询
- 缓存未命中时才调用后端 lookup API

但这只是 workaround，后端 lookup API 仍需正常工作以支持：
- 跨设备同步
- 缓存重建
- 数据一致性验证

## 请求

请后端团队协助：

1. [ ] 确认 lookup API 的实现状态
2. [ ] 检查 SurrealDB 索引配置
3. [ ] 如有必要，重新索引现有数据
4. [ ] 提供一个可用的测试 source_id 供验证
5. [ ] 更新 API 文档（如有参数变更）

## 联系方式

如有疑问或需要更多调试信息，请联系前端团队。

---

**相关文档**:
- [API-CONTRACT.md](../docs/API-CONTRACT.md)
- [MEMORY-ID-CACHE-DESIGN.md](../docs/MEMORY-ID-CACHE-DESIGN.md)
- [plugin-lookup-api-request-20260409.md](./plugin-lookup-api-request-20260409.md)

**相关 Backlog**:
- BL-CA-33: 实现 memory_id 缓存机制 ✅
- BL-CA-34: 后端 Memory Lookup API 实现 ⏳
