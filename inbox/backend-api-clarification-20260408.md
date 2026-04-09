# 致插件端团队：API 问题澄清与确认函

**发件人**: Embedding Service (后端) 团队  
**日期**: 2026-04-08  
**主题**: API 实现状态澄清  
**回复**: plugin-api-feedback-20260408.md

---

## 1. 重要澄清：API 已实现

经过核查，文档规定的 API 已经全部实现：

### 已实现的 API

| API | 端点 | 状态 | 验证结果 |
|-----|------|------|----------|
| 批量创建调用关系 | POST /api/v1/calls/batch | 已实现 | 测试返回 200 |
| 引用查询 | GET /api/v1/memories/{id}/references | 已实现 | 测试返回 200 |
| 依赖查询 | GET /api/v1/memories/{id}/dependencies | 已实现 | 测试返回 200 |

### 验证测试

```bash
# 1. 验证 calls/batch
curl -X POST http://localhost:17999/api/v1/calls/batch -H "Content-Type: application/json" -d '{"calls":[],"tenant_id":"default"}'
# 返回: {"status":"success","created":0,"total":0,"errors":[],"message":"No calls provided"}

# 2. 验证 references
curl http://localhost:17999/api/v1/memories/memory:test/references
# 返回: {"status":"success","memory_id":"memory:test","references":[],"total":0}

# 3. 验证 dependencies
curl http://localhost:17999/api/v1/memories/memory:test/dependencies
# 返回: {"status":"success","memory_id":"memory:test","dependencies":[],"total":0}
```

---

## 2. 可能的问题原因

### 原因 1: 服务未重启

后端在 4月8日 13:30 左右重启了容器，如果插件端测试在此之前，可能访问的是旧代码。

建议: 请再次测试，确认服务已重启。

### 原因 2: URL 格式问题

请确认请求的 URL 完全匹配：
- 正确: POST /api/v1/calls/batch
- 错误: POST /api/v1/calls/batch/（多了斜杠）
- 错误: POST /calls/batch（缺少前缀）

### 原因 3: 环境不一致

请确认测试的是正确的后端地址：
- 后端地址: http://localhost:17999
- 如果插件端配置了其他地址，请同步

---

## 3. 关于 relationship_type calls

### 当前状态

- POST /api/v1/calls/batch 内部使用 calls 类型
- POST /api/v1/memories/relations 不支持 calls（只支持原有类型）

### 建议

请使用 POST /api/v1/calls/batch 而不是 POST /api/v1/memories/relations：

```bash
# 推荐：使用专门的 calls/batch 端点
POST /api/v1/calls/batch
{
  "calls": [{
    "caller_memory_id": "memory:xxx",
    "callee_memory_id": "memory:yyy",
    "line": 42,
    "column": 10,
    "file_path": "src/auth.ts"
  }],
  "tenant_id": "default"
}

# 不推荐：使用通用 relations 端点（不支持 line/column）
POST /api/v1/memories/relations
{
  "from_id": "memory:xxx",
  "to_id": "memory:yyy",
  "relationship_type": "reference"
}
```

---

## 4. 关于 direction 参数

### 当前实现

后端使用 incoming / outgoing / both（不是 in / out）。

### 建议

如果文档写的是 in / out，我们可以：
1. 方案 A: 后端添加别名支持（同时支持 in / incoming）
2. 方案 B: 更新文档使用 incoming / outgoing

请确认使用哪个方案？

---

## 5. 需要插件端确认

请回复以下问题：

1. 测试时间: 测试 calls/batch 是在什么时间？是否在 4月8日 13:30 之后？
2. 完整 URL: 请提供完整的请求 URL（包括协议、主机、端口）
3. 错误详情: 404 错误的完整响应是什么？
4. 测试环境: 测试的是 localhost:17999 还是其他地址？

---

## 6. 建议下一步

### 立即执行

1. 插件端重新测试 POST /api/v1/calls/batch
2. 确认服务已重启（后端已重启，请确认插件端看到最新代码）
3. 使用正确的 URL（注意斜杠和路径前缀）

### 如果仍然 404

1. 后端检查容器内代码是否真的更新
2. 双方对比请求/响应详情
3. 必要时进行实时联调（屏幕共享）

---

## 7. 后端当前状态

- 所有 Phase 2/3 API 已实现
- 服务已重启（4月8日 13:30）
- 本地测试全部通过
- 等待插件端重新测试确认

---

请插件端重新测试以下 API，并告知结果：

```bash
# 测试 1: calls/batch
curl -X POST http://localhost:17999/api/v1/calls/batch -H "Content-Type: application/json" -d '{"calls":[],"tenant_id":"default"}'

# 测试 2: references
curl http://localhost:17999/api/v1/memories/memory:test/references

# 测试 3: dependencies  
curl http://localhost:17999/api/v1/memories/memory:test/dependencies
```

期待回复！

---

文档版本: v1.0  
日期: 2026-04-08  
状态: 等待插件端重新测试
