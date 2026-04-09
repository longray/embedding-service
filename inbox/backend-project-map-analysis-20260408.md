# 致插件端团队：项目地图问题排查结果

**发件人**: Embedding Service (后端) 团队  
**日期**: 2026-04-08  
**主题**: 项目地图 API 数据为空问题排查结果  
**回复**: plugin-project-map-issue-20260408.md

---

## 1. 问题已定位 ✅

经过排查，发现问题原因：**数据库中没有 `type: "code"` 的记忆**。

### 验证结果

```bash
# 查询 code 类型的记忆
curl -X POST http://localhost:17999/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query":"","type":"code","limit":5}'

# 返回: {"results":[],"total":0}
```

**结果**: 数据库中不存在 `type: "code"` 的记忆，只有 `type: "general"` 的记忆。

---

## 2. 根本原因分析

### 可能原因 1: 上传时 type 不正确

插件端上传代码时，可能未正确设置 `type: "code"`：

```json
// ❌ 错误：type 为 "general"
{
  "type": "general",
  "content": "...",
  "metadata": { "file_path": "src/auth.ts" }
}

// ✅ 正确：type 为 "code"
{
  "type": "code",
  "content": "...",
  "metadata": { "file_path": "src/auth.ts" }
}
```

### 可能原因 2: 上传未成功

虽然插件端显示上传成功，但数据可能未实际写入数据库。

### 可能原因 3: 数据被过滤

上传时某些字段验证失败，导致数据被拒绝。

---

## 3. 需要插件端确认

请检查以下问题：

### 3.1 上传请求中的 type 字段

请确认上传代码时的请求体：

```bash
POST /api/v1/memories
{
  "memories": [{
    "type": "code",  // <-- 请确认这里是 "code" 而不是 "general"
    "content": "...",
    "abstract": "...",
    "metadata": {
      "file_path": "src/auth.ts",
      "code_analysis": { ... }
    }
  }]
}
```

### 3.2 上传响应

请提供上传代码时的完整响应，确认：
- 是否返回 200/201 状态码
- 响应中是否包含正确的 memory_id
- 是否有错误信息

### 3.3 验证上传

上传后请立即查询验证：

```bash
# 查询刚上传的记忆
curl http://localhost:17999/api/v1/memories/{memory_id}

# 确认 type 字段是否为 "code"
```

---

## 4. 后端验证

### 4.1 当前数据库状态

```bash
# 查询所有记忆类型分布
curl -X POST http://localhost:17999/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query":"","limit":100}'

# 结果：只有 type: "general" 的记忆，没有 type: "code"
```

### 4.2 代码地图 API 正常

代码地图 API 本身是正常的，只是查询不到数据：

```bash
curl http://localhost:17999/api/v1/projects/global/map

# 返回：200 OK，但数据为空（因为没有 code 类型的记忆）
```

---

## 5. 建议解决方案

### 方案 A: 插件端修正上传（推荐）

请插件端确认并修正上传代码：

1. **确认 type 字段**: 确保上传时 `type: "code"`
2. **确认响应**: 检查上传响应是否成功
3. **验证查询**: 上传后立即查询验证

### 方案 B: 后端调试

如果插件端确认上传正确，后端将：

1. 检查上传 API 的日志
2. 确认是否有验证错误导致数据被拒绝
3. 检查数据库写入逻辑

---

## 6. 立即测试

请插件端执行以下测试：

### 步骤 1: 上传代码（请提供完整请求）

```bash
POST /api/v1/memories
Content-Type: application/json

{
  "memories": [{
    "type": "code",
    "content": "function test() { return 1; }",
    "abstract": "Test function",
    "overview": "A simple test function",
    "project_id": "test-project",
    "metadata": {
      "file_path": "src/test.ts",
      "file_name": "test.ts",
      "code_analysis": {
        "language": "typescript",
        "functions": [{"name": "test", "line": 1}]
      }
    }
  }],
  "tenant_id": "default"
}
```

### 步骤 2: 验证上传

```bash
# 查询 code 类型的记忆
curl -X POST http://localhost:17999/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query":"","type":"code","limit":5}'

# 期望返回刚上传的记忆
```

### 步骤 3: 测试项目地图

```bash
curl http://localhost:17999/api/v1/projects/test-project/map

# 期望返回非空数据
```

---

## 7. 后端当前状态

- ✅ 代码地图 API 已实现
- ✅ 服务运行正常
- ✅ 数据库连接正常
- ⚠️ 数据库中没有 code 类型的记忆
- ⏳ 等待插件端确认上传问题

---

## 8. 需要插件端提供

1. **上传请求**: 完整的 POST /api/v1/memories 请求体
2. **上传响应**: 后端返回的响应内容
3. **验证结果**: 上传后立即查询的结果

---

**请插件端检查上传时的 type 字段，并提供上述信息！**

---

文档版本: v1.0  
日期: 2026-04-08  
状态: 等待插件端确认上传问题
