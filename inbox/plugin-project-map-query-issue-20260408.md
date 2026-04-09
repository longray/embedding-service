# 插件端反馈：项目地图查询返回空数据
**发件人**: OpenCode Memory Plugin (插件端)  
**日期**: 2026-04-08  
**主题**: 后端修复验证 - 项目地图端点返回空结果  
**优先级**: P1 (High)  
**状态**: 🔍 需要后端排查
---
## 1. 验证结果摘要

感谢后端团队快速修复所有问题！我们已完成验证，发现一个新问题：

| 修复项 | 状态 | 备注 |
|--------|------|------|
| 上传成功但数据未写入 | ✅ **已修复** | 新文件正确插入 |
| 代码数据 hash 去重 | ✅ **已修复** | code 类型跳过去重 |
| 项目地图边数据 | ⚠️ **部分修复** | 关系存储正常，但查询返回空 |
| API 字段名统一 | ✅ **已修复** | abstract/overview 统一 |
| 部署验证 | ✅ **已修复** | 容器已重启 |

---
## 2. 发现问题：项目地图端点返回空数据

### 2.1 现象

**数据已正确存储：**
- ✅ 3 个文件上传成功，带有 `project_id: "test-integration-project"`
- ✅ 调用关系创建成功，存储在 `memory_relation` 表
- ✅ 直接查询关系 API 返回数据正常

**但项目地图返回空：**
```bash
GET /api/v1/projects/test-integration-project/map
Response: {
  "file_tree": {},
  "module_dependencies": [],
  "hot_files": [],
  "statistics": {
    "total_files": 0,
    "total_functions": 0,
    "total_classes": 0,
    "avg_complexity": 0
  }
}
```

### 2.2 验证步骤

**步骤 1: 上传文件** ✅
```javascript
POST /api/v1/memories
{
  "memories": [{
    "type": "code",
    "content": "export function hashPassword...",
    "abstract": "Password hashing utilities",
    "overview": "Utility functions for password hashing",
    "project_id": "test-integration-project",
    "metadata": {
      "file_path": "src/utils/crypto.ts",
      "language": "typescript"
    }
  }],
  "tenant_id": "default"
}
// Response: 200 OK, memory_ids: ["memory:xxx"]
```

**步骤 2: 创建调用关系** ✅
```javascript
POST /api/v1/calls/batch
{
  "calls": [
    {"caller_id": "memory:auth", "callee_id": "memory:crypto", "relation_type": "calls"}
  ]
}
// Response: 200 OK, {"success": true, "created": 2}
```

**步骤 3: 直接查询关系** ✅
```bash
GET /api/v1/memories/memory:auth/relations
Response: {
  "relations": [
    {"target_id": "memory:crypto", "type": "calls"}
  ]
}
// ✅ 关系存在！
```

**步骤 4: 查询项目地图** ❌
```bash
GET /api/v1/projects/test-integration-project/map
Response: {
  "file_tree": {},  // 空！
  "module_dependencies": [],  // 空！
  ...
}
// ❌ 返回空数据
```

---
## 3. 根因分析（推测）

### 可能原因 1：项目地图查询条件不匹配

项目地图端点可能使用了错误的查询条件，导致无法找到已存储的数据。

**建议检查：**
- 查询是否使用了正确的 `project_id` 字段名
- 是否使用了 `type::record()` 处理 RecordID
- 查询条件是否区分大小写

### 可能原因 2：SurrealDB 查询语法问题

类似于之前的 `metadata->file_path` vs `metadata.file_path` 问题。

**建议检查：**
- 项目地图查询中的 SurrealDB 语法
- 是否正确使用了 `->` 和 `.` 操作符

### 可能原因 3：数据关联问题

文件和关系可能存储在不同的表中，但项目地图查询时未正确关联。

**建议检查：**
- `memories` 表和 `memory_relation` 表的关联查询
- 是否使用了正确的 JOIN 条件

---
## 4. 需要后端协助

### 4.1 调试信息

请提供以下信息以便排查：

1. **项目地图查询的 SurrealDB SQL 语句**
2. **手动执行查询的结果**
   ```sql
   SELECT * FROM memories WHERE project_id = "test-integration-project";
   ```
3. **memory_relation 表中的数据示例**
   ```sql
   SELECT * FROM memory_relation WHERE relationship_type = "calls" LIMIT 5;
   ```

### 4.2 测试建议

建议后端在修复后进行以下测试：

```bash
# 1. 上传测试文件
POST /api/v1/memories
{"memories": [{"type": "code", "project_id": "test-project", ...}]}

# 2. 创建调用关系
POST /api/v1/calls/batch
{"calls": [...]}

# 3. 验证数据存在
GET /api/v1/memories/{id}
GET /api/v1/memories/{id}/relations

# 4. 验证项目地图
GET /api/v1/projects/test-project/map
# 应该返回非空的 file_tree 和 module_dependencies
```

---
## 5. 测试环境

**测试项目**: `test-integration-project`  
**Tenant**: `default`  
**上传文件**: 3 个 TypeScript 文件  
**调用关系**: 2 条（已确认创建成功）  
**后端地址**: http://localhost:17999  
**测试时间**: 2026-04-08

---
## 6. 附件

### 6.1 测试脚本
完整测试脚本：`D:\github\opencode-memory-plugin\test-integration\verify-project-map-fix.js`

### 6.2 相关 Memory IDs
- crypto.ts: `memory:01JQ...` (示例)
- auth.ts: `memory:01JR...` (示例)
- api.ts: `memory:01JS...` (示例)

---
## 7. 下一步

**插件端**: 等待后端修复项目地图查询问题  
**后端**: 请检查项目地图端点的查询逻辑  
**最终联调**: 2026-04-11 16:00（建议在此之前修复）

---

**文档版本**: v1.0  
**状态**: 已发送，等待后端回复
