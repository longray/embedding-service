# 插件端反馈：项目地图边（edges）数据缺失
**发件人**: OpenCode Memory Plugin (插件端)  
**日期**: 2026-04-08  
**主题**: Scene 2 测试 - 项目地图返回节点但缺少边数据  
**优先级**: P1 (High)  
**状态**: 🔍 需要后端排查
---
## 1. 测试概览
**测试场景**: Scene 2 - 项目地图可视化  
**测试时间**: 2026-04-08  
**后端版本**: 2.4.1 (已部署修复)  
**测试结果**: ⚠️ 部分通过 - 节点正常，边数据缺失
---
## 2. 测试步骤
### 2.1 上传测试文件
成功上传 3 个 TypeScript 文件：
- `src/utils/crypto.ts` - 3 个工具函数
- `src/auth.ts` - AuthService 类（调用 crypto）
- `src/api.ts` - ApiService 类（调用 auth）

**上传响应**: ✅ 200 OK，返回 memory_ids

### 2.2 创建调用关系
使用 `POST /api/v1/calls/batch` 创建 6 条调用关系：
```json
{
  "calls": [
    {"caller_id": "auth.ts", "callee_id": "crypto.ts", "relation_type": "calls"},
    {"caller_id": "api.ts", "callee_id": "auth.ts", "relation_type": "calls"}
  ]
}
```

**创建响应**: ✅ 200 OK，返回成功消息

### 2.3 查询项目地图
使用 `GET /api/v1/projects/test-integration-project/map`

**响应结果**:
```json
{
  "file_tree": {
    "src": {
      "utils": {
        "crypto.ts": { "id": "memory:xxx", "type": "code" }
      },
      "auth.ts": { "id": "memory:yyy", "type": "code" },
      "api.ts": { "id": "memory:zzz", "type": "code" }
    }
  },
  "module_dependencies": [],  // ❌ 空数组！
  "hot_files": [
    { "file_path": "src/utils/crypto.ts", "access_count": 1 },
    { "file_path": "src/auth.ts", "access_count": 1 },
    { "file_path": "src/api.ts", "access_count": 1 }
  ]
}
```

---
## 3. 发现的问题
### 3.1 问题：module_dependencies 为空
**现象**: `module_dependencies` 数组为空，但已成功创建调用关系

**期望**: 应该返回 6 条边（调用关系）：
```json
"module_dependencies": [
  {"source": "auth.ts", "target": "crypto.ts", "type": "calls"},
  {"source": "api.ts", "target": "auth.ts", "type": "calls"}
]
```

**实际**: 返回空数组 `[]`

### 3.2 问题：/api/v1/calls 端点 404
**现象**: 尝试查询调用关系时，`GET /api/v1/calls` 返回 404

**问题**: 是否有查询调用关系的端点？还是只能通过 references/dependencies 查询？

### 3.3 问题：统计信息为 0
**现象**: 项目统计返回 `total_functions: 0` 和 `total_classes: 0`

**期望**: 应该统计代码中的函数和类数量

**实际**: 始终返回 0

---
## 4. 根因分析（推测）
### 可能原因 1：调用关系未关联到项目地图
调用关系可能存储在独立的表中，但项目地图查询时未关联这些数据。

**建议检查**:
- `calls` 表/集合中的数据是否正确关联到项目
- 项目地图查询是否 JOIN 了 calls 表

### 可能原因 2：不同的关系类型
我们使用 `relation_type: "calls"`，但项目地图可能期望其他类型（如 `"depends_on"` 或 `"imports"`）。

**建议检查**:
- 项目地图期望的关系类型是什么
- 是否需要转换关系类型

### 可能原因 3：缺少代码分析数据
项目统计为 0 可能是因为：
- 未解析 `metadata.code_analysis` 字段
- 统计逻辑未实现

---
## 5. 需要后端协助
### 5.1 确认问题
1. 调用关系是否正确存储在数据库中？
2. 项目地图查询是否包含调用关系？
3. 统计信息是否已实现？

### 5.2 提供调试信息
请提供以下信息以便排查：
1. SurrealDB 中 `calls` 表的数据示例
2. 项目地图查询的 SQL/查询语句
3. 统计信息的计算逻辑

### 5.3 可能的修复方案
**方案 A**: 修改项目地图查询，JOIN calls 表获取边数据
**方案 B**: 提供单独的调用关系查询端点
**方案 C**: 在创建调用关系时，同时更新项目地图的缓存

---
## 6. 测试环境
**测试项目**: `test-integration-project`  
**Tenant**: `default`  
**测试文件**: 3 个 TypeScript 文件  
**调用关系**: 6 条（已确认创建成功）  
**后端地址**: http://localhost:17999

---
## 7. 附件
### 7.1 测试脚本
测试脚本位于：`D:\github\opencode-memory-plugin\test-integration\scene2-project-map-test.js`

可重复运行以验证问题。

### 7.2 相关 API 调用日志
```bash
# 创建调用关系
POST /api/v1/calls/batch
Body: {"calls": [...]}
Response: 200 {"success": true, "created": 6}

# 查询项目地图
GET /api/v1/projects/test-integration-project/map
Response: 200 {"file_tree": {...}, "module_dependencies": [], ...}
```

---
## 8. 下一步
**插件端**: 等待后端修复或提供替代方案  
**后端**: 请确认问题并提供修复时间估计  
**联调会议**: 2026-04-11 16:00（可讨论此问题）

---
**文档版本**: v1.0  
**状态**: 已发送，等待后端回复
