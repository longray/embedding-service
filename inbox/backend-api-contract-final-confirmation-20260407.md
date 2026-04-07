# 致插件端团队：代码分析 v1.4 API 契约最终确认函

**发件人**: Embedding Service (后端) 团队  
**日期**: 2026-04-07  
**主题**: API 契约细节最终确认  
**回复**: plugin-reply-to-backend-20260407.md

---

## 1. API 契约细节确认 ✅

所有 API 契约细节已确认，双方达成一致：

### 1.1 批量创建调用关系（BL-CA-20）

**端点**: `POST /api/v1/calls/batch`

**请求格式**（已确认）：
```json
{
  "calls": [
    {
      "caller_memory_id": "memory:def456",  // 调用者记忆 ID（必需）
      "callee_memory_id": "memory:abc123",  // 被调用者记忆 ID（必需）
      "line": 42,                            // 调用位置行号（可选）
      "column": 10,                          // 调用位置列号（可选）
      "file_path": "src/auth.ts"           // 调用所在文件（可选，用于调试）
    }
  ],
  "tenant_id": "default"                   // 租户 ID（可选，默认 default）
}
```

**响应格式**（已确认）：
```json
{
  "status": "success",
  "created": 5,
  "errors": [
    {
      "index": 2,
      "caller_memory_id": "memory:xxx",
      "callee_memory_id": "memory:yyy",
      "error": "callee_memory_id not found"
    }
  ]
}
```

**约束条件**（已确认）：
- ✅ **最大批量**: 100 条/批次
- ✅ **错误处理**: callee_memory_id 不存在时返回错误列表，跳过不存在的调用，继续处理其他
- ✅ **存储方式**: SurrealDB `memory_relation` 表，`relationship_type="calls"`

### 1.2 引用查询 API（BL-CA-21）

**端点**: `GET /api/v1/memories/{id}/references?tenant_id=default&limit=50`

**响应格式**（已确认）：
```json
{
  "status": "success",
  "memory_id": "memory:abc123",
  "references": [
    {
      "memory_id": "memory:def456",        // 调用者记忆 ID
      "file_path": "src/auth.ts",         // 调用者文件路径
      "line": 42,                          // 调用位置
      "column": 10,
      "caller_function": "validateUser",   // 调用函数名
      "confidence": 0.95                   // 置信度（预留，默认 0.95）
    }
  ],
  "total": 5
}
```

**约束条件**（已确认）：
- ✅ **多条记录**: 如果函数被调用多次，返回多条记录
- ✅ **分页**: 默认返回全部，支持 `limit` 参数（默认 50，最大 200）
- ✅ **查询深度**: Phase 2 仅支持单层查询

### 1.3 依赖查询 API（BL-CA-22）

**端点**: `GET /api/v1/memories/{id}/dependencies?tenant_id=default&limit=50`

**响应格式**（已确认）：
```json
{
  "status": "success",
  "memory_id": "memory:def456",
  "dependencies": [
    {
      "memory_id": "memory:abc123",        // 被调用者记忆 ID
      "file_path": "src/utils/crypto.ts", // 被调用者文件路径
      "line": 42,                          // 调用位置
      "column": 10,
      "callee_function": "hashPassword",   // 被调用函数名
      "type": "internal"                   // 依赖类型
    }
  ],
  "total": 3
}
```

**依赖类型判断**（已确认）：
- ✅ **internal**: 同一项目内的文件（file_path 以项目路径开头）
- ✅ **external**: 外部包（npm/pip/cargo 等，通过 import 路径判断）
- ✅ **builtin**: 内置模块（node:fs, os, sys 等）

**判断逻辑**（后端实现）：
```python
def get_dependency_type(import_path: str, project_id: str) -> str:
    if import_path.startswith("node:") or import_path in BUILTIN_MODULES:
        return "builtin"
    elif import_path.startswith(".") or import_path.startswith("/"):
        return "internal"
    else:
        return "external"
```

### 1.4 代码地图 API（BL-CA-23，已实现）

**端点**: `GET /api/v1/projects/{id}/map?tenant_id=default`

**状态**: ✅ **已实现，可直接使用**

### 1.5 代码统计 API（BL-CA-25，已实现）

**端点**: `GET /api/v1/projects/{id}/stats?tenant_id=default`

**状态**: ✅ **已实现，可直接使用**

---

## 2. 实施计划最终确认 ✅

| 日期 | 任务 | 负责人 | 状态 |
|------|------|--------|------|
| **04-07** | ✅ 确认 API 契约 | 双方 | **已完成** |
| **04-08** | 🔄 后端实现 BL-CA-20 | 后端 | **后端今日完成** |
| **04-08** | 🔄 插件端实现 memory_id 缓存 | 插件端 | 插件端今日完成 |
| **04-09** | 🔄 后端实现 BL-CA-21/22 | 后端 | **后端明日完成** |
| **04-10** | 🔄 后端单元测试 | 后端 | **后端完成** |
| **04-10** | 🔄 插件端准备测试脚本 | 插件端 | 插件端完成 |
| **04-11** | 📅 **联调测试** | **双方** | **已确认** |

**后端承诺**：
- 04-08 完成 BL-CA-20（批量创建调用关系）
- 04-09 完成 BL-CA-21/22（引用/依赖查询）
- 04-10 完成单元测试
- 04-11 准时参加联调

---

## 3. 联调安排最终确认 ✅

**时间**: 2026-04-11（周五）16:00-17:00  
**形式**: 线上会议 + 实时调试  
**地点**: 双方开发环境（后端 localhost:17999，插件端本地）

### 联调议程（已确认）

| 时间 | 内容 | 负责人 |
|------|------|--------|
| **16:00-16:10** | 环境确认 | 双方 |
| | - 后端服务状态检查 | 后端 |
| | - 插件端连接测试 | 插件端 |
| **16:10-16:40** | 端到端测试 | 双方 |
| | - 上传代码文件 → 获取 memory_id | 插件端 |
| | - 分析调用关系 → 批量上传 | 插件端 |
| | - 查询引用/依赖 → 验证结果 | 双方 |
| **16:40-17:00** | 问题讨论 | 双方 |
| | - 性能优化建议 | 双方 |
| | - 错误处理策略 | 双方 |
| | - 下一步计划 | 双方 |

---

## 4. 测试数据确认 ✅

插件端提供的测试数据已确认：

```json
{
  "project_id": "github.com/opencode-memory-plugin/test",
  "files": [
    {
      "file_path": "src/utils/crypto.ts",
      "memory_id": "memory:abc123",
      "functions": [{"name": "hashPassword", "line": 10}]
    },
    {
      "file_path": "src/auth.ts",
      "memory_id": "memory:def456",
      "functions": [
        {
          "name": "validateUser",
          "calls": [
            {
              "target": "hashPassword",
              "target_file": "src/utils/crypto.ts",
              "line": 42,
              "column": 10
            }
          ]
        }
      ]
    }
  ]
}
```

**后端已准备**：
- ✅ 测试环境（localhost:17999）
- ✅ 测试数据库（SurrealDB + Meilisearch）
- ✅ 测试项目空间

---

## 5. 风险应对确认 ✅

| 风险 | 应对措施 | 状态 |
|------|---------|------|
| 后端 API 延迟 | 后端承诺 04-11 前完成，插件端准备 mock 数据备用 | ✅ 已确认 |
| memory_id 缓存失效 | 后端 API 返回错误列表，插件端记录警告 | ✅ 已确认 |
| 跨文件调用解析失败 | 后端跳过不存在的调用，继续处理其他 | ✅ 已确认 |

---

## 6. 最终确认清单 ✅

| 项目 | 状态 | 说明 |
|------|------|------|
| API 契约细节 | ✅ 已确认 | 所有字段、格式、约束已达成一致 |
| 批量上传限制 | ✅ 已确认 | 100 条/批次 |
| 错误处理策略 | ✅ 已确认 | 返回错误列表，跳过不存在调用 |
| 依赖类型判断 | ✅ 已确认 | 后端根据文件路径判断 |
| 联调时间 | ✅ 已确认 | 2026-04-11 16:00-17:00 |
| 测试数据 | ✅ 已确认 | 15 条调用关系 |
| 实施计划 | ✅ 已确认 | 双方按计划执行 |

---

## 7. 后端今日开始实施

后端团队立即开始实施：

1. **BL-CA-20**: `POST /api/v1/calls/batch`
   - 实现批量创建调用关系
   - 支持 100 条/批次限制
   - 返回错误列表

2. **BL-CA-21**: `GET /api/v1/memories/{id}/references`
   - 实现引用查询
   - 支持 limit 参数

3. **BL-CA-22**: `GET /api/v1/memories/{id}/dependencies`
   - 实现依赖查询
   - 支持依赖类型判断

---

## 8. 联系方式

- **后端负责人**: Embedding Service Team
- **技术对接**: 通过本信函回复或 GitHub Issue
- **紧急联系**: 在 BACKLOG.md 中标注阻塞问题

---

**API 契约已最终确认，后端立即开始实施！**

期待 04-11 的联调，共同推进 v1.4 的顺利实施！

---

*文档版本: v1.0*  
*日期: 2026-04-07*  
*状态: API 契约最终确认，后端开始实施*
