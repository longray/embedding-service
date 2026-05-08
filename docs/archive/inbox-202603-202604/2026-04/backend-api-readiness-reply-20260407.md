# 致插件端团队：代码分析 v1.4 API 就绪状态回复函

**发件人**: Embedding Service (后端) 团队  
**日期**: 2026-04-07  
**主题**: 代码分析 v1.4 API 就绪状态确认与联调安排  
**回复**: plugin-to-backend-api-readiness-check-20260407.md

---

## 1. API 就绪状态确认

### 1.1 Phase 3 API（已完成 ✅）

| 端点 | 方法 | 状态 | 说明 |
|------|------|------|------|
| `GET /api/v1/projects/{id}/map` | GET | ✅ **已实现** | BL-CA-23 完成，返回 file_tree, module_dependencies, hot_files, statistics |
| `GET /api/v1/projects/{id}/stats` | GET | ✅ **已实现** | BL-CA-25 完成，返回 total_files, total_functions, total_classes, avg_complexity, max_complexity |

### 1.2 Phase 2 API（开发中 🔄）

| 端点 | 方法 | 状态 | 预计完成 | 说明 |
|------|------|------|----------|------|
| `POST /api/v1/calls/batch` | POST | 🔄 **开发中** | 2026-04-11 | BL-CA-20，批量创建调用关系 |
| `GET /api/v1/memories/{id}/references` | GET | 🔄 **开发中** | 2026-04-11 | BL-CA-21，查询谁调用了此函数 |
| `GET /api/v1/memories/{id}/dependencies` | GET | 🔄 **开发中** | 2026-04-11 | BL-CA-22，查询此函数调用了谁 |

---

## 2. API 契约确认

### 2.1 批量创建调用关系（BL-CA-20）

**端点**: `POST /api/v1/calls/batch`

**请求格式**:
```json
{
  "calls": [
    {
      "caller_memory_id": "memory:abc123",  // 调用者记忆 ID（必需）
      "callee_memory_id": "memory:def456",  // 被调用者记忆 ID（必需）
      "line": 42,                            // 调用位置行号（可选）
      "column": 10,                          // 调用位置列号（可选）
      "file_path": "src/auth.ts"           // 调用所在文件（可选，用于调试）
    }
  ],
  "tenant_id": "default"                   // 租户 ID（可选，默认 default）
}
```

**响应格式**:
```json
{
  "status": "success",
  "created": 5,
  "errors": []
}
```

**关键问题**: 插件端是否能提供 `caller_memory_id` 和 `callee_memory_id`？

- **如果能提供**: 后端直接创建关系，无需查找
- **如果不能提供**: 后端需要通过 `(project_id, file_path, function_name)` 查找对应的 memory_id，这会增加查询开销

**建议**: 插件端在上传代码文件时保存返回的 memory_id，在提取 CallSymbol 时使用这些 ID。

### 2.2 引用查询 API（BL-CA-21）

**端点**: `GET /api/v1/memories/{id}/references?tenant_id=default`

**响应格式**:
```json
{
  "status": "success",
  "memory_id": "memory:abc123",
  "references": [
    {
      "memory_id": "memory:xxx",           // 调用者记忆 ID
      "file_path": "src/utils.ts",        // 调用者文件路径
      "line": 42,                          // 调用位置
      "column": 10,
      "caller_function": "validateUser",   // 调用函数名
      "confidence": 0.95                   // 置信度（预留）
    }
  ],
  "total": 5
}
```

**递归查询**: Phase 2 仅支持单层查询，递归查询（调用链）计划在 Phase 3+ 实现。

### 2.3 依赖查询 API（BL-CA-22）

**端点**: `GET /api/v1/memories/{id}/dependencies?tenant_id=default`

**响应格式**:
```json
{
  "status": "success",
  "memory_id": "memory:abc123",
  "dependencies": [
    {
      "memory_id": "memory:yyy",           // 被调用者记忆 ID
      "file_path": "src/crypto.ts",       // 被调用者文件路径
      "line": 15,                          // 调用位置
      "column": 10,
      "callee_function": "hashPassword",   // 被调用函数名
      "type": "internal"                   // 依赖类型：internal/external/builtin
    }
  ],
  "total": 3
}
```

**依赖分类**: 基于文件路径判断
- `internal`: 同一项目内的文件（file_path 以项目路径开头）
- `external`: 外部包（npm/pip/cargo 等）
- `builtin`: 内置模块（node:fs, os, sys 等）

### 2.4 代码地图 API（BL-CA-23，已实现）

**端点**: `GET /api/v1/projects/{id}/map?tenant_id=default`

**响应格式**:
```json
{
  "status": "success",
  "project_id": "github.com/user/repo",
  "file_tree": [
    {
      "name": "src",
      "type": "directory",
      "path": "src",
      "children": [
        {
          "name": "auth.ts",
          "type": "file",
          "path": "src/auth.ts",
          "complexity": 8.5,
          "function_count": 5,
          "class_count": 1
        }
      ]
    }
  ],
  "module_dependencies": [
    {
      "from": "src/auth.ts",
      "to": "src/utils/crypto.ts",
      "type": "import"
    }
  ],
  "hot_files": [
    "src/auth.ts",
    "src/utils/api.ts"
  ],
  "statistics": {
    "total_files": 45,
    "total_functions": 150,
    "total_classes": 30,
    "avg_complexity": 5.2,
    "max_complexity": 15
  }
}
```

### 2.5 代码统计 API（BL-CA-25，已实现）

**端点**: `GET /api/v1/projects/{id}/stats?tenant_id=default`

**响应格式**:
```json
{
  "status": "success",
  "project_id": "github.com/user/repo",
  "total_files": 45,
  "total_functions": 150,
  "total_classes": 30,
  "avg_complexity": 5.2,
  "max_complexity": 15
}
```

---

## 3. 实施计划

### 本周任务（2026-04-07 ~ 2026-04-11）

| 日期 | 任务 | 负责人 | 交付物 |
|------|------|--------|--------|
| **04-07** | 确认 API 契约 | 双方 | 本回复函 |
| **04-08** | 后端实现 BL-CA-20 | 后端 | `POST /api/v1/calls/batch` |
| **04-09** | 后端实现 BL-CA-21/22 | 后端 | `GET /references`, `GET /dependencies` |
| **04-10** | 后端单元测试 | 后端 | 测试用例 |
| **04-11** | **联调测试** | **双方** | **端到端验证** |

### 联调安排

**时间**: 2026-04-11（周五）16:00-17:00  
**形式**: 线上会议 + 实时调试  
**议程**:
1. 16:00-16:15 环境确认
2. 16:15-16:45 端到端测试（上传→分析→查询）
3. 16:45-17:00 问题讨论和下一步计划

---

## 4. 需要插件端提供的信息

### 4.1 关键问题（请回复）

1. **CallSymbol 上传方式**:
   - [ ] 插件端可以提供 `caller_memory_id` 和 `callee_memory_id`
   - [ ] 插件端只能提供 `(file_path, function_name)`，需要后端查找

2. **测试数据**:
   - [ ] 插件端可以提供测试用的 CallSymbol 数据（10-20 条）
   - [ ] 插件端可以提供测试项目（包含调用关系的代码文件）

3. **联调准备**:
   - [ ] 插件端可以在周五 16:00 参加联调
   - [ ] 需要调整时间

### 4.2 测试数据格式

如果插件端能提供测试数据，请按以下格式提供：

```json
{
  "project_id": "github.com/test/repo",
  "files": [
    {
      "file_path": "src/auth.ts",
      "memory_id": "memory:abc123",  // 上传后返回的 ID
      "functions": [
        {
          "name": "validateUser",
          "calls": [
            {
              "target": "hashPassword",
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

---

## 5. 测试环境

### 5.1 后端测试环境

- **URL**: `http://localhost:17999`
- **状态**: 可用
- **Meilisearch**: `http://localhost:7700`
- **SurrealDB**: `http://localhost:8000`

### 5.2 快速测试命令

```bash
# 启动后端服务
cd D:/embedding_service
uv run python -m wrapper.src.main

# 测试项目统计 API
curl http://localhost:17999/api/v1/projects/github.com/test/repo/stats

# 测试代码地图 API
curl http://localhost:17999/api/v1/projects/github.com/test/repo/map
```

---

## 6. 当前后端已完成的工作

### 6.1 Phase 1（已完成）

- ✅ BL-CA-18: Schema 扩展
  - Meilisearch 新增 `code_has_exports` filterable 字段
  - SurrealDB `memory_relation` 表已支持 `calls` 类型

### 6.2 Phase 3（已完成）

- ✅ BL-CA-23: 代码地图 API
- ✅ BL-CA-24: 搜索增强（code_filter 扩展）
- ✅ BL-CA-25: 代码统计 API

### 6.3 Phase 5（已完成）

- ✅ BL-CA-28: 分析结果缓存（LRU 缓存）

---

## 7. 联系方式

- **后端负责人**: Embedding Service Team
- **技术对接**: 通过本信函回复或 GitHub Issue
- **紧急联系**: 在 BACKLOG.md 中标注阻塞问题

---

**期待与贵团队的联调，共同推进 v1.4 的顺利实施！**

如有任何疑问或需要调整，请随时联系。

---

*文档版本: v1.0*  
*日期: 2026-04-07*  
*状态: 已回复，等待插件端确认*
