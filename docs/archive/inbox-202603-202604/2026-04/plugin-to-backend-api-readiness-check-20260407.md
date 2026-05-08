# 致后端团队：代码分析 v1.4 API 就绪状态确认函

**发件人**: OpenCode Memory Plugin (插件端) 团队  
**日期**: 2026-04-07  
**主题**: 代码分析 v1.4 API 就绪状态确认与联调请求

---

## 1. 背景

插件端已完成 BL-CA-18（调用关系提取与可视化）的全部实现：

- ✅ Oxc 路径（JS/TS）调用关系提取
- ✅ Tree-sitter 路径（Python/Go/Rust/Java）调用关系提取
- ✅ 调用关系可视化输出（表格 + 树形格式）

现在需要与后端 API 对接，实现调用关系的存储和查询功能。

---

## 2. 需要确认的 API 端点

根据 [BACKEND-ALIGNMENT-v1.4.md](../docs/BACKEND-ALIGNMENT-v1.4.md)，需要以下 API：

### Phase 2: 调用关系 API（BL-CA-20~22）

| 端点 | 方法 | 状态 | 说明 |
|------|------|------|------|
| `POST /api/v1/calls/batch` | POST | ❓ 待确认 | 批量创建调用关系 |
| `GET /api/v1/memories/{id}/references` | GET | ❓ 待确认 | 查询谁调用了此函数 |
| `GET /api/v1/memories/{id}/dependencies` | GET | ❓ 待确认 | 查询此函数调用了谁 |

### Phase 3: 代码地图 API（BL-CA-23~25）

| 端点 | 方法 | 状态 | 说明 |
|------|------|------|------|
| `GET /api/v1/projects/{id}/map` | GET | ❓ 待确认 | 项目代码地图 |
| `GET /api/v1/projects/{id}/stats` | GET | ❓ 待确认 | 项目统计 |

---

## 3. 插件端已准备的数据格式

### CallSymbol 结构（已确认）

```json
{
  "target": "hashPassword",
  "file_path": "src/utils/crypto.ts",
  "line": 42,
  "column": 10
}
```

### 批量上传示例

```json
POST /api/v1/calls/batch
{
  "calls": [
    {
      "from_memory_id": "mem_xxx",
      "to_memory_id": "mem_yyy",
      "target": "hashPassword",
      "file_path": "src/auth.ts",
      "line": 42,
      "column": 10
    }
  ]
}
```

---

## 4. 需要后端确认的问题

### 4.1 API 就绪状态

1. **Phase 2 API** (`/api/v1/calls/batch`, `/references`, `/dependencies`)
   - 是否已实现？
   - 预计何时可用？

2. **Phase 3 API** (`/projects/{id}/map`, `/projects/{id}/stats`)
   - 是否已实现？
   - 预计何时可用？

### 4.2 API 契约确认

1. **批量创建调用关系**
   - 请求格式是否正确？
   - 是否需要 `from_memory_id` 和 `to_memory_id`？
   - 还是通过 `(project_id, file_path, function_name)` 查找？

2. **引用查询**
   - 返回格式是什么？
   - 是否支持递归查询（深度可配置）？

3. **依赖查询**
   - 返回格式是什么？
   - 是否支持分类（internal/external/builtin）？

### 4.3 测试环境

- 后端测试环境是否可用？
- 是否需要提供测试数据？

---

## 5. 插件端测试准备

插件端已准备测试代码：

```javascript
// 测试批量上传调用关系
const testCalls = [
  {
    target: "helper",
    file_path: "src/test.js",
    line: 8,
    column: 12
  },
  {
    target: "obj.method",
    file_path: "src/test.js",
    line: 10,
    column: 12
  }
];

// 等待后端 API 就绪后测试
```

---

## 6. 请求

1. **确认 API 就绪状态** - 请回复每个 API 的实现状态
2. **提供 API 文档** - 如有更新，请提供最新 API 契约
3. **安排联调时间** - 建议本周五 16:00 进行联调
4. **提供测试环境** - 确认测试环境可用性

---

## 7. 联系方式

- **插件端负责人**: OpenCode Memory Plugin Team
- **技术对接**: 通过本信函回复或 GitHub Issue
- **紧急联系**: 在 BACKLOG.md 中标注阻塞问题

---

期待贵团队的回复，让我们共同推进 v1.4 的顺利实施！

---

*文档版本: v1.0*  
*日期: 2026-04-07*  
*状态: 待后端确认*
