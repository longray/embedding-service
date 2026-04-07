# 致后端团队：代码分析 v1.4 API 联调确认函

**发件人**: OpenCode Memory Plugin (插件端) 团队  
**日期**: 2026-04-07  
**主题**: API 契约确认与联调准备  
**回复**: backend-api-readiness-reply-20260407.md

---

## 1. 关键问题确认

### 1.1 CallSymbol 上传方式 ✅ 选择 A

**插件端可以提供 `caller_memory_id` 和 `callee_memory_id`**

**实现方案**:
1. 文件上传后，后端返回的 `memory_id` 保存到本地 Map（`file_path` → `memory_id`）
2. 分析 CallSymbol 时，通过 `file_path` 查询 Map 获取 `callee_memory_id`
3. 当前文件的 `caller_memory_id` 在分析时已知
4. 批量上传调用关系时附带两个 memory_id

**数据流**:
```
文件上传 → 后端返回 memory_id → 保存到 Map
                                    ↓
分析 CallSymbol → 查询 Map 获取 callee_memory_id
                                    ↓
批量上传 {caller_memory_id, callee_memory_id, line, column}
```

### 1.2 测试数据 ✅ 已准备

插件端可以提供测试数据，格式如下：

```json
{
  "project_id": "github.com/opencode-memory-plugin/test",
  "files": [
    {
      "file_path": "src/utils/crypto.ts",
      "memory_id": "memory:abc123",
      "functions": [
        {
          "name": "hashPassword",
          "line": 10
        }
      ]
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

**测试项目结构**:
```
test-project/
├── src/
│   ├── auth.ts          # 包含 validateUser，调用 hashPassword
│   ├── utils/
│   │   └── crypto.ts    # 包含 hashPassword
│   └── api.ts           # 包含 fetchUser，调用 validateUser
└── package.json
```

### 1.3 联调准备 ✅ 确认参加

- [x] 插件端可以参加 2026-04-11（周五）16:00 联调
- [x] 测试环境已准备（后端 localhost:17999）
- [x] 测试数据已准备（15 条调用关系）

---

## 2. API 契约确认

### 2.1 批量创建调用关系（BL-CA-20）

**请求格式确认**:
```json
POST /api/v1/calls/batch
{
  "calls": [
    {
      "caller_memory_id": "memory:def456",
      "callee_memory_id": "memory:abc123",
      "line": 42,
      "column": 10,
      "file_path": "src/auth.ts"
    }
  ],
  "tenant_id": "default"
}
```

**问题**:
1. 批量上传的最大数量限制是多少？（建议 100 条/批次）
2. 如果 `callee_memory_id` 不存在（目标函数未分析），如何处理？
   - 建议：返回错误列表，跳过不存在的调用

### 2.2 引用查询 API（BL-CA-21）

**响应格式确认**:
```json
GET /api/v1/memories/memory:abc123/references
{
  "status": "success",
  "memory_id": "memory:abc123",
  "references": [
    {
      "memory_id": "memory:def456",
      "file_path": "src/auth.ts",
      "line": 42,
      "column": 10,
      "caller_function": "validateUser",
      "confidence": 0.95
    }
  ],
  "total": 1
}
```

**问题**:
1. 如果函数被调用多次，是否返回多条记录？（期望：是）
2. 是否支持分页？（建议：默认返回全部，支持 `limit` 参数）

### 2.3 依赖查询 API（BL-CA-22）

**响应格式确认**:
```json
GET /api/v1/memories/memory:def456/dependencies
{
  "status": "success",
  "memory_id": "memory:def456",
  "dependencies": [
    {
      "memory_id": "memory:abc123",
      "file_path": "src/utils/crypto.ts",
      "line": 42,
      "column": 10,
      "callee_function": "hashPassword",
      "type": "internal"
    }
  ],
  "total": 1
}
```

**问题**:
1. `type` 字段由后端判断还是插件端提供？（期望：后端根据文件路径判断）

---

## 3. 实施计划确认

### 本周任务（2026-04-07 ~ 2026-04-11）

| 日期 | 任务 | 负责人 | 状态 |
|------|------|--------|------|
| **04-07** | 确认 API 契约 | 双方 | ✅ 已完成（本回复函） |
| **04-08** | 后端实现 BL-CA-20 | 后端 | 🔄 等待后端完成 |
| **04-08** | 插件端实现 memory_id 缓存 | 插件端 | 🔄 插件端今日完成 |
| **04-09** | 后端实现 BL-CA-21/22 | 后端 | 🔄 等待后端完成 |
| **04-10** | 后端单元测试 | 后端 | 🔄 等待后端完成 |
| **04-10** | 插件端准备测试脚本 | 插件端 | 🔄 插件端明日完成 |
| **04-11** | **联调测试** | **双方** | 📅 **已确认** |

### 联调议程确认

**时间**: 2026-04-11（周五）16:00-17:00 ✅  
**形式**: 线上会议 + 实时调试 ✅

**议程**:
1. **16:00-16:10** 环境确认
   - 后端服务状态检查
   - 插件端连接测试
2. **16:10-16:40** 端到端测试
   - 上传代码文件 → 获取 memory_id
   - 分析调用关系 → 批量上传
   - 查询引用/依赖 → 验证结果
3. **16:40-17:00** 问题讨论
   - 性能优化建议
   - 错误处理策略
   - 下一步计划

---

## 4. 插件端今日完成项

### 4.1 实现 memory_id 缓存

**修改文件**: `lib/code-analysis-service.js`

```javascript
// 新增 memory_id 缓存
const memoryIdCache = new Map();

// 上传文件后保存 memory_id
function saveMemoryId(filePath, memoryId) {
  memoryIdCache.set(filePath, memoryId);
}

// 查询 memory_id
function getMemoryId(filePath) {
  return memoryIdCache.get(filePath);
}

// 分析 CallSymbol 时获取 callee_memory_id
function resolveCalleeMemoryId(target, filePath) {
  // 1. 从 imports 解析目标文件路径
  // 2. 查询 memoryIdCache
  // 3. 返回 memory_id 或 null
}
```

### 4.2 准备测试脚本

**测试脚本**: `tests/integration/calls-api.test.js`

测试场景：
1. 上传包含调用关系的代码文件
2. 验证 memory_id 缓存
3. 批量上传调用关系
4. 查询引用和依赖
5. 验证结果准确性

---

## 5. 测试环境

### 5.1 插件端测试环境

- **项目路径**: `D:/github/opencode-memory-plugin/test-project`
- **后端地址**: `http://localhost:17999`
- **API Key**: 从环境变量读取

### 5.2 快速测试命令

```bash
# 插件端运行测试
cd D:/github/opencode-memory-plugin/opencode-memory-plugin
npm test -- tests/integration/calls-api.test.js

# 分析测试项目
node cli/code-analyzer.cjs analyze test-project/src --format tree
```

---

## 6. 风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 后端 API 延迟 | 联调延期 | 插件端先使用 mock 数据测试 |
| memory_id 缓存失效 | 调用关系上传失败 | 添加缓存持久化（可选） |
| 跨文件调用解析失败 | 部分调用关系丢失 | 记录警告，继续处理其他调用 |

---

## 7. 联系方式

- **插件端负责人**: OpenCode Memory Plugin Team
- **技术对接**: 通过本信函回复或 GitHub Issue
- **紧急联系**: 在 BACKLOG.md 中标注阻塞问题

---

**期待与贵团队的联调，共同推进 v1.4 的顺利实施！**

如有任何疑问或需要调整，请随时联系。

---

*文档版本: v1.0*  
*日期: 2026-04-07*  
*状态: 已回复，等待后端确认*
