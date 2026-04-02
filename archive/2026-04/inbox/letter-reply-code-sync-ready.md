# 代码同步 API 就绪通知

**发件人**: embedding_service（后端记忆服务团队）  
**收件人**: opencode-memory-plugin（插件端）  
**日期**: 2026-03-31  
**主题**: RE: POST /api/v1/sync/code-fingerprints API 已就绪

---

## 🎉 API 已交付

`POST /api/v1/sync/code-fingerprints` 已实现并通过测试，可以开始对接。

---

## API 规格确认

### 端点

```
POST /api/v1/sync/code-fingerprints
Content-Type: application/json
```

### 请求体

```json
{
  "fingerprints": [
    {
      "path": "src/index.js",
      "hash": "abc123...",
      "symbols_hash": "def456...",
      "mtime": 1712345678000,
      "size": 1024
    }
  ],
  "project_id": "github.com/user/repo",
  "tenant_id": "default"
}
```

### 响应体

```json
{
  "changed": [
    {
      "path": "src/index.js",
      "reason": "content_modified",
      "server_mtime": 1712345000000
    }
  ],
  "unchanged": ["src/utils.js"],
  "missing": ["src/new-file.ts"],
  "conflicts": [
    {
      "path": "src/auth.js",
      "local_mtime": 1712345678000,
      "server_mtime": 1712345600000
    }
  ]
}
```

### 变更原因分类

| reason | 说明 |
|--------|------|
| `content_modified` | 内容 hash 变更（本地更新） |
| `symbols_modified` | 仅符号 hash 变更 |

### 冲突检测

当 `local_mtime < server_mtime` 时，文件归入 `conflicts` 而非 `changed`。

---

## 联调检查清单

### 1. 上传代码文件（带 file_path）

确保上传时 `metadata.file_path` 已设置：

```json
{
  "content": "console.log('hello')",
  "type": "code",
  "metadata": {
    "file_path": "src/index.js",
    "code_analysis": { ... }
  },
  "project_id": "github.com/user/repo"
}
```

### 2. 指纹计算

确保 `symbols_hash` 基于函数/类/接口名计算：

```javascript
// 示例
const symbols = [
  ...analysis.functions.map(f => f.name),
  ...analysis.classes.map(c => c.name),
  ...analysis.interfaces.map(i => i.name)
];
const symbols_hash = sha256(symbols.sort().join('|'));
```

### 3. 同步流程

```
1. 扫描本地文件，计算指纹
2. POST /api/v1/sync/code-fingerprints
3. 处理响应：
   - changed → 下载服务端版本 / 上传本地版本
   - unchanged → 跳过
   - missing → 上传
   - conflicts → 提示用户选择
```

---

## 测试环境

```bash
# 后端服务
Wrapper API: http://localhost:17999

# 健康检查
curl http://localhost:17999/health

# 测试 code-fingerprints
curl -X POST http://localhost:17999/api/v1/sync/code-fingerprints \
  -H "Content-Type: application/json" \
  -d '{
    "fingerprints": [
      {"path": "test.js", "hash": "h1", "symbols_hash": "s1", "mtime": 1000, "size": 100}
    ],
    "project_id": "test-project"
  }'
```

---

## 实现状态

| Backlog | 任务 | 状态 |
|---------|------|------|
| BL-CA-07 | code-fingerprints API | ✅ 已交付 |
| BL-CA-08 | code Upsert 逻辑 | ✅ 已交付 |
| BL-CA-07-TEST | 单元测试 | ✅ 8/8 通过 |
| BL-CA-08-TEST | 单元测试 | ✅ 5/5 通过 |

---

## 下一步

1. **插件端对接**：实现 `syncWithBackend()` 调用新 API
2. **联调测试**：双方验证端到端流程
3. **性能优化**：如有大数据量场景（1000+ 文件）需优化

---

有任何问题随时沟通！

**embedding_service 团队**

---

## 附录：后端变更摘要

```
feat: Phase 2 code sync (BL-CA-07/08) + tests

- POST /api/v1/sync/code-fingerprints endpoint
- CodeFingerprint / CodeSyncRequest / CodeSyncResponse Pydantic models
- MemoryManager.sync_code_fingerprints() method
- Code file upsert logic (file_path + project_id)
- 13 unit tests for sync and upsert
```
