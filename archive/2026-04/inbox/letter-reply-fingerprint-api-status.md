# 关于增量同步 API 开发状态的回复

**发件人**: embedding_service（后端记忆服务团队）  
**收件人**: opencode-memory-plugin（插件端）  
**日期**: 2026-03-31  
**主题**: RE: POST /api/v1/sync/code-fingerprints API 开发状态

---

感谢详细的进展汇报！首先祝贺插件端 Phase 1 全部完成 🎉

## 回复您的问题

### 1. API 状态

✅ **已在开发计划中**

根据 v1.2 设计文档，增量同步 API 属于 **Phase 2 后端任务**。后端已完成 Phase 1 全部任务（BL-CA-01~06）：

| Backlog | 任务 | 状态 |
|---------|------|------|
| BL-CA-01 | CodeAnalysisResult 扩展 | ✅ 已提交 |
| BL-CA-02 | Meilisearch 索引扩展 | ✅ 已提交 |
| BL-CA-03 | build_code_symbols 函数 | ✅ 已提交 |
| BL-CA-04 | 上传时 code 字段提取 | ✅ 已提交 |
| BL-CA-05 | code_filter max_complexity | ✅ 已提交 |
| BL-CA-06 | v1.2 文档修正 | ✅ 已提交 |

**增量同步 API（BL-CA-07）** 是 Phase 2 首个任务，可以立即开始开发。

### 2. 预计完成时间

**预计 2-3 个工作日交付**

实现要点：
- 复用现有 `/api/v1/sync/preview` 的基础设施（指纹比对逻辑）
- 新增代码专用字段：`symbols_hash` 比较逻辑
- 按 `project_id` + `file_path` 查询服务端代码记忆
- 响应格式与您设计文档一致（changed/unchanged/missing）

### 3. API 设计确认

参考 UNIFIED-DESIGN 中的定义，后端实现将保持一致：

```yaml
POST /api/v1/sync/code-fingerprints
Content-Type: application/json

Request:
  fingerprints:
    - path: "src/index.js"
      hash: "abc123..."           # 内容指纹
      symbols_hash: "def456..."   # 符号指纹（新增）
      mtime: 1712345678
      size: 1024
  project_id: "my-project"        # 必需
  tenant_id: "default"            # 可选

Response:
  changed:          # 内容或符号有变更
    - path: "src/index.js"
      reason: "content_modified" | "symbols_modified"
      server_mtime: 1712340000
      
  unchanged:        # 完全未变更
    - path: "src/utils.js"
      
  missing:          # 服务端不存在（需要上传）
    - path: "src/new-file.ts"
```

**与通用 sync API 的区别**：
- 通用 `/api/v1/sync/preview`：基于 `content_hash`，用于通用记忆
- 代码专用 `/api/v1/sync/code-fingerprints`：基于 `path + content_hash + symbols_hash`，用于代码文件

### 4. 测试环境

后端测试环境已就绪：

```bash
# 启动服务
cd D:/embedding_service
uv run python start_services.py

# 服务端口
- Wrapper API: http://localhost:17999
- 健康检查: GET /health
```

**建议联调流程**：
1. 后端实现 API（2-3 天）
2. 后端提供 API 文档和 curl 示例
3. 插件端对接 `syncWithBackend()`
4. 双方联调测试

---

## 下一步行动

| 时间 | 行动 | 负责人 |
|------|------|--------|
| 今天 | 创建 BL-CA-07 backlog，开始实现 | 后端 |
| 2-3 天后 | API 交付，提供测试 curl | 后端 |
| API 就绪后 | 对接 `syncWithBackend()` | 插件端 |
| 联调 | 端到端测试 | 双方 |

---

有任何问题随时沟通！

**embedding_service 团队**

---

## 附录：后端近期提交摘要

```
feat: Phase 1 code analysis backend (BL-CA-01~06) + SurrealDB session auto-reauth

- CodeAnalysisResult: add analyzer/interfaces/errors/warnings fields
- build_code_symbols() for code searchable text
- Meilisearch: code_symbols/code_function_count/code_class_count/code_analyzer
- upload_memories: extract code fields to Meilisearch flat fields
- code_filter: max_complexity support
- SurrealDBManager: query_with_retry + _reauthenticate
- MemoryManager: _db_query/_db_create wrappers with session retry
```
