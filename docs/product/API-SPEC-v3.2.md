# 后端 API 规范 (v3.2 + v3.3)

> **版本**: v3.2.0 → v3.3.0
> **日期**: 2026-04-10（v3.2）/ 2026-04-29（v3.3）
> **服务端口**: 18008
> **基础路径**: `/api/v1`

---

## 1. 基础配置

### 1.1 服务端点

| 环境 | 地址 |
|------|------|
| 开发 | `http://localhost:18008/api/v1` |
| 生产 | `https://memory.example.com/api/v1` |

### 1.2 认证

```http
Authorization: Bearer {API_KEY}
Content-Type: application/json
```

### 1.3 通用响应格式

**成功响应**:
```json
{
  "success": true,
  "data": { ... },
  "timestamp": "2026-04-10T10:30:00Z"
}
```

**错误响应**:
```json
{
  "success": false,
  "error": {
    "code": "CONN_001",
    "message": "WebSocket connection failed",
    "details": { ... },
    "recoverable": true
  },
  "timestamp": "2026-04-10T10:30:00Z"
}
```

---

## 2. Memory API

### 2.1 创建记忆

**端点**: `POST /api/v1/memories`

**请求体**:
```json
{
  "type": "code",
  "title": "utils.ts",
  "abstract": "Utility functions for file operations",
  "overview": {
    "language": "typescript",
    "lines_of_code": 150,
    "function_count": 5
  },
  "content": "full content here...",
  "project": "my-project",
  "tenant_id": "default",
  "tags": ["typescript", "utils"]
}
```

**响应** (201):
```json
{
  "id": "entity:01HQ...",
  "type": "code",
  "created_at": "2026-04-10T10:30:00Z"
}
```

**验证规则**:
- `type` 必须为 `memory` | `backlog` | `wiki` | `code`
- `abstract` 长度 ≤ 100 字符
- `content` 为必填字段

---

### 2.2 搜索记忆

**端点**: `GET /api/v1/memories/search`

**查询参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 搜索关键词 |
| `project` | string | 否 | 项目过滤 |
| `type` | string | 否 | 类型过滤 |
| `tenant_id` | string | 否 | 租户 ID (默认 "default") |
| `limit` | int | 否 | 返回数量 (默认 20) |

**响应** (200):
```json
{
  "hits": [
    {
      "id": "entity:01HQ...",
      "title": "utils.ts",
      "abstract": "Utility functions...",
      "score": 0.95
    }
  ],
  "total": 1
}
```

---

### 2.3 获取记忆

**端点**: `GET /api/v1/memories/{id}`

**响应** (200):
```json
{
  "id": "entity:01HQ...",
  "type": "code",
  "title": "utils.ts",
  "abstract": "Utility functions...",
  "overview": { ... },
  "content": "...",
  "atoms": ["atom:func-1", "atom:func-2"]
}
```

---

### 2.4 更新记忆

**端点**: `PATCH /api/v1/memories/{id}`

**请求体**:
```json
{
  "abstract": "Updated abstract",
  "tags": ["typescript", "utils", "file"]
}
```

**响应** (200):
```json
{
  "id": "entity:01HQ...",
  "updated_at": "2026-04-10T10:35:00Z"
}
```

---

### 2.5 删除记忆

**端点**: `DELETE /api/v1/memories/{id}`

**响应** (204): 无内容

---

## 3. Code Analysis API

### 3.1 触发预计算

**端点**: `POST /api/v1/code/precompute`

**请求体**:
```json
{
  "file_path": "src/utils.ts",
  "source_code": "...",
  "language": "typescript",
  "tenant_id": "default"
}
```

**响应** (200):
```json
{
  "entity_id": "entity:code-src-utils",
  "atoms_count": 5,
  "duration_ms": 120,
  "memory_mb": 15.5,
  "success": true
}
```

**性能要求**:
- `duration_ms` < 10000 (10 秒)
- `memory_mb` < 100 (100MB)

---

### 3.2 代码导航

**端点**: `GET /api/v1/code/navigate`

**查询参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `symbol` | string | 是 | 符号名称 |
| `action` | string | 是 | `goto_definition` |

**响应** (200):
```json
{
  "symbol": "analyzeCode",
  "file_path": "src/utils.ts",
  "line": 85,
  "column": 10
}
```

---

### 3.3 爆炸半径分析

**端点**: `GET /api/v1/code/impact`

**查询参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `symbol` | string | 是 | 符号名称 |
| `depth` | int | 否 | 分析深度 (默认 2) |
| `direction` | string | 否 | `upstream` | `downstream` | `both` |

**响应** (200):
```json
{
  "symbol": "analyzeCode",
  "impacted_symbols": [
    {"name": "parseSync", "depth": 1},
    {"name": "validateConfig", "depth": 2}
  ]
}
```

---

### 3.4 代码搜索

**端点**: `GET /api/v1/code/search`

**查询参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 搜索关键词 |
| `language` | string | 否 | 语言过滤 |
| `hybrid` | bool | 否 | 混合搜索 (默认 true) |

**响应** (200):
```json
{
  "hits": [
    {
      "id": "atom:func-analyzeCode",
      "name": "analyzeCode",
      "signature": "async analyzeCode(filePath: string)",
      "score": 0.92
    }
  ]
}
```

---

## 4. WebSocket API

### 4.1 连接建立

**端点**: `ws://localhost:18008/ws`

**连接参数**:
```
?token={API_KEY}&tenant_id=default
```

### 4.2 心跳消息

**Ping** (客户端 → 服务端):
```json
{
  "type": "ping",
  "timestamp": 1712745000.123
}
```

**Pong** (服务端 → 客户端):
```json
{
  "type": "pong",
  "timestamp": 1712745000.123
}
```

**配置**:
- 心跳间隔: 30 秒
- 最大未响应: 2 次
- 触发重连: 2 次未响应后

---

### 4.3 订阅变更

**订阅请求**:
```json
{
  "action": "subscribe",
  "query": "LIVE SELECT * FROM entity WHERE id = \"entity:test\""
}
```

**DIFF 模式订阅**:
```json
{
  "action": "subscribe",
  "query": "LIVE SELECT DIFF FROM entity WHERE id = \"entity:test\""
}
```

**DIFF 响应**:
```json
{
  "type": "diff",
  "entity_id": "entity:test",
  "patches": [
    {"op": "replace", "path": "/abstract", "value": "new abstract"}
  ]
}
```

**性能要求**: 数据传输减少 ≥ 90%

---

### 4.4 消息确认 (ACK)

**发送消息**:
```json
{
  "action": "update",
  "data": { ... },
  "_msgId": "msg-12345",
  "_requiresAck": true
}
```

**确认响应**:
```json
{
  "type": "ack",
  "_ackId": "msg-12345",
  "_ackData": { ... }
}
```

**配置**:
- 超时时间: 5 秒
- 最大重试: 3 次
- 退避策略: 指数退避

---

## 5. 健康检查

### 5.1 服务健康

**端点**: `GET /health`

**响应** (200):
```json
{
  "status": "healthy",
  "version": "3.2.0",
  "timestamp": "2026-04-10T10:30:00Z"
}
```

---

### 5.2 数据库健康

**端点**: `GET /health/db`

**响应** (200):
```json
{
  "surrealdb": "connected",
  "meilisearch": "connected",
  "latency_ms": {
    "surrealdb": 5,
    "meilisearch": 3
  }
}
```

---

### 5.3 WebSocket 健康

**端点**: `GET /health/ws`

**响应** (200):
```json
{
  "connections": 5,
  "uptime": 3600
}
```

---

### 5.4 Prometheus 指标

**端点**: `GET /metrics`

**响应**: Prometheus 格式指标数据

---

## 6. 错误码规范

### 6.1 错误码分类

| 错误码 | 类型 | 说明 | HTTP 状态 | 处理策略 |
|--------|------|------|-----------|----------|
| `CONN_001` | 连接错误 | WebSocket 连接失败 | 503 | 自动重试 + 指数退避 |
| `CONN_002` | 连接错误 | 服务不可用 | 503 | 切换到备份端点 |
| `MSG_001` | 消息错误 | 消息发送超时 | 504 | ACK 超时 + 重试 |
| `MSG_002` | 消息错误 | 消息格式错误 | 400 | 验证 + 拒绝 |
| `MSG_003` | 消息错误 | 消息处理失败 | 500 | 队列保留 + 人工处理 |
| `AUTH_001` | 认证错误 | API 密钥无效 | 401 | 提示用户检查配置 |
| `AUTH_002` | 认证错误 | Token 过期 | 401 | 刷新 token |
| `RECN_001` | 重连错误 | 达到最大重试次数 | 503 | 降级模式 |
| `RECN_002` | 重连错误 | 状态恢复失败 | 500 | 重新初始化 |

---

### 6.2 HTTP 状态码

| 状态码 | 含义 | 处理建议 |
|--------|------|----------|
| 200 | 成功 | - |
| 201 | 创建成功 | - |
| 204 | 删除成功 | - |
| 400 | 请求错误 | 检查参数 |
| 401 | 未授权 | 检查 API Key |
| 404 | 未找到 | 检查 ID |
| 500 | 服务器错误 | 重试或报告 |
| 503 | 服务不可用 | 稍后重试 |
| 504 | 网关超时 | 检查网络 |

---

## 7. 性能基准

### 7.1 WebSocket 性能

| 指标 | 基准值 | 测试条件 |
|------|--------|----------|
| 并发连接数 | ≥ 1000 | 单服务器实例 |
| 消息吞吐量 | ≥ 10,000 msg/s | 1000 并发连接 |
| 消息延迟 (p99) | < 100ms | 局域网环境 |
| 心跳成功率 | ≥ 99% | 30s 间隔，1 小时 |
| 重连时间 | < 5s | 首次重连 |
| 内存使用 | < 500MB | 1000 并发连接 |

---

### 7.2 PrecomputeService 性能

| 指标 | 基准值 | 测试条件 |
|------|--------|----------|
| 处理速度 | > 1000 行/秒 | 标准代码文件 |
| 内存占用 | < 100MB | 处理 1000 行代码 |
| 批处理时间 | < 10s | 批大小 100 |
| 增量识别率 | > 95% | 文件指纹对比 |

---

## 8. 统一搜索 API (v3.3)

> **新增于 v3.3** — 跨 Entity（Meilisearch）和 Atom（SurrealDB）的统一搜索端点。

### 8.1 统一搜索

**端点**: `POST /api/v1/search`

**请求体**:

```json
{
  "query": "setup函数",
  "mode": "hybrid",
  "scope": "all",
  "types": null,
  "atom_types": null,
  "max_level": null,
  "limit": 20,
  "level": 1,
  "tenant_id": "default"
}
```

**请求参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `query` | string | 是 | - | 搜索查询（≥1字符） |
| `mode` | string | 否 | `hybrid` | 搜索模式: `vector` \| `keyword` \| `hybrid` |
| `scope` | string | 否 | `all` | 搜索范围: `all` \| `memory` \| `code` \| `backlog` \| `atom` \| `entity` |
| `types` | list[string] | 否 | null | 过滤 Entity 类型 |
| `atom_types` | list[string] | 否 | null | 过滤 Atom 类型 |
| `max_level` | int | 否 | null | 最大标题层级过滤（1-6），仅返回 `heading_level <= max_level` 的 Atom |
| `limit` | int | 否 | 20 | 返回数量限制（1-100） |
| `level` | int | 否 | 1 | 返回层级: 0=abstract, 1=abstract+overview, 2=full |
| `tenant_id` | string | 否 | `default` | 租户 ID |

**响应** (200):

```json
{
  "results": [
    {
      "type": "entity",
      "id": "entity:01HQ...",
      "entity_type": "memory",
      "abstract": "Vue3 Composition API 指南",
      "score": 0.95
    },
    {
      "type": "atom",
      "local_id": "01SEC001",
      "atom_id": "atom:01HQ...",
      "atom_type": "section",
      "name": "1.1 setup() 函数",
      "content": "setup 是 Composition API 的入口...",
      "heading_level": 2,
      "parent_id": "01CHAP001",
      "order": "a0",
      "tags": ["vue", "composition-api"],
      "entity_id": "entity:01HQ...",
      "score": 0.5
    }
  ],
  "total": 2,
  "mode": "hybrid",
  "query": "setup函数"
}
```

**scope 路由规则**:

| scope 值 | 搜索 Entity | 搜索 Atom |
|----------|:-----------:|:---------:|
| `all` | ✅ | ✅ |
| `memory` | ✅ | ✅ |
| `code` | ✅ | ✅ |
| `backlog` | ✅ | ✅ |
| `entity` | ✅ | ❌ |
| `atom` | ❌ | ✅ |

---

## 9. Entity API (v3.3 扩展)

> v3.2 中 Entity CRUD 端点已存在，v3.3 新增内联 Atom 创建和跨 Entity Atom 链接。

### 9.1 创建 Entity（含内联 Atom）

**端点**: `POST /api/v1/entities`

**请求体**:

```json
{
  "type": "memory",
  "abstract": "Vue3 Composition API 指南",
  "overview": {},
  "atoms": [
    "atom:01HQ...",
    {
      "type": "chapter",
      "content": "章节概述...",
      "name": "第1章：Composition API",
      "local_id": "01CHAP001",
      "heading_level": 1,
      "parent_id": null,
      "order": "a0"
    },
    {
      "type": "section",
      "content": "详细说明...",
      "name": "1.1 基本用法",
      "local_id": "01SEC001",
      "heading_level": 2,
      "parent_id": "01CHAP001",
      "order": "a0"
    }
  ],
  "tags": ["vue", "javascript"],
  "tenant_id": "default"
}
```

**`atoms` 字段说明** (v3.3 新增):

- 支持双格式：字符串（已有 Atom ID）或对象（内联创建 `AtomInlineCreate`）
- 内联创建的 Atom 自动注入 `tenant_id` 和 `entity_id`
- 整个操作在事务内执行，确保原子性

**响应** (201):

```json
{
  "id": "entity:01HQ...",
  "type": "memory",
  "tenant_id": "default",
  "abstract": "Vue3 Composition API 指南",
  "overview": {},
  "atoms": ["atom:01HQ...", "atom:01NEW1...", "atom:01NEW2..."],
  "tags": ["vue", "javascript"],
  "created_at": "2026-04-29T10:30:00Z"
}
```

---

### 9.2 跨 Entity 获取 Atom (v3.3 新增)

**端点**: `GET /api/v1/entities/{entity_id}/atoms/{atom_id}`

**查询参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `tenant_id` | string | 否 | 租户 ID (默认 "default") |

**说明**:

- 验证 Atom 的 `entity_id` 字段与路径参数匹配
- 用于跨 Entity 的 Atom 链接解析（`[[local_id]]` 引用）
- 返回完整 Atom 数据

**响应** (200):

```json
{
  "id": "atom:01HQ...",
  "type": "section",
  "content": "详细说明...",
  "name": "1.1 基本用法",
  "heading_level": 2,
  "parent_id": "01CHAP001",
  "order": "a0",
  "entity_id": "entity:01HQ...",
  "tenant_id": "default",
  "created_at": "2026-04-29T10:30:00Z"
}
```

**错误响应**:

- `404`: Atom 不存在或不属于该 Entity

---

## 10. Atom API (v3.3 新增)

> **新增于 v3.3** — 原子级知识单元的完整 CRUD 管理。

### 10.1 创建 Atom

**端点**: `POST /api/v1/atoms`

**请求体**:

```json
{
  "type": "function",
  "content": "async function fetchData(url: string) { ... }",
  "name": "fetchData",
  "signature": "async fetchData(url: string): Promise<Response>",
  "params": [{"name": "url", "type": "string"}],
  "return_type": "Promise<Response>",
  "is_async": true,
  "is_exported": true,
  "complexity": 3,
  "start_line": 10,
  "end_line": 15,
  "docstring": {"summary": "Fetch data from URL"},
  "tags": ["api", "async"],
  "heading_level": null,
  "parent_id": null,
  "order": null,
  "entity_id": "entity:01HQ...",
  "tenant_id": "default"
}
```

**Atom 有效类型**:

| 类型 | 说明 | heading_level |
|------|------|---------------|
| `function` | 函数定义 | - |
| `class` | 类定义 | - |
| `interface` | 接口定义 | - |
| `import` | 导入语句 | - |
| `goal` | 目标 | - |
| `scope` | 范围 | - |
| `task` | 任务 | - |
| `note` | 笔记 | - |
| `chapter` | 章节 (v3.3) | 1-6 |
| `section` | 小节 (v3.3) | 1-6 |

**响应** (201):

```json
{
  "id": "atom:01HQ...",
  "type": "function",
  "content": "async function fetchData(url: string) { ... }",
  "name": "fetchData",
  "signature": "async fetchData(url: string): Promise<Response>",
  "tenant_id": "default",
  "version": 1,
  "created_at": "2026-04-29T10:30:00Z"
}
```

---

### 10.2 列出 Atom

**端点**: `GET /api/v1/atoms`

**查询参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `query` | string | 否 | null | 按名称过滤（子串匹配） |
| `type` | string | 否 | null | Atom 类型过滤 |
| `project` | string | 否 | null | 项目过滤 |
| `tenant_id` | string | 否 | `default` | 租户 ID |
| `max_level` | int | 否 | null | 最大标题层级过滤（1-6），仅返回 `heading_level <= max_level` 的 Atom |
| `page` | int | 否 | 1 | 页码 |
| `page_size` | int | 否 | 50 | 每页大小（1-100） |
| `limit` | int | 否 | null | 返回数量限制（向后兼容） |
| `offset` | int | 否 | null | 偏移量（向后兼容） |

**响应** (200):

```json
{
  "data": [
    {
      "id": "atom:01HQ...",
      "type": "function",
      "content": "...",
      "name": "fetchData",
      "heading_level": null,
      "parent_id": null,
      "entity_id": "entity:01HQ...",
      "tenant_id": "default",
      "created_at": "2026-04-29T10:30:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 50,
  "has_more": false
}
```

---

### 10.3 获取 Atom 详情

**端点**: `GET /api/v1/atoms/{atom_id}`

**查询参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `tenant_id` | string | 否 | 租户 ID (默认 "default") |

**响应** (200): 完整 Atom 对象

---

### 10.4 更新 Atom

**端点**: `PUT /api/v1/atoms/{atom_id}`

**请求体** (部分更新):

```json
{
  "content": "更新后的内容",
  "tags": ["updated-tag"]
}
```

**说明**:

- 自动递增 `version` 字段
- 自动更新 `updated_at` 时间戳
- 使用事务保证原子性

**响应** (200): 更新后的完整 Atom 对象

---

### 10.5 删除 Atom

**端点**: `DELETE /api/v1/atoms/{atom_id}`

**查询参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `tenant_id` | string | 否 | 租户 ID (默认 "default") |

**响应** (200):

```json
{
  "success": true,
  "message": "Atom 已删除"
}
```

---

### 10.6 批量创建 Atom

**端点**: `POST /api/v1/atoms/batch`

**请求体**:

```json
{
  "atoms": [
    {"type": "chapter", "content": "...", "name": "第1章"},
    {"type": "section", "content": "...", "name": "1.1 节"}
  ],
  "tenant_id": "default"
}
```

**限制**: 单次批量最多 100 个 Atom

**响应** (200):

```json
{
  "success": [
    {"id": "atom:01HQ...", "type": "chapter", ...}
  ],
  "failed": [],
  "total": 2,
  "success_count": 2,
  "failed_count": 0
}
```

---

### 10.7 上下文预算 (v3.3 新增)

**端点**: `POST /api/v1/atoms/budget`

在 token 预算内选择最相关的 Atoms，支持两种策略。

**请求体**:

```json
{
  "entity_id": "entity:01HQ...",
  "query": "setup函数",
  "max_tokens": 4000,
  "strategy": "relevance",
  "max_level": null,
  "tenant_id": "default"
}
```

**请求参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `entity_id` | string | 是 | - | Entity ID |
| `query` | string | 否 | null | 搜索关键词（relevance 策略必需） |
| `max_tokens` | int | 否 | 4000 | Token 预算上限（100-100000） |
| `strategy` | string | 否 | `relevance` | 选择策略: `relevance` \| `hierarchy` |
| `max_level` | int | 否 | null | 最大标题层级过滤 |
| `tenant_id` | string | 否 | `default` | 租户 ID |

**策略说明**:

| 策略 | 排序方式 | 适用场景 |
|------|----------|----------|
| `relevance` | BM25 评分 + heading_level 加权 | 需要精准匹配的问答场景 |
| `hierarchy` | heading_level → order → name | 需要完整结构化的文档浏览 |

**祖先链保证**: 无论哪种策略，选中的 Atom 都会自动包含其祖先链（parent → grandparent → ...），确保上下文完整。

**响应** (200):

```json
{
  "atoms": [
    {
      "id": "atom:01HQ...",
      "type": "section",
      "name": "1.1 setup() 函数",
      "content": "详细说明...",
      "heading_level": 2,
      "parent_id": "01CHAP001",
      "order": "a0"
    }
  ],
  "total_atoms": 15,
  "selected_count": 3,
  "used_tokens": 1200,
  "max_tokens": 4000,
  "strategy": "relevance",
  "budget_exhausted": false
}
```

**Token 估算规则**: `max(1, len(name + content + signature) // 2)`

---

## 11. v3.3 数据模型

> **新增于 v3.3** — Atom Architecture 相关数据模型。

### 11.1 AtomInlineCreate

内联创建 Atom 请求，可嵌入 Entity 创建/更新的 `atoms` 字段。

```json
{
  "type": "chapter",
  "content": "章节概述...",
  "name": "第1章：Composition API",
  "local_id": "01CHAP001",
  "heading_level": 1,
  "parent_id": null,
  "order": "a0",
  "aliases": ["Composition API 概述"],
  "tags": ["vue", "composition-api"],
  "signature": null,
  "params": null,
  "return_type": null,
  "is_exported": null,
  "is_async": null,
  "complexity": null,
  "start_line": null,
  "end_line": null,
  "docstring": null,
  "metadata": null,
  "project": null,
  "fingerprint": null
}
```

**字段说明**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | Atom 类型 (function, class, chapter, section, etc.) |
| `content` | string | 是 | Atom 内容 |
| `name` | string | 否 | Atom 名称 |
| `local_id` | string | 否 | 客户端侧 ID（用于树结构引用） |
| `heading_level` | int | 否 | 标题层级 1-6 |
| `parent_id` | string | 否 | 父 Atom 的 local_id |
| `order` | string | 否 | 排序键（如 a0, aV） |
| `aliases` | list[string] | 否 | 别名列表 |
| `tags` | list[string] | 否 | 标签列表 |
| `signature` | string | 否 | 函数签名 |
| `params` | list[dict] | 否 | 参数列表 |
| `return_type` | string | 否 | 返回类型 |
| `is_exported` | bool | 否 | 是否导出 |
| `is_async` | bool | 否 | 是否异步 |
| `complexity` | int | 否 | 复杂度 |
| `start_line` | int | 否 | 起始行号 |
| `end_line` | int | 否 | 结束行号 |
| `docstring` | dict | 否 | 文档字符串 |
| `metadata` | dict | 否 | 元数据 |
| `project` | string | 否 | 项目 ID |
| `fingerprint` | string | 否 | 内容指纹 |

---

### 11.2 UnifiedSearchRequest / UnifiedSearchResponse

统一搜索端点的请求和响应模型。

**UnifiedSearchRequest**:

```json
{
  "query": "搜索查询",
  "mode": "hybrid",
  "scope": "all",
  "types": null,
  "atom_types": null,
  "max_level": null,
  "limit": 20,
  "level": 1,
  "tenant_id": "default"
}
```

**UnifiedSearchResponse**:

```json
{
  "results": [],
  "total": 0,
  "mode": "hybrid",
  "query": "搜索查询"
}
```

---

### 11.3 AtomBudgetRequest / AtomBudgetResponse

Atom 上下文预算端点的请求和响应模型。

**AtomBudgetRequest**:

```json
{
  "entity_id": "entity:01HQ...",
  "query": "搜索关键词",
  "max_tokens": 4000,
  "strategy": "relevance",
  "max_level": null,
  "tenant_id": "default"
}
```

**AtomBudgetResponse**:

```json
{
  "atoms": [],
  "total_atoms": 0,
  "selected_count": 0,
  "used_tokens": 0,
  "max_tokens": 4000,
  "strategy": "relevance",
  "budget_exhausted": false
}
```

---

## 附录

### A. 配置项清单

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `PORT` | int | 18008 | 服务端口 |
| `HOST` | string | 0.0.0.0 | 服务主机 |
| `WORKERS` | int | 4 | Uvicorn 工作进程数 |
| `LOG_LEVEL` | string | INFO | 日志级别 |
| `WS_HEARTBEAT_INTERVAL` | int | 30 | 心跳间隔 (秒) |
| `WS_RECONNECT_MAX_ATTEMPTS` | int | 10 | 最大重连次数 |
| `WS_ACK_TIMEOUT` | float | 5.0 | 消息确认超时 (秒) |
| `PRECOMPUTE_BATCH_SIZE` | int | 100 | 批处理大小 |
| `PRECOMPUTE_INTERVAL` | int | 300 | 处理间隔 (秒) |
| `PRECOMPUTE_MAX_CONCURRENT` | int | 5 | 最大并发数 |

---

### B. 数据模型

**Atom (原子单元)**:

```json
{
  "id": "atom:01HQ...",
  "type": "function",
  "content": "...",
  "name": "analyzeCode",
  "signature": "...",
  "tenant_id": "default",
  "version": 1,
  "tags": [],
  "heading_level": null,
  "parent_id": null,
  "order": null,
  "aliases": [],
  "entity_id": "entity:01HQ..."
}
```

**Entity (实体)**:

```json
{
  "id": "entity:01HQ...",
  "type": "code",
  "abstract": "...",
  "overview": { ... },
  "atoms": ["atom:func-1"],
  "tenant_id": "default"
}
```

**Reference (关系)**:

```json
{
  "in": "atom:caller",
  "out": "atom:callee",
  "type": "calls",
  "weight": 0.5
}
```

---

_文档版本: v3.2.0 + v3.3.0_
_最后更新: 2026-04-29 (v3.3 Atom Architecture)_
_服务端口: 18008_
