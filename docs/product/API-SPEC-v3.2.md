# 后端 API 规范 (v3.2)

> **版本**: v3.2.0  
> **日期**: 2026-04-10  
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
  "tenant_id": "default"
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

_文档版本: v3.2.0_  
_最后更新: 2026-04-10_  
_服务端口: 18008_
