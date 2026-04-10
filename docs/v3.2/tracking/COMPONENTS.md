# v3.2 组件清单（可追踪）

> **文档**: v3.2 设计文档结构分析  
> **生成时间**: 2026-04-10  
> **版本**: v3.2.0

---

## 1. API 端点统计

### 1.1 原子级操作 (Atom API)

| # | 端点 | 方法 | 说明 | 来源文档 |
|---|------|------|------|----------|
| 1 | `/api/v1/atoms` | POST | 创建 Atom | UNIFIED-ARCHITECTURE-v3.2.md |
| 2 | `/api/v1/atoms/{atom_id}` | PATCH | 更新 Atom | UNIFIED-ARCHITECTURE-v3.2.md |
| 3 | `/api/v1/atoms/{atom_id}` | DELETE | 删除 Atom | UNIFIED-ARCHITECTURE-v3.2.md |
| 4 | `/api/v1/atoms` | GET | 搜索 Atom | UNIFIED-ARCHITECTURE-v3.2.md |

**小计**: 4 个端点

### 1.2 实体操作 (Entity API)

| # | 端点 | 方法 | 说明 | 来源文档 |
|---|------|------|------|----------|
| 5 | `/api/v1/entities` | POST | 创建 Entity | UNIFIED-ARCHITECTURE-v3.2.md |
| 6 | `/api/v1/entities/{id}` | GET | 获取 Entity | PLUGIN-v3.2-API.md |
| 7 | `/api/v1/entities/{id}` | PATCH | 更新 Entity | PLUGIN-v3.2-API.md |
| 8 | `/api/v1/entities/{id}` | DELETE | 删除 Entity | PLUGIN-v3.2-API.md |

**小计**: 4 个端点

### 1.3 关系操作 (Relation API)

| # | 端点 | 方法 | 说明 | 来源文档 |
|---|------|------|------|----------|
| 9 | `/api/v1/relations` | POST | 创建关系 | UNIFIED-ARCHITECTURE-v3.2.md |

**小计**: 1 个端点

### 1.4 记忆操作 (Memory API)

| # | 端点 | 方法 | 说明 | 来源文档 |
|---|------|------|------|----------|
| 10 | `/api/v1/memories` | POST | 创建记忆 | PLUGIN-v3.2-API.md |
| 11 | `/api/v1/memories/search` | GET | 搜索记忆 | PLUGIN-v3.2-API.md |
| 12 | `/api/v1/memories/{id}` | GET | 获取记忆 | PLUGIN-v3.2-API.md |
| 13 | `/api/v1/memories/{id}` | PATCH | 更新记忆 | PLUGIN-v3.2-API.md |
| 14 | `/api/v1/memories/{id}` | DELETE | 删除记忆 | PLUGIN-v3.2-API.md |

**小计**: 5 个端点

### 1.5 代码分析 API (Code Analysis API)

| # | 端点 | 方法 | 说明 | 来源文档 |
|---|------|------|------|----------|
| 15 | `/api/v1/code/precompute` | POST | 触发预计算 | PLUGIN-v3.2-API.md |
| 16 | `/api/v1/code/navigate` | GET | 代码导航 | PLUGIN-v3.2-API.md |
| 17 | `/api/v1/code/impact` | GET | 爆炸半径分析 | PLUGIN-v3.2-API.md |
| 18 | `/api/v1/code/search` | GET | 代码搜索 | PLUGIN-v3.2-API.md |

**小计**: 4 个端点

### 1.6 WebSocket API

| # | 端点 | 类型 | 说明 | 来源文档 |
|---|------|------|------|----------|
| 19 | `/ws` | WebSocket | 实时同步连接 | BACKEND-v3.2-WEBSOCKET.md |

**小计**: 1 个端点

### 1.7 健康检查与监控

| # | 端点 | 方法 | 说明 | 来源文档 |
|---|------|------|------|----------|
| 20 | `/health` | GET | 服务健康状态 | DEPLOYMENT-v3.2.md |
| 21 | `/health/db` | GET | 数据库连接状态 | DEPLOYMENT-v3.2.md |
| 22 | `/health/ws` | GET | WebSocket 状态 | DEPLOYMENT-v3.2.md |
| 23 | `/metrics` | GET | Prometheus 指标 | DEPLOYMENT-v3.2.md |

**小计**: 4 个端点

**API 端点总计**: 23 个

---

## 2. 数据库表统计

### 2.1 核心表 (Core Tables)

| # | 表名 | 类型 | 说明 | ChangeFeed | 来源文档 |
|---|------|------|------|------------|----------|
| 1 | `atom` | NORMAL | 原子级数据 | 7d | DATABASE-v3.2-SCHEMA.md |
| 2 | `entity` | NORMAL | 实体级数据 | 7d | DATABASE-v3.2-SCHEMA.md |
| 3 | `reference` | RELATION | 原子/实体间关系 | 7d | DATABASE-v3.2-SCHEMA.md |

**小计**: 3 个核心表

### 2.2 辅助表 (Auxiliary Tables)

| # | 表名 | 类型 | 说明 | ChangeFeed | 来源文档 |
|---|------|------|------|------------|----------|
| 4 | `timeline` | NORMAL | 时间线索引 | - | DATABASE-v3.2-SCHEMA.md |
| 5 | `stats` | NORMAL | 统计信息 | - | DATABASE-v3.2-SCHEMA.md |
| 6 | `project` | NORMAL | 项目元数据 | - | DATABASE-v3.2-SCHEMA.md |
| 7 | `config` | NORMAL | 系统配置 | - | DATABASE-v3.2-SCHEMA.md |

**小计**: 4 个辅助表

### 2.3 性能监控表

| # | 表名 | 类型 | 说明 | 来源文档 |
|---|------|------|------|----------|
| 8 | `performance_log` | NORMAL | 性能日志 | BACKEND-v3.2-PRECOMPUTE.md |

**小计**: 1 个监控表

**数据库表总计**: 8 个

---

## 3. 配置项统计

### 3.1 服务配置 (Service Config)

| # | 配置项 | 类型 | 默认值 | 说明 | 来源文档 |
|---|--------|------|--------|------|----------|
| 1 | PORT | int | 18008 | 服务端口 | BACKEND-v3.2-IMPLEMENTATION.md |
| 2 | HOST | string | 0.0.0.0 | 服务主机 | BACKEND-v3.2-IMPLEMENTATION.md |
| 3 | WORKERS | int | 4 | Uvicorn 工作进程数 | DEPLOYMENT-v3.2.md |
| 4 | LOG_LEVEL | string | INFO | 日志级别 | DEPLOYMENT-v3.2.md |

**小计**: 4 个服务配置

### 3.2 WebSocket 配置

| # | 配置项 | 类型 | 默认值 | 说明 | 来源文档 |
|---|--------|------|--------|------|----------|
| 5 | WS_HEARTBEAT_INTERVAL | int | 30 | 心跳间隔(秒) | BACKEND-v3.2-WEBSOCKET.md |
| 6 | WS_RECONNECT_MAX_ATTEMPTS | int | 10 | 最大重连次数 | DEPLOYMENT-v3.2.md |
| 7 | WS_MAX_MISSED_PONGS | int | 2 | 最大未响应次数 | BACKEND-v3.2-WEBSOCKET.md |
| 8 | WS_BASE_DELAY | float | 1.0 | 重连基础延迟(秒) | BACKEND-v3.2-WEBSOCKET.md |
| 9 | WS_MAX_DELAY | float | 300.0 | 最大重连延迟(秒) | BACKEND-v3.2-WEBSOCKET.md |
| 10 | WS_ACK_TIMEOUT | float | 5.0 | 消息确认超时(秒) | BACKEND-v3.2-WEBSOCKET.md |
| 11 | WS_ACK_MAX_RETRIES | int | 3 | 消息重试次数 | BACKEND-v3.2-WEBSOCKET.md |

**小计**: 7 个 WebSocket 配置

### 3.3 数据库配置

| # | 配置项 | 类型 | 默认值 | 说明 | 来源文档 |
|---|--------|------|--------|------|----------|
| 12 | SURREALDB_URL | string | - | SurrealDB 连接地址 | DEPLOYMENT-v3.2.md |
| 13 | SURREALDB_NS | string | opencode | Namespace | DEPLOYMENT-v3.2.md |
| 14 | SURREALDB_DB | string | memory | Database | DEPLOYMENT-v3.2.md |
| 15 | SURREALDB_USER | string | root | 用户名 | DEPLOYMENT-v3.2.md |
| 16 | SURREALDB_PASS | string | root | 密码 | DEPLOYMENT-v3.2.md |
| 17 | MEILISEARCH_URL | string | - | Meilisearch 地址 | DEPLOYMENT-v3.2.md |
| 18 | MEILISEARCH_API_KEY | string | - | API 密钥 | DEPLOYMENT-v3.2.md |

**小计**: 7 个数据库配置

### 3.4 预计算配置

| # | 配置项 | 类型 | 默认值 | 说明 | 来源文档 |
|---|--------|------|--------|------|----------|
| 19 | PRECOMPUTE_BATCH_SIZE | int | 100 | 批处理大小 | DEPLOYMENT-v3.2.md |
| 20 | PRECOMPUTE_INTERVAL | int | 300 | 处理间隔(秒) | DEPLOYMENT-v3.2.md |
| 21 | PRECOMPUTE_MAX_CONCURRENT | int | 5 | 最大并发数 | BACKEND-v3.2-PRECOMPUTE.md |
| 22 | PRECOMPUTE_MAX_FILE_SIZE | int | 10MB | 最大文件大小 | DATABASE-v3.2-SCHEMA.md |

**小计**: 4 个预计算配置

### 3.5 外部服务配置

| # | 配置项 | 类型 | 默认值 | 说明 | 来源文档 |
|---|--------|------|--------|------|----------|
| 23 | MODELSCOPE_API_KEY | string | - | ModelScope API 密钥 | DEPLOYMENT-v3.2.md |

**小计**: 1 个外部服务配置

**配置项总计**: 23 个

---

## 4. 代码模块统计

### 4.1 后端 Python 模块

| # | 模块名 | 类型 | 说明 | 来源文档 |
|---|--------|------|------|----------|
| 1 | main.py | Entry | FastAPI 入口 | BACKEND-v3.2-IMPLEMENTATION.md |
| 2 | config.py | Config | 配置管理 | BACKEND-v3.2-IMPLEMENTATION.md |
| 3 | models.py | Model | Pydantic 模型 | BACKEND-v3.2-IMPLEMENTATION.md |

**核心模块小计**: 3 个

#### 4.1.1 WebSocket 模块

| # | 模块名 | 说明 | 来源文档 |
|---|--------|------|----------|
| 4 | websocket.py | WebSocket 端点 | BACKEND-v3.2-WEBSOCKET.md |
| 5 | reliable_client.py | 可靠客户端 | BACKEND-v3.2-WEBSOCKET.md |
| 6 | ack_system.py | 确认系统 | BACKEND-v3.2-WEBSOCKET.md |
| 7 | state_recovery.py | 状态恢复 | BACKEND-v3.2-WEBSOCKET.md |
| 8 | persistent_queue.py | 持久化队列 | BACKEND-v3.2-WEBSOCKET.md |

**WebSocket 小计**: 5 个模块

#### 4.1.2 服务模块

| # | 模块名 | 说明 | 来源文档 |
|---|--------|------|----------|
| 9 | precompute.py | 预计算服务 | BACKEND-v3.2-PRECOMPUTE.md |
| 10 | performance_monitor.py | 性能监控 | BACKEND-v3.2-PRECOMPUTE.md |
| 11 | concurrency_control.py | 并发控制 | BACKEND-v3.2-PRECOMPUTE.md |

**服务模块小计**: 3 个模块

#### 4.1.3 工具模块

| # | 模块名 | 说明 | 来源文档 |
|---|--------|------|----------|
| 12 | meili_client.py | Meilisearch 客户端 | BACKEND-v3.2-MEILISEARCH.md |
| 13 | code_analyzer.py | 代码分析器 | BACKEND-v3.2-IMPLEMENTATION.md |
| 14 | diff/subscription.py | DIFF 订阅 | BACKEND-v3.2-WEBSOCKET.md |
| 15 | diff/patch_applier.py | 补丁应用 | BACKEND-v3.2-WEBSOCKET.md |

**工具模块小计**: 4 个模块

**后端模块总计**: 15 个

### 4.2 前端/插件模块

| # | 模块名 | 类型 | 说明 | 来源文档 |
|---|--------|------|------|----------|
| 16 | wrapper-client.js | Client | API 客户端 | PLUGIN-v3.2-IMPLEMENTATION.md |
| 17 | websocket-client.js | Client | WebSocket 客户端 | PLUGIN-v3.2-IMPLEMENTATION.md |
| 18 | config.js | Config | 配置管理 | PLUGIN-v3.2-IMPLEMENTATION.md |
| 19 | acks.js | Utility | ACK 管理 | PLUGIN-v3.2-IMPLEMENTATION.md |
| 20 | reconnection.js | Utility | 重连管理 | PLUGIN-v3.2-IMPLEMENTATION.md |
| 21 | error-handler.js | Utility | 错误处理 | PLUGIN-v3.2-IMPLEMENTATION.md |

**前端模块总计**: 6 个

**代码模块总计**: 21 个

---

## 5. 组件统计汇总

| 类别 | 数量 | 备注 |
|------|------|------|
| API 端点 | 23 | 含 WebSocket |
| 数据库表 | 8 | 含监控表 |
| 配置项 | 23 | 环境变量 |
| 代码模块 | 21 | Python + JS |
| **总计** | **75** | 可追踪组件 |

---

## 6. 实现追踪矩阵

### 6.1 追踪状态定义

| 状态 | 符号 | 说明 |
|------|------|------|
| 未开始 | [ ] | 尚未开始实现 |
| 进行中 | [/] | 正在开发中 |
| 已完成 | [x] | 已实现并测试 |
| 已验证 | [v] | 已通过验证测试 |
| 已部署 | [d] | 已部署到生产 |

### 6.2 按文档追踪

| 文档 | API端点 | 数据库表 | 配置项 | 代码模块 | 状态 |
|------|---------|----------|--------|----------|------|
| UNIFIED-ARCHITECTURE-v3.2.md | 9 | 0 | 0 | 0 | [ ] |
| DATABASE-v3.2-SCHEMA.md | 0 | 8 | 1 | 0 | [ ] |
| BACKEND-v3.2-WEBSOCKET.md | 1 | 0 | 7 | 7 | [ ] |
| BACKEND-v3.2-PRECOMPUTE.md | 0 | 1 | 4 | 3 | [ ] |
| BACKEND-v3.2-IMPLEMENTATION.md | 0 | 0 | 4 | 5 | [ ] |
| BACKEND-v3.2-MEILISEARCH.md | 0 | 0 | 0 | 1 | [ ] |
| BACKEND-v3.2-MIGRATION.md | 0 | 0 | 4 | 0 | [ ] |
| PLUGIN-v3.2-API.md | 9 | 0 | 0 | 0 | [ ] |
| PLUGIN-v3.2-IMPLEMENTATION.md | 0 | 0 | 0 | 6 | [ ] |
| DEPLOYMENT-v3.2.md | 4 | 0 | 7 | 0 | [ ] |

---

## 7. 文档依赖关系

```
UNIFIED-ARCHITECTURE-v3.2.md (根文档)
├── DATABASE-v3.2-SCHEMA.md
├── BACKEND-v3.2-WEBSOCKET.md
├── BACKEND-v3.2-PRECOMPUTE.md
├── BACKEND-v3.2-IMPLEMENTATION.md
│   ├── BACKEND-v3.2-WEBSOCKET.md
│   ├── BACKEND-v3.2-PRECOMPUTE.md
│   └── BACKEND-v3.2-MEILISEARCH.md
├── BACKEND-v3.2-MIGRATION.md
├── PLUGIN-v3.2-API.md
├── PLUGIN-v3.2-IMPLEMENTATION.md
│   └── PLUGIN-v3.2-API.md
└── DEPLOYMENT-v3.2.md
    └── BACKEND-v3.2-IMPLEMENTATION.md
```

---

*此文档由自动化分析生成，用于 v3.2 实现追踪*
