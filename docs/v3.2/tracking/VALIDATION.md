# v3.2 验证标准（可测试）

> **文档**: v3.2 设计文档结构分析  
> **生成时间**: 2026-04-10  
> **版本**: v3.2.0

---

## 1. 性能指标验证标准

### 1.1 WebSocket 性能指标

| 指标ID | 指标名称 | 基准值 | 测试条件 | 验证方法 | 来源文档 |
|--------|----------|--------|----------|----------|----------|
| PERF-WS-01 | 并发连接数 | >= 1000 | 单服务器实例 | Artillery/k6 测试 | PLUGIN-v3.2-API.md |
| PERF-WS-02 | 消息吞吐量 | >= 10,000 msg/s | 1000 并发连接 | 压力测试 | PLUGIN-v3.2-API.md |
| PERF-WS-03 | 消息延迟(p99) | < 100ms | 局域网环境 | 延迟统计 | BACKEND-v3.2-WEBSOCKET.md |
| PERF-WS-04 | 心跳成功率 | >= 99% | 30s间隔,1小时 | 成功率统计 | BACKEND-v3.2-WEBSOCKET.md |
| PERF-WS-05 | 首次重连时间 | < 5s | 首次重连 | 计时测试 | BACKEND-v3.2-WEBSOCKET.md |
| PERF-WS-06 | 内存使用 | < 500MB | 1000 并发连接 | 内存监控 | PLUGIN-v3.2-API.md |

### 1.2 预计算服务性能指标

| 指标ID | 指标名称 | 基准值 | 测试条件 | 验证方法 | 来源文档 |
|--------|----------|--------|----------|----------|----------|
| PERF-PC-01 | 处理速度 | > 1000 行/秒 | 标准代码文件 | 吞吐量测试 | BACKEND-v3.2-PRECOMPUTE.md |
| PERF-PC-02 | 内存占用 | < 100MB | 大文件处理 | 内存监控 | BACKEND-v3.2-PRECOMPUTE.md |
| PERF-PC-03 | 批量插入 | > 100 条/批次 | 批量操作 | 批量测试 | BACKEND-v3.2-PRECOMPUTE.md |
| PERF-PC-04 | 增量识别率 | > 95% | 文件指纹对比 | 准确率测试 | BACKEND-v3.2-PRECOMPUTE.md |

### 1.3 系统整体性能指标

| 指标ID | 指标名称 | 基准值 | 测试条件 | 验证方法 | 来源文档 |
|--------|----------|--------|----------|----------|----------|
| PERF-SYS-01 | API响应时间 | < 100ms | 标准请求 | 响应时间测试 | UNIFIED-ARCHITECTURE-v3.2.md |
| PERF-SYS-02 | 搜索延迟 | < 200ms | 混合搜索 | 搜索性能测试 | UNIFIED-ARCHITECTURE-v3.2.md |
| PERF-SYS-03 | 数据库连接 | < 50ms | 初始连接 | 连接时间测试 | DEPLOYMENT-v3.2.md |

---

## 2. 功能要求验证标准

### 2.1 WebSocket 功能验证

| 功能ID | 功能名称 | 验证标准 | 测试方法 | 来源文档 |
|--------|----------|----------|----------|----------|
| FUNC-WS-01 | 心跳机制 | 30s间隔,2次未响应触发重连 | 心跳超时测试 | BACKEND-v3.2-WEBSOCKET.md |
| FUNC-WS-02 | 指数退避重连 | 1s->2s->4s...最大300s | 重连延迟测试 | BACKEND-v3.2-WEBSOCKET.md |
| FUNC-WS-03 | 消息确认 | 5s超时,3次重试 | ACK超时测试 | BACKEND-v3.2-WEBSOCKET.md |
| FUNC-WS-04 | DIFF模式 | 减少90%数据传输 | 数据量对比测试 | BACKEND-v3.2-WEBSOCKET.md |
| FUNC-WS-05 | 连接恢复 | session+offset恢复 | 断线重连测试 | BACKEND-v3.2-WEBSOCKET.md |
| FUNC-WS-06 | 持久化队列 | 7天过期清理 | 队列持久化测试 | BACKEND-v3.2-WEBSOCKET.md |
| FUNC-WS-07 | 文件锁 | portalocker跨平台锁 | 并发写入测试 | BACKEND-v3.2-WEBSOCKET.md |

### 2.2 预计算服务功能验证

| 功能ID | 功能名称 | 验证标准 | 测试方法 | 来源文档 |
|--------|----------|----------|----------|----------|
| FUNC-PC-01 | AST解析 | tree-sitter Query模式 | 解析准确性测试 | BACKEND-v3.2-PRECOMPUTE.md |
| FUNC-PC-02 | 符号提取 | 函数/类/接口提取 | 符号完整性测试 | BACKEND-v3.2-PRECOMPUTE.md |
| FUNC-PC-03 | 批量创建 | 批量插入Atoms | 批量性能测试 | BACKEND-v3.2-PRECOMPUTE.md |
| FUNC-PC-04 | 双向引用 | Entity<->Atoms双向 | 引用一致性测试 | BACKEND-v3.2-PRECOMPUTE.md |
| FUNC-PC-05 | 性能监控 | 耗时/内存/CPU记录 | 指标记录测试 | BACKEND-v3.2-PRECOMPUTE.md |
| FUNC-PC-06 | 增量更新 | SHA256指纹检测 | 变更检测测试 | BACKEND-v3.2-PRECOMPUTE.md |
| FUNC-PC-07 | 并发控制 | 最大5并发 | 并发限制测试 | BACKEND-v3.2-PRECOMPUTE.md |
| FUNC-PC-08 | 循环调用检测 | DFS算法检测环 | 循环检测测试 | BACKEND-v3.2-PRECOMPUTE.md |
| FUNC-PC-09 | 权重计算 | 多因素权重算法 | 权重准确性测试 | BACKEND-v3.2-PRECOMPUTE.md |

### 2.3 数据库功能验证

| 功能ID | 功能名称 | 验证标准 | 测试方法 | 来源文档 |
|--------|----------|----------|----------|----------|
| FUNC-DB-01 | RELATE语法 | 原生图关系支持 | 关系创建测试 | DATABASE-v3.2-SCHEMA.md |
| FUNC-DB-02 | ChangeFeed | 7天变更追踪 | 变更事件测试 | DATABASE-v3.2-SCHEMA.md |
| FUNC-DB-03 | 复合索引 | tenant_id+字段组合 | 查询性能测试 | DATABASE-v3.2-SCHEMA.md |
| FUNC-DB-04 | 唯一约束 | project.name/config.key唯一 | 重复插入测试 | DATABASE-v3.2-SCHEMA.md |
| FUNC-DB-05 | 自动更新 | updated_at自动更新 | 时间戳测试 | DATABASE-v3.2-SCHEMA.md |
| FUNC-DB-06 | Timeline事件 | Entity创建自动创建Timeline | 事件触发测试 | DATABASE-v3.2-SCHEMA.md |

---

## 3. 接口契约验证标准

### 3.1 原子操作接口契约

#### 3.1.1 创建 Atom (POST /api/v1/atoms)

**请求格式**:
```json
{
  "type": "function",
  "content": "async function analyzeCode(...) { ... }",
  "name": "analyzeCode",
  "signature": "async analyzeCode(filePath: string): Promise<AnalysisResult>",
  "params": [{"name": "filePath", "type": "string"}],
  "return_type": "Promise<AnalysisResult>",
  "is_exported": true,
  "is_async": true,
  "complexity": 5,
  "start_line": 85,
  "end_line": 125,
  "project": "backlog-api",
  "tenant_id": "default",
  "tags": ["typescript", "analysis"]
}
```

**响应格式**:
```json
{
  "id": "atom:01HQ...",
  "type": "function",
  "created_at": "2026-04-10T10:30:00Z"
}
```

**验证标准**:
- [ ] 请求必须包含 type, content 字段
- [ ] type 必须在枚举范围内
- [ ] 响应包含 ULID 格式的 id
- [ ] 响应包含 ISO8601 格式的 created_at

#### 3.1.2 更新 Atom (PATCH /api/v1/atoms/{atom_id})

**请求格式**:
```json
{
  "content": "更新后的内容",
  "version": 2
}
```

**响应格式**:
```json
{
  "id": "atom:01HQ...",
  "version": 2,
  "updated_at": "2026-04-10T10:35:00Z"
}
```

**验证标准**:
- [ ] version 必须递增
- [ ] 响应包含 updated_at 时间戳

### 3.2 实体操作接口契约

#### 3.2.1 创建 Entity (POST /api/v1/entities)

**请求格式**:
```json
{
  "type": "code",
  "title": "utils.ts",
  "abstract": "TypeScript file with 5 functions",
  "overview": {
    "language": "typescript",
    "lines_of_code": 150,
    "function_count": 5
  },
  "atoms": ["atom:func-1", "atom:func-2"],
  "file_path": "src/utils.ts",
  "language": "typescript",
  "project": "backlog-api",
  "tenant_id": "default",
  "tags": ["typescript", "utils"]
}
```

**响应格式**:
```json
{
  "id": "entity:01HQ...",
  "type": "code",
  "created_at": "2026-04-10T10:30:00Z"
}
```

**验证标准**:
- [ ] 请求必须包含 type, abstract 字段
- [ ] abstract 长度 <= 100 字符
- [ ] type 必须在 [memory, backlog, wiki, code] 范围内
- [ ] atoms 必须是有效的 atom ID 数组

### 3.3 WebSocket 消息契约

#### 3.3.1 心跳消息

**Ping 消息**:
```json
{
  "type": "ping",
  "timestamp": 1712745000.123
}
```

**Pong 消息**:
```json
{
  "type": "pong",
  "timestamp": 1712745000.123
}
```

**验证标准**:
- [ ] timestamp 为 Unix 时间戳(秒)
- [ ] 30s 间隔发送 ping
- [ ] 2次未响应触发重连

#### 3.3.2 带确认的消息

**发送消息**:
```json
{
  "action": "update",
  "data": {...},
  "_msgId": "msg-12345",
  "_requiresAck": true
}
```

**确认消息**:
```json
{
  "type": "ack",
  "_ackId": "msg-12345",
  "_ackData": {...}
}
```

**验证标准**:
- [ ] _msgId 全局唯一
- [ ] 5s 内收到 ack
- [ ] 3次重试后失败报错

#### 3.3.3 DIFF 消息

**订阅请求**:
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

**验证标准**:
- [ ] patches 符合 JSON Patch RFC 6902
- [ ] 仅发送变更字段
- [ ] 数据量减少 >= 90%

### 3.4 代码分析接口契约

#### 3.4.1 触发预计算 (POST /api/v1/code/precompute)

**请求格式**:
```json
{
  "file_path": "src/utils.ts",
  "source_code": "...",
  "language": "typescript",
  "tenant_id": "default"
}
```

**响应格式**:
```json
{
  "entity_id": "entity:code-src-utils",
  "atoms_count": 5,
  "duration_ms": 120,
  "memory_mb": 15.5,
  "success": true
}
```

**验证标准**:
- [ ] duration_ms < 10000 (10秒)
- [ ] memory_mb < 100 (100MB)
- [ ] atoms_count >= 0
- [ ] success 为布尔值

#### 3.4.2 代码导航 (GET /api/v1/code/navigate)

**查询参数**:
```
?symbol=analyzeCode&action=goto_definition
```

**响应格式**:
```json
{
  "symbol": "analyzeCode",
  "file_path": "src/utils.ts",
  "line": 85,
  "column": 10
}
```

**验证标准**:
- [ ] line 和 column 为非负整数
- [ ] file_path 为有效路径
- [ ] 404 时返回未找到错误

#### 3.4.3 爆炸半径分析 (GET /api/v1/code/impact)

**查询参数**:
```
?symbol=analyzeCode&depth=2&direction=both
```

**响应格式**:
```json
{
  "symbol": "analyzeCode",
  "impacted_symbols": [
    {"name": "parseSync", "depth": 1},
    {"name": "validateConfig", "depth": 2}
  ]
}
```

**验证标准**:
- [ ] depth 与请求参数一致
- [ ] impacted_symbols 按 depth 排序
- [ ] direction 支持 [upstream, downstream, both]

### 3.5 健康检查接口契约

#### 3.5.1 服务健康 (GET /health)

**响应格式**:
```json
{
  "status": "healthy",
  "version": "3.2.0",
  "timestamp": "2026-04-10T10:30:00Z"
}
```

**验证标准**:
- [ ] status 为 healthy/degraded/unhealthy
- [ ] version 符合 semver
- [ ] HTTP 200 状态码

#### 3.5.2 数据库健康 (GET /health/db)

**响应格式**:
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

**验证标准**:
- [ ] surrealdb 状态为 connected/disconnected
- [ ] meilisearch 状态为 connected/disconnected
- [ ] latency_ms < 100

---

## 4. 错误码验证标准

### 4.1 错误码分类

| 错误码 | 类型 | 说明 | HTTP状态码 | 处理策略 |
|--------|------|------|------------|----------|
| CONN_001 | 连接错误 | WebSocket连接失败 | 503 | 自动重试+指数退避 |
| CONN_002 | 连接错误 | 服务不可用 | 503 | 切换到备份端点 |
| MSG_001 | 消息错误 | 消息发送超时 | 504 | ACK超时+重试 |
| MSG_002 | 消息错误 | 消息格式错误 | 400 | 验证+拒绝 |
| MSG_003 | 消息错误 | 消息处理失败 | 500 | 队列保留+人工处理 |
| AUTH_001 | 认证错误 | API密钥无效 | 401 | 提示用户检查配置 |
| AUTH_002 | 认证错误 | Token过期 | 401 | 刷新token |
| RECN_001 | 重连错误 | 达到最大重试次数 | 503 | 降级模式 |
| RECN_002 | 重连错误 | 状态恢复失败 | 500 | 重新初始化 |

### 4.2 错误响应格式

```json
{
  "error": {
    "code": "CONN_001",
    "message": "WebSocket connection failed",
    "details": {...},
    "recoverable": true,
    "timestamp": "2026-04-10T10:30:00Z"
  }
}
```

**验证标准**:
- [ ] code 符合错误码规范
- [ ] message 为可读描述
- [ ] recoverable 为布尔值
- [ ] timestamp 为 ISO8601 格式

---

## 5. 测试用例清单

### 5.1 单元测试用例

| 用例ID | 测试目标 | 测试输入 | 预期输出 | 优先级 |
|--------|----------|----------|----------|--------|
| UT-WS-01 | ReliableWebSocketClient连接 | URL, token | 连接成功 | P0 |
| UT-WS-02 | 心跳机制 | 30s间隔 | 正常收发pong | P0 |
| UT-WS-03 | 指数退避重连 | 断开连接 | 延迟递增重连 | P0 |
| UT-PC-01 | PrecomputeService预计算 | 源代码 | Entity+Atoms创建 | P0 |
| UT-PC-02 | 指纹计算 | 文件内容 | SHA256哈希 | P1 |
| UT-PC-03 | 循环调用检测 | 调用关系 | 检测出环 | P1 |
| UT-DB-01 | RELATE创建关系 | from_id, to_id | 关系创建成功 | P0 |
| UT-DB-02 | ChangeFeed订阅 | LIVE SELECT | 收到变更事件 | P1 |

### 5.2 集成测试用例

| 用例ID | 测试目标 | 测试场景 | 预期结果 | 优先级 |
|--------|----------|----------|----------|--------|
| IT-01 | 完整文件保存流程 | 保存->预计算->存储 | 数据正确存储 | P0 |
| IT-02 | WebSocket实时同步 | 修改->推送->接收 | 客户端收到更新 | P0 |
| IT-03 | 搜索功能 | 关键词搜索 | 返回相关结果 | P0 |
| IT-04 | 代码导航 | 跳转到定义 | 定位正确位置 | P1 |
| IT-05 | 爆炸半径分析 | 修改函数分析 | 返回影响范围 | P1 |

### 5.3 性能测试用例

| 用例ID | 测试目标 | 测试负载 | 通过标准 | 优先级 |
|--------|----------|----------|----------|--------|
| PT-01 | WebSocket并发 | 1000连接 | 成功率>95% | P0 |
| PT-02 | 消息吞吐量 | 10000msg/s | 无丢包 | P0 |
| PT-03 | 预计算性能 | 1000行代码 | <10s | P1 |
| PT-04 | 搜索性能 | 10000文档 | <200ms | P1 |

---

*此文档由自动化分析生成，用于 v3.2 验证标准追踪*
