# WebSocket ACK 消息协议定义

> **版本**: v3.2.0  
> **日期**: 2026-04-14  
> **状态**: 实施版  
> **范围**: ACK 消息确认机制协议规范

---

## 目录

1. [协议概述](#1-协议概述)
2. [ACK 消息格式](#2-ack-消息格式)
3. [客户端 ACK 发送时机](#3-客户端-ack-发送时机)
4. [服务端 ACK 处理流程](#4-服务端-ack-处理流程)
5. [错误处理规范](#5-错误处理规范)
6. [示例](#6-示例)

---

## 1. 协议概述

### 1.1 设计目标

ACK（Acknowledgement）消息协议用于确保 WebSocket 消息的可靠投递，防止消息丢失。

### 1.2 核心机制

| 机制 | 说明 | 参数 |
|------|------|------|
| 消息确认 | 客户端收到消息后发送 ACK | - |
| 超时重试 | 服务端未收到 ACK 自动重发 | 5s 超时，3 次重试 |
| 唯一标识 | 每条消息附带唯一 ACK ID | UUID v4 |

---

## 2. ACK 消息格式

### 2.1 服务端发送消息（带 ACK 请求）

```json
{
  "type": "data",
  "payload": {
    "action": "CREATE",
    "result": {
      "id": "memory:abc123",
      "content": "..."
    }
  },
  "_ackId": "550e8400-e29b-41d4-a716-446655440000"
}
```text

**字段说明**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | 消息类型：`data` |
| `payload` | object | 是 | 消息内容 |
| `_ackId` | string | 是 | ACK 唯一标识符（UUID） |

### 2.2 客户端 ACK 响应

```json
{
  "type": "ack",
  "_ackId": "550e8400-e29b-41d4-a716-446655440000"
}
```text

**字段说明**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | 固定值：`ack` |
| `_ackId` | string | 是 | 对应消息的 ACK ID |

---

## 3. 客户端 ACK 发送时机

### 3.1 标准流程

```text
服务端发送消息 ──► 客户端接收处理 ──► 客户端发送 ACK
     │                    │                    │
     │                    │                    │
   启动定时器           业务处理            清除定时器
  (5s 超时)           (建议 <100ms)        (停止重试)
```text

### 3.2 发送时机规则

| 场景 | 行为 | 说明 |
|------|------|------|
| 消息成功处理 | 立即发送 ACK | 业务逻辑处理完成后 |
| 消息处理失败 | 仍发送 ACK | 防止服务端无限重试 |
| 连接断开 | 不发送 ACK | 服务端自动重连后重发 |

### 3.3 客户端实现建议

```javascript
// JavaScript 示例
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  // 1. 检查是否需要 ACK
  if (message._ackId) {
    // 2. 处理业务逻辑
    processMessage(message.payload);
    
    // 3. 发送 ACK（无论业务处理成功与否）
    ws.send(JSON.stringify({
      type: "ack",
      _ackId: message._ackId
    }));
  }
};
```text

---

## 4. 服务端 ACK 处理流程

### 4.1 消息发送流程

```python
# Python 伪代码
async def send_message_with_ack(websocket, message):
    ack_id = generate_uuid()
    message_with_ack = {**message, "_ackId": ack_id}
    
    # 1. 发送消息
    await websocket.send_json(message_with_ack)
    
    # 2. 启动超时定时器
    start_timeout_timer(ack_id, timeout=5.0)
    
    # 3. 等待 ACK 或超时
    if await wait_for_ack(ack_id):
        # 3a. 收到 ACK，清理资源
        cleanup(ack_id)
        return True
    else:
        # 3b. 超时，重试
        return await retry_send(websocket, message, ack_id)
```text

### 4.2 ACK 接收处理

```python
# Python 伪代码
async def handle_message(websocket, raw_message):
    message = json.loads(raw_message)
    
    if message.get("type") == "ack":
        ack_id = message.get("_ackId")
        if ack_id:
            # 标记 ACK 已收到
            ack_manager.handle_ack(ack_id)
```text

### 4.3 重试机制

| 参数 | 值 | 说明 |
|------|-----|------|
| 超时时间 | 5.0s | 等待 ACK 的最大时间 |
| 最大重试次数 | 3 | 超过后放弃发送 |
| 重试间隔 | 即时 | 超时后立即重试 |

---

## 5. 错误处理规范

### 5.1 服务端错误处理

| 错误场景 | 处理方式 | 日志级别 |
|----------|----------|----------|
| ACK 超时 | 自动重试 | WARNING |
| 超过最大重试 | 放弃发送，记录失败 | ERROR |
| 收到未知 ACK ID | 忽略 | DEBUG |
| 连接断开 | 停止所有 pending ACK | INFO |

### 5.2 客户端错误处理

| 错误场景 | 处理方式 |
|----------|----------|
| 收到重复消息 | 正常处理，发送 ACK |
| ACK 发送失败 | 忽略（服务端会重试） |
| 消息格式错误 | 发送 ACK（防止服务端重试） |

### 5.3 边界情况

```text
情况 1: ACK 丢失
  服务端 ──► 客户端（消息到达）
  客户端 ──► 服务端（ACK 丢失）
  服务端超时重发 ──► 客户端
  客户端再次 ACK

情况 2: 消息重复
  服务端 ──► 客户端（消息到达）
  客户端 ACK ──► 服务端（ACK 延迟）
  服务端超时重发 ──► 客户端
  客户端再次 ACK（幂等处理）

情况 3: 连接断开
  服务端发送消息
  连接断开（客户端未收到）
  服务端检测到断开
  重连后服务端重新发送
```text

---

## 6. 示例

### 6.1 完整交互示例

```text
# 1. 连接建立
Client ──► Server: {"type": "connect", "tenant_id": "default"}
Server ──► Client: {"type": "connected", "session_id": "sess-xxx"}

# 2. 服务端推送消息（带 ACK）
Server ──► Client: {
  "type": "data",
  "payload": {"action": "CREATE", "result": {...}},
  "_ackId": "550e8400-e29b-41d4-a716-446655440000"
}

# 3. 客户端确认
Client ──► Server: {
  "type": "ack",
  "_ackId": "550e8400-e29b-41d4-a716-446655440000"
}

# 4. 超时重试场景（ACK 未收到）
Server ──► Client: {
  "type": "data",
  "payload": {"action": "UPDATE", "result": {...}},
  "_ackId": "660e8400-e29b-41d4-a716-446655440001"
}
# ... 5s 后未收到 ACK ...
Server ──► Client: {
  "type": "data",
  "payload": {"action": "UPDATE", "result": {...}},
  "_ackId": "660e8400-e29b-41d4-a716-446655440001"
}

# 5. 客户端确认重试消息
Client ──► Server: {
  "type": "ack",
  "_ackId": "660e8400-e29b-41d4-a716-446655440001"
}
```text

### 6.2 DIFF 模式配置

**连接参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `mode` | string | 否 | 同步模式：`diff`（增量）或 `full`（完整），默认 `full` |
| `session_id` | string | 否 | 恢复已有 Session（断线重连时使用） |
| `tenant_id` | string | 否 | 租户 ID，默认 `default` |
| `token` | string | 否 | 认证 token |

**连接示例**:

```javascript
// 完整模式（默认）
const ws1 = new WebSocket('ws://localhost:18008/ws/memories/live');

// 增量模式
const ws2 = new WebSocket('ws://localhost:18008/ws/memories/live?mode=diff');

// 断线重连（恢复 Session）
const ws3 = new WebSocket('ws://localhost:18008/ws/memories/live?mode=diff&session_id=sess-xxx');
```text

**模式对比**:

| 特性 | full 模式 | diff 模式 |
|------|-----------|-----------|
| 数据格式 | 完整数据对象 | JSON Patch |
| 带宽占用 | 高 | 低（节省 50%+） |
| CPU 占用 | 低 | 中等 |
| 适用场景 | 初始同步、小数据量 | 实时更新、大数据量 |
| 向后兼容 | ✅ 默认 | 需显式指定 |

### 6.3 代码示例

**服务端（Python）**:

```python
from wrapper.src.websocket import ReliableWebSocketServer

server = ReliableWebSocketServer(
    websocket=websocket,
    diff_mode="diff",  # 或 "full"
)
await server.accept()

# 发送带 ACK 的消息
success = await server.send_json_with_ack({
    "type": "data",
    "payload": {...}
}, timeout=5.0)

if success:
    print("消息已确认送达")
else:
    print("消息发送失败（超过最大重试次数）")

# 动态切换模式
server.set_diff_mode("full")
```text

**客户端（JavaScript）**:

```javascript
// 连接（使用 diff 模式）
const ws = new WebSocket('ws://localhost:18008/ws/memories/live?mode=diff');

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  // 处理消息
  if (message.type === 'change') {
    if (message.action === 'CREATE') {
      // 处理新建
      handleCreate(message.data);
    } else if (message.action === 'UPDATE') {
      // 处理更新（应用 patches）
      applyPatches(message.patches);
    } else if (message.action === 'DELETE') {
      // 处理删除
      handleDelete(message.id);
    }
  }
  
  // 发送 ACK
  if (message._ackId) {
    ws.send(JSON.stringify({
      type: 'ack',
      _ackId: message._ackId
    }));
  }
};

// 应用 JSON Patch
function applyPatches(patches) {
  patches.forEach(patch => {
    if (patch.op === 'replace') {
      // 替换值
      setValueAtPath(patch.path, patch.value);
    } else if (patch.op === 'add') {
      // 添加值
      addValueAtPath(patch.path, patch.value);
    } else if (patch.op === 'remove') {
      // 删除值
      removeValueAtPath(patch.path);
    }
  });
}
```text

---

## 附录

### A. 相关文件

| 文件 | 说明 |
|------|------|
| `wrapper/src/websocket/ack_manager.py` | ACK 管理器实现 |
| `wrapper/src/websocket/diff_manager.py` | DIFF 管理器实现 |
| `wrapper/src/websocket/live_diff_handler.py` | LIVE SELECT DIFF 处理器 |
| `wrapper/src/websocket/reliable_server.py` | 可靠 WebSocket 服务器 |
| `wrapper/src/routers/websocket.py` | WebSocket 路由端点 |
| `tests/test_websocket_ack.py` | ACK 单元测试 |
| `tests/test_websocket_integration.py` | ACK 集成测试 |
| `tests/test_websocket_diff_integration.py` | DIFF 集成测试 |
| `tests/test_websocket_live_diff.py` | LIVE SELECT DIFF 测试 |

### B. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v3.2.0 | 2026-04-14 | 初始版本（ACK 协议） |
| v3.2.1 | 2026-04-14 | 添加 DIFF 模式支持 |
