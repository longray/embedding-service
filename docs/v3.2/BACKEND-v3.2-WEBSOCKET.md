# 后端 v3.2 WebSocket 详细设计

> **版本**: v3.2.0  
> **日期**: 2026-04-10  
> **状态**: 实施版  
> **目标**: 实现完整的 WebSocket 可靠性保障（心跳、指数退避重连、消息确认、DIFF 模式）

---

## 目录

1. [设计目标](#1-设计目标)
2. [架构设计](#2-架构设计)
3. [核心组件](#3-核心组件)
4. [实现细节](#4-实现细节)
5. [测试验证](#5-测试验证)

---

## 1. 设计目标

### 1.1 可靠性要求

| 功能             | 要求                         | 实现方式                |
| ---------------- | ---------------------------- | ----------------------- |
| **心跳机制**     | 30s 间隔，2 次未响应触发重连 | ReliableWebSocketClient |
| **指数退避重连** | 1s → 2s → 4s... 最大 300s    | \_schedule_reconnect()  |
| **消息确认**     | 5s 超时，3 次重试            | AcknowledgementSystem   |
| **DIFF 模式**    | 减少 90% 数据传输            | DiffSubscription        |
| **连接恢复**     | session + offset 恢复        | ConnectionStateRecovery |

### 1.2 性能指标

- 并发连接：1000+
- 心跳成功率：> 99%
- 消息延迟：< 100ms
- 重连时间：< 5s（首次）

---

## 2. 架构设计

### 2.1 组件关系

```
┌─────────────────────────────────────────────────────────────┐
│                    WebSocket 架构                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │  Client (Node)  │◄──►│  Server (Python)│                │
│  │                 │    │                 │                │
│  │ • ws 8.20       │    │ • FastAPI       │                │
│  │ • autoPong      │    │ • websockets    │                │
│  │                 │    │                 │                │
│  └────────┬────────┘    └────────┬────────┘                │
│           │                      │                         │
│           ▼                      ▼                         │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ ReliableClient  │    │ ReliableServer  │                │
│  │                 │    │                 │                │
│  │ • heartbeat     │    │ • heartbeat     │                │
│  │ • reconnect     │    │ • ack_system    │                │
│  │ • ack_handler   │    │ • diff_sub      │                │
│  │                 │    │                 │                │
│  └─────────────────┘    └─────────────────┘                │
│                                                             │
│  ┌─────────────────────────────────────────┐               │
│  │           PersistentQueue               │               │
│  │  • 离线消息持久化                        │               │
│  │  • 文件锁（portalocker）                │               │
│  │  • 7 天过期清理                          │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
1. 连接建立
   Client ──► Server
   └── 认证 (token)
   └── 订阅 (LIVE SELECT)

2. 心跳维持
   Server ──ping──► Client
   Client ──pong──► Server
   └── 30s 间隔
   └── 2 次未响应 = 断开

3. 消息推送
   Server ──data──► Client
   └── 包含 messageId
   └── 需要 ACK

4. 消息确认
   Client ──ack──► Server
   └── 包含 ackId
   └── 5s 超时重试

5. DIFF 模式
   Server ──diff──► Client
   └── JSON Patch 格式
   └── 客户端应用补丁
```

---

## 3. 核心组件

### 3.1 ReliableWebSocketClient

```python
import asyncio
import json
import time
import random
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class ReliableWebSocketClient:
    """
    可靠的 WebSocket 客户端

    功能：
    - 心跳机制（30s 间隔）
    - 指数退避重连（1s → 2s → 4s... 最大 300s）
    - 连接状态恢复
    """

    def __init__(
        self,
        url: str,
        token: str,
        tenant_id: str = "default",
        heartbeat_interval: float = 30.0,
        max_missed_pongs: int = 2,
        base_delay: float = 1.0,
        max_delay: float = 300.0,
        max_retries: int = 10
    ):
        self.url = url
        self.token = token
        self.tenant_id = tenant_id

        # 心跳配置
        self.heartbeat_interval = heartbeat_interval
        self.max_missed_pongs = max_missed_pongs
        self.missed_pongs = 0
        self.heartbeat_timer: Optional[asyncio.Task] = None

        # 重连配置
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.retry_count = 0

        # 状态
        self.ws = None
        self.connected = False
        self.latency = 0.0

        # 回调
        self.on_message: Optional[Callable] = None
        self.on_connect: Optional[Callable] = None
        self.on_disconnect: Optional[Callable] = None

    async def connect(self) -> bool:
        """建立连接"""
        try:
            import websockets

            # 构建 URL（带认证参数）
            auth_url = f"{self.url}?token={self.token}&tenant_id={self.tenant_id}"

            self.ws = await websockets.connect(auth_url)
            self.connected = True
            self.retry_count = 0
            self.missed_pongs = 0

            logger.info(f"WebSocket connected: {self.url}")

            # 启动心跳
            self.heartbeat_timer = asyncio.create_task(self._heartbeat_loop())

            # 触发回调
            if self.on_connect:
                await self.on_connect()

            # 启动消息接收
            asyncio.create_task(self._receive_loop())

            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            await self._schedule_reconnect()
            return False

    async def _heartbeat_loop(self):
        """心跳循环"""
        while self.connected:
            try:
                # 检查未响应次数
                if self.missed_pongs >= self.max_missed_pongs:
                    logger.warning(f"Missed {self.missed_pongs} pongs, reconnecting...")
                    await self.disconnect()
                    await self._schedule_reconnect()
                    return

                # 发送 ping
                ping_data = json.dumps({
                    "type": "ping",
                    "timestamp": time.time()
                })
                await self.ws.send(ping_data)
                self.missed_pongs += 1

                # 等待 pong（在 _receive_loop 中处理）
                await asyncio.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await self.disconnect()
                await self._schedule_reconnect()
                return

    async def _receive_loop(self):
        """接收消息循环"""
        while self.connected:
            try:
                message = await self.ws.recv()
                data = json.loads(message)

                # 处理 pong
                if data.get("type") == "pong":
                    self.missed_pongs = 0
                    self.latency = time.time() - data.get("timestamp", 0)
                    logger.debug(f"Pong received, latency: {self.latency:.3f}s")
                    continue

                # 触发消息回调
                if self.on_message:
                    await self.on_message(data)

            except Exception as e:
                logger.error(f"Receive error: {e}")
                await self.disconnect()
                await self._schedule_reconnect()
                return

    async def _schedule_reconnect(self):
        """指数退避重连"""
        if self.retry_count >= self.max_retries:
            logger.error(f"Max retries ({self.max_retries}) exceeded")
            return

        # 计算延迟：base * 2^retry + jitter
        delay = min(
            self.base_delay * (2 ** self.retry_count),
            self.max_delay
        )
        delay += random.uniform(0, 1)  # 添加抖动

        self.retry_count += 1
        logger.info(f"Reconnecting in {delay:.2f}s (attempt {self.retry_count})")

        await asyncio.sleep(delay)
        await self.connect()

    async def send(self, data: dict) -> bool:
        """发送消息"""
        if not self.connected:
            logger.error("Cannot send: not connected")
            return False

        try:
            await self.ws.send(json.dumps(data))
            return True
        except Exception as e:
            logger.error(f"Send error: {e}")
            return False

    async def disconnect(self):
        """断开连接"""
        self.connected = False

        if self.heartbeat_timer:
            self.heartbeat_timer.cancel()
            self.heartbeat_timer = None

        if self.ws:
            await self.ws.close()
            self.ws = None

        logger.info("WebSocket disconnected")

        if self.on_disconnect:
            await self.on_disconnect()
```

### 3.2 AcknowledgementSystem

```python
import asyncio
from typing import Dict, Callable
from dataclasses import dataclass


@dataclass
class PendingAck:
    """待确认消息"""
    message_id: str
    future: asyncio.Future
    timeout_handle: asyncio.Handle
    retry_count: int = 0


class AcknowledgementSystem:
    """
    消息确认系统

    功能：
    - 发送消息并等待确认
    - 超时重试（指数退避）
    - 最大 3 次重试
    """

    def __init__(
        self,
        client: ReliableWebSocketClient,
        timeout: float = 5.0,
        max_retries: int = 3
    ):
        self.client = client
        self.timeout = timeout
        self.max_retries = max_retries
        self.pending: Dict[str, PendingAck] = {}
        self.message_counter = 0

    async def send_with_ack(self, data: dict) -> dict:
        """
        发送消息并等待确认

        Args:
            data: 消息数据

        Returns:
            确认响应

        Raises:
            TimeoutError: 超过最大重试次数
        """
        for attempt in range(self.max_retries):
            try:
                # 生成消息 ID
                self.message_counter += 1
                message_id = f"msg-{self.message_counter}"

                # 创建 Future
                future = asyncio.get_event_loop().create_future()

                # 设置超时
                timeout_delay = self.timeout * (2 ** attempt)  # 指数增加
                timeout_handle = asyncio.get_event_loop().call_later(
                    timeout_delay,
                    lambda: future.set_exception(TimeoutError())
                )

                # 保存待确认
                self.pending[message_id] = PendingAck(
                    message_id=message_id,
                    future=future,
                    timeout_handle=timeout_handle,
                    retry_count=attempt
                )

                # 发送消息
                message = {
                    **data,
                    "_msgId": message_id,
                    "_requiresAck": True
                }
                await self.client.send(message)

                # 等待确认
                result = await future
                return result

            except TimeoutError:
                logger.warning(f"Ack timeout (attempt {attempt + 1})")
                if attempt == self.max_retries - 1:
                    raise TimeoutError(f"Message failed after {self.max_retries} retries")

                # 退避延迟
                await asyncio.sleep(0.5 * (2 ** attempt))
                continue

    def handle_ack(self, message: dict):
        """处理确认响应"""
        ack_id = message.get("_ackId")
        if not ack_id:
            return

        pending = self.pending.pop(ack_id, None)
        if not pending:
            return

        # 取消超时
        pending.timeout_handle.cancel()

        # 设置结果
        if not pending.future.done():
            pending.future.set_result(message.get("_ackData", {}))

        logger.debug(f"Ack received for {ack_id}")
```

### 3.3 ConnectionStateRecovery

```python
import json
from pathlib import Path


class ConnectionStateRecovery:
    """
    连接状态恢复

    功能：
    - session ID 生成
    - last offset 记录
    - 丢失消息同步
    """

    def __init__(self, state_file: str = ".opencode/ws-state.json"):
        self.state_file = Path(state_file)
        self.session_id = self._generate_session_id()
        self.last_offset = "0"
        self._load_state()

    def _generate_session_id(self) -> str:
        """生成 session ID"""
        import uuid
        return f"sess-{int(time.time())}-{uuid.uuid4().hex[:9]}"

    def _load_state(self):
        """加载状态"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.session_id = state.get("session_id", self.session_id)
                    self.last_offset = state.get("last_offset", "0")
            except Exception:
                pass

    def save_state(self):
        """保存状态"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump({
                "session_id": self.session_id,
                "last_offset": self.last_offset
            }, f)

    async def sync_missed_messages(self, client: ReliableWebSocketClient, from_offset: str):
        """同步丢失的消息"""
        # 调用后端 API 获取丢失的消息
        sync_request = {
            "action": "sync",
            "sessionId": self.session_id,
            "lastOffset": from_offset
        }
        await client.send(sync_request)
```

### 3.4 DiffSubscription

```python
import json
from typing import Dict, Callable


class DiffSubscription:
    """
    DIFF 模式订阅

    功能：
    - LIVE SELECT DIFF 订阅
    - 本地缓存维护
    - JSON Patch 应用
    """

    def __init__(self, client: ReliableWebSocketClient):
        self.client = client
        self.local_cache: Dict[str, dict] = {}
        self.subscriptions: Dict[str, str] = {}  # entity_id -> query_uuid
        self.on_update: Optional[Callable] = None

    async def subscribe(self, entity_id: str, tenant_id: str = "default"):
        """
        订阅 DIFF 模式

        Args:
            entity_id: 实体 ID
            tenant_id: 租户 ID
        """
        # 先获取完整数据缓存
        # 注意：这里应该调用后端 API 获取完整数据
        # 简化示例：
        self.local_cache[entity_id] = {}

        # 订阅 DIFF 模式
        subscribe_request = {
            "action": "subscribe",
            "query": f'LIVE SELECT DIFF FROM entity WHERE id = "{entity_id}" AND tenant_id = "{tenant_id}"'
        }
        await self.client.send(subscribe_request)

        logger.info(f"Subscribed to DIFF for {entity_id}")

    def apply_diff(self, entity_id: str, patches: list):
        """
        应用差异更新

        Args:
            entity_id: 实体 ID
            patches: JSON Patch 格式的差异
        """
        import jsonpatch

        current = self.local_cache.get(entity_id)
        if not current:
            return

        try:
            # 应用补丁
            updated = jsonpatch.apply_patch(current, patches)
            self.local_cache[entity_id] = updated

            # 触发更新回调
            if self.on_update:
                self.on_update(entity_id, updated, patches)

            logger.debug(f"Applied diff to {entity_id}")

        except Exception as e:
            logger.error(f"Failed to apply diff: {e}")
```

### 3.5 PersistentMessageQueue

```python
import json
import time
from pathlib import Path
from typing import List, Optional


class PersistentMessageQueue:
    """
    持久化消息队列

    功能：
    - 离线消息持久化
    - 文件锁（portalocker）
    - 7 天过期清理
    """

    def __init__(self, queue_file: str = ".opencode/ws-queue.json"):
        self.queue_file = Path(queue_file)
        self.lock_file = self.queue_file.with_suffix(".lock")
        self.max_age = 7 * 24 * 60 * 60 * 1000  # 7 天（毫秒）

    def _load(self) -> List[dict]:
        """加载队列（带文件锁）"""
        import portalocker

        if not self.queue_file.exists():
            return []

        with portalocker.Lock(str(self.lock_file), timeout=5):
            try:
                with open(self.queue_file, 'r') as f:
                    data = json.load(f)
                    # 清理过期消息
                    now = time.time() * 1000
                    return [
                        msg for msg in data
                        if now - msg.get("_queuedAt", 0) < self.max_age
                    ]
            except Exception:
                return []

    def _save(self, queue: List[dict]):
        """保存队列（原子写入）"""
        import portalocker

        self.queue_file.parent.mkdir(parents=True, exist_ok=True)

        with portalocker.Lock(str(self.lock_file), timeout=5):
            temp_file = self.queue_file.with_suffix(".tmp")
            with open(temp_file, 'w') as f:
                json.dump(queue, f)
            temp_file.replace(self.queue_file)

    def push(self, message: dict):
        """入队"""
        queue = self._load()
        queue.append({
            **message,
            "_queuedAt": time.time() * 1000
        })
        self._save(queue)

    def pop(self) -> Optional[dict]:
        """出队"""
        queue = self._load()
        if not queue:
            return None

        item = queue.pop(0)
        self._save(queue)
        return item

    def peek_all(self) -> List[dict]:
        """查看所有消息（不出队）"""
        return self._load()
```

---

## 4. 实现细节

### 4.1 FastAPI WebSocket 端点

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket 端点

    功能：
    - 认证
    - 心跳处理
    - 消息路由
    """
    await websocket.accept()

    try:
        while True:
            # 接收消息
            data = await websocket.receive_json()

            # 处理 ping
            if data.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": data.get("timestamp")
                })
                continue

            # 处理订阅
            if data.get("action") == "subscribe":
                query = data.get("query")
                # 启动 LIVE SELECT
                # ...
                continue

            # 处理确认
            if data.get("type") == "ack":
                ack_id = data.get("ackId")
                # 通知 AcknowledgementSystem
                # ...
                continue

            # 其他消息处理
            # ...

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
```

### 4.2 配置

```python
# config.py
class WebSocketConfig:
    """WebSocket 配置"""

    # 心跳
    HEARTBEAT_INTERVAL = 30.0  # 秒
    MAX_MISSED_PONGS = 2

    # 重连
    BASE_DELAY = 1.0  # 秒
    MAX_DELAY = 300.0  # 秒
    MAX_RETRIES = 10

    # 确认
    ACK_TIMEOUT = 5.0  # 秒
    ACK_MAX_RETRIES = 3

    # 队列
    QUEUE_FILE = ".opencode/ws-queue.json"
    QUEUE_MAX_AGE = 7 * 24 * 60 * 60 * 1000  # 7 天
```

---

## 5. 测试验证

### 5.1 单元测试

```python
import pytest
import asyncio


@pytest.mark.asyncio
async def test_reliable_client():
    """测试 ReliableWebSocketClient"""
    client = ReliableWebSocketClient(
        url="ws://localhost:18008/ws",
        token="test-token",
        tenant_id="default"
    )

    # 测试连接
    result = await client.connect()
    assert result is True

    # 测试心跳
    await asyncio.sleep(35)  # 等待心跳
    assert client.missed_pongs == 0

    # 测试断开
    await client.disconnect()
    assert client.connected is False


@pytest.mark.asyncio
async def test_ack_system():
    """测试 AcknowledgementSystem"""
    client = ReliableWebSocketClient(...)
    ack_system = AcknowledgementSystem(client)

    # 测试发送并等待确认
    with pytest.raises(TimeoutError):
        await ack_system.send_with_ack({"test": "data"})
```

### 5.2 集成测试

```python
@pytest.mark.asyncio
async def test_websocket_full_flow():
    """测试完整 WebSocket 流程"""

    # 1. 建立连接
    client = ReliableWebSocketClient(...)
    await client.connect()

    # 2. 订阅 DIFF
    diff_sub = DiffSubscription(client)
    await diff_sub.subscribe("entity:test-001")

    # 3. 发送带确认的消息
    ack_system = AcknowledgementSystem(client)
    result = await ack_system.send_with_ack({"action": "test"})

    # 4. 断开并重连
    await client.disconnect()
    await asyncio.sleep(1)
    await client.connect()

    # 5. 验证状态恢复
    assert client.retry_count == 0
```

### 5.3 性能测试

```python
@pytest.mark.asyncio
async def test_websocket_performance():
    """测试 WebSocket 性能"""

    # 并发连接测试
    clients = []
    for i in range(100):
        client = ReliableWebSocketClient(...)
        await client.connect()
        clients.append(client)

    # 发送 1000 条消息
    for i in range(1000):
        await clients[0].send({"test": i})

    # 验证延迟
    assert clients[0].latency < 0.1  # < 100ms
```

---

## 参考文档

- [UNIFIED-ARCHITECTURE-v3.2.md](./UNIFIED-ARCHITECTURE-v3.2.md)
- [BACKEND-v3.2-IMPLEMENTATION.md](./BACKEND-v3.2-IMPLEMENTATION.md)
- [BACKEND-v3.2-PRECOMPUTE.md](./BACKEND-v3.2-PRECOMPUTE.md)

---

_文档版本: v3.2.0_  
_最后更新: 2026-04-10_
