# 插件端 API 规范

> **版本**: v3.2.0 + v3.3.0 增补  
> **日期**: 2026-04-10（v3.2）/ 2026-04-29（v3.3）  
> **状态**: 实施版  
> **目标**: 定义插件端与后端 v3.2/v3.3 的 API 接口

---

## 目录

1. [基础配置](#1-基础配置)
2. [Memory API](#2-memory-api)
3. [Code Analysis API](#3-code-analysis-api)
4. [WebSocket API](#4-websocket-api)
5. [Entity API](#5-entity-api)
6. [Atom API](#6-atom-api) *(v3.3)*
7. [统一搜索 API](#7-统一搜索-api) *(v3.3)*
8. [数据模型](#8-数据模型) *(v3.3 增补)*
9. [错误处理](#9-错误处理)

---

## 1. 基础配置

### 1.1 服务端点

```javascript
// config.js
export const config = {
  // v3.2 新端口
  API_BASE: "http://localhost:18008/api/v1",
  WS_BASE: "ws://localhost:18008/api/v1/ws",

  // 向后兼容（已废弃）
  LEGACY_API_BASE: "http://localhost:18008/api/v1",

  // 认证
  API_KEY: process.env.OPENCODE_API_KEY,

  // 超时
  TIMEOUT: 30000,
  WS_HEARTBEAT_INTERVAL: 30000,
};
```

### 1.2 请求封装

```javascript
// lib/api-client.js
import { config } from "./config.js";

export class ApiClient {
  constructor() {
    this.baseUrl = config.API_BASE;
    this.headers = {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.API_KEY}`,
    };
  }

  async request(method, path, data = null) {
    const url = `${this.baseUrl}${path}`;
    const options = {
      method,
      headers: this.headers,
    };

    if (data) {
      options.body = JSON.stringify(data);
    }

    const response = await fetch(url, options);

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  }

  get(path) {
    return this.request("GET", path);
  }

  post(path, data) {
    return this.request("POST", path, data);
  }

  patch(path, data) {
    return this.request("PATCH", path, data);
  }

  delete(path) {
    return this.request("DELETE", path);
  }
}
```

---

## 2. Memory API

### 2.1 创建记忆

```javascript
// POST /api/v1/memories
const memory = await apiClient.post('/memories', {
  type: 'code',
  title: 'utils.ts',
  abstract: 'Utility functions for file operations',
  overview: {
    language: 'typescript',
    lines: 150,
    functions: 5
  },
  content: 'full content here...',
  project: 'my-project',
  tenant_id: 'default',
  tags: ['typescript', 'utils']
});

// 响应
{
  "id": "entity:01HQ...",
  "type": "code",
  "created_at": "2026-04-10T10:30:00Z"
}
```

### 2.2 搜索记忆

```javascript
// GET /api/v1/memories/search
const results = await apiClient.get('/memories/search?query=file+operations&project=my-project');

// 响应
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

### 2.3 获取记忆

```javascript
// GET /api/v1/memories/{id}
const memory = await apiClient.get('/memories/entity:01HQ...');

// 响应
{
  "id": "entity:01HQ...",
  "type": "code",
  "title": "utils.ts",
  "abstract": "Utility functions...",
  "overview": {...},
  "content": "...",
  "atoms": ["atom:func-1", "atom:func-2"]
}
```

### 2.4 更新记忆

```javascript
// PATCH /api/v1/memories/{id}
await apiClient.patch("/memories/entity:01HQ...", {
  abstract: "Updated abstract",
  tags: ["typescript", "utils", "file"],
});
```

### 2.5 删除记忆

```javascript
// DELETE /api/v1/memories/{id}
await apiClient.delete("/memories/entity:01HQ...");
```

---

## 3. Code Analysis API

### 3.1 触发预计算

```javascript
// POST /api/v1/code/precompute
const result = await apiClient.post('/code/precompute', {
  file_path: 'src/utils.ts',
  source_code: '...',
  language: 'typescript',
  tenant_id: 'default'
});

// 响应
{
  "entity_id": "entity:code-src-utils",
  "atoms_count": 5,
  "duration_ms": 120,
  "memory_mb": 15.5,
  "success": true
}
```

### 3.2 代码导航

```javascript
// GET /api/v1/code/navigate
const result = await apiClient.get('/code/navigate?symbol=analyzeCode&action=goto_definition');

// 响应
{
  "symbol": "analyzeCode",
  "file_path": "src/utils.ts",
  "line": 85,
  "column": 10
}
```

### 3.3 爆炸半径分析

```javascript
// GET /api/v1/code/impact
const result = await apiClient.get('/code/impact?symbol=analyzeCode&depth=2&direction=both');

// 响应
{
  "symbol": "analyzeCode",
  "impacted_symbols": [
    {"name": "parseSync", "depth": 1},
    {"name": "validateConfig", "depth": 2}
  ]
}
```

### 3.4 代码搜索

```javascript
// GET /api/v1/code/search
const results = await apiClient.get('/code/search?query=async+file&language=typescript&hybrid=true');

// 响应
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

```javascript
// lib/websocket-client.js
import WebSocket from "ws";
import { config } from "./config.js";

export class MemoryWebSocket {
  constructor() {
    this.url = `${config.WS_BASE}?token=${config.API_KEY}&tenant_id=default`;
    this.ws = null;
    this.subscriptions = new Map();
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);

      this.ws.on("open", () => {
        console.log("WebSocket connected");
        resolve();
      });

      this.ws.on("message", (data) => {
        const message = JSON.parse(data);
        this.handleMessage(message);
      });

      this.ws.on("error", reject);
    });
  }

  handleMessage(message) {
    const { type, data } = message;

    switch (type) {
      case "update":
        this.handleUpdate(data);
        break;
      case "diff":
        this.handleDiff(data);
        break;
      case "ack":
        this.handleAck(data);
        break;
    }
  }
}
```

### 4.2 订阅变更

```javascript
// 订阅实体变更
subscribe(entityId) {
  const message = {
    action: 'subscribe',
    query: `LIVE SELECT * FROM entity WHERE id = "${entityId}"`
  };

  this.ws.send(JSON.stringify(message));
  this.subscriptions.set(entityId, callback);
}

// 订阅 DIFF 模式
subscribeDiff(entityId) {
  const message = {
    action: 'subscribe',
    query: `LIVE SELECT DIFF FROM entity WHERE id = "${entityId}"`
  };

  this.ws.send(JSON.stringify(message));
}
```

### 4.3 发送带确认的消息

```javascript
// 发送消息并等待确认
async sendWithAck(data, timeout = 5000) {
  const messageId = `msg-${Date.now()}`;

  const message = {
    ...data,
    _msgId: messageId,
    _requiresAck: true
  };

  return new Promise((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      reject(new Error('Ack timeout'));
    }, timeout);

    this.pendingAcks.set(messageId, (ackData) => {
      clearTimeout(timeoutId);
      resolve(ackData);
    });

    this.ws.send(JSON.stringify(message));
  });
}
```

---

### 4.4 WebSocket 性能测试

#### 4.4.1 性能指标基准

| 指标 | 基准值 | 测试条件 | 说明 |
|------|--------|----------|------|
| **并发连接数** | ≥ 1000 | 单服务器实例 | 同时保持的 WebSocket 连接数 |
| **消息吞吐量** | ≥ 10,000 msg/s | 1000 并发连接 | 每秒处理的消息总数 |
| **消息延迟** | p99 < 100ms | 局域网环境 | 消息从发送到接收的时间 |
| **心跳成功率** | ≥ 99% | 30s 间隔，持续 1 小时 | 心跳响应成功率 |
| **重连时间** | < 5s | 首次重连 | 从断开到重新连接的时间 |
| **内存使用** | < 500MB | 1000 并发连接 | 服务器端内存占用 |

#### 4.4.2 测试环境要求

**硬件要求**:
- CPU: 4 核及以上
- 内存: 8GB 及以上
- 网络: 千兆以太网（测试客户端与服务器同机房）

**软件要求**:
- Node.js: 18+
- 测试工具: Artillery 或 k6
- 监控工具: Prometheus + Grafana（可选）

#### 4.4.3 测试工具配置

#### 方案 A: Artillery 测试

```yaml
# websocket-load-test.yml
config:
  target: "ws://localhost:18008/ws"
  phases:
    - duration: 60
      arrivalRate: 10
      name: "Ramp up"
    - duration: 300
      arrivalRate: 50
      name: "Sustained load"
  ws:
    headers:
      Authorization: "Bearer ${API_KEY}"

scenarios:
  - name: "WebSocket message exchange"
    weight: 100
    engine: ws
    
    beforeScenario:
      - function: "setupConnection"
    
    flow:
      # 发送订阅请求
      - send:
          action: "subscribe"
          query: 'LIVE SELECT * FROM entity'
      
      # 等待消息
      - think: 30
      
      # 发送心跳
      - send:
          type: "ping"
      
      # 等待响应
      - think: 5
      
      # 发送带确认的消息
      - send:
          action: "update"
          data:
            entity_id: "entity:test"
            content: "Test update"
          _requiresAck: true
      
      # 等待确认
      - think: 2

afterScenario:
  - function: "cleanupConnection"
```

运行测试:
```bash
# 安装 Artillery
npm install -g artillery

# 运行测试
artillery run websocket-load-test.yml

# 生成报告
artillery run websocket-load-test.yml --output report.json
artillery report report.json
```

#### 方案 B: k6 测试

```javascript
// websocket-load-test.js
import ws from 'k6/ws';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '1m', target: 100 },   //  ramp up
    { duration: '5m', target: 1000 },  //  sustained load
    { duration: '1m', target: 0 },     //  ramp down
  ],
  thresholds: {
    ws_connecting_duration: ['p(95)<500'],  // 95% 连接时间 < 500ms
    ws_msgs_received: ['count>10000'],      // 接收消息数 > 10000
    ws_msgs_sent: ['count>10000'],          // 发送消息数 > 10000
  },
};

const WS_URL = 'ws://localhost:18008/ws';
const API_KEY = __ENV.API_KEY || 'test-key';

export default function () {
  const url = `${WS_URL}?token=${API_KEY}&tenant_id=default`;
  
  const res = ws.connect(url, null, function (socket) {
    socket.on('open', function () {
      console.log('WebSocket connected');
      
      // 发送订阅请求
      socket.send(JSON.stringify({
        action: 'subscribe',
        query: 'LIVE SELECT * FROM entity'
      }));
      
      // 定时发送心跳
      socket.setInterval(function () {
        socket.send(JSON.stringify({ type: 'ping' }));
      }, 30000);
      
      // 接收消息
      socket.on('message', function (message) {
        check(message, {
          'message received': (m) => m !== null,
        });
      });
      
      // 5分钟后关闭连接
      socket.setTimeout(function () {
        socket.close();
      }, 300000);
    });
    
    socket.on('close', function () {
      console.log('WebSocket disconnected');
    });
    
    socket.on('error', function (e) {
      console.error('WebSocket error:', e.error());
    });
  });
  
  check(res, {
    'connection established': (r) => r && r.status === 101,
  });
  
  sleep(1);
}
```

运行测试:
```bash
# 安装 k6
# macOS: brew install k6
# Windows: choco install k6
# Linux: sudo apt install k6

# 运行测试
k6 run --env API_KEY=your-api-key websocket-load-test.js

# 生成 HTML 报告
k6 run --out html=report.html websocket-load-test.js
```

#### 4.4.4 自定义测试工具

```javascript
// tests/performance/ws-benchmark.js
import WebSocket from 'ws';
import { performance } from 'perf_hooks';

class WebSocketBenchmark {
  constructor(options = {}) {
    this.targetUrl = options.url || 'ws://localhost:18008/ws';
    this.apiKey = options.apiKey;
    this.concurrentConnections = options.connections || 1000;
    this.duration = options.duration || 300; // 5 minutes
    this.results = {
      connections: { success: 0, failed: 0 },
      messages: { sent: 0, received: 0, latency: [] },
      heartbeats: { sent: 0, received: 0 },
      errors: [],
    };
  }

  async run() {
    console.log(`Starting WebSocket benchmark...`);
    console.log(`Target: ${this.targetUrl}`);
    console.log(`Connections: ${this.concurrentConnections}`);
    console.log(`Duration: ${this.duration}s`);

    const startTime = performance.now();
    
    // 创建连接池
    const connections = await this.createConnections();
    
    // 运行测试
    await this.runTest(connections);
    
    // 收集结果
    const endTime = performance.now();
    this.results.duration = (endTime - startTime) / 1000;
    
    // 生成报告
    this.generateReport();
    
    // 清理
    await this.cleanup(connections);
  }

  async createConnections() {
    const connections = [];
    const batchSize = 100;
    
    for (let i = 0; i < this.concurrentConnections; i += batchSize) {
      const batch = [];
      for (let j = 0; j < batchSize && i + j < this.concurrentConnections; j++) {
        batch.push(this.createConnection(i + j));
      }
      
      const results = await Promise.allSettled(batch);
      results.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          connections.push(result.value);
          this.results.connections.success++;
        } else {
          this.results.connections.failed++;
          this.results.errors.push(result.reason);
        }
      });
      
      // 批量创建间隔，避免瞬间冲击
      await this.sleep(100);
    }
    
    return connections;
  }

  createConnection(id) {
    return new Promise((resolve, reject) => {
      const url = `${this.targetUrl}?token=${this.apiKey}&tenant_id=default`;
      const ws = new WebSocket(url);
      
      const timeout = setTimeout(() => {
        reject(new Error(`Connection ${id} timeout`));
      }, 5000);
      
      ws.on('open', () => {
        clearTimeout(timeout);
        resolve(ws);
      });
      
      ws.on('error', (err) => {
        clearTimeout(timeout);
        reject(err);
      });
    });
  }

  async runTest(connections) {
    const testDuration = this.duration * 1000;
    const startTime = performance.now();
    
    // 为每个连接设置消息处理
    connections.forEach((ws, index) => {
      ws.on('message', (data) => {
        this.results.messages.received++;
        
        // 计算延迟（假设消息包含时间戳）
        try {
          const msg = JSON.parse(data);
          if (msg._timestamp) {
            const latency = performance.now() - msg._timestamp;
            this.results.messages.latency.push(latency);
          }
        } catch (e) {
          // 忽略非 JSON 消息
        }
      });
      
      // 定时发送消息
      const interval = setInterval(() => {
        if (performance.now() - startTime > testDuration) {
          clearInterval(interval);
          return;
        }
        
        ws.send(JSON.stringify({
          type: 'ping',
          _timestamp: performance.now(),
        }));
        
        this.results.messages.sent++;
        this.results.heartbeats.sent++;
      }, 30000 + Math.random() * 10000); // 30-40s 随机间隔
    });
    
    // 等待测试完成
    await this.sleep(testDuration);
  }

  generateReport() {
    const { connections, messages, heartbeats, duration } = this.results;
    
    // 计算延迟统计
    const latencies = messages.latency.sort((a, b) => a - b);
    const p50 = latencies[Math.floor(latencies.length * 0.5)] || 0;
    const p95 = latencies[Math.floor(latencies.length * 0.95)] || 0;
    const p99 = latencies[Math.floor(latencies.length * 0.99)] || 0;
    
    const throughput = messages.received / duration;
    
    console.log('\n========== WebSocket Benchmark Report ==========');
    console.log(`Duration: ${duration.toFixed(2)}s`);
    console.log(`\nConnections:`);
    console.log(`  Success: ${connections.success}`);
    console.log(`  Failed: ${connections.failed}`);
    console.log(`  Success Rate: ${((connections.success / this.concurrentConnections) * 100).toFixed(2)}%`);
    console.log(`\nMessages:`);
    console.log(`  Sent: ${messages.sent}`);
    console.log(`  Received: ${messages.received}`);
    console.log(`  Throughput: ${throughput.toFixed(2)} msg/s`);
    console.log(`\nLatency (ms):`);
    console.log(`  p50: ${p50.toFixed(2)}`);
    console.log(`  p95: ${p95.toFixed(2)}`);
    console.log(`  p99: ${p99.toFixed(2)}`);
    console.log(`\nHeartbeats:`);
    console.log(`  Sent: ${heartbeats.sent}`);
    console.log(`  Received: ${heartbeats.received}`);
    console.log(`  Success Rate: ${((heartbeats.received / heartbeats.sent) * 100).toFixed(2)}%`);
    console.log('================================================\n');
    
    // 基准检查
    const passed = 
      connections.success >= this.concurrentConnections * 0.95 &&
      throughput >= 10000 &&
      p99 < 100;
    
    console.log(`Benchmark ${passed ? 'PASSED ✅' : 'FAILED ❌'}`);
    
    return passed;
  }

  async cleanup(connections) {
    connections.forEach(ws => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    });
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// 运行测试
const benchmark = new WebSocketBenchmark({
  url: 'ws://localhost:18008/ws',
  apiKey: process.env.API_KEY,
  connections: 1000,
  duration: 300,
});

benchmark.run().catch(console.error);
```

运行测试:
```bash
# 设置环境变量
export API_KEY=your-api-key

# 运行测试
node tests/performance/ws-benchmark.js
```

#### 4.4.5 结果分析

**通过标准**:
- ✅ 并发连接数 ≥ 1000（成功率 ≥ 95%）
- ✅ 消息吞吐量 ≥ 10,000 msg/s
- ✅ p99 延迟 < 100ms
- ✅ 心跳成功率 ≥ 99%
- ✅ 内存使用 < 500MB

**性能优化建议**:
- 如果连接数不达标：检查服务器 ulimit 和文件描述符限制
- 如果延迟过高：检查网络延迟和服务器处理能力
- 如果内存过高：检查消息缓存和连接泄漏

#### 4.4.6 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 连接失败率高 | 服务器资源不足 | 增加 CPU/内存，优化代码 |
| 消息延迟高 | 网络拥塞 | 使用内网测试，优化消息处理 |
| 心跳丢失 | 服务器过载 | 增加心跳间隔，优化性能 |
| 内存泄漏 | 连接未正确关闭 | 检查连接生命周期管理 |

---

## 5. Entity API

> **版本**: v3.2 + v3.3 增补

### 5.1 创建 Entity

```javascript
// POST /api/v1/entities
const entity = await apiClient.post('/entities', {
  type: 'memory',
  tenant_id: 'default',
  abstract: 'Vue3 Composition API 指南',
  overview: {
    language: 'typescript',
    sections: 5
  },
  tags: ['vue', 'javascript'],
  // v3.3: atoms 字段 — 支持内联创建 Atom 对象
  atoms: [
    // 格式 1: 内联创建 Atom（AtomInlineCreate 对象）
    {
      type: 'chapter',
      name: '第1章：setup() 函数',
      content: 'setup 是 Composition API 的入口...',
      heading_level: 1,
      order: 'a0',
      parent_id: null,
      tags: ['vue', 'composition-api']
    },
    {
      type: 'section',
      name: '1.1 基本用法',
      content: '在组件中定义 setup 函数...',
      heading_level: 2,
      order: 'a0',
      parent_id: '01CHAP001'
    },
    // 格式 2: 引用已有 Atom ID（字符串）
    'atom:existing-atom-id'
  ]
});

// 响应
{
  "id": "entity:01HQ...",
  "type": "memory",
  "tenant_id": "default",
  "abstract": "Vue3 Composition API 指南",
  "overview": {...},
  "atoms": ["atom:01ABC...", "atom:01DEF...", "atom:existing-atom-id"],
  "tags": ["vue", "javascript"],
  "created_by": "system",
  "created_at": "2026-04-29T10:30:00Z"
}
```

### 5.2 获取 Entity

```javascript
// GET /api/v1/entities/{id}?level=0|1|2
const entity = await apiClient.get('/entities/entity:01HQ...?level=2');

// 响应（level=2 返回完整数据）
{
  "id": "entity:01HQ...",
  "type": "memory",
  "abstract": "...",
  "overview": {...},
  "atoms": ["atom:01ABC...", "atom:01DEF..."],
  "tags": [...],
  "created_at": "..."
}
```

### 5.3 列出 Entities

```javascript
// GET /api/v1/entities?type=memory&page=1&page_size=50
const result = await apiClient.get('/entities?type=memory&page=1&page_size=50');

// 响应
{
  "data": [...],
  "total": 42,
  "page": 1,
  "page_size": 50,
  "has_more": false
}
```

### 5.4 更新 Entity

```javascript
// PUT /api/v1/entities/{id}
await apiClient.put('/entities/entity:01HQ...', {
  abstract: 'Updated abstract',
  tags: ['vue', 'javascript', 'composition-api'],
  // v3.3: 也支持内联创建 Atom
  atoms: ['atom:existing-id', { type: 'note', content: '新笔记', name: '补充' }]
});
```

### 5.5 删除 Entity

```javascript
// DELETE /api/v1/entities/{id}
await apiClient.delete('/entities/entity:01HQ...');
```

### 5.6 跨 Entity Atom 链接 *(v3.3)*

```javascript
// GET /api/v1/entities/{entity_id}/atoms/{atom_id}
// 验证 atom 的 entity_id 与路径匹配，用于跨 Entity 的 Atom 引用解析
const atom = await apiClient.get('/entities/entity:01HQ.../atoms/atom:01ABC...');

// 响应（完整 Atom 数据）
{
  "id": "atom:01ABC...",
  "type": "section",
  "name": "1.1 基本用法",
  "content": "在组件中定义 setup 函数...",
  "entity_id": "entity:01HQ...",
  "heading_level": 2,
  "parent_id": "01CHAP001",
  "order": "a0",
  "tags": ["vue"],
  "tenant_id": "default"
}
```

**错误响应**:

- `404`: Atom 不存在或不属于该 Entity

---

## 6. Atom API *(v3.3)*

> **版本**: v3.3.0 新增

### 6.1 创建 Atom

```javascript
// POST /api/v1/atoms
const atom = await apiClient.post('/atoms', {
  type: 'function',
  content: 'function analyzeCode(filePath: string) { ... }',
  tenant_id: 'default',
  name: 'analyzeCode',
  signature: 'analyzeCode(filePath: string)',
  params: [{ name: 'filePath', type: 'string' }],
  return_type: 'Promise<AnalysisResult>',
  is_exported: true,
  is_async: true,
  complexity: 5,
  start_line: 85,
  end_line: 120,
  // v3.3 层级化字段
  tags: ['analyzer', 'core'],
  heading_level: null,
  parent_id: null,
  order: null,
  aliases: ['parseCode'],
  entity_id: 'entity:01HQ...'
});

// 响应
{
  "id": "atom:01ABC...",
  "type": "function",
  "content": "function analyzeCode(filePath: string) { ... }",
  "name": "analyzeCode",
  "signature": "analyzeCode(filePath: string)",
  "version": 1,
  "tenant_id": "default",
  "created_at": "2026-04-29T10:30:00Z"
}
```

**有效 Atom 类型**:

| 类型 | 说明 | 版本 |
|------|------|------|
| `function` | 函数定义 | v3.2 |
| `class` | 类定义 | v3.2 |
| `interface` | 接口定义 | v3.2 |
| `import` | 导入语句 | v3.2 |
| `goal` | 目标 | v3.2 |
| `scope` | 范围 | v3.2 |
| `task` | 任务 | v3.2 |
| `note` | 笔记 | v3.2 |
| `chapter` | 章节 | v3.3 |
| `section` | 小节 | v3.3 |

### 6.2 列出 Atoms

```javascript
// GET /api/v1/atoms?type=function&max_level=2&page=1&page_size=50
const result = await apiClient.get('/atoms?type=function&max_level=2&page=1&page_size=50');

// 响应
{
  "data": [...],
  "total": 15,
  "page": 1,
  "page_size": 50,
  "has_more": false
}
```

**查询参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `query` | string | 按名称过滤（子串匹配） |
| `type` | string | Atom 类型过滤 |
| `project` | string | 项目过滤 |
| `max_level` | int (1-6) | 最大标题层级过滤，仅返回 `heading_level <= max_level` 的 Atom *(v3.3)* |
| `tenant_id` | string | 租户 ID（默认 `default`） |
| `page` | int | 页码（默认 1） |
| `page_size` | int | 每页大小（默认 50，最大 100） |

### 6.3 获取 Atom

```javascript
// GET /api/v1/atoms/{atom_id}
const atom = await apiClient.get('/atoms/atom:01ABC...');

// 响应
{
  "id": "atom:01ABC...",
  "type": "function",
  "content": "...",
  "name": "analyzeCode",
  "version": 1,
  "created_at": "...",
  "updated_at": "..."
}
```

### 6.4 上下文预算管理 *(v3.3)*

```javascript
// POST /api/v1/atoms/budget
const budget = await apiClient.post('/atoms/budget', {
  entity_id: 'entity:01HQ...',
  query: 'setup function reactive',   // relevance 策略需要
  max_tokens: 4000,
  strategy: 'relevance',              // relevance | hierarchy
  max_level: 3,                       // 可选，层级过滤
  tenant_id: 'default'
});

// 响应
{
  "atoms": [
    {
      "id": "atom:01CHAP...",
      "type": "chapter",
      "name": "第1章：setup() 函数",
      "content": "...",
      "heading_level": 1,
      "order": "a0"
    },
    {
      "id": "atom:01SEC...",
      "type": "section",
      "name": "1.1 基本用法",
      "content": "...",
      "heading_level": 2,
      "parent_id": "01CHAP001",
      "order": "a0"
    }
  ],
  "total_atoms": 25,
  "selected_count": 8,
  "used_tokens": 3200,
  "max_tokens": 4000,
  "strategy": "relevance",
  "budget_exhausted": false
}
```

**预算策略说明**:

| 策略 | 说明 | 排序依据 |
|------|------|----------|
| `relevance` | 基于 BM25 评分 + 层级加权 | 相关度降序，自动补充祖先链 |
| `hierarchy` | 按树结构层级排序 | heading_level → order → name |

**关键行为**:

- `relevance` 策略自动解析祖先链（parent → grandparent），确保上下文完整
- 至少返回 1 个 atom（即使超出预算）
- `budget_exhausted=true` 表示还有未选中的 atom

### 6.5 批量创建 Atoms

```javascript
// POST /api/v1/atoms/batch
const result = await apiClient.post('/atoms/batch', {
  atoms: [
    { type: 'function', content: '...', name: 'fn1' },
    { type: 'class', content: '...', name: 'Cls1' }
  ],
  tenant_id: 'default'
});

// 响应
{
  "success": [...],
  "failed": [{"index": 1, "error": "..."}],
  "total": 2,
  "success_count": 1,
  "failed_count": 1
}
```

> 批量上限: 100 条/请求

---

## 7. 统一搜索 API *(v3.3)*

> **版本**: v3.3.0 新增

### 7.1 跨 Entity + Atom 搜索

```javascript
// POST /api/v1/search
const result = await apiClient.post('/search', {
  query: 'setup function reactive',
  mode: 'hybrid',       // vector | keyword | hybrid
  scope: 'all',         // all | memory | code | backlog | atom | entity
  types: null,          // 过滤 Entity 类型（可选）
  atom_types: null,     // 过滤 Atom 类型（可选）
  max_level: null,      // 最大标题层级过滤（可选）
  limit: 20,
  level: 1,             // 返回层级: 0=abstract, 1=abstract+overview, 2=full
  tenant_id: 'default'
});

// 响应
{
  "results": [
    // Entity 结果
    {
      "type": "entity",
      "id": "entity:01HQ...",
      "entity_type": "memory",
      "abstract": "Vue3 Composition API 指南",
      "score": 0.95
    },
    // Atom 结果
    {
      "type": "atom",
      "local_id": "01SEC001",
      "atom_id": "atom:01ABC...",
      "atom_type": "section",
      "name": "1.1 基本用法",
      "content": "在组件中定义 setup 函数...",
      "heading_level": 2,
      "parent_id": "01CHAP001",
      "order": "a0",
      "tags": ["vue"],
      "entity_id": "entity:01HQ...",
      "score": 0.5
    }
  ],
  "total": 2,
  "mode": "hybrid",
  "query": "setup function reactive"
}
```

### 7.2 搜索参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | string (必填) | - | 搜索查询 |
| `mode` | string | `hybrid` | 搜索模式: `vector` / `keyword` / `hybrid` |
| `scope` | string | `all` | 搜索范围: `all` / `memory` / `code` / `backlog` / `atom` / `entity` |
| `types` | list\<string\> | null | 过滤 Entity 类型（如 `["memory", "wiki"]`） |
| `atom_types` | list\<string\> | null | 过滤 Atom 类型（如 `["function", "section"]`） |
| `max_level` | int (1-6) | null | 最大标题层级过滤 |
| `limit` | int (1-100) | 20 | 返回数量限制 |
| `level` | int (0-2) | 1 | 返回层级 |
| `tenant_id` | string | `default` | 租户 ID |

### 7.3 scope 路由逻辑

| scope | 搜索 Entity | 搜索 Atom |
|-------|-------------|-----------|
| `all` | ✅ | ✅ |
| `memory` | ✅ | ✅ |
| `code` | ✅ | ✅ |
| `backlog` | ✅ | ✅ |
| `entity` | ✅ | ❌ |
| `atom` | ❌ | ✅ |

> `types` 参数可进一步缩小范围：`types=["atom"]` 跳过 Entity 搜索，`types=["memory"]` 跳过 Atom 搜索。

---

## 8. 数据模型

### 8.1 AtomInlineCreate *(v3.3)*

用于 Entity 创建/更新时内联创建 Atom 的模型：

```javascript
// 嵌入 POST /api/v1/entities 的 atoms 数组中
{
  type: 'section',              // 必填: Atom 类型
  content: '详细内容...',       // 必填: Atom 内容
  name: '1.1 基本用法',         // 可选: Atom 名称
  local_id: '01SEC001',         // 可选: 客户端侧 ID（树结构）
  heading_level: 2,             // 可选: 标题层级 1-6
  parent_id: '01CHAP001',       // 可选: 父 Atom 的 local_id
  order: 'a0',                  // 可选: 排序键
  aliases: ['基础用法'],        // 可选: 别名
  tags: ['vue', 'setup'],       // 可选: 标签
  signature: null,              // 可选: 函数签名
  params: null,                 // 可选: 参数列表
  return_type: null,            // 可选: 返回类型
  is_exported: null,            // 可选: 是否导出
  is_async: null,               // 可选: 是否异步
  complexity: null,             // 可选: 复杂度
  start_line: null,             // 可选: 起始行号
  end_line: null,               // 可选: 结束行号
  docstring: null,              // 可选: 文档字符串
  metadata: null,               // 可选: 元数据
  project: null,                // 可选: 项目 ID
  fingerprint: null             // 可选: 内容指纹
}
```

### 8.2 EntityCreateRequest (v3.3 更新)

```javascript
{
  type: 'memory',               // 必填: memory | backlog | wiki | code
  tenant_id: 'default',
  abstract: '摘要',             // 必填: ≤100字符
  overview: {},                 // 概览: 结构化数据
  atoms: [                      // v3.3: 双格式
    'atom:existing-id',         //   格式1: 已有 Atom ID（字符串）
    { type: 'section', ... }    //   格式2: 内联创建（AtomInlineCreate）
  ],
  // 以下为类型特定字段
  title: null,                  // wiki 类型
  aliases: [],                  // wiki 类型
  priority: null,               // backlog 类型: P0-P3
  status: null,                 // backlog 类型: backlog|todo|in_progress|done
  scene: null,                  // backlog 类型
  estimated_hours: null,        // backlog 类型
  actual_hours: null,           // backlog 类型
  file_path: null,              // code 类型
  language: null,               // code 类型
  quality_score: null,          // code 类型
  complexity_metrics: null,     // code 类型
  tags: [],
  project: null,
  created_by: 'system'
}
```

---

## 9. 错误处理

### 9.1 错误码

| 状态码 | 含义       | 处理建议     |
| ------ | ---------- | ------------ |
| 200    | 成功       | -            |
| 400    | 请求错误   | 检查参数     |
| 401    | 未授权     | 检查 API Key |
| 404    | 未找到     | 检查 ID      |
| 500    | 服务器错误 | 重试或报告   |

### 9.2 错误处理示例

```javascript
// lib/error-handler.js
export class ApiError extends Error {
  constructor(status, message, details = null) {
    super(message);
    this.status = status;
    this.details = details;
  }
}

export async function handleApiError(error) {
  if (error instanceof ApiError) {
    switch (error.status) {
      case 401:
        console.error("Authentication failed. Check API key.");
        break;
      case 404:
        console.error("Resource not found.");
        break;
      case 500:
        console.error("Server error. Retrying...");
        // 重试逻辑
        break;
      default:
        console.error(`API error ${error.status}:`, error.message);
    }
  } else {
    console.error("Network error:", error.message);
  }
}
```

---

## 参考文档

- [BACKEND-v3.2-IMPLEMENTATION.md](./BACKEND-v3.2-IMPLEMENTATION.md)
- [PLUGIN-v3.2-IMPLEMENTATION.md](./PLUGIN-v3.2-IMPLEMENTATION.md)
- [API-CONTRACT.md](../API-CONTRACT.md)

---

_文档版本: v3.2.0 + v3.3.0 增补_  
_最后更新: 2026-04-29_
