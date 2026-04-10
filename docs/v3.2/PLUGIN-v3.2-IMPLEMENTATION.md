# 插件端 v3.2 实施指南

> **版本**: v3.2.0  
> **日期**: 2026-04-10  
> **状态**: 实施版  
> **目标**: 升级插件端依赖，适配后端 v3.2

---

## 目录

1. [依赖升级](#1-依赖升级)
2. [配置更新](#2-配置更新)
3. [WebSocket 客户端](#3-websocket-客户端)
4. [测试验证](#4-测试验证)

---

## 1. 依赖升级

### 1.1 package.json 更新

```json
{
  "name": "@csuwl/opencode-memory-plugin",
  "version": "3.2.0",
  "dependencies": {
    "@opencode-ai/plugin": "^1.0.0",
    "chokidar": "^5.0.0",
    "oxc-parser": "^0.121.0",

    "ws": "^8.20.0",
    "pino": "^9.5.0",
    "dotenv": "^16.4.5"
  },
  "devDependencies": {
    "@types/node": "^22.0.0",
    "@types/ws": "^8.5.13",
    "pino-pretty": "^13.0.0"
  }
}
```

### 1.2 安装依赖

```bash
npm install
```

---

## 2. 配置更新

### 2.1 环境变量配置

```javascript
// lib/config.js
import dotenv from "dotenv";

dotenv.config();

export const config = {
  // 服务端口（v3.2 更新）
  API_PORT: process.env.API_PORT || 18008,
  API_HOST: process.env.API_HOST || "localhost",

  // WebSocket
  WS_HEARTBEAT_INTERVAL: 30000,
  WS_ACK_TIMEOUT: 5000,

  // 日志
  LOG_LEVEL: process.env.LOG_LEVEL || "info",

  // 向后兼容
  LEGACY_PORT: 17999,
};
```

### 2.2 WrapperClient 更新

```javascript
// lib/wrapper-client.js
import { config } from "./config.js";

export class WrapperClient {
  constructor(options = {}) {
    this.port = options.port || config.API_PORT;
    this.host = options.host || config.API_HOST;
    this.baseUrl = `http://${this.host}:${this.port}/api/v1`;
  }

  // ... 其他方法
}
```

---

## 3. WebSocket 客户端

### 3.1 ReliableWebSocketClient

```javascript
// lib/websocket-client.js
import WebSocket from "ws";
import { config } from "./config.js";

export class ReliableWebSocketClient {
  constructor(url, options = {}) {
    this.url = url;
    this.heartbeatInterval = options.heartbeatInterval || 30000;
    this.maxMissedPongs = 2;
    this.baseDelay = 1000;
    this.maxDelay = 300000;
    this.maxRetries = 10;
    this.retryCount = 0;

    this.ws = null;
    this.connected = false;
    this.missedPongs = 0;
    this.heartbeatTimer = null;
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);

      this.ws.on("open", () => {
        this.connected = true;
        this.retryCount = 0;
        this.startHeartbeat();
        resolve();
      });

      this.ws.on("message", (data) => {
        const msg = JSON.parse(data);

        if (msg.type === "pong") {
          this.missedPongs = 0;
        }

        if (this.onMessage) {
          this.onMessage(msg);
        }
      });

      this.ws.on("close", () => {
        this.connected = false;
        this.stopHeartbeat();
        this.scheduleReconnect();
      });

      this.ws.on("error", (err) => {
        reject(err);
      });
    });
  }

  startHeartbeat() {
    this.heartbeatTimer = setInterval(() => {
      if (this.missedPongs >= this.maxMissedPongs) {
        this.ws.close();
        return;
      }

      this.ws.send(
        JSON.stringify({
          type: "ping",
          timestamp: Date.now(),
        }),
      );

      this.missedPongs++;
    }, this.heartbeatInterval);
  }

  stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  scheduleReconnect() {
    if (this.retryCount >= this.maxRetries) {
      console.error("Max retries exceeded");
      return;
    }

    const delay = Math.min(
      this.baseDelay * Math.pow(2, this.retryCount),
      this.maxDelay,
    );

    this.retryCount++;
    console.log(`Reconnecting in ${delay}ms (attempt ${this.retryCount})`);

    setTimeout(() => this.connect(), delay);
  }

  send(data) {
    if (!this.connected) {
      throw new Error("Not connected");
    }
    this.ws.send(JSON.stringify(data));
  }

  disconnect() {
    this.stopHeartbeat();
    if (this.ws) {
      this.ws.close();
    }
  }
}
```

---

## 4. 错误处理

### 4.1 错误类型分类

| 错误类型 | 代码 | 说明 | 处理策略 |
| -------- | ---- | ---- | -------- |
| **连接错误** | CONN_001 | WebSocket 连接失败 | 自动重试 + 指数退避 |
| **连接错误** | CONN_002 | 服务不可用 | 切换到备份端点 |
| **消息错误** | MSG_001 | 消息发送超时 | ACK 超时 + 重试 |
| **消息错误** | MSG_002 | 消息格式错误 | 验证 + 拒绝 |
| **消息错误** | MSG_003 | 消息处理失败 | 队列保留 + 人工处理 |
| **认证错误** | AUTH_001 | API 密钥无效 | 提示用户检查配置 |
| **认证错误** | AUTH_002 | Token 过期 | 刷新 token |
| **重连错误** | RECN_001 | 达到最大重试次数 | 降级模式 |
| **重连错误** | RECN_002 | 状态恢复失败 | 重新初始化 |

### 4.2 连接失败处理

```javascript
// lib/websocket-client.js
export class ReliableWebSocketClient {
  constructor(url, options = {}) {
    // ... 已有属性
    
    // 错误处理配置
    this.errorHandlers = {
      onConnectionFailed: options.onConnectionFailed || this._defaultConnFailed,
      onMaxRetriesReached: options.onMaxRetriesReached || this._defaultMaxRetries,
      onAuthError: options.onAuthError || this._defaultAuthError,
    };
  }

  async connect() {
    return new Promise((resolve, reject) => {
      try {
        this._connectInternal(resolve, reject);
      } catch (error) {
        this._handleError("CONN_001", error);
        reject(error);
      }
    });
  }

  _handleError(code, error, context = {}) {
    const errorInfo = {
      code,
      message: error.message,
      timestamp: Date.now(),
      context,
      recoverable: this._isRecoverable(code),
    };
    
    console.error(`[WebSocket Error] ${code}:`, errorInfo);
    
    // 触发回调
    if (code === "CONN_001") {
      this.errorHandlers.onConnectionFailed(errorInfo);
    } else if (code.startsWith("AUTH")) {
      this.errorHandlers.onAuthError(errorInfo);
    }
    
    return errorInfo;
  }

  _isRecoverable(code) {
    const recoverableCodes = ["CONN_001", "MSG_001", "RECN_002"];
    return recoverableCodes.includes(code);
  }

  _defaultConnFailed(errorInfo) {
    console.error("连接失败，触发自动重试...");
    this.scheduleReconnect();
  }

  _defaultMaxRetries(errorInfo) {
    console.error("已达到最大重试次数，切换到降级模式");
    this.connected = false;
    this.degradedMode = true;
  }

  _defaultAuthError(errorInfo) {
    console.error("认证错误，请检查 API 密钥配置");
    process.exit(1);
  }
}
```

### 4.3 消息超时处理

```javascript
// lib/acks.js
export class AckManager {
  constructor(options = {}) {
    this.timeout = options.timeout || 5000;
    this.pendingAcks = new Map();
  }

  async sendWithAck(messageId, sendFn, data) {
    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        this.pendingAcks.delete(messageId);
        reject(new Error(`MSG_001: 消息 ${messageId} 超时`));
      }, this.timeout);

      this.pendingAcks.set(messageId, {
        resolve,
        reject,
        timeoutId,
        timestamp: Date.now(),
      });

      // 发送消息
      sendFn(data).then((response) => {
        // 收到响应，清除超时
        clearTimeout(timeoutId);
        this.pendingAcks.delete(messageId);
        resolve(response);
      }).catch((error) => {
        clearTimeout(timeoutId);
        this.pendingAcks.delete(messageId);
        reject(error);
      });
    });
  }

  handleAck(messageId, response) {
    const pending = this.pendingAcks.get(messageId);
    if (pending) {
      clearTimeout(pending.timeoutId);
      pending.resolve(response);
      this.pendingAcks.delete(messageId);
    }
  }

  cancelPending(messageId) {
    const pending = this.pendingAcks.get(messageId);
    if (pending) {
      clearTimeout(pending.timeoutId);
      pending.reject(new Error("MSG_002: 消息已取消"));
      this.pendingAcks.delete(messageId);
    }
  }
}
```

### 4.4 重连失败处理

```javascript
// lib/reconnection.js
export class ReconnectionManager {
  constructor(options = {}) {
    this.maxRetries = options.maxRetries || 10;
    this.baseDelay = options.baseDelay || 1000;
    this.maxDelay = options.maxDelay || 300000;
    
    this.retryCount = 0;
    this.retryHistory = [];
    this.state = "idle"; // idle, reconnecting, degraded, failed
  }

  async scheduleReconnect(onReconnect) {
    if (this.retryCount >= this.maxRetries) {
      this.state = "failed";
      this._handleMaxRetries();
      return;
    }

    this.state = "reconnecting";
    
    // 计算延迟（指数退避）
    const delay = Math.min(
      this.baseDelay * Math.pow(2, this.retryCount),
      this.maxDelay
    );
    
    this.retryCount++;
    this.retryHistory.push({
      attempt: this.retryCount,
      delay,
      timestamp: Date.now(),
    });

    console.log(`[Reconnect] 尝试 ${this.retryCount}/${this.maxRetries}，${delay}ms 后重试...`);
    
    await new Promise((resolve) => setTimeout(resolve, delay));
    
    try {
      await onReconnect();
      this.retryCount = 0;
      this.state = "idle";
      console.log("[Reconnect] 连接恢复成功");
    } catch (error) {
      // 继续重试
      await this.scheduleReconnect(onReconnect);
    }
  }

  _handleMaxRetries() {
    console.error("[Reconnect] 达到最大重试次数");
    
    // 检查降级模式是否启用
    if (process.env.ENABLE_DEGRADED_MODE === "true") {
      this.state = "degraded";
      console.log("[Reconnect] 切换到降级模式（仅本地存储）");
    } else {
      throw new Error("RECN_001: 达到最大重试次数，无法自动恢复");
    }
  }

  getStatus() {
    return {
      state: this.state,
      retryCount: this.retryCount,
      maxRetries: this.maxRetries,
      retryHistory: this.retryHistory.slice(-10),
    };
  }
}
```

### 4.5 故障排查指南

#### 场景 1: 连接被拒绝

```
症状: Error: connect ECONNREFUSED 127.0.0.1:18008
排查:
1. 检查服务是否启动: docker-compose ps
2. 检查端口占用: netstat -tlnp | grep 18008
3. 检查防火墙: firewall-cmd --list-ports
4. 检查服务日志: docker-compose logs api
```

#### 场景 2: 认证��败

```
症状: Error: AUTH_001 API 密钥无效
排查:
1. 检查环境变量: echo $WRAPPER_MEILI_API_KEY
2. 检查配置文件: cat ~/.opencode/memory/memory-config.json
3. 验证密钥格式: 应为 32 位字符串
4. 检查后端日志: docker-compose logs api | grep auth
```

#### 场景 3: 消息超时

```
症状: Error: MSG_001 消息超时
排查:
1. 检查网络延迟: ping localhost
2. 检查服务负载: docker stats
3. 检查消息队列: curl localhost:18008/api/v1/queue/status
4. 增加超时时间重试
```

#### 场景 4: WebSocket 断连

```
症状: WebSocket connection closed unexpectedly
排查:
1. 检查心跳配置: curl localhost:18008/api/v1/ws/config
2. 检查连接数上限: ulimit -n
3. 检查 Nginx 配置: proxy_read_timeout
4. 查看断开原因: docker-compose logs api | grep "ws close"
```

#### 场景 5: 数据库连接失败

```
症状: Error: 连接 SurrealDB 失败
排查:
1. 检查 SurrealDB: docker-compose ps surrealdb
2. 测试连接: curl http://localhost:8000/health
3. 检查认证: SURREALDB_USER/SURREALDB_PASS
4. 查看日志: docker-compose logs surrealdb
```

#### 场景 6: 搜索服务异常

```
症状: 搜索返回空结果
排查:
1. 检查 Meilisearch: curl http://localhost:7700/health
2. 检查索引: curl http://localhost:7700/indexes
3. 重建索引: curl -X POST localhost:18008/api/v1/reindex
4. 检查权限: MEILISEARCH_API_KEY
```

#### 场景 7: 内存溢出

```
症状: JavaScript heap out of memory
排查:
1. 增加 Node 内存: NODE_OPTIONS="--max-old-space-size=4096"
2. 检查大文件: ls -lh ~/.opencode/memory/
3. 清理缓存: rm -rf ~/.opencode/memory/.cache
4. 分批处理: 使用 limit/offset
```

#### 场景 8: 权限错误

```
症状: Error: EACCES permission denied
排查:
1. 检查文件权限: ls -la ~/.opencode/memory/
2. 修复权限: chown -R $USER ~/.opencode/
3. 检查 .npm 目录: ls -la ~/.npm
4. 清除重新安装: npm cache clean -f
```

---

## 5. 测试验证

### 5.1 依赖检查

```bash
# 检查版本
npm list ws pino dotenv

# 预期输出
# ws@8.20.0
# pino@9.5.0
# dotenv@16.4.5
```

### 5.2 连接测试

```javascript
// test-connection.js
import { WrapperClient } from "./lib/wrapper-client.js";

const client = new WrapperClient();

// 测试 HTTP 连接
const health = await fetch(`${client.baseUrl}/health`);
console.log("Health:", await health.json());

// 测试 WebSocket 连接
import { ReliableWebSocketClient } from "./lib/websocket-client.js";

const ws = new ReliableWebSocketClient(`ws://localhost:18008/ws`);
await ws.connect();
console.log("WebSocket connected");
```

### 5.3 运行测试

```bash
npm test
```

---

## 参考文档

- [UNIFIED-ARCHITECTURE-v3.2.md](./UNIFIED-ARCHITECTURE-v3.2.md)
- [BACKEND-v3.2-IMPLEMENTATION.md](./BACKEND-v3.2-IMPLEMENTATION.md)

---

_文档版本: v3.2.0_  
_最后更新: 2026-04-10_
