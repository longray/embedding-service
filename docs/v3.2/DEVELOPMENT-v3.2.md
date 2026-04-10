# v3.2 开发指南

> **版本**: v3.2.0  
> **日期**: 2026-04-10  
> **状态**: 实施版  
> **目标**: 帮助开发者理解和扩展 v3.2 系统

---

## 目录

1. [开发环境搭建](#1-开发环境搭建)
2. [项目结构](#2-项目结构)
3. [代码规范](#3-代码规范)
4. [核心模块详解](#4-核心模块详解)
5. [测试指南](#5-测试指南)
6. [调试技巧](#6-调试技巧)
7. [贡献指南](#7-贡献指南)

---

## 1. 开发环境搭建

### 1.1 前置要求

| 工具           | 版本  | 说明       |
| -------------- | ----- | ---------- |
| Python         | 3.10+ | 主开发语言 |
| Node.js        | 18+   | 插件开发   |
| Docker         | 24+   | 服务容器化 |
| Docker Compose | 2.20+ | 多服务编排 |
| Git            | 2.40+ | 版本控制   |

### 1.2 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/csuwl/opencode-memory-plugin.git
cd opencode-memory-plugin

# 2. 安装后端依赖
cd embedding_service/wrapper
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"

# 3. 安装插件依赖
cd ../../opencode-memory-plugin
npm install

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 设置 API 密钥

# 5. 启动开发环境
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### 1.3 IDE 配置

#### VS Code 推荐配置

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "./embedding_service/wrapper/venv/bin/python",
  "python.analysis.typeCheckingMode": "basic",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.mypyEnabled": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": "explicit"
  }
}
```

#### PyCharm 配置

1. **Project Interpreter**: 选择 `embedding_service/wrapper/venv`
2. **Code Style**: 导入 `.editorconfig`
3. **Inspections**: 启用 Mypy、Pylint
4. **Run/Debug**: 配置 Uvicorn 启动

---

## 2. 项目结构

### 2.1 整体架构

```
opencode-memory-plugin/
├── docs/                          # 开发文档
│   ├── v3.2/                      # v3.2 架构文档
│   ├── API-CONTRACT.md            # API 契约
│   └── ...
│
├── embedding_service/             # 后端服务
│   └── wrapper/
│       ├── src/
│       │   ├── main.py            # FastAPI 入口
│       │   ├── config.py          # 配置管理
│       │   ├── models.py          # Pydantic 模型
│       │   ├── routers/           # API 路由
│       │   │   ├── websocket.py   # WebSocket 端点
│       │   │   ├── code.py        # 代码分析路由
│       │   │   └── ...
│       │   ├── services/          # 业务服务
│       │   │   ├── precompute.py  # 预计算服务
│       │   │   └── ...
│       │   ├── utils/             # 工具函数
│       │   │   ├── websocket/     # WebSocket 工具
│       │   │   ├── diff/          # DIFF 工具
│       │   │   └── ...
│       │   └── db/                # 数据库
│       │       └── migrations/    # 迁移脚本
│       ├── tests/                 # 测试文件
│       ├── pyproject.toml         # Python 依赖
│       └── Dockerfile             # 容器配置
│
├── opencode-memory-plugin/        # OpenCode 插件
│   ├── lib/                       # 核心库
│   │   ├── memory-core.js         # 记忆核心
│   │   ├── wrapper-client.js      # 后端客户端
│   │   ├── ws-client.js           # WebSocket 客户端
│   │   └── ...
│   ├── tools/                     # 工具实现
│   │   ├── core.js                # memory_write/read
│   │   ├── search.js              # memory_search
│   │   └── ...
│   ├── agents/                    # 自定义代理
│   ├── cli/                       # CLI 工具
│   └── plugin.js                  # 插件入口
│
└── docker-compose.yml             # 服务编排
```

### 2.2 后端模块说明

| 模块         | 文件            | 职责                               |
| ------------ | --------------- | ---------------------------------- |
| **Main**     | `main.py`       | FastAPI 应用入口、中间件、生命周期 |
| **Config**   | `config.py`     | 环境变量、配置验证、默认值         |
| **Models**   | `models.py`     | Pydantic 模型、请求/响应 Schema    |
| **Routers**  | `routers/*.py`  | API 端点定义、请求处理             |
| **Services** | `services/*.py` | 业务逻辑、核心算法                 |
| **Utils**    | `utils/*.py`    | 工具函数、客户端封装               |
| **DB**       | `db/*.py`       | 数据库连接、迁移脚本               |

---

## 3. 代码规范

### 3.1 Python 规范

#### 代码风格

```python
# ✅ 正确：类型注解 + 文档字符串
from typing import Optional, List
from pydantic import BaseModel

class AtomCreateRequest(BaseModel):
    """创建 Atom 的请求模型。

    Attributes:
        type: Atom 类型（function, class, task 等）
        content: 内容文本
        tenant_id: 租户 ID（v3.2 预留字段）
    """
    type: str
    content: str
    tenant_id: str = "default"

    class Config:
        json_schema_extra = {
            "example": {
                "type": "function",
                "content": "async def analyze(): ...",
                "tenant_id": "default"
            }
        }


# ✅ 正确：异步函数 + 异常处理
async def create_atom(
    db: Surreal,
    request: AtomCreateRequest
) -> Atom:
    """创建 Atom 记录。

    Args:
        db: SurrealDB 连接
        request: 创建请求

    Returns:
        创建的 Atom 对象

    Raises:
        ValueError: 当类型无效时
        SurrealDBError: 当数据库操作失败时
    """
    try:
        result = await db.create("atom", request.model_dump())
        return Atom(**result[0])
    except Exception as e:
        logger.error(f"Failed to create atom: {e}")
        raise
```

#### 导入规范

```python
# ✅ 正确：分组导入
# 标准库
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

# 第三方库
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from surrealdb import Surreal

# 本地模块
from src.config import settings
from src.models import Atom, Entity
from src.utils.logger import logger
```

### 3.2 JavaScript 规范

#### 代码风格

```javascript
// ✅ 正确：JSDoc + 类型注解
/**
 * 写入记忆条目
 * @param {Object} options - 选项
 * @param {string} options.content - 内容
 * @param {string} options.type - 类型
 * @param {string[]} [options.tags] - 标签
 * @returns {Promise<string>} 记忆 ID
 */
async function writeMemory({ content, type, tags = [] }) {
  // 验证参数
  if (!content || typeof content !== "string") {
    throw new Error("Content is required");
  }

  // 构建条目
  const entry = buildEntry({ content, type, tags });

  // 写入文件
  await writeEntryToTimeline(entry);

  // 同步到后端
  await syncToBackend(entry);

  return entry.id;
}

// ✅ 正确：异步错误处理
async function fetchWithRetry(url, options, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetch(url, options);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      if (i === maxRetries - 1) {
        throw error;
      }
      await sleep(1000 * (i + 1)); // 指数退避
    }
  }
}
```

### 3.3 命名规范

| 类型     | 规范             | 示例                                 |
| -------- | ---------------- | ------------------------------------ |
| **类名** | PascalCase       | `AtomService`, `WebSocketManager`    |
| **函数** | snake_case       | `create_atom`, `handle_message`      |
| **变量** | snake_case       | `atom_count`, `is_connected`         |
| **常量** | UPPER_SNAKE_CASE | `MAX_RETRY_COUNT`, `DEFAULT_TIMEOUT` |
| **文件** | snake_case       | `precompute.py`, `ws_client.js`      |
| **路由** | kebab-case       | `/api/v1/atoms`, `/ws/connection`    |

---

## 4. 核心模块详解

### 4.1 WebSocket 模块

#### 架构

```
WebSocket 架构
├── ReliableWebSocket          # 可靠连接管理
│   ├── ConnectionManager      # 连接生命周期
│   ├── HeartbeatManager       # 心跳机制
│   └── ReconnectManager       # 自动重连
├── AckSystem                  # ACK 确认机制
│   ├── MessageTracker         # 消息追踪
│   └── RetryQueue             # 重试队列
├── StateRecovery              # 状态恢复
│   ├── SnapshotManager        # 快照管理
│   └── DiffApplier            # 差异应用
└── SubscriptionManager        # 订阅管理
    ├── TopicRegistry          # 主题注册
    └── BroadcastHandler       # 广播处理
```

#### 关键代码示例

```python
# src/utils/websocket/reliable_ws.py
class ReliableWebSocket:
    """可靠 WebSocket 连接管理器。"""

    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.ack_system = AckSystem()
        self.heartbeat = HeartbeatManager(interval=30)
        self.reconnect = ReconnectManager(max_attempts=5)

    async def connect(self, websocket: WebSocket, client_id: str):
        """建立连接。"""
        await websocket.accept()
        self.connections[client_id] = websocket

        # 启动心跳
        asyncio.create_task(self.heartbeat.start(client_id, websocket))

        # 发送初始状态
        await self.send_initial_state(websocket)

    async def send_with_ack(
        self,
        client_id: str,
        message: dict,
        timeout: float = 5.0
    ) -> bool:
        """发送消息并等待 ACK。"""
        msg_id = generate_ulid()
        message["id"] = msg_id
        message["requires_ack"] = True

        # 注册待确认消息
        await self.ack_system.register(msg_id, client_id)

        # 发送消息
        websocket = self.connections.get(client_id)
        if not websocket:
            return False

        await websocket.send_json(message)

        # 等待 ACK
        return await self.ack_system.wait_for_ack(msg_id, timeout)
```

### 4.2 预计算服务

#### 架构

```
PrecomputeService
├── TaskQueue                    # 任务队列
│   ├── PriorityQueue            # 优先级队列
│   └── BatchProcessor           # 批处理器
├── CodeAnalyzer                 # 代码分析
│   ├── TreeSitterParser         # AST 解析
│   ├── ComplexityCalculator     # 复杂度计算
│   └── SymbolExtractor          # 符号提取
├── IncrementalEngine            # 增量引擎
│   ├── FingerprintCalculator    # 指纹计算
│   ├── DiffGenerator            # 差异生成
│   └── ChangeDetector           # 变更检测
└── PerformanceMonitor           # 性能监控
    ├── MetricsCollector         # 指标收集
    └── AlertManager             # 告警管理
```

#### 关键代码示例

```python
# src/services/precompute.py
class PrecomputeService:
    """预计算服务 - 文件保存时自动分析。"""

    def __init__(self):
        self.queue = TaskQueue(max_size=1000)
        self.analyzer = CodeAnalyzer()
        self.incremental = IncrementalEngine()
        self.monitor = PerformanceMonitor()

    async def process_file(
        self,
        file_path: str,
        content: str,
        project_id: str
    ) -> AnalysisResult:
        """处理单个文件。"""
        start_time = time.time()

        try:
            # 1. 计算指纹
            fingerprint = self.incremental.calculate_fingerprint(content)

            # 2. 检查变更
            if not self.incremental.has_changed(file_path, fingerprint):
                return AnalysisResult(skipped=True, reason="unchanged")

            # 3. 解析代码
            ast = await self.analyzer.parse(content, language="typescript")

            # 4. 提取符号
            symbols = self.analyzer.extract_symbols(ast)

            # 5. 计算复杂度
            complexity = self.analyzer.calculate_complexity(ast)

            # 6. 保存结果
            result = AnalysisResult(
                file_path=file_path,
                symbols=symbols,
                complexity=complexity,
                fingerprint=fingerprint,
                project_id=project_id
            )
            await self.save_result(result)

            # 7. 记录指标
            duration = time.time() - start_time
            self.monitor.record(file_path, duration, len(symbols))

            return result

        except Exception as e:
            self.monitor.record_error(file_path, str(e))
            raise

    async def process_batch(
        self,
        files: List[FileTask],
        batch_size: int = 10
    ) -> BatchResult:
        """批量处理文件。"""
        results = []

        # 使用信号量控制并发
        semaphore = asyncio.Semaphore(batch_size)

        async def process_with_limit(task: FileTask):
            async with semaphore:
                return await self.process_file(
                    task.path,
                    task.content,
                    task.project_id
                )

        # 并发处理
        tasks = [process_with_limit(f) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return BatchResult(results=results)
```

### 4.3 插件端 WebSocket 客户端

#### 架构

```
WebSocketClient
├── ConnectionManager            # 连接管理
│   ├── connect()                # 建立连接
│   ├── disconnect()             # 断开连接
│   └── reconnect()              # 自动重连
├── MessageHandler               # 消息处理
│   ├── handleMessage()          # 处理消息
│   ├── handleDiff()             # 处理差异
│   └── handleAck()              # 处理 ACK
├── StateManager                 # 状态管理
│   ├── applyPatch()             # 应用补丁
│   ├── createSnapshot()         # 创建快照
│   └── restoreFromSnapshot()    # 从快照恢复
└── SubscriptionManager          # 订阅管理
    ├── subscribe()              # 订阅主题
    ├── unsubscribe()            # 取消订阅
    └── notify()                 # 通知监听者
```

#### 关键代码示例

```javascript
// opencode-memory-plugin/lib/ws-client.js
class WebSocketClient {
  /**
   * WebSocket 客户端 - v3.2 可靠连接实现
   */
  constructor(options = {}) {
    this.url = options.url || "ws://localhost:18008/ws";
    this.heartbeatInterval = options.heartbeatInterval || 30000;
    this.reconnectMaxAttempts = options.reconnectMaxAttempts || 5;

    this.ws = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.pendingMessages = new Map();
    this.subscriptions = new Map();
  }

  /**
   * 建立连接
   */
  async connect() {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this.startHeartbeat();
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(JSON.parse(event.data));
        };

        this.ws.onclose = () => {
          this.isConnected = false;
          this.stopHeartbeat();
          this.attemptReconnect();
        };

        this.ws.onerror = (error) => {
          reject(error);
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * 发送消息并等待 ACK
   */
  async sendWithAck(message, timeout = 5000) {
    const msgId = generateULID();
    message.id = msgId;
    message.requiresAck = true;

    return new Promise((resolve, reject) => {
      // 设置超时
      const timer = setTimeout(() => {
        this.pendingMessages.delete(msgId);
        reject(new Error("ACK timeout"));
      }, timeout);

      // 注册待确认消息
      this.pendingMessages.set(msgId, {
        resolve: (data) => {
          clearTimeout(timer);
          resolve(data);
        },
        reject,
      });

      // 发送消息
      this.ws.send(JSON.stringify(message));
    });
  }

  /**
   * 处理收到的消息
   */
  handleMessage(message) {
    switch (message.type) {
      case "ack":
        this.handleAck(message);
        break;
      case "diff":
        this.handleDiff(message);
        break;
      case "snapshot":
        this.handleSnapshot(message);
        break;
      default:
        this.notifySubscribers(message);
    }
  }

  /**
   * 处理 ACK
   */
  handleAck(message) {
    const pending = this.pendingMessages.get(message.ackId);
    if (pending) {
      pending.resolve(message.data);
      this.pendingMessages.delete(message.ackId);
    }
  }

  /**
   * 自动重连
   */
  async attemptReconnect() {
    if (this.reconnectAttempts >= this.reconnectMaxAttempts) {
      console.error("Max reconnection attempts reached");
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);

    console.log(
      `Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`,
    );

    setTimeout(() => {
      this.connect().catch(() => {
        // 重连失败，继续尝试
      });
    }, delay);
  }
}
```

---

## 5. 测试指南

### 5.1 测试结构

```
tests/
├── unit/                          # 单元测试
│   ├── test_models.py             # 模型测试
│   ├── test_services.py           # 服务测试
│   └── test_utils.py              # 工具测试
├── integration/                   # 集成测试
│   ├── test_api.py                # API 测试
│   ├── test_websocket.py          # WebSocket 测试
│   └── test_database.py           # 数据库测试
├── e2e/                           # 端到端测试
│   └── test_workflow.py           # 工作流测试
└── conftest.py                    # 测试配置
```

### 5.2 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/unit/test_models.py

# 运行带覆盖率
pytest --cov=src --cov-report=html

# 运行性能测试
pytest tests/ -m "slow"

# 调试模式
pytest tests/ -v --pdb
```

### 5.3 测试示例

```python
# tests/unit/test_precompute.py
import pytest
from unittest.mock import Mock, AsyncMock
from src.services.precompute import PrecomputeService

@pytest.fixture
def precompute_service():
    service = PrecomputeService()
    service.analyzer = Mock()
    service.incremental = Mock()
    return service

@pytest.mark.asyncio
async def test_process_file_unchanged(precompute_service):
    """测试文件未变更时跳过处理。"""
    # 准备
    precompute_service.incremental.has_changed.return_value = False

    # 执行
    result = await precompute_service.process_file(
        file_path="test.ts",
        content="console.log('test')",
        project_id="test-project"
    )

    # 验证
    assert result.skipped is True
    assert result.reason == "unchanged"

@pytest.mark.asyncio
async def test_process_file_success(precompute_service):
    """测试成功处理文件。"""
    # 准备
    precompute_service.incremental.has_changed.return_value = True
    precompute_service.analyzer.parse.return_value = {"type": "Program"}
    precompute_service.analyzer.extract_symbols.return_value = [
        {"name": "test", "type": "function"}
    ]

    # 执行
    result = await precompute_service.process_file(
        file_path="test.ts",
        content="function test() {}",
        project_id="test-project"
    )

    # 验证
    assert result.skipped is False
    assert len(result.symbols) == 1
```

---

## 6. 调试技巧

### 6.1 后端调试

#### VS Code 调试配置

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["src.main:app", "--reload", "--port", "18008"],
      "jinja": true,
      "justMyCode": false,
      "env": {
        "PYTHONPATH": "${workspaceFolder}/embedding_service/wrapper/src"
      }
    }
  ]
}
```

#### 日志调试

```python
# 在关键位置添加日志
import logging

logger = logging.getLogger(__name__)

async def process_file(self, file_path: str, content: str):
    logger.debug(f"Processing file: {file_path}")

    try:
        result = await self.analyze(content)
        logger.info(f"Analysis complete: {len(result.symbols)} symbols found")
        return result
    except Exception as e:
        logger.exception(f"Failed to process {file_path}")
        raise
```

### 6.2 插件调试

#### Node.js 调试

```bash
# 使用 inspect 模式
node --inspect-brk opencode-memory-plugin/cli/index.cjs

# 然后在 Chrome DevTools 中调试
# chrome://inspect
```

#### 日志输出

```javascript
// 启用详细日志
const DEBUG = process.env.DEBUG === "true";

function debugLog(...args) {
  if (DEBUG) {
    console.log("[DEBUG]", ...args);
  }
}

// 使用
debugLog("WebSocket connecting to:", url);
```

### 6.3 数据库调试

```bash
# 连接 SurrealDB 控制台
docker-compose exec surrealdb surreal sql \
  --conn http://localhost:8000 \
  --user root --pass root \
  --ns opencode --db memory

# 常用查询
SELECT * FROM atom LIMIT 10;
SELECT * FROM entity WHERE type = 'code';
SELECT * FROM reference WHERE type = 'calls';

# 查看表结构
INFO FOR TABLE atom;
INFO FOR TABLE entity;
```

---

## 7. 贡献指南

### 7.1 提交规范

```bash
# 分支命名
feature/BL-CA-36-websocket-rewrite
fix/websocket-reconnect-bug
docs/api-documentation

# 提交信息格式
type(scope): subject

# 示例
feat(websocket): implement heartbeat mechanism
fix(precompute): handle empty file content
docs(api): add WebSocket protocol documentation
test(websocket): add reconnection tests
```

### 7.2 代码审查清单

- [ ] 代码符合项目规范
- [ ] 包含适当的测试
- [ ] 文档已更新
- [ ] 所有测试通过
- [ ] 没有引入新的警告
- [ ] 性能影响已评估

### 7.3 发布流程

```bash
# 1. 更新版本号
# pyproject.toml
# package.json

# 2. 更新 CHANGELOG.md

# 3. 创建发布分支
git checkout -b release/v3.2.0

# 4. 运行完整测试
pytest
cd opencode-memory-plugin && npm test

# 5. 构建 Docker 镜像
docker build -t opencode-memory:v3.2.0 .

# 6. 打标签
git tag v3.2.0
git push origin v3.2.0

# 7. 合并到主分支
git checkout main
git merge release/v3.2.0
```

---

## 附录

### A. 常用命令速查

| 命令                         | 说明                 |
| ---------------------------- | -------------------- |
| `docker-compose up -d`       | 启动服务             |
| `docker-compose logs -f api` | 查看 API 日志        |
| `pytest`                     | 运行测试             |
| `pytest --cov`               | 运行测试并生成覆盖率 |
| `black src/`                 | 格式化 Python 代码   |
| `mypy src/`                  | 类型检查             |
| `npm run lint`               | 检查 JS 代码规范     |
| `npm run test`               | 运行 JS 测试         |

### B. 参考文档

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [SurrealDB 文档](https://surrealdb.com/docs)
- [Pydantic 文档](https://docs.pydantic.dev/)
- [OpenCode 插件开发](https://docs.opencode.ai)

### C. 获取帮助

- **GitHub Issues**: https://github.com/csuwl/opencode-memory-plugin/issues
- **Discussions**: https://github.com/csuwl/opencode-memory-plugin/discussions
