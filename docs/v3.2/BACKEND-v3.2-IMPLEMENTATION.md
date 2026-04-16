# 后端 v3.2 实施指南

> **版本**: v3.2.0  
> **日期**: 2026-04-10  
> **状态**: ✅ 已完成（2026-04-16）
> **目标**: 完成后端服务从 17999 到 18008 的完全替换

---

## 目录

1. [项目结构](#1-项目结构)
2. [依赖升级](#2-依赖升级)
3. [服务架构](#3-服务架构)
4. [实施步骤](#4-实施步骤)
5. [测试验证](#5-测试验证)

---

## 1. 项目结构

```
embedding_service/
├── wrapper/
│   ├── src/
│   │   ├── main.py                    # FastAPI 入口（端口 18008）
│   │   ├── config.py                  # 配置管理
│   │   ├── models.py                  # Pydantic 模型
│   │   ├── routers/
│   │   │   ├── websocket.py           # WebSocket 端点（重写）
│   │   │   ├── code.py                # 预计算路由（新增）
│   │   │   └── ...
│   │   ├── services/                  # 新增目录
│   │   │   ├── precompute.py          # PrecomputeService
│   │   │   ├── performance_monitor.py
│   │   │   └── concurrency_control.py
│   │   ├── utils/
│   │   │   ├── websocket/             # 新增目录
│   │   │   │   ├── reliable_client.py
│   │   │   │   ├── ack_system.py
│   │   │   │   ├── state_recovery.py
│   │   │   │   └── persistent_queue.py
│   │   │   ├── diff/                  # 新增目录
│   │   │   │   ├── subscription.py
│   │   │   │   └── patch_applier.py
│   │   │   ├── meili_client.py        # 重写（SDK 0.40）
│   │   │   └── code_analyzer.py       # 解耦
│   │   └── db/
│   │       └── migrations/            # 新增目录
│   │           ├── v3.2_schema.sql
│   │           └── migrate_v2_to_v3.2.py
│   ├── pyproject.toml                 # 依赖更新
│   └── docker-compose.yml             # 端口更新
└── ...
```

---

## 2. 依赖升级

### pyproject.toml 更新

```toml
[project]
name = "opencode-memory-service"
version = "3.2.0"
description = "OpenCode Memory Service v3.2"
requires-python = ">=3.10"

dependencies = [
    # 现有依赖（保持不变）
    "fastapi>=0.115.0,<0.116.0",
    "uvicorn[standard]>=0.32.0,<0.33.0",
    "pydantic>=2.9.0,<2.10.0",
    "transformers>=4.48.0",
    "torch==2.4.0+cu121",
    
    # 更新依赖（版本锁定）
    "surrealdb>=1.0.8,<1.1.0",           # 🔒 缩小范围，使用稳定版
    "meilisearch>=0.40.0,<0.41.0",       # ✅ SDK 升级
    "websockets>=12.0,<13.0",            # 🔒 添加上限，防止重大变更
    
    # 新增依赖（版本锁定）🔼 升级到 tree-sitter 0.25.x
    "tree-sitter>=0.25.0,<0.26.0",       # 🔼 AST 解析核心 0.25.x
    "tree-sitter-python>=0.25.0,<0.26.0", # 🔼 Python 语言包 0.25.x
    "tree-sitter-javascript>=0.25.0,<0.26.0", # 🔼 JavaScript 语言包 0.25.x
    "tree-sitter-typescript>=0.23.0,<0.24.0", # 🔼 TypeScript 语言包 0.23.x
    "tree-sitter-go>=0.25.0,<0.26.0",    # 🔼 Go 语言包 0.25.x
    "tree-sitter-rust>=0.24.0,<0.25.0",  # 🔼 Rust 语言包 0.24.x
    "tree-sitter-java>=0.23.0,<0.24.0",  # 🔼 Java 语言包 0.23.x
    
    # 其他依赖
    "fast-json-patch>=1.32",         # DIFF 模式
    "portalocker>=2.7",              # 文件锁
    "psutil>=5.9",                   # 性能监控
    "aiofiles>=23.0",                # 异步文件操作
]
```

---

## 3. 服务架构

### 3.1 核心服务

| 服务            | 文件                              | 说明                |
| --------------- | --------------------------------- | ------------------- |
| **WebSocket**   | `utils/websocket/`                | 可靠 WebSocket 连接 |
| **预计算**      | `services/precompute.py`          | 代码分析服务        |
| **Meilisearch** | `utils/meili_client.py`           | 搜索客户端          |
| **性能监控**    | `services/performance_monitor.py` | 指标收集            |

### 3.2 端口配置

```python
# config.py
class Config:
    PORT = 18008  # 从 17999 迁移
    HOST = "0.0.0.0"

    # 向后兼容
    LEGACY_PORT = 17999  # 可选
```

---

## 4. 实施步骤

### Phase 1: 依赖升级（1 天）

**目标**: 完成 pyproject.toml 更新和依赖安装验证

#### 任务分解

| # | 任务 | 工时 | 交付物 | 依赖 |
| --- | ---- | ---- | ---- | ---- |
| T1.1 | 备份现有 pyproject.toml | 0.5h | backup/pyproject.toml.bak | - |
| T1.2 | 更新 Python 版本要求为 3.10+ | 0.5h | pyproject.toml | T1.1 |
| T1.3 | 锁定 SurrealDB 版本为 1.0.8-1.1.0 | 1h | pyproject.toml | T1.1 |
| T1.4 | 锁定 Meilisearch SDK 为 0.40.0-0.41.0 | 1h | pyproject.toml | T1.1 |
| T1.5 | 锁定 WebSocket 版本为 12.0-13.0 | 0.5h | pyproject.toml | T1.1 |
| T1.6 | 添加 tree-sitter 0.25.x 依赖 | 2h | pyproject.toml | T1.1 |
| T1.7 | 添加 psutil/aiofiles 依赖 | 0.5h | pyproject.toml | T1.1 |
| T1.8 | 运行 pip install -e . | 1h | - | T1.2-T1.7 |
| T1.9 | 验证依赖安装（pip list） | 0.5h | 验证报告 | T1.8 |
| T1.10 | 运行 pytest 验证兼容性 | 1h | 测试报告 | T1.8 |

#### 实施命令

```bash
# 备份
cp pyproject.toml pyproject.toml.bak

# 安装依赖
cd wrapper
pip install -e ".[dev]"

# 验证
pip list | grep -E "surrealdb|meilisearch|websockets|tree-sitter"
```

#### 风险点

- tree-sitter 0.25.x 与现有代码兼容性问题
- SurrealDB 1.0.8 断开连接处理变化

---

### Phase 2: WebSocket 重构（2-3 天）

**目标**: 实现可靠的 WebSocket 连接（重试、心跳、ACK、状态恢复）

#### 任务分解

| # | 任务 | 工时 | 交付物 | 依赖 |
| --- | ---- | ---- | ---- | ---- |
| T2.1 | 创建 utils/websocket/ 目录 | 0.5h | utils/websocket/ | - |
| T2.2 | 实现 ReliableWebSocketClient 类 | 4h | reliable_client.py | T2.1 |
| T2.3 | 实现 ACK 确认系统 | 3h | ack_system.py | T2.2 |
| T2.4 | 实现心跳/保活机制 | 2h | reliable_client.py | T2.2 |
| T2.5 | 实现指数退避重连 | 2h | reliable_client.py | T2.2 |
| T2.6 | 实现状态恢复机制 | 3h | state_recovery.py | T2.3 |
| T2.7 | 实现持久化队列 | 2h | persistent_queue.py | T2.6 |
| T2.8 | 更新 REST API 集成 WebSocket | 2h | routers/websocket.py | T2.7 |
| T2.9 | 单元测试覆盖 | 3h | tests/test_websocket.py | T2.8 |
| T2.10 | 集成测试 | 2h | tests/integration/ws_test.py | T2.9 |

#### 详细设计

详见 [BACKEND-v3.2-WEBSOCKET.md](./BACKEND-v3.2-WEBSOCKET.md)

```python
# utils/websocket/reliable_client.py
class ReliableWebSocketClient:
    """可靠的 WebSocket 客户端"""
    
    def __init__(self, url, options=None):
        self.url = url
        self.options = options or {}
        
        # 连接配置
        self.base_delay = 1000  # 1秒基础延迟
        self.max_delay = 300000  # 5分钟最大延迟
        self.max_retries = 10
        
        # 心跳配置
        self.heartbeat_interval = 30000  # 30秒
        self.max_missed_pongs = 2
        
    async def connect(self):
        """建立连接并启动心跳"""
        ...
    
    async def send_with_ack(self, message, ack_timeout=5000):
        """发送消息并等待 ACK"""
        ...
    
    async def _reconnect_with_backoff(self):
        """指数退避重连"""
        delay = min(self.base_delay * (2 ** self.retry_count), self.max_delay)
        await asyncio.sleep(delay / 1000)
        ...
```

#### 风险点

- WebSocket 连接泄漏（未正确关闭）
- ACK 超时导致重复发送
- 状态恢复丢失消息

---

### Phase 3: PrecomputeService（2-3 天）

**目标**: 实现代码分析预计算服务，减少实时延迟

#### 任务分解

| # | 任务 | 工时 | 交付物 | 依赖 |
| --- | ---- | ---- | ---- | ---- |
| T3.1 | 创建 services/ 目录 | 0.5h | services/ | - |
| T3.2 | 实现 PrecomputeService 基类 | 3h | precompute.py | - |
| T3.3 | 实现代码分析批处理 | 4h | precompute.py | T3.2 |
| T3.4 | 实现性能监控服务 | 2h | performance_monitor.py | - |
| T3.5 | 实现并发控制服务 | 2h | concurrency_control.py | - |
| T3.6 | 集成 tree-sitter 0.25.x | 3h | precompute.py | Phase1 |
| T3.7 | 实现增量预计算触发 | 2h | precompute.py | T3.3 |
| T3.8 | 实现定时预计算任务 | 2h | precompute.py | T3.7 |
| T3.9 | 单元测试覆盖 | 2h | tests/test_precompute.py | T3.8 |
| T3.10 | 性能基准测试 | 1h | benchmark_report.md | T3.9 |

#### 详细设计

详见 [BACKEND-v3.2-PRECOMPUTE.md](./BACKEND-v3.2-PRECOMPUTE.md)

```python
# services/precompute.py
class PrecomputeService:
    """代码分析预计算服务"""
    
    def __init__(self, batch_size=100, interval=300):
        self.batch_size = batch_size
        self.interval = interval  # 秒
        self.queue = asyncio.Queue()
        self.running = False
    
    async def start(self):
        """启动预计算服务"""
        self.running = True
        asyncio.create_task(self._process_loop())
        asyncio.create_task(self._schedule_task())
    
    async def submit(self, file_path, content):
        """提交分析任务"""
        await self.queue.put((file_path, content))
    
    async def _process_loop(self):
        """批处理循环"""
        batch = []
        while self.running:
            # 收集批次
            while len(batch) < self.batch_size:
                try:
                    task = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=1.0
                    )
                    batch.append(task)
                except asyncio.TimeoutError:
                    break
            
            # 处理批次
            if batch:
                await self._analyze_batch(batch)
                batch = []
```

#### 风险点

- 批处理导致分析延迟增加
- 内存占用随队列增长

---

### Phase 4: Meilisearch SDK 升级（1 天）

**目标**: 升级 Meilisearch Python SDK 0.40.x 并更新 API 调用

#### 任务分解

| # | 任务 | 工时 | 交付物 | 依赖 |
| --- | ---- | ---- | ---- | ---- |
| T4.1 | 检查 0.40.x API 变更 | 1h | api_changes.md | Phase1 |
| T4.2 | 更新 meili_client.py 导入 | 1h | meili_client.py | - |
| T4.3 | 替换已废弃方法调用 | 2h | meili_client.py | T4.1 |
| T4.4 | 更新索引创建逻辑 | 1h | meili_client.py | T4.3 |
| T4.5 | 更新文档索引逻辑 | 1h | meili_client.py | T4.3 |
| T4.6 | 更新搜索逻辑 | 1h | meili_client.py | T4.3 |
| T4.7 | 添加批量操作支持 | 2h | meili_client.py | T4.6 |
| T4.8 | 错误处理更新 | 1h | meili_client.py | T4.7 |
| T4.9 | 单元测试覆盖 | 2h | tests/test_meili.py | T4.8 |
| T4.10 | 端到端测试 | 1h | tests/e2e/meili_test.py | T4.9 |

#### 详细设计

详见 [BACKEND-v3.2-MEILISEARCH.md](./BACKEND-v3.2-MEILISEARCH.md)

```python
# utils/meili_client.py
from meilisearch import Client

# 0.40.x 使用新 API
class MeiliClient:
    def __init__(self, url, api_key):
        # 0.40.x 初始化方式
        self.client = Client(url, api_key)
    
    def create_index(self, uid, primary_key="id"):
        # 0.40.x 方式
        self.client.create_index(uid, {"primaryKey": primary_key})
    
    def add_documents(self, index_uid, documents):
        # 0.40.x 方式
        index = self.client.index(index_uid)
        index.add_documents(documents)
    
    def search(self, index_uid, query, **options):
        # 0.40.x 方式
        index = self.client.index(index_uid)
        return index.search(query, **options)
```

#### 风险点

- 0.40.x 与 0.31.x API 不兼容
- 索引设置格式变更

---

### Phase 5: 端口迁移（1 天）

**目标**: 从端口 17999 迁移到 18008，包含配置和文档更新

#### 任务分解

| # | 任务 | 工时 | 交付物 | 依赖 |
| --- | ---- | ---- | ---- | ---- |
| T5.1 | 更新 config.py 端口常量 | 0.5h | config.py | - |
| T5.2 | 更新 Dockerfile EXPOSE | 0.5h | Dockerfile | - |
| T5.3 | 更新 docker-compose 端口映射 | 0.5h | docker-compose.yml | - |
| T5.4 | 更新健康检查端点 | 1h | main.py | T5.1 |
| T5.5 | 更新服务发现配置 | 1h | config.py | T5.1 |
| T5.6 | 添加向后兼容支持（可选 17999） | 1h | config.py | T5.1 |
| T5.7 | 更新 OpenAPI 文档 | 0.5h | main.py | T5.4 |
| T5.8 | 更新所有引用 17999 的文档 | 2h | docs/*.md | - |
| T5.9 | 运行集成测试 | 1h | 测试报告 | T5.3-T5.8 |
| T5.10 | 更新客户端演示 | 1h | examples/ | Phase1 |

#### 详细设计

详见 [BACKEND-v3.2-MIGRATION.md](./BACKEND-v3.2-MIGRATION.md)

```python
# config.py
class Config:
    # v3.2 新端口
    PORT = 18008
    HOST = "0.0.0.0"
    
    # 向后兼容（可选）
    LEGACY_PORT = 17999
    ENABLE_LEGACY_PORT = os.getenv("ENABLE_LEGACY_PORT", "false").lower() == "true"
```

```bash
# 迁移验证
curl http://localhost:18008/health
# 预期: {"status": "healthy", "version": "3.2.0"}
```

#### 风险点

- 客户端缓存旧端口
- 负载均衡器配置未更新

---

### 实施时间线

```
Week 1:
├── Mon: Phase 1 - 依赖升级
├── Tue: Phase 2 - WebSocket 重构 (1/2)
├── Wed: Phase 2 - WebSocket 重构 (2/2)
├── Thu: Phase 3 - PrecomputeService (1/2)
└── Fri: Phase 3 - PrecomputeService (2/2)

Week 2:
├── Mon: Phase 4 - Meilisearch SDK 升级
├── Tue: Phase 5 - 端口迁移
├── Wed: 集成测试
├── Thu: 修复问题
└── Fri: 部署验证
```

### 里程碑检查点

| 检查点 | 标准 | 时间 |
| ---- | ---- | ---- |
| CP1 | Phase 1 完成，依赖测试通过 | Day 1 |
| CP2 | Phase 2 完成，WebSocket 测试通过 | Day 3 |
| CP3 | Phase 3 完成，预计算测试通过 | Day 6 |
| CP4 | Phase 4 完成，Meilisearch 测试通过 | Day 7 |
| CP5 | Phase 5 完成，端口迁移完成 | Day 8 |
| CP6 | 集成测试通过 | Day 9 |

---

## 5. 测试验证

### 5.1 单元测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块
pytest tests/test_websocket.py -v
pytest tests/test_precompute.py -v
```

### 5.2 集成测试

```bash
# 启动服务
uvicorn wrapper.src.main:app --port 18008

# 运行集成测试
pytest tests/integration/ -v
```

### 5.3 端到端测试

```bash
# 完整流程测试
python tests/e2e/test_full_flow.py
```

---

## 6. 详细实施步骤

### 6.1 环境准备

**步骤 1: 克隆代码库**

```bash
git clone https://github.com/your-org/embedding_service.git
cd embedding_service
```

**步骤 2: 安装 uv 包管理器**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**步骤 3: 创建虚拟环境并安装依赖**

```bash
uv venv
uv pip install -e ".[dev]"
```

**步骤 4: 启动依赖服务**

```bash
# 使用 Docker Compose 启动 SurrealDB 和 Meilisearch
docker-compose up -d surrealdb meilisearch

# 等待服务就绪
sleep 15
```

**步骤 5: 初始化数据库**

```bash
uv run python scripts/init_all.py
```

### 6.2 服务启动

**开发模式启动**

```bash
# 启动包装服务（端口 18008）
uv run python -m wrapper.src.main

# 或使用 uvicorn（热重载）
uv run uvicorn wrapper.src.main:app --reload --port 18008
```

**生产模式启动**

```bash
# 设置环境变量
export SURREAL_URL=ws://localhost:18002/rpc
export SURREAL_NS=memory_ns
export SURREAL_DB=memory_db
export SURREAL_USER=runtime_user
export SURREAL_PASS=runtime_pass
export WRAPPER_MEILI_ENABLED=true
export WRAPPER_MEILI_URL=http://localhost:7700

# 启动服务
uv run uvicorn wrapper.src.main:app --host 0.0.0.0 --port 18008 --workers 4
```

### 6.3 验证部署

**健康检查**

```bash
# 检查服务健康
curl http://localhost:18008/health

# 预期响应
{"status": "healthy", "version": "3.2.0"}
```

**功能测试**

```bash
# 测试 Embedding 接口
curl -X POST http://localhost:18008/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "测试文本", "model": "Qwen3-Embedding-0.6B"}'

# 测试记忆搜索
curl -X POST http://localhost:18008/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "测试", "mode": "keyword", "tenant_id": "default"}'
```

---

## 7. 最佳实践

### 7.1 配置管理

**环境变量分层**

```bash
# .env.development（开发环境）
DEBUG=true
LOG_LEVEL=debug
WRAPPER_AUTH_ENABLED=false

# .env.production（生产环境）
DEBUG=false
LOG_LEVEL=warning
WRAPPER_AUTH_ENABLED=true
WRAPPER_API_KEYS="your-key:read;write"
```

**配置加载顺序**

1. 默认值（代码中）
2. 环境变量
3. `.env` 文件
4. 运行时参数

### 7.2 性能优化

**数据库连接池**

```python
# config.py
class DatabaseConfig:
    # 连接池配置
    POOL_SIZE = 10
    MAX_OVERFLOW = 20
    POOL_TIMEOUT = 30
```

**缓存策略**

```python
# 嵌入结果缓存
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text: str) -> List[float]:
    # 缓存最近 1000 个嵌入结果
    return embedding_service.encode(text)
```

**批量处理**

```python
# 使用批量操作减少数据库往返
async def batch_create_memories(memories: List[Dict]):
    batch_size = 100
    for i in range(0, len(memories), batch_size):
        batch = memories[i:i + batch_size]
        await db.query("CREATE memory CONTENT $batch", {"batch": batch})
```

### 7.3 错误处理

**全局异常处理**

```python
# main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )
```

**重试机制**

```python
# 数据库操作重试
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def query_with_retry(db, query: str):
    return await db.query(query)
```

### 7.4 监控告警

**性能指标收集**

```python
# 使用 PerformanceMonitor
from wrapper.src.services.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor(db=db)

with monitor.monitor("operation_name", {"tenant_id": tenant_id}):
    # 执行业务逻辑
    result = await process_data()
```

**日志规范**

```python
import structlog

logger = structlog.get_logger()

# 结构化日志
logger.info(
    "memory_created",
    memory_id=memory_id,
    tenant_id=tenant_id,
    duration_ms=duration,
)
```

---

## 8. FAQ

### Q1: 如何切换回旧端口 17999？

**A**: 修改 `wrapper/src/config.py`：

```python
class WrapperConfig:
    PORT = 17999  # 改为 17999
```

或在启动时指定：

```bash
uv run uvicorn wrapper.src.main:app --port 17999
```

### Q2: WebSocket 连接失败怎么办？

**A**: 检查以下几点：

1. **服务是否启动**
   ```bash
   curl http://localhost:18008/health
   ```

2. **防火墙设置**
   ```bash
   # 检查端口是否开放
   netstat -an | grep 18008
   ```

3. **客户端配置**
   ```javascript
   // 确保使用正确的 URL
   const ws = new WebSocket('ws://localhost:18008/ws/memories/live');
   ```

### Q3: 如何禁用 Meilisearch？

**A**: 设置环境变量：

```bash
export WRAPPER_MEILI_ENABLED=false
```

服务将自动回退到 SurrealDB 的 BM25 搜索。

### Q4: 数据库迁移失败如何处理？

**A**: 

1. **检查数据库连接**
   ```bash
   curl http://localhost:18002/health
   ```

2. **查看迁移日志**
   ```bash
   uv run python scripts/migrate_v2_to_v32.py --dry-run
   ```

3. **手动修复**
   ```bash
   # 重新初始化数据库
   uv run python scripts/init_database.py
   ```

### Q5: 性能测试失败怎么办？

**A**: 

1. **检查资源使用**
   ```bash
   # 监控 CPU 和内存
   htop
   ```

2. **调整测试参数**
   ```bash
   # 使用快速模式
   uv run python tests/performance/benchmark.py --quick
   ```

3. **查看详细日志**
   ```bash
   # 启用调试日志
   export LOG_LEVEL=debug
   uv run python tests/performance/benchmark.py --report
   ```

### Q6: 如何添加新的语言支持？

**A**: 

1. **安装语言包**
   ```bash
   uv pip install tree-sitter-go
   ```

2. **注册语言**
   ```python
   # wrapper/src/services/code_parser.py
   from tree_sitter_go import language_go
   
   self._languages["go"] = language_go()
   ```

3. **添加查询规则**
   ```python
   # 在 QUERIES 中添加 Go 查询
   "go": """
       (function_declaration
         name: (identifier) @function.name)
   """
   ```

### Q7: 如何配置 SSL？

**A**: 参考 [SSL-SETUP.md](../SSL-SETUP.md)：

```bash
# 使用 Docker Compose 启动带 SSL 的服务
docker-compose -f docker-compose.ssl.yml up -d
```

### Q8: 如何排查内存泄漏？

**A**: 

1. **使用内存分析器**
   ```bash
   # 安装 memory_profiler
   uv pip install memory_profiler
   
   # 运行分析
   python -m memory_profiler wrapper/src/main.py
   ```

2. **检查连接池**
   ```python
   # 确保连接正确关闭
   async with db_pool.acquire() as conn:
       await conn.query("...")
   ```

---

## 参考文档

- [BACKEND-v3.2-WEBSOCKET.md](./BACKEND-v3.2-WEBSOCKET.md)
- [BACKEND-v3.2-PRECOMPUTE.md](./BACKEND-v3.2-PRECOMPUTE.md)
- [BACKEND-v3.2-MIGRATION.md](./BACKEND-v3.2-MIGRATION.md)
- [PLUGIN-v3.2-IMPLEMENTATION.md](./PLUGIN-v3.2-IMPLEMENTATION.md)
- [SSL-SETUP.md](../SSL-SETUP.md)

---

_文档版本: v3.2.0_  
_最后更新: 2026-04-10_
