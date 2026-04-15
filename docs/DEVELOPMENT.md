# OpenCode Memory Service - 开发文档

> **版本**: v3.2.0  
> **面向**: 开发人员、架构师、贡献者  
> **目标**: 说明架构设计、开发规范、实现细节

---

## 架构概览

### 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenCode Memory Stack                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Wrapper    │  │  SurrealDB   │  │  Meilisearch │      │
│  │   (18008)    │  │   (18002)    │  │   (7700)     │      │
│  │              │  │              │  │              │      │
│  │ • FastAPI    │  │ • 图数据库   │  │ • 全文搜索   │      │
│  │ • WebSocket  │  │ • 向量存储   │  │ • 索引管理   │      │
│  │ • Precompute │  │ • ChangeFeed │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │  Embedding   │  │     LLM      │                        │
│  │   (18000)    │  │   (18001)    │                        │
│  │              │  │              │                        │
│  │ • Qwen3      │  │ • Qwen3      │                        │
│  │ • 向量化     │  │ • 文本生成   │                        │
│  └──────────────┘  └──────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 技术栈

| 层级 | 技术 | 版本 |
|------|------|------|
| **Web 框架** | FastAPI | 0.115.x |
| **数据库** | SurrealDB | 3.0+ |
| **搜索引擎** | Meilisearch | 1.4+ |
| **WebSocket** | websockets | 12.x |
| **AST 解析** | tree-sitter | 0.25.x |
| **模型推理** | ModelScope | 1.17.x |

---

## 项目结构

```
embedding_service/
├── src/                          # Embedding & LLM 服务
│   └── qwen3_embedding_service/
│       ├── embedding_service.py  # 端口 18000
│       └── llm_service.py        # 端口 18001
├── wrapper/                      # 主服务（端口 18008）
│   └── src/
│       ├── main.py               # FastAPI 入口
│       ├── config.py             # 配置管理
│       ├── models.py             # Pydantic 模型
│       ├── routers/              # API 路由
│       │   ├── health.py
│       │   ├── embeddings.py
│       │   ├── memories.py
│       │   ├── search.py
│       │   ├── relations.py
│       │   ├── websocket.py      # WebSocket 端点
│       │   └── stubs.py          # Stub 端点（11个）
│       ├── services/             # 业务逻辑
│       │   ├── precompute.py     # ⚠️ 部分实现
│       │   ├── performance_monitor.py
│       │   ├── concurrency_control.py
│       │   ├── relation_builder.py
│       │   ├── cycle_detector.py
│       │   └── weight_calculator.py
│       ├── websocket/            # WebSocket 实现
│       │   ├── reliable_server.py
│       │   ├── heartbeat.py
│       │   ├── ack_manager.py
│       │   ├── diff_manager.py
│       │   └── ...
│       └── utils/                # 工具类
│           ├── meili_client.py
│           └── memory_manager/
├── tests/                        # 测试套件
├── docs/                         # 文档
│   ├── PRODUCT.md                # 产品文档
│   ├── DEVELOPMENT.md            # 本文档
│   ├── BACKLOG.md                # 任务追踪
│   └── v3.2/                     # v3.2 设计文档
└── k8s/                          # Kubernetes 配置
```

---

## 开发规范

### 代码规范

- **Python**: 使用 `ruff` 进行代码格式化和检查
- **类型注解**: 所有函数必须添加类型注解
- **文档字符串**: 使用 Google 风格文档字符串

```bash
# 代码检查
uv run ruff check .
uv run ruff format .

# 类型检查
uv run pyright
```

### 提交规范

```
<type>(<scope>): <subject>

<body>

<footer>
```

**类型**:
- `feat`: 新功能
- `fix`: 修复
- `docs`: 文档
- `test`: 测试
- `refactor`: 重构
- `perf`: 性能优化

**示例**:
```
feat(websocket): add heartbeat manager

- Implement 30s interval heartbeat
- Add 2 missed pongs detection
- Add exponential backoff reconnection

Closes BL-B-1
```

---

## 实现状态

### 已完成 ✅

| 模块 | 文件 | 状态 |
|------|------|------|
| WebSocket 核心 | `websocket/*.py` | 100% |
| Meilisearch SDK | `utils/meili_client.py` | 100% |
| 关系构建器 | `services/relation_builder.py` | 100% |
| 循环检测器 | `services/cycle_detector.py` | 100% |
| 权重计算器 | `services/weight_calculator.py` | 100% |
| 性能监控 | `services/performance_monitor.py` | 100% |
| 并发控制 | `services/concurrency_control.py` | 100% |

### 开发中 🚧

| 模块 | 文件 | 状态 | 备注 |
|------|------|------|------|
| 预计算服务 | `services/precompute.py` | 60% | 3 个 TODO 待完成 |

**TODO 列表**:
1. [ ] 初始化资源（tree-sitter, config, DB 连接）
2. [ ] 清理资源
3. [ ] 文件处理逻辑（解析、符号提取）

### Stub 实现 ⏳

| 端点 | 文件 | 优先级 |
|------|------|--------|
| `/hnsw/stats` | `routers/stubs.py` | P1 |
| `/hnsw/optimize` | `routers/stubs.py` | P1 |
| `/hnsw/rebuild` | `routers/stubs.py` | P1 |
| `/cache/*` | `routers/stubs.py` | P2 |
| `/prefetch/*` | `routers/stubs.py` | P3 |
| `/memories/cluster/leiden` | `routers/stubs.py` | P3 |

---

## 测试

### 运行测试

```bash
# 所有测试
uv run pytest tests/ -v

# 特定模块
uv run pytest tests/test_websocket_*.py -v
uv run pytest tests/test_precompute_*.py -v

# 性能测试
uv run pytest tests/performance/ -v
```

### 测试覆盖率

```bash
uv run pytest tests/ --cov=wrapper/src --cov-report=html
```

---

## 部署

### 本地开发

```bash
# 启动依赖服务
docker-compose up -d surrealdb meilisearch

# 初始化数据库
uv run python scripts/init_all.py

# 启动服务
uv run python -m wrapper.src.main
```

### 生产部署

```bash
# Docker Compose
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Kubernetes
kubectl apply -k k8s/
```

---

## 贡献指南

1. **Fork** 仓库
2. **创建分支** `git checkout -b feature/xxx`
3. **提交更改** `git commit -m "feat: xxx"`
4. **推送分支** `git push origin feature/xxx`
5. **创建 PR**

---

## 参考

- [产品文档](./PRODUCT.md)
- [BACKLOG](./BACKLOG.md)
- [v3.2 设计文档](./v3.2/)

---

_文档版本: v3.2.0_  
_最后更新: 2026-04-15_
