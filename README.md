# Embedding Service

本地化 AI 推理 + 记忆管理服务，基于 Qwen3-Embedding 和 SurrealDB。

## 开发状态

**当前版本**: v1.1.0
**已完成**: P0 + P1 + P2 + P3-1 + P3-2

| 阶段 | 内容 | 状态 |
|------|------|------|
| P0 | Embedding + LLM 核心服务 | ✅ 已完成 |
| P1 | 缓存 + 连接池 + 测试套件 | ✅ 已完成 |
| P2 | Docker + CI/CD + 文档 | ✅ 已完成 |
| P3-1 | Docker Compose 一键部署 | ✅ 已完成 |
| P3-2 | HNSW 向量索引优化 | ✅ 已完成 |
| P3-3 | 监控告警（Alertmanager） | ⏳ 待开始 |
| P3-4 | Kubernetes 部署 | ⏳ 待开始 |
| P3-5 | 审计日志 | ⏳ 待开始 |

详细路线见 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 架构

```
客户端 → 包装服务 (17999) → Embedding 服务 (18000)
              ↕                    ↕
          SurrealDB (8000)    LLM 服务 (18001)
```

| 服务 | 端口 | 模型/技术 | 说明 |
|------|------|----------|------|
| 包装服务 | 17999 | FastAPI | 缓存 + 连接池 + SurrealDB 集成 |
| Embedding | 18000 | Qwen3-Embedding-0.6B | 文本 → 1024维向量 |
| LLM | 18001 | MiniCPM4-0.5B | 对话补全 |
| SurrealDB | 8000 | — | 向量存储 + HNSW 索引 |

## API 端点

包装服务（端口 17999）：

| 端点 | 方法 | 功能 |
|------|------|------|
| `/health` | GET | 健康检查（Embedding + SurrealDB 状态） |
| `/v1/embeddings` | POST | 文本嵌入（带 LRU 缓存） |
| `/api/v1/memories` | POST | 批量上传记忆 |
| `/api/v1/memories/search` | POST | 搜索记忆（vector/keyword/hybrid） |

## 快速开始

### 启动包装服务

```bash
# 启动服务
uv run python -m wrapper.src.main
```

### Docker Compose 一键启动

```bash
docker-compose up -d
```

### 运行测试

```bash
# 推荐：核心 API 测试
uv run pytest tests/test_wrapper_api.py -v

# 全部测试
uv run pytest tests/ -v
```

### 代码检查

```bash
uv run ruff check .
uv run pyright
```

## 配置

通过环境变量或 `.env` 文件配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WRAPPER_PORT` | 17999 | 包装服务端口 |
| `WRAPPER_HOST` | 0.0.0.0 | 监听地址 |
| `WRAPPER_CACHE_ENABLED` | true | 启用 LRU 缓存 |
| `WRAPPER_EMBEDDING_SERVICE_URL` | http://localhost:18000 | Embedding 服务地址 |
| `WRAPPER_SURREALDB_URL` | ws://localhost:8000/rpc | SurrealDB 地址 |
| `WRAPPER_SEARCH_VECTOR_THRESHOLD` | 0.75 | 向量搜索阈值 |
| `WRAPPER_SEARCH_HYBRID_THRESHOLD` | 0.75 | 混合搜索阈值 |
| `WRAPPER_SEARCH_KEYWORD_THRESHOLD` | 0.0 | 关���词搜索阈值 |

## 核心功能

- ✅ **文本嵌入**：Qwen3-Embedding-0.6B 模型，1024 维向量
- ✅ **LLM 对话**：MiniCPM4-0.5B 模型，OpenAI 兼容接口
- ✅ **记忆管理**：SurrealDB 向量存储，支持向量/关键词/混合搜索
- ✅ **HNSW 索引**：向量搜索 O(log n)，延迟 < 50ms
- ✅ **LRU 缓存**：线程安全，TTL 过期
- ✅ **HTTP 连接池**：httpx 连接复用
- ✅ **Docker 部署**：docker-compose 一键启动

## 项目结构

```
embedding_service/
├── src/qwen3_embedding_service/    # 模型推理服务
│   ├── embedding_service.py        # Embedding 服务
│   └── llm_service.py              # LLM 服务
├── wrapper/src/                    # 最小化包装服务
│   ├── main.py                     # FastAPI 主程序
│   ├── config.py                   # 配置管理
│   └── utils/                      # 工具模块
├── tests/                          # 测试套件
├── scripts/                        # 工具脚本
├── docs/                           # 项目文档
├── docker-compose.yml              # Docker 编排
└── pyproject.toml                  # 项目配置
```
