# Embedding Service - Agent 指南

## 项目注意事项

### Python 环境管理

**⚠️ 重要：不要删除 Python 虚拟环境**
- PyTorch 体积很大，重新下载浪费流量
- 如果包有问题，使用 `uv` 管理包修复

### 包管理

**使用 uv 管理依赖**：
```bash
# 安装包
uv pip install package_name

# 运行 Python 脚本
uv run python script.py

# 运行测试
uv run pytest tests/ -v
```

## 项目结构

```
embedding_service/
├── src/                                # 模型推理服务
│   └── qwen3_embedding_service/
│       ├── embedding_service.py        # Embedding 服务（端口 18000）
│       ├── llm_service.py              # LLM 服务（端口 18001）
│       ├── start_embedding.py          # Embedding 启动脚本
│       ├── start_llm.py                # LLM 启动脚本
│       └── download_model.py           # 模型下载工具
├── wrapper/                            # 最小化包装服务（端口 17999）
│   └── src/
│       ├── main.py                     # FastAPI 主程序
│       ├── config.py                   # 配置管理（dataclass + 环境变量）
│       └── utils/
│           ├── cache.py                # 线程安全 LRU 缓存
│           ├── http_pool.py            # HTTP 连接池（httpx）
│           ├── memory_manager.py       # 记忆管理器（SurrealDB 操作）
│           └── exceptions.py           # 统一异常类
├── tests/                              # 测试套件
│   ├── conftest.py                     # Pytest 配置和 fixtures
│   ├── test_wrapper_api.py             # 包装服务 API 测试（推荐）
│   ├── test_wrapper_minimal.py         # 最小化包装服务测试
│   ├── test_embedding_service.py       # Embedding 基础测试
│   ├── test_llm_service.py             # LLM 基础测试
│   ├── test_performance.py             # 性能测试
│   ├── test_security.py                # 安全测试
│   └── ...                             # 更多测试文件
├── scripts/                            # 工具脚本
├── docs/                               # 项目文档
│   ├── ROADMAP.md                      # 路线图
│   ├── API_SPECIFICATION.md            # API 接口规范
│   ├── START_GUIDE.md                  # 启动指南
│   └── architecture/                   # 架构设计文档
├── docker-compose.yml                  # Docker 编排
├── Dockerfile.embedding                # Embedding 服务镜像
├── Dockerfile.llm                      # LLM 服务镜像
├── pyproject.toml                      # 项目配置
└── .env                                # 环境变量（不提交）
```

## 服务架构

```
客户端 → 包装服务(17999) → Embedding 服务(18000)
              ↕
          SurrealDB(8000)
```

**包装服务 API 端点**（端口 17999）：
- `GET  /health` — 健康检查（含 Embedding 和 SurrealDB 状态）
- `POST /v1/embeddings` — 文本嵌入（带 LRU 缓存）
- `POST /api/v1/memories` — 批量上传记忆
- `POST /api/v1/memories/search` — 搜索记忆（vector/keyword/hybrid）

## 开发命令

```bash
# 启动包装服务
uv run python -m wrapper.src.main

# 运行推荐测试
uv run pytest tests/test_wrapper_api.py -v

# 运行所有测试
uv run pytest tests/ -v

# 代码检查
uv run ruff check .
uv run pyright
```

## 当前版本状态

**版本**: v1.1.0
**已完成**: P0 + P1 + P2 + P3-1 + P3-2

### 核心特性
- ✅ Embedding 服务（Qwen3-Embedding-0.6B，端口 18000）
- ✅ LLM 服务（MiniCPM4-0.5B，端口 18001）
- ✅ 最小化包装服务（端口 17999，LRU 缓存 + HTTP 连接池 + SurrealDB）
- ✅ 记忆管理系统（向量搜索 + 关键词搜索 + 混合搜索）
- ✅ Docker Compose 一键部署
- ✅ HNSW 向量索引优化

### 已移除/不使用的功能（旧 wrapper-service 遗留）
- ❌ 熔断器（当前最小化包装不使用）
- ❌ prometheus_client 依赖（已移除）
- ❌ structlog（当前使用 print 日志）
- ❌ API 认证（当前最小化包装不使用）
- ❌ `/v1/chat/completions` 端点（当前包装不代理 LLM）
