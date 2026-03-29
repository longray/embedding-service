# 服务启动指南

## 🚀 统一启动脚本

使用 `start_services.py` 可以一键启动所有服务。

### 基本用法

```bash
# 方式1：只启动 Embedding + 包装层（推荐）
uv run python start_services.py

# 方式2：启动所有服务（Embedding + LLM + 包装层）
uv run python start_services.py --with-llm

# 方式3：只启动后端服务（测试用）
uv run python start_services.py --no-wrapper
uv run python start_services.py --with-llm --no-wrapper
```

### 启动流程

```
1. 启动 Embedding 服务（必需）
   ├── 端口：18000
   └── 等待就绪（约10-30秒）

2. 启动 LLM 服务（可选）
   ├── 端口：18001
   └── 等待就绪（约10-30秒）

3. 启动包装层服务（推荐）
   ├── 端口：3001
   └── 等待就绪（<5秒）
```

### 服务访问地址

启动成功后，可以通过以下地址访问：

| 服务 | 地址 | 说明 |
|------|------|------|
| **包装层** | http://localhost:17999 | 推荐使用（带缓存、熔断器、Meilisearch 全文搜索） |
| Embedding | http://localhost:18000 | 直接访问后端 |
| LLM | http://localhost:18001 | 直接访问后端 |
| SurrealDB | ws://localhost:18002/rpc | 向量搜索 + 图关系 + 数据存储 |
| Meilisearch | http://localhost:7700 | 全文搜索 + CJK 中文分词（可选） |

### 停止服务

按 `Ctrl+C` 停止所有服务。

### 依赖关系

```
包装层服务 (17999)
    ├── 必须依赖：Embedding 服务 (18000)
    ├── 必须依赖：SurrealDB (18002) — 向量搜索 + 图关系
    ├── 可选依赖：LLM 服务 (18001)
    └── 可选依赖：Meilisearch (7700) — 全文搜索（不可用时回退到 SurrealDB BM25）
```

### 故障排查

**问题1：Embedding服务启动失败**

- 检查端口18000是否被占用
- 检查模型文件是否存在
- 查看错误日志

**问题2：服务未能就绪**

- 首次启动需要下载模型（约1.2GB）
- GPU模式需要更长的启动时间
- 检查健康检查端点：`curl http://localhost:18000/health`

**问题3：包装层服务启动失败**

- 确保后端服务已启动
- 检查端口 17999 是否被占用
- 检查环境变量配置（SurrealDB、Meilisearch 地址等）

## 📝 手动启动（不推荐）

如果需要手动启动各个服务：

```bash
# 终端1：Embedding服务
uv run python src/qwen3_embedding_service/embedding_service.py

# 终端2：LLM服务（可选）
uv run python src/qwen3_embedding_service/llm_service.py

# 终端3：包装层服务
uv run python -m wrapper.src.main

## 🎯 推荐配置

**开发环境**：
```bash
uv run python start_services.py
```

- 只启动必需的服务
- 快速启动，节省资源

**生产环境**：

```bash
uv run python start_services.py --with-llm
```

- 启动所有服务
- 提供完整功能

## ⚙️ 环境变量配置

启动前可以配置以下环境变量：

```bash
# Embedding服务
export EMB_MAX_BATCH_SIZE=256
export EMB_CACHE_SIZE=1000

# LLM服务
export LLM_CACHE_SIZE=100

# 包装层服务
export WRAPPER_PORT=17999
export WRAPPER_CACHE_MAX_SIZE=1000
export WRAPPER_CACHE_TTL=3600

# SurrealDB
export WRAPPER_SURREALDB_URL=ws://localhost:18002/rpc
export WRAPPER_SURREALDB_NAMESPACE=memory_ns
export WRAPPER_SURREALDB_DATABASE=memory_db

# Meilisearch（可选，不配置则回退到 SurrealDB BM25）
export WRAPPER_MEILI_ENABLED=true
export WRAPPER_MEILI_URL=http://127.0.0.1:7700
export WRAPPER_MEILI_API_KEY=your_master_key
export WRAPPER_MEILI_INDEX_NAME=memories
export WRAPPER_MEILI_TIMEOUT=30.0
```

## 📦 数据迁移（首次启用 Meilisearch 时）

如果之前已有 SurrealDB 中的记忆数据，需要同步到 Meilisearch：

```bash
# 设置 SurrealDB 连接信息
export SURREAL_URL=ws://localhost:18002/rpc
export SURREAL_NS=memory_ns
export SURREAL_DB=memory_db

# 运行迁移脚本（幂等，可重复运行）
uv run python scripts/migrate_to_meilisearch.py --batch-size 200
```

详细配置说明见 `README.md`。
