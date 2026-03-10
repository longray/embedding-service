# 服务启动指南

## 🚀 快速启动

### 方式 1：直接启动包装服务（推荐开发用）

```bash
# 启动最小化包装服务（端口 17999）
uv run python -m wrapper.src.main
```

> 前提：需要先启动 Embedding 服务（端口 18000）和 SurrealDB（端口 8000）。

### 方式 2：Docker Compose 一键启动

```bash
docker-compose up -d
```

### 方式 3：Windows 批处理脚本

```batch
:: 启动所有服务（Embedding + SurrealDB）
start_all_services.bat

:: 或单独启动
start_embedding_service.bat
start_surrealdb.bat
```

## 服务访问地址

| 服务 | 地址 | 说明 |
|------|------|------|
| **包装服务** | http://localhost:17999 | 推荐使用（带缓存） |
| Embedding | http://localhost:18000 | 直接访问后端 |
| LLM | http://localhost:18001 | 直接访问后端 |
| SurrealDB | ws://localhost:8000/rpc | 数据库 |

## 依赖关系

```
包装服务 (17999)
    ├── 必须依赖：Embedding 服务 (18000)
    └── 必须依赖：SurrealDB (8000)
```

> LLM 服务 (18001) 可独立启动，当前包装服务不代理 LLM。

## 手动启动各服务

如果需要分别启动：

```bash
# 终端 1：Embedding 服务
uv run python src/qwen3_embedding_service/embedding_service.py

# 终端 2：LLM 服务（可选）
uv run python src/qwen3_embedding_service/llm_service.py

# 终端 3：包装服务
uv run python -m wrapper.src.main
```

## 环境变量配置

启动前可配置（也可写入 `.env` 文件）：

```bash
# 包装服务
export WRAPPER_PORT=17999
export WRAPPER_CACHE_ENABLED=true
export WRAPPER_EMBEDDING_SERVICE_URL=http://localhost:18000
export WRAPPER_SURREALDB_URL=ws://localhost:8000/rpc

# 搜索阈值
export WRAPPER_SEARCH_VECTOR_THRESHOLD=0.75
export WRAPPER_SEARCH_HYBRID_THRESHOLD=0.75
export WRAPPER_SEARCH_KEYWORD_THRESHOLD=0.0
```

## 停止服务

按 `Ctrl+C` 停止服务。Docker 方式使用：

```bash
docker-compose down
```

## 故障排查

**Embedding 服务启动失败**
- 检查端口 18000 是否被占用：`netstat -ano | findstr ":18000"`
- 检查模型文件是否存在
- 首次启动需下载模型（约 1.2GB）

**包装服务启动失败**
- 确保 Embedding 服务和 SurrealDB 已启动
- 检查端口 17999 是否被占用
- 查看 `.env` 配置是否正确

**健康检查**
```bash
# 检查包装服务
curl http://localhost:17999/health

# 检查 Embedding 服务
curl http://localhost:18000/health
```
