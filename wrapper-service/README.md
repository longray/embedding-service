# Embedding Wrapper Service

包装服务，为Embedding和LLM服务提供统一入口和增强功能。

## 功能特性

- ✅ **熔断器保护**：防止级联故障
- ✅ **连接池管理**：提高性能和资源利用
- ✅ **智能缓存**：线程安全的LRU缓存
- ✅ **结构化日志**：便于监控和调试
- ✅ **Prometheus指标**：完整的监控支持
- ✅ **统一异常处理**：标准化错误响应

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

```bash
export WRAPPER_PORT=3001
export WRAPPER_EMBEDDING_SERVICE_URL=http://localhost:18000
export WRAPPER_LLM_SERVICE_URL=http://localhost:18001
export WRAPPER_LOG_LEVEL=INFO
```

### 启动服务

```bash
python -m src.main
```

## API端点

- `GET /health` - 健康检查
- `POST /v1/embeddings` - 创建文本嵌入
- `POST /v1/chat/completions` - 聊天补全
- `GET /metrics` - Prometheus指标

### API使用示例

#### 创建文本嵌入

```bash
curl -X POST http://localhost:3001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, world!",
    "model": "Qwen3-Embedding-0.6B"
  }'
```

**响应示例**:
```json
{
  "object": "list",
  "data": [{
    "object": "embedding",
    "embedding": [0.123, -0.456, ...],
    "index": 0
  }],
  "model": "Qwen3-Embedding-0.6B",
  "usage": {
    "prompt_tokens": 3,
    "total_tokens": 3
  }
}
```

#### 聊天补全

```bash
curl -X POST http://localhost:3001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "model": "qwen3"
  }'
```

#### 健康检查

```bash
curl http://localhost:3001/health
```

**响应示例**:
```json
{
  "status": "healthy",
  "cache_stats": {
    "max_size": 1000,
    "current_size": 42,
    "hits": 156,
    "misses": 23,
    "hit_rate": 87.15
  },
  "circuit_breakers": {
    "embedding": "closed",
    "llm": "closed"
  }
}
```

## 架构设计

详见 `WRAPPER_SERVICE_DESIGN.md` 和 `WRAPPER_SERVICE_REVIEW.md`

## 配置说明

### 环境变量

| 变量名 | 默认值 | 说明 |
|---------|--------|------|
| `WRAPPER_PORT` | 3001 | 服务端口 |
| `WRAPPER_HOST` | 0.0.0.0 | 监听地址 |
| `WRAPPER_EMBEDDING_SERVICE_URL` | http://localhost:18000 | Embedding服务地址 |
| `WRAPPER_LLM_SERVICE_URL` | http://localhost:18001 | LLM服务地址 |
| `WRAPPER_LOG_LEVEL` | INFO | 日志级别 (DEBUG/INFO/WARNING/ERROR) |
| `WRAPPER_JSON_LOGS` | false | 是否输出JSON格式日志 |
| `WRAPPER_CACHE_ENABLED` | true | 是否启用缓存 |
| `WRAPPER_CACHE_MAX_SIZE` | 1000 | 缓存最大条目数 |
| `WRAPPER_CACHE_TTL` | 3600 | 缓存过期时间（秒） |
| `WRAPPER_HTTP_TIMEOUT` | 30.0 | HTTP请求超时（秒） |
| `WRAPPER_HTTP_CONNECT_TIMEOUT` | 5.0 | 连接超时（秒） |
| `WRAPPER_RATE_LIMIT_ENABLED` | true | 是否启用限流 |
| `WRAPPER_RATE_LIMIT_REQUESTS` | 100 | 限流请求数 |
| `WRAPPER_RATE_LIMIT_WINDOW` | 60 | 限流时间窗口（秒） |
| `WRAPPER_CIRCUIT_BREAKER_ENABLED` | true | 是否启用熔断器 |
| `WRAPPER_CIRCUIT_BREAKER_THRESHOLD` | 5 | 熔断器失败阈值 |
| `WRAPPER_CIRCUIT_BREAKER_TIMEOUT` | 60 | 熔断器超时（秒） |

## 部署指南

### 生产环境部署

**使用 uvicorn 多进程模式**:
```bash
uvicorn src.main:app --host 0.0.0.0 --port 3001 --workers 4
```

**使用 gunicorn + uvicorn worker**:
```bash
gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:3001
```

## 监控

### Prometheus 指标

访问 `http://localhost:3001/metrics` 获取指标。

**主要指标**:
- `wrapper_requests_total` - 请求总数
- `wrapper_request_duration_seconds` - 请求延迟
- `wrapper_cache_hits_total` - 缓存命中数
- `wrapper_circuit_breaker_state` - 熔断器状态

## 故障排查

### 服务启动失败

1. **检查端口是否被占用**: `netstat -an | grep 3001`
2. **检查后端服务是否可达**: `curl http://localhost:18000/health`
3. **查看日志**: 设置 `WRAPPER_LOG_LEVEL=DEBUG`

### 熔断器打开

如果熔断器打开，请求会返回 503 错误。

**解决方法**:
1. 检查后端服务是否正常
2. 等待熔断器自动恢复（默认60秒）
3. 查看 `/health` 端点的熔断器状态
