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

## 架构设计

详见 `WRAPPER_SERVICE_DESIGN.md` 和 `WRAPPER_SERVICE_REVIEW.md`
