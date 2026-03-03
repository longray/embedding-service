# Embedding Service 项目

基于 Qwen3 的文本嵌入和大语言模型服务，包含包装层服务提供统一入口和增强功能。

## 项目概述

本项目提供两层服务架构：

1. **后端服务层**：
   - Embedding 服务（端口 18000）- 基于 Qwen3-Embedding-0.6B
   - LLM 服务（端口 18001）- 基于 MiniCPM4-0.5B

2. **包装层服务**（端口 3001）：
   - 统一入口和API代理
   - 熔断器保护，防止级联故障
   - 智能缓存，提高响应速度
   - 连接池管理，优化资源利用
   - 完整的监控和日志

## 项目结构

```
embedding_service/
├── src/qwen3_embedding_service/    # 后端服务
│   ├── embedding_service.py        # Embedding 服务
│   └── llm_service.py               # LLM 服务
├── wrapper-service/                 # 包装层服务
│   ├── src/
│   │   ├── main.py                  # 主服务程序
│   │   ├── config.py                # 配置管理
│   │   └── utils/                   # 工具模块
│   │       ├── cache.py             # 线程安全缓存
│   │       ├── circuit_breaker.py   # 熔断器
│   │       ├── http_pool.py         # HTTP连接池
│   │       ├── logging.py           # 结构化日志
│   │       ├── metrics.py           # Prometheus指标
│   │       └── exceptions.py        # 异常处理
│   └── tests/                       # 测试文件
├── WRAPPER_SERVICE_DESIGN.md        # 包装服务设计文档
└── WRAPPER_SERVICE_REVIEW.md        # 包装服务评审文档
```

## 快速开始

### 1. 启动后端服务

```bash
# 启动 Embedding 服务
python src/qwen3_embedding_service/embedding_service.py

# 启动 LLM 服务
python src/qwen3_embedding_service/llm_service.py
```

### 2. 启动包装层服务

```bash
cd wrapper-service
pip install -r requirements.txt
python -m src.main
```

### 3. 测试服务

```bash
# 健康检查
curl http://localhost:3001/health

# 创建文本嵌入
curl -X POST http://localhost:3001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "model": "Qwen3-Embedding-0.6B"}'
```

## 核心功能

### 包装层服务特性

- ✅ **熔断器保护**：三状态保护机制，防止级联故障
- ✅ **智能缓存**：线程安全的LRU缓存，支持TTL
- ✅ **连接池管理**：HTTP连接复用，提高性能
- ✅ **结构化日志**：structlog支持，便于监控和调试
- ✅ **Prometheus指标**：完整的监控指标收集
- ✅ **统一异常处理**：标准化错误响应

### API端点

| 端点 | 方法 | 功能 | 端口 |
|------|------|------|------|
| `/v1/embeddings` | POST | 文本嵌入（带缓存+熔断） | 3001 |
| `/v1/chat/completions` | POST | 聊天补全（带熔断） | 3001 |
| `/health` | GET | 健康检查+熔断器状态 | 3001 |
| `/metrics` | GET | Prometheus指标 | 3001 |

## 配置说明

包装层服务支持通过环境变量配置：

```bash
export WRAPPER_PORT=3001
export WRAPPER_EMBEDDING_SERVICE_URL=http://localhost:18000
export WRAPPER_LLM_SERVICE_URL=http://localhost:18001
export WRAPPER_LOG_LEVEL=INFO
export WRAPPER_CACHE_MAX_SIZE=1000
export WRAPPER_CACHE_TTL=3600
```

完整配置说明见 [wrapper-service/README.md](wrapper-service/README.md)

## 监控

访问 `http://localhost:3001/metrics` 获取 Prometheus 指标。

主要指标：
- 请求总数和延迟
- 缓存命中率
- 熔断器状态
- 后端错误数

## 文档

- [包装服务设计文档](WRAPPER_SERVICE_DESIGN.md) - 完整的设计方案和实施状态
- [包装服务评审文档](WRAPPER_SERVICE_REVIEW.md) - 深度分析和重构建议
- [包装服务使用指南](wrapper-service/README.md) - API使用、配置和部署

## 开发状态

**当前版本**: v0.1.0 (项目版本) / Embedding服务 v2.0.1
**实施阶段**: P0 + P1 核心功能已完成

### 已完成
- ✅ P0级别Bug修复（配置管理、缓存、异常处理）
- ✅ P1级别功能（熔断器、连接池、日志、监控）
- ✅ 主服务程序和API端点
- ✅ 完整的测试和验证

### 待实现（P2）
- ⏳ API认证授权
- ⏳ API限流机制
- ⏳ 分布式追踪
- ⏳ 完善监控告警

## 技术栈

- **后端服务**: Python 3.11+, Qwen3 模型
- **包装层**: FastAPI, httpx, structlog, prometheus-client
- **配置管理**: pydantic-settings
- **监控**: Prometheus + Grafana

## 许可证

[待添加]

## 贡献

[待添加]
