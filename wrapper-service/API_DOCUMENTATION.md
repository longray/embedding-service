# 包装层服务 API 文档

**服务名称**: Embedding Wrapper Service  
**版本**: 1.0.0  
**默认端口**: 17999  
**基础URL**: http://localhost:17999

---

## 目录

1. [服务概述](#服务概述)
2. [核心功能](#核心功能)
3. [API 接口列表](#api-接口列表)
4. [接口详细说明](#接口详细说明)
5. [错误处理](#错误处理)
6. [监控指标](#监控指标)

---

## 服务概述

包装层服务是一个中间层服务，为后端的 Embedding 服务和 LLM 服务提供统一入口和增强功能。它不直接处理模型推理，而是作为代理层提供以下增强功能：

- **统一入口**：单一端点访问多个后端服务
- **熔断器保护**：防止级联故障
- **智能缓存**：提高响应速度，减少后端负载
- **连接池管理**：优化资源利用
- **监控和日志**：完整的可观测性

---

## 核心功能

### 1. 熔断器保护 (Circuit Breaker)

**功能说明**：
- 监控后端服务的健康状态
- 当后端服务连续失败达到阈值时，自动打开熔断器
- 熔断器打开后，直接返回错误，不再调用后端服务
- 超时后自动尝试恢复（半开状态）

**配置参数**：
- 失败阈值：5次连续失败
- 超时时间：60秒
- 半开状态最大调用次数：3次

**熔断器状态**：
- `closed`：正常状态，请求正常转发
- `open`：熔断状态，直接返回错误
- `half_open`：尝试恢复，限制请求数量

### 2. 智能缓存 (LRU Cache)

**功能说明**：
- 线程安全的 LRU（最近最少使用）缓存
- 仅对 Embedding 接口启用缓存
- 基于输入文本生成缓存键
- 自动过期机制

**配置参数**：
- 最大缓存条目：1000个
- 缓存过期时间：3600秒（1小时）

**缓存策略**：
- 缓存键格式：`emb:{input_text}`
- 命中：直接返回缓存结果
- 未命中：调用后端服务并缓存结果

### 3. 连接池管理

**功能说明**：
- HTTP 连接复用
- 自动重试机制
- 超时控制

**配置参数**：
- 请求超时：30秒
- 连接超时：5秒

### 4. 监控和日志

**日志功能**：
- 结构化日志（structlog）
- 支持 JSON 格式输出
- 可配置日志级别

**监控指标**：
- Prometheus 格式指标
- 请求计数、延迟、错误率
- 缓存命中率
- 熔断器状态

---

## API 接口列表

| 端点 | 方法 | 功能 | 缓存 | 熔断器 |
|------|------|------|------|--------|
| `/health` | GET | 健康检查 | ❌ | ❌ |
| `/v1/embeddings` | POST | 文本嵌入 | ✅ | ✅ |
| `/v1/chat/completions` | POST | 聊天补全 | ❌ | ✅ |
| `/metrics` | GET | Prometheus指标 | ❌ | ❌ |

---

## 接口详细说明

### 1. 健康检查

**端点**: `GET /health`

**功能描述**：
检查包装层服务的健康状态，包括缓存统计和熔断器状态。

**请求示例**：
```bash
curl http://localhost:17999/health
```

**响应示例**：
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

**响应字段说明**：
- `status`: 服务状态（"healthy"）
- `cache_stats`: 缓存统计信息
  - `max_size`: 最大缓存容量
  - `current_size`: 当前缓存条目数
  - `hits`: 缓存命中次数
  - `misses`: 缓存未命中次数
  - `hit_rate`: 缓存命中率（百分比）
- `circuit_breakers`: 熔断器状态
  - `embedding`: Embedding服务熔断器状态
  - `llm`: LLM服务熔断器状态

**状态码**：
- `200 OK`: 服务正常

---

### 2. 创建文本嵌入

**端点**: `POST /v1/embeddings`

**功能描述**：
将文本转换为向量表示（embedding）。此接口启用了智能缓存和熔断器保护。

**增强功能**：
- ✅ **智能缓存**：相同输入直接返回缓存结果
- ✅ **熔断器保护**：后端服务故障时快速失败
- ✅ **自动重试**：网络临时故障自动重试
- ✅ **性能监控**：记录请求延迟和错误率

**请求格式**：
```json
{
  "input": "string | string[]",
  "model": "string"
}
```

**请求字段说明**：
- `input` (必需): 要嵌入的文本
  - 类型：字符串或字符串数组
  - 单个文本：`"Hello, world!"`
  - 批量文本：`["text1", "text2", "text3"]`
- `model` (可选): 模型名称
  - 默认：`"Qwen3-Embedding-0.6B"`

**请求示例**：

单个文本：
```bash
curl -X POST http://localhost:17999/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, world!",
    "model": "Qwen3-Embedding-0.6B"
  }'
```

批量文本：
```bash
curl -X POST http://localhost:17999/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["First text", "Second text", "Third text"],
    "model": "Qwen3-Embedding-0.6B"
  }'
```

**响应示例**：
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, 0.789, ...],
      "index": 0
    }
  ],
  "model": "Qwen3-Embedding-0.6B",
  "usage": {
    "prompt_tokens": 3,
    "total_tokens": 3
  }
}
```

**响应字段说明**：
- `object`: 响应对象类型（"list"）
- `data`: 嵌入结果数组
  - `object`: 数据对象类型（"embedding"）
  - `embedding`: 向量数组（浮点数）
  - `index`: 在批量请求中的索引
- `model`: 使用的模型名称
- `usage`: 令牌使用统计
  - `prompt_tokens`: 输入令牌数
  - `total_tokens`: 总令牌数

**状态码**：
- `200 OK`: 请求成功
- `422 Unprocessable Entity`: 请求参数错误
- `503 Service Unavailable`: 后端服务不可用或熔断器打开

**缓存行为**：
- 缓存键：基于输入文本生成
- 缓存命中：直接返回缓存结果，响应时间 < 10ms
- 缓存未命中：调用后端服务，响应时间约 100-500ms

**性能指标**：
- 缓存命中响应时间：< 10ms
- 缓存未命中响应时间：100-500ms
- 批量处理：支持最多 50 条文本

---

### 3. 聊天补全

**端点**: `POST /v1/chat/completions`

**功能描述**：
生成聊天对话的补全响应。此接口启用了熔断器保护，但不启用缓存（因为每次对话都是唯一的）。

**增强功能**：
- ✅ **熔断器保护**：后端服务故障时快速失败
- ✅ **自动重试**：网络临时故障自动重试
- ✅ **性能监控**：记录请求延迟和错误率
- ❌ **缓存**：不启用（对话具有唯一性）

**请求格式**：
```json
{
  "messages": [
    {
      "role": "string",
      "content": "string"
    }
  ],
  "model": "string",
  "max_tokens": "integer (optional)",
  "temperature": "float (optional)"
}
```

**请求字段说明**：
- `messages` (必需): 对话消息数组
  - `role`: 角色（"user" | "assistant" | "system"）
  - `content`: 消息内容
- `model` (可选): 模型名称
  - 默认：`"MiniCPM4-0.5B"`
- `max_tokens` (可选): 最大生成令牌数
- `temperature` (可选): 采样温度（0.0-2.0）

**请求示例**：

简单对话：
```bash
curl -X POST http://localhost:17999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "你好，请介绍一下自己"}
    ],
    "model": "MiniCPM4-0.5B"
  }'
```

多轮对话：
```bash
curl -X POST http://localhost:17999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "你好"},
      {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
      {"role": "user", "content": "介绍一下Python"}
    ],
    "model": "MiniCPM4-0.5B",
    "max_tokens": 100
  }'
```

**响应示例**：
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "MiniCPM4-0.5B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！我是一个AI助手..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

**响应字段说明**：
- `id`: 请求唯一标识符
- `object`: 响应对象类型（"chat.completion"）
- `created`: 创建时间戳
- `model`: 使用的模型名称
- `choices`: 生成结果数组
  - `index`: 结果索引
  - `message`: 生成的消息
    - `role`: 角色（"assistant"）
    - `content`: 生成的内容
  - `finish_reason`: 完成原因（"stop" | "length"）
- `usage`: 令牌使用统计

**状态码**：
- `200 OK`: 请求成功
- `422 Unprocessable Entity`: 请求参数错误
- `503 Service Unavailable`: 后端服务不可用或熔断器打开

**性能指标**：
- 平均响应时间：1-5秒（取决于生成长度）
- 最大响应时间：30秒（超时）

---

### 4. Prometheus 指标

**端点**: `GET /metrics`

**功能描述**：
提供 Prometheus 格式的监控指标，用于服务监控和告警。

**请求示例**：
```bash
curl http://localhost:17999/metrics
```

**响应格式**：
Prometheus 文本格式

**主要指标**：

1. **请求计数**：
   - `wrapper_requests_total{method="POST",endpoint="/v1/embeddings"}` - 请求总数
   - `wrapper_requests_total{method="POST",endpoint="/v1/chat/completions"}` - 请求总数

2. **请求延迟**：
   - `wrapper_request_duration_seconds_bucket` - 请求延迟分布
   - `wrapper_request_duration_seconds_sum` - 请求延迟总和
   - `wrapper_request_duration_seconds_count` - 请求计数

3. **缓存指标**：
   - `wrapper_cache_hits_total` - 缓存命中总数
   - `wrapper_cache_misses_total` - 缓存未命中总数

4. **后端错误**：
   - `wrapper_backend_errors_total{service="embedding",error_type="HTTPError"}` - 后端错误总数

5. **熔断器状态**：
   - `wrapper_circuit_breaker_state{service="embedding"}` - 熔断器状态（0=closed, 1=open, 2=half_open）

**状态码**：
- `200 OK`: 成功返回指标

---

## 错误处理

### 统一错误响应格式

所有错误响应都遵循统一格式：

```json
{
  "error": "错误消息",
  "details": "详细错误信息（可选）"
}
```

### 常见错误码

| 状态码 | 错误类型 | 说明 | 解决方法 |
|--------|----------|------|----------|
| 400 | Bad Request | 请求格式错误 | 检查请求参数 |
| 422 | Unprocessable Entity | 参数验证失败 | 检查必需字段和数据类型 |
| 503 | Service Unavailable | 后端服务不可用 | 等待服务恢复或检查后端服务 |
| 503 | Circuit Breaker Open | 熔断器打开 | 等待熔断器自动恢复（60秒） |

### 错误示例

**缺失必需字段**：
```json
{
  "error": "Validation error",
  "details": "Field 'input' is required"
}
```

**后端服务不可用**：
```json
{
  "error": "Embedding service unavailable",
  "details": null
}
```

**熔断器打开**：
```json
{
  "error": "Embedding service unavailable",
  "details": "Circuit breaker is open"
}
```

---

## 监控指标

### 缓存性能

**查看缓存命中率**：
```bash
curl http://localhost:17999/health | jq '.cache_stats.hit_rate'
```

**预期值**：
- 良好：> 80%
- 正常：60-80%
- 需优化：< 60%

### 熔断器状态

**查看熔断器状态**：
```bash
curl http://localhost:17999/health | jq '.circuit_breakers'
```

**状态说明**：
- `closed`: 正常，所有请求正常转发
- `open`: 熔断，直接返回错误
- `half_open`: 尝试恢复，限制请求数量

### Prometheus 查询示例

**请求成功率**：
```promql
rate(wrapper_requests_total{status="success"}[5m]) 
/ 
rate(wrapper_requests_total[5m])
```

**平均响应时间**：
```promql
rate(wrapper_request_duration_seconds_sum[5m]) 
/ 
rate(wrapper_request_duration_seconds_count[5m])
```

**缓存命中率**：
```promql
rate(wrapper_cache_hits_total[5m]) 
/ 
(rate(wrapper_cache_hits_total[5m]) + rate(wrapper_cache_misses_total[5m]))
```

---

## 配置参考

### 环境变量

完整的环境变量配置请参考 `README.md` 中的配置说明部分。

### 推荐配置

**生产环境**：
```bash
WRAPPER_PORT=17999
WRAPPER_LOG_LEVEL=INFO
WRAPPER_JSON_LOGS=true
WRAPPER_CACHE_ENABLED=true
WRAPPER_CACHE_MAX_SIZE=10000
WRAPPER_CACHE_TTL=3600
WRAPPER_CIRCUIT_BREAKER_ENABLED=true
```

**开发环境**：
```bash
WRAPPER_PORT=17999
WRAPPER_LOG_LEVEL=DEBUG
WRAPPER_JSON_LOGS=false
WRAPPER_CACHE_ENABLED=true
WRAPPER_CACHE_MAX_SIZE=100
WRAPPER_CACHE_TTL=300
```

---

## 最佳实践

### 1. 缓存使用

- ✅ 对于相同的输入文本，使用缓存可以提升 10-50 倍性能
- ✅ 批量请求时，相同文本会自动使用缓存
- ⚠️ 缓存仅对 Embedding 接口有效，Chat 接口不缓存

### 2. 错误处理

- ✅ 始终检查响应状态码
- ✅ 实现重试机制（指数退避）
- ✅ 监控熔断器状态，及时发现后端服务问题

### 3. 性能优化

- ✅ 使用批量请求减少网络开销
- ✅ 复用相同的输入文本以利用缓存
- ✅ 监控 Prometheus 指标，及时发现性能瓶颈

---

**文档版本**: 1.0.0  
**最后更新**: 2026-03-03
