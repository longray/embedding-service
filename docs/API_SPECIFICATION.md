# Embedding Service API 接口规范

**版本**: 1.0.0
**生成日期**: 2026-03-04
**适用项目**: D:\embedding_service

---

## 📋 目录

- [一、API概述](#一api概述)
- [二、API版本管理](#二api版本管理)
- [三、认证与授权](#三认证与授权)
- [四、错误处理规范](#四错误处理规范)
- [五、核心API端点](#五核心api端点)
  - [5.1 Embedding服务](#51-embedding服务端口-18000)
  - [5.2 LLM服务](#52-llm服务端口-18001)
  - [5.3 包装层服务](#53-包装层服务端口-3001)
- [六、数据模型定义](#六数据模型定义)
- [七、插件API规范](#七插件api规范)
- [八、OpenAPI/Swagger文档](#八openapiswagger文档)
- [九、最佳实践建议](#九最佳实践建议)

---

## 一、API概述

### 1.1 架构设计

本项目采用**两层服务架构**：

```
┌─────────────────────────────────────────────────────────────┐
│                       客户端                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              包装层服务 (端口 3001)                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                │
│  │ 缓存    │  │ 熔断器  │  │ 连接池  │                │
│  └─────────┘  └─────────┘  └─────────┘                │
└─────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │Embedding│ │   LLM    │ │SurrealDB │
        │ 18000   │ │  18001   │ │  18002   │
        │(Qwen3)  │ │(MiniCPM) │ │          │
        └──────────┘ └──────────┘ └──────────┘
```

### 1.2 服务列表

| 服务 | 端口 | 功能 | 模型 |
|------|------|------|------|
| Embedding服务 | 18000 | 文本→向量转换 | Qwen3-Embedding-0.6B |
| LLM服务 | 18001 | 对话补全生成 | MiniCPM4-0.5B |
| 包装层服务 | 3001 | 统一入口+增强功能 | - |

### 1.3 核心特性

- ✅ **熔断器保护**：防止级联故障
- ✅ **智能缓存**：线程安全LRU缓存（TTL过期）
- ✅ **连接池管理**：HTTP连接复用
- ✅ **结构化日志**：structlog支持
- ✅ **Prometheus指标**：完整监控
- ✅ **记忆管理**：SurrealDB向量存储

---

## 二、API版本管理

### 2.1 版本策略

本项目采用**URL路径版本控制**：

```yaml
版本格式: /v{major}/endpoint

示例:
  - /v1/embeddings        # 主版本1
  - /v1/chat/completions # 主版本1
  - /api/v1/memories      # 记忆API主版本1
```

### 2.2 版本兼容性原则

| 约定 | 说明 |
|------|------|
| **主版本号 (major)** | 不兼容的API修改 |
| **次版本号 (minor)** | 向下兼容的功能性新增 |
| **修订号 (patch)** | 向下兼容的问题修正 |

### 2.3 版本生命周期

```
Current (当前版本) → Supported (支持版本) → Deprecated (弃用版本) → Retired (退役版本)

示例:
  - v1.0: Current  (2026-03-04)
  - v0.9: Supported (至少支持6个月)
  - v0.8: Deprecated (提前3个月通知)
  - v0.7: Retired (不再支持)
```

### 2.4 版本升级指南

**添加新字段**：
- ✅ 向后兼容：客户端可忽略新字段
- ✅ 新字段必须可选或提供默认值

**删除字段**：
- ❌ 破坏性变更：必须升级主版本号
- ⚠️ 提前3个月发布弃用通知

**修改字段类型**：
- ❌ 破坏性变更：必须升级主版本号
- ✅ 向下兼容：扩大类型范围（如 int32 → int64）

---

## 三、认证与授权

### 3.1 当前状态

**⚠️ 注意**：当前项目**未实现**认证授权机制

- 所有端点均无API Key验证
- CORS配置允许所有来源（`allow_origins=["*"]`）
- 认证机制为P2待实现功能

### 3.2 推荐的认证方案

#### 方案1: API Key认证（推荐）

**实现方式**：HTTP Header

```http
POST /v1/embeddings HTTP/1.1
Host: localhost:3001
Authorization: Bearer sk-xxxxxxxxxxxxxxxxxxxx
Content-Type: application/json
```

**FastAPI实现示例**：

```python
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """验证API Key"""
    api_key = credentials.credentials

    # 从数据库或配置中验证API Key
    if not is_valid_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return api_key

# 使用认证
@app.post("/v1/embeddings")
async def create_embeddings(
    request: EmbeddingRequest,
    api_key: str = Depends(verify_api_key)
):
    # 处理请求
    pass
```

#### 方案2: JWT Token认证

**实现方式**：Bearer Token

```http
POST /v1/embeddings HTTP/1.1
Host: localhost:3001
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json
```

**JWT Payload示例**：

```json
{
  "sub": "user123",
  "iat": 1234567890,
  "exp": 1234567890 + 3600,
  "scopes": ["embeddings:read", "chat:write"]
}
```

#### 方案3: OAuth2.0

**授权码流程**：

```
1. 客户端重定向到授权服务器
2. 用户授权
3. 授权服务器返回授权码
4. 客户端用授权码换取访问令牌
5. 客户端使用访问令牌访问API
```

### 3.3 权限模型（推荐）

```yaml
权限范围 (scopes):
  - embeddings:read    # 读取embedding
  - embeddings:write   # 创建embedding
  - chat:read         # 读取聊天记录
  - chat:write        # 发送聊天请求
  - memories:read     # 读取记忆
  - memories:write    # 写入记忆
  - admin:*           # 管理员权限（所有权限）
```

### 3.4 速率限制

**推荐实现**：基于令牌桶算法

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# 限制每个IP每分钟最多100次请求
@app.post("/v1/embeddings")
@limiter.limit("100/minute")
async def create_embeddings(request: Request):
    pass
```

---

## 四、错误处理规范

### 4.1 HTTP状态码使用

| 状态码 | 说明 | 使用场景 |
|--------|------|----------|
| 200 | OK | 请求成功 |
| 400 | Bad Request | 请求参数错误 |
| 401 | Unauthorized | 未认证或认证失败 |
| 403 | Forbidden | 无权限访问 |
| 404 | Not Found | 资源不存在 |
| 422 | Unprocessable Entity | 参数验证失败 |
| 429 | Too Many Requests | 超出速率限制 |
| 500 | Internal Server Error | 服务器内部错误 |
| 503 | Service Unavailable | 服务不可用/熔断器打开 |

### 4.2 错误响应格式

**推荐格式**（基于RFC 7807 Problem Details）：

```json
{
  "type": "https://api.example.com/errors/validation-error",
  "title": "Validation Error",
  "status": 400,
  "detail": "The request contains invalid parameters",
  "instance": "/v1/embeddings",
  "errors": [
    {
      "field": "input",
      "message": "Input cannot be empty",
      "code": "EMPTY_INPUT"
    }
  ],
  "timestamp": "2026-03-04T14:30:00Z",
  "request_id": "req_abc123xyz"
}
```

### 4.3 统一异常类设计

**当前实现**（参考 `wrapper-service/src/utils/exceptions.py`）：

```python
class WrapperServiceError(Exception):
    """包装服务基础异常"""
    def __init__(self, message: str, status_code: int = 500, details: Optional[dict] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class ServiceUnavailableError(WrapperServiceError):
    """服务不可用异常 (503)"""
    pass

class CircuitBreakerError(WrapperServiceError):
    """熔断器打开异常 (503)"""
    pass

class ValidationError(WrapperServiceError):
    """验证错误异常 (400)"""
    pass

class RateLimitExceededError(WrapperServiceError):
    """限流异常 (429)"""
    pass
```

### 4.4 全局异常处理器

**FastAPI实现**：

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(WrapperServiceError)
async def wrapper_error_handler(request: Request, exc: WrapperServiceError):
    """统一异常处理"""
    error_id = generate_error_id()

    logger.error(
        "api_error",
        error_id=error_id,
        error=str(exc),
        status_code=exc.status_code,
        path=request.url.path,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": f"https://api.example.com/errors/{type(exc).__name__.lower()}",
            "title": type(exc).__name__,
            "status": exc.status_code,
            "detail": exc.message,
            "instance": str(request.url.path),
            "errors": exc.details.get("errors", []) if exc.details else [],
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": error_id,
        },
    )
```

---

## 五、核心API端点

### 5.1 Embedding服务（端口 18000）

#### 5.1.1 创建文本嵌入

**端点**：`POST /v1/embeddings`

**请求示例**：

```json
{
  "input": "Hello, world!",
  "model": "Qwen3-Embedding-0.6B",
  "encoding_format": "float",
  "dimensions": 1024,
  "normalize": true
}
```

**请求参数**：

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| input | string \| string[] | ✅ | 文本或文本列表 |
| model | string | ❌ | 模型名称（默认：Qwen3-Embedding-0.6B） |
| encoding_format | string | ❌ | 编码格式：float \| base64 |
| dimensions | int | ❌ | 输出维度（32-1024） |
| normalize | boolean | ❌ | 是否归一化（默认：true） |

**响应示例**：

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.1234, -0.5678, 0.9012, ...]
    }
  ],
  "model": "Qwen3-Embedding-0.6B",
  "usage": {
    "prompt_tokens": 3,
    "total_tokens": 3,
    "processing_time_ms": 123.45
  }
}
```

**批量请求示例**：

```json
{
  "input": [
    "Hello, world!",
    "How are you?",
    "Good morning"
  ],
  "model": "Qwen3-Embedding-0.6B"
}
```

**响应示例**：

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.1234, -0.5678, ...]
    },
    {
      "object": "embedding",
      "index": 1,
      "embedding": [0.2345, -0.6789, ...]
    },
    {
      "object": "embedding",
      "index": 2,
      "embedding": [0.3456, -0.7890, ...]
    }
  ],
  "model": "Qwen3-Embedding-0.6B",
  "usage": {
    "prompt_tokens": 12,
    "total_tokens": 12,
    "processing_time_ms": 456.78
  }
}
```

**错误响应**：

```json
{
  "type": "https://api.example.com/errors/validation-error",
  "title": "Validation Error",
  "status": 400,
  "detail": "Input validation failed",
  "instance": "/v1/embeddings",
  "errors": [
    {
      "field": "input",
      "message": "Text length cannot exceed 32768 characters",
      "code": "TEXT_TOO_LONG"
    }
  ],
  "timestamp": "2026-03-04T14:30:00Z",
  "request_id": "req_abc123"
}
```

#### 5.1.2 健康检查

**端点**：`GET /health`

**响应示例**：

```json
{
  "status": "healthy",
  "service": "embedding",
  "version": "2.0.1",
  "device": "cuda",
  "max_batch_size": 256,
  "max_length": 2048,
  "model": "/path/to/Qwen/Qwen3-Embedding-0___6B",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3080",
  "gpu_memory_total_gb": 10.0,
  "gpu_memory_used_mb": 1234.56,
  "gpu_memory_reserved_mb": 2048.0
}
```

#### 5.1.3 模型列表

**端点**：`GET /v1/models`

**响应示例**：

```json
{
  "data": [
    {
      "id": "Qwen3-Embedding-0.6B",
      "object": "model",
      "created": 1700000000,
      "owned_by": "Alibaba",
      "dimensions": 1024,
      "max_batch_size": 256
    }
  ],
  "object": "list"
}
```

#### 5.1.4 统计信息

**端点**：`GET /stats`

**响应示例**：

```json
{
  "service": "embedding",
  "cache": {
    "hits": 1234,
    "misses": 567,
    "maxsize": 1000,
    "currsize": 890,
    "hit_rate": 68.5
  },
  "config": {
    "max_batch_size": 256,
    "max_length": 2048,
    "device": "cuda"
  },
  "model_loaded": true
}
```

---

### 5.2 LLM服务（端口 18001）

#### 5.2.1 对话补全（OpenAI兼容）

**端点**：`POST /v1/chat/completions`

**请求示例**：

```json
{
  "model": "MiniCPM4-0.5B",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "top_p": 0.7,
  "max_tokens": 512,
  "stream": false,
  "do_sample": true
}
```

**请求参数**：

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| model | string | ❌ | 模型名称（默认：MiniCPM4-0.5B） |
| messages | array | ✅ | 对话消息列表 |
| temperature | float | ❌ | 采样温度（0.0-2.0） |
| top_p | float | ❌ | 核采样参数（0.0-1.0） |
| max_tokens | int | ❌ | 最大生成token数 |
| stream | boolean | ❌ | 是否流式输出 |
| do_sample | boolean | ❌ | 是否使用采样 |

**响应示例**：

```json
{
  "id": "chatcmpl-local",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "MiniCPM4-0.5B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！有什么我可以帮助你的吗？"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 50,
    "total_tokens": 70
  }
}
```

#### 5.2.2 简单生成（支持缓存）

**端点**：`POST /generate`

**请求示例**：

```json
{
  "prompt": "Tell me a joke",
  "temperature": 0.7,
  "top_p": 0.7,
  "max_new_tokens": 512,
  "use_cache": true
}
```

**请求参数**：

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| prompt | string | ✅ | 提示词 |
| temperature | float | ❌ | 采样温度（0.0-2.0） |
| top_p | float | ❌ | 核采样参数（0.0-1.0） |
| max_new_tokens | int | ❌ | 最大生成长度 |
| use_cache | boolean | ❌ | 是否使用缓存（默认：true） |

**响应示例**：

```json
{
  "response": "Why don't scientists trust atoms? Because they make up everything!",
  "model": "MiniCPM4-0.5B",
  "usage": {
    "completion_tokens": 25,
    "from_cache": false
  },
  "generation_time_ms": 1234.56
}
```

#### 5.2.3 健康检查

**端点**：`GET /health`

**响应示例**：

```json
{
  "status": "healthy",
  "service": "llm",
  "version": "1.0.0",
  "device": "cuda",
  "model": "OpenBMB/MiniCPM4-0.5B",
  "max_batch_size": 2,
  "max_new_tokens": 1024,
  "max_length": 2048,
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3080",
  "gpu_memory_total_gb": 10.0,
  "gpu_memory_used_mb": 2345.67,
  "gpu_memory_reserved_mb": 4096.0
}
```

---

### 5.3 包装层服务（端口 3001）

#### 5.3.1 健康检查（含SurrealDB状态）

**端点**：`GET /health`

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
  },
  "surrealdb": "healthy"
}
```

#### 5.3.2 创建文本嵌入（带缓存+熔断）

**端点**：`POST /v1/embeddings`

**请求/响应格式**：与Embedding服务相同

**增强特性**：
- ✅ 智能缓存（LRU + TTL）
- ✅ 熔断器保护
- ✅ 连接池复用

#### 5.3.3 聊天补全（带熔断）

**端点**：`POST /v1/chat/completions`

**请求/响应格式**：与LLM服务相同

**增强特性**：
- ✅ 熔断器保护
- ✅ 连接池复用

#### 5.3.4 批量上传记忆

**端点**：`POST /api/v1/memories`

**请求示例**：

```json
{
  "memories": [
    {
      "content": "User prefers TypeScript for development",
      "metadata": {
        "source": "user",
        "project": "projectA"
      },
      "entities": [
        {
          "name": "TypeScript",
          "type": "language"
        }
      ],
      "relations": [
        {
          "from": "user",
          "to": "TypeScript",
          "type": "prefers"
        }
      ]
    }
  ]
}
```

**请求参数**：

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| memories | array | ✅ | 记忆列表 |
| memories[].content | string | ✅ | 记忆内容 |
| memories[].metadata | object | ❌ | 元数据 |
| memories[].entities | array | ❌ | 实体列表 |
| memories[].relations | array | ❌ | 关系列表 |

**响应示例**：

```json
{
  "total": 1,
  "success": 1,
  "failed": 0,
  "memory_ids": ["mem_abc123"]
}
```

#### 5.3.5 搜索记忆

**端点**：`POST /api/v1/memories/search`

**请求示例**：

```json
{
  "query": "TypeScript preferences",
  "mode": "hybrid",
  "limit": 10,
  "threshold": 0.7
}
```

**请求参数**：

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| query | string | ✅ | 搜索查询 |
| mode | string | ❌ | 搜索模式：vector \| keyword \| hybrid |
| limit | int | ❌ | 结果数量限制（默认：10） |
| threshold | float | ❌ | 相似度阈值（0.0-1.0） |

**响应示例**：

```json
{
  "results": [
    {
      "id": "mem_abc123",
      "content": "User prefers TypeScript for development",
      "score": 0.85,
      "metadata": {
        "source": "user"
      }
    }
  ],
  "total": 1
}
```

#### 5.3.6 Prometheus指标

**端点**：`GET /metrics`

**返回**：Prometheus格式的指标数据

```
# HELP wrapper_requests_total Total number of requests
# TYPE wrapper_requests_total counter
wrapper_requests_total{method="POST",endpoint="/v1/embeddings",status="200"} 1234

# HELP wrapper_request_duration_seconds Request duration in seconds
# TYPE wrapper_request_duration_seconds histogram
wrapper_request_duration_seconds_bucket{le="0.1"} 100
wrapper_request_duration_seconds_bucket{le="0.5"} 500
wrapper_request_duration_seconds_bucket{le="1.0"} 800
```

---

## 六、数据模型定义

### 6.1 Embedding相关模型

```python
from pydantic import BaseModel, Field, validator
from typing import Literal, List, Union

class EmbeddingRequest(BaseModel):
    """Embedding请求模型"""
    input: Union[str, List[str]] = Field(..., description="文本或文本列表")
    model: str = Field("Qwen3-Embedding-0.6B", description="模型名称")
    encoding_format: Literal["float", "base64"] = Field("float", description="编码格式")
    dimensions: int = Field(None, ge=32, le=1024, description="输出维度")
    normalize: bool = Field(True, description="是否归一化")

    @validator("input")
    def validate_input(cls, v):
        if isinstance(v, list) and not v:
            raise ValueError("输入列表不能为空")
        return v

class EmbeddingObject(BaseModel):
    """Embedding对象"""
    object: str = "embedding"
    index: int
    embedding: List[float]

class Usage(BaseModel):
    """使用情况"""
    prompt_tokens: int
    total_tokens: int
    processing_time_ms: float

class EmbeddingResponse(BaseModel):
    """Embedding响应"""
    object: str = "list"
    data: List[EmbeddingObject]
    model: str
    usage: Usage
```

### 6.2 LLM相关模型

```python
class Message(BaseModel):
    """消息模型"""
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    """对话补全请求"""
    model: str = "MiniCPM4-0.5B"
    messages: List[Message] = Field(..., min_items=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(None, ge=1, le=8192)
    stream: bool = False
    do_sample: bool = True

class Choice(BaseModel):
    """选择对象"""
    index: int
    message: Message
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    """对话补全响应"""
    id: str = "chatcmpl-local"
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
```

### 6.3 记忆相关模型

```python
class Entity(BaseModel):
    """实体模型"""
    name: str
    type: str
    properties: Optional[dict] = None

class Relation(BaseModel):
    """关系模型"""
    from: str
    to: str
    type: str
    properties: Optional[dict] = None

class Memory(BaseModel):
    """记忆模型"""
    content: str
    metadata: Optional[dict] = None
    entities: Optional[List[Entity]] = None
    relations: Optional[List[Relation]] = None

class MemoryUploadRequest(BaseModel):
    """记忆上传请求"""
    memories: List[Memory] = Field(..., min_items=1)

class MemoryUploadResponse(BaseModel):
    """记忆上传响应"""
    total: int
    success: int
    failed: int
    memory_ids: List[str]
```

---

## 七、插件API规范

### 7.1 插件通信模式

#### 7.1.1 HTTP API（推荐）

**优势**：
- ✅ 跨语言支持
- ✅ 易于调试
- ✅ 标准化协议

**实现方式**：RESTful API + JSON

```http
POST /api/v1/plugins/{plugin_id}/invoke HTTP/1.1
Host: localhost:3001
Content-Type: application/json

{
  "action": "search_memories",
  "params": {
    "query": "TypeScript",
    "limit": 10
  }
}
```

#### 7.1.2 WebSocket（实时通信）

**优势**：
- ✅ 双向通信
- ✅ 实时推送
- ✅ 低延迟

**实现方式**：WebSocket + JSON

```javascript
const ws = new WebSocket('ws://localhost:3001/ws/plugins');

ws.onopen = () => {
  ws.send(JSON.stringify({
    action: 'subscribe',
    params: {
      topic: 'memory_updates'
    }
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### 7.2 插件注册与生命周期

#### 7.2.1 插件清单（plugin.json）

```json
{
  "name": "memory-search-plugin",
  "version": "1.0.0",
  "description": "Memory search plugin for embedding service",
  "author": "Your Name",
  "license": "MIT",
  "main": "index.js",
  "capabilities": [
    "memory:search",
    "memory:upload",
    "memory:delete"
  ],
  "permissions": [
    "read:memories",
    "write:memories",
    "network:request"
  ],
  "dependencies": {
    "embedding-service": ">=1.0.0"
  }
}
```

#### 7.2.2 插件生命周期

```
安装 → 初始化 → 注册 → 激活 → 运行 → 停用 → 卸载
                                    ↓
                               错误恢复
```

**状态转换图**：

```
[UNINSTALLED] → [INSTALLED] → [INITIALIZING] → [ACTIVE] → [DISABLED] → [UNINSTALLED]
                                           ↓
                                        [ERROR]
                                           ↓
                                    [RECOVERING] → [ACTIVE]
```

### 7.3 插件权限模型

```yaml
权限范围:
  - read:memories     # 读取记忆
  - write:memories    # 写入记忆
  - delete:memories   # 删除记忆
  - network:request   # 网络请求
  - admin:*          # 管理员权限

权限级别:
  - public:          # 公开权限（无需授权）
  - user:            # 用户权限（需要认证）
  - admin:           # 管理员权限（需要管理员角色）
```

### 7.4 插件API设计原则

#### 7.4.1 最小权限原则

```python
# ✅ 正确：仅请求必要的权限
plugin_permissions = ["read:memories"]

# ❌ 错误：请求过多权限
plugin_permissions = ["admin:*"]
```

#### 7.4.2 向后兼容性

```python
# ✅ 正确：保留旧API，新增可选参数
def search_memories(query, limit=10, threshold=None):
    """搜索记忆"""
    pass

# ❌ 错误：修改现有参数
def search_memories(query, max_results):
    """搜索记忆（破坏性修改）"""
    pass
```

#### 7.4.3 版本化API

```python
# 插件API版本化
class PluginAPIv1:
    def search_memories(self, query: str, limit: int = 10):
        pass

class PluginAPIv2(PluginAPIv1):
    def search_memories(self, query: str, limit: int = 10, filters: dict = None):
        # 向后兼容：新增可选参数
        pass
```

### 7.5 插件间通信

#### 7.5.1 事件总线（Event Bus）

```python
# 发布事件
event_bus.publish('memory:created', {
    'memory_id': 'mem_123',
    'content': 'User preference',
    'timestamp': '2026-03-04T14:30:00Z'
})

# 订阅事件
@event_bus.subscribe('memory:created')
def on_memory_created(event):
    """处理记忆创建事件"""
    print(f"Memory created: {event['memory_id']}")
```

#### 7.5.2 服务发现（Service Discovery）

```python
# 注册服务
service_registry.register({
    'name': 'memory-search-plugin',
    'version': '1.0.0',
    'endpoints': {
        'search': '/api/v1/plugins/memory-search/search'
    }
})

# 发现服务
services = service_registry.discover('memory:*')
```

---

## 八、OpenAPI/Swagger文档

### 8.1 OpenAPI规范结构

```yaml
openapi: 3.0.3
info:
  title: Embedding Service API
  version: 1.0.0
  description: |
    Embedding服务API接口规范

    ## 功能
    - 文本嵌入生成
    - 对话补全
    - 记忆管理
    - 向量搜索

  contact:
    name: API Support
    email: support@example.com
  license:
    name: MIT

servers:
  - url: http://localhost:3001
    description: 本地开发环境
  - url: https://api.example.com
    description: 生产环境

tags:
  - name: embeddings
    description: 文本嵌入相关接口
  - name: chat
    description: 对话补全相关接口
  - name: memories
    description: 记忆管理相关接口
  - name: health
    description: 健康检查相关接口
```

### 8.2 端点定义示例

```yaml
paths:
  /v1/embeddings:
    post:
      tags:
        - embeddings
      summary: 创建文本嵌入
      description: 生成文本的嵌入向量
      operationId: createEmbeddings
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/EmbeddingRequest'
            examples:
              single:
                summary: 单条文本
                value:
                  input: "Hello, world!"
                  model: "Qwen3-Embedding-0.6B"
              batch:
                summary: 批量文本
                value:
                  input: ["Hello", "World"]
                  model: "Qwen3-Embedding-0.6B"
      responses:
        '200':
          description: 成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/EmbeddingResponse'
              example:
                object: "list"
                data:
                  - object: "embedding"
                    index: 0
                    embedding: [0.1234, -0.5678, 0.9012]
                model: "Qwen3-Embedding-0.6B"
                usage:
                  prompt_tokens: 3
                  total_tokens: 3
                  processing_time_ms: 123.45
        '400':
          $ref: '#/components/responses/BadRequest'
        '429':
          $ref: '#/components/responses/RateLimitExceeded'
        '500':
          $ref: '#/components/responses/InternalError'

  /health:
    get:
      tags:
        - health
      summary: 健康检查
      description: 检查服务健康状态
      operationId: healthCheck
      responses:
        '200':
          description: 服务健康
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthResponse'
```

### 8.3 组件定义

```yaml
components:
  schemas:
    EmbeddingRequest:
      type: object
      required:
        - input
      properties:
        input:
          oneOf:
            - type: string
            - type: array
              items:
                type: string
          description: 文本或文本列表
        model:
          type: string
          default: Qwen3-Embedding-0.6B
        encoding_format:
          type: string
          enum: [float, base64]
          default: float
        dimensions:
          type: integer
          minimum: 32
          maximum: 1024
        normalize:
          type: boolean
          default: true

    EmbeddingResponse:
      type: object
      required:
        - object
        - data
        - model
        - usage
      properties:
        object:
          type: string
          default: list
        data:
          type: array
          items:
            $ref: '#/components/schemas/EmbeddingObject'
        model:
          type: string
        usage:
          $ref: '#/components/schemas/Usage'

    HealthResponse:
      type: object
      required:
        - status
      properties:
        status:
          type: string
          enum: [healthy, degraded, unhealthy]
        cache_stats:
          type: object
        circuit_breakers:
          type: object
        surrealdb:
          type: string

  responses:
    BadRequest:
      description: 请求参数错误
      content:
        application/json:
          schema:
            type: object
            properties:
              type:
                type: string
              title:
                type: string
              status:
                type: integer
              detail:
                type: string
              errors:
                type: array

    RateLimitExceeded:
      description: 超出速率限制
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'

    InternalError:
      description: 服务器内部错误
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'
```

### 8.4 安全方案

```yaml
components:
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: Authorization
      description: API Key认证
      scheme: bearer
      bearerFormat: JWT

    OAuth2:
      type: oauth2
      flows:
        authorizationCode:
          authorizationUrl: https://example.com/oauth/authorize
          tokenUrl: https://example.com/oauth/token
          scopes:
            embeddings:read: 读取embedding
            embeddings:write: 创建embedding
            chat:write: 发送聊天请求
            memories:read: 读取记忆
            memories:write: 写入记忆

security:
  - ApiKeyAuth: []
  - OAuth2: []
```

---

## 九、最佳实践建议

### 9.1 API设计原则

#### 9.1.1 RESTful设计

| 约定 | 说明 |
|------|------|
| **资源命名** | 使用名词复数：`/memories`, `/embeddings` |
| **HTTP方法** | GET（读取）、POST（创建）、PUT（更新）、DELETE（删除） |
| **状态码** | 正确使用HTTP状态码 |
| **版本控制** | 使用URL路径版本：`/v1/`, `/v2/` |

#### 9.1.2 分页、过滤、排序

**分页**：

```http
GET /api/v1/memories?page=1&limit=20
```

**响应头**：

```http
X-Total-Count: 100
X-Page-Count: 5
X-Current-Page: 1
X-Per-Page: 20
```

**过滤**：

```http
GET /api/v1/memories?status=active&type=preference
```

**排序**：

```http
GET /api/v1/memories?sort=created_at&order=desc
```

### 9.2 性能优化

#### 9.2.1 缓存策略

```python
# LRU缓存 + TTL
from functools import lru_cache
from datetime import datetime, timedelta

CACHE_TTL = timedelta(hours=1)

@lru_cache(maxsize=1000)
def get_embedding(text: str) -> List[float]:
    """获取文本嵌入（带缓存）"""
    cache_key = hashlib.md5(text.encode()).hexdigest()

    # 检查缓存
    if cache_key in cache_store:
        entry = cache_store[cache_key]
        if datetime.now() - entry['timestamp'] < CACHE_TTL:
            return entry['embedding']

    # 计算embedding
    embedding = compute_embedding(text)

    # 存入缓存
    cache_store[cache_key] = {
        'embedding': embedding,
        'timestamp': datetime.now()
    }

    return embedding
```

#### 9.2.2 连接池

```python
import httpx

# HTTP连接池
http_client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    timeout=30.0
)
```

#### 9.2.3 批量处理

```python
# 批量上传记忆（推荐）
async def batch_upload_memories(memories: List[Memory], batch_size: int = 10):
    """批量上传记忆"""
    for i in range(0, len(memories), batch_size):
        batch = memories[i:i + batch_size]
        await upload_batch(batch)
```

### 9.3 安全建议

#### 9.3.1 输入验证

```python
from pydantic import BaseModel, Field, validator

class EmbeddingRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=32768)

    @validator("input")
    def validate_input(cls, v):
        # 防止XSS攻击
        if "<script>" in v.lower():
            raise ValueError("Invalid input")
        return v
```

#### 9.3.2 输出编码

```python
from fastapi.responses import JSONResponse

class SafeJSONResponse(JSONResponse):
    """安全的JSON响应（防止XSS）"""
    def render(self, content) -> bytes:
        # 转义特殊字符
        return super().render(content).replace(b"<", b"\\u003c")
```

#### 9.3.3 速率限制

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/v1/embeddings")
@limiter.limit("100/minute")
async def create_embeddings(request: Request):
    pass
```

### 9.4 监控与日志

#### 9.4.1 结构化日志

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "api_request",
    method="POST",
    endpoint="/v1/embeddings",
    status_code=200,
    duration_ms=123.45,
    request_id="req_abc123"
)
```

#### 9.4.2 Prometheus指标

```python
from prometheus_client import Counter, Histogram

# 请求计数器
request_count = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)

# 请求延迟
request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['endpoint']
)

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    with request_duration.labels(endpoint="/v1/embeddings").time():
        # 处理请求
        pass

    request_count.labels(
        method="POST",
        endpoint="/v1/embeddings",
        status="200"
    ).inc()
```

### 9.5 测试建议

#### 9.5.1 单元测试

```python
import pytest
from fastapi.testclient import TestClient

client = TestClient(app)

def test_create_embeddings():
    response = client.post(
        "/v1/embeddings",
        json={"input": "Hello, world!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) == 1
```

#### 9.5.2 集成测试

```python
import pytest
import httpx

@pytest.mark.asyncio
async def test_full_workflow():
    """测试完整工作流"""
    async with httpx.AsyncClient() as client:
        # 1. 上传记忆
        upload_response = await client.post(
            "http://localhost:3001/api/v1/memories",
            json={
                "memories": [{"content": "Test memory"}]
            }
        )
        assert upload_response.status_code == 200

        # 2. 搜索记忆
        search_response = await client.post(
            "http://localhost:3001/api/v1/memories/search",
            json={"query": "Test"}
        )
        assert search_response.status_code == 200
```

---

## 十、附录

### 10.1 环境变量配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| **Embedding服务** |||
| `EMB_MAX_BATCH_SIZE` | 自动(64-256) | 批量大小 |
| `EMB_MODEL_PATH` | Qwen/Qwen3-Embedding-0___6B | 模型路径 |
| `EMB_CACHE_SIZE` | 1000 | 缓存大小 |
| **LLM服务** |||
| `LLM_MAX_BATCH_SIZE` | 自动(1-4) | 批量大小 |
| `LLM_MODEL_PATH` | OpenBMB/MiniCPM4-0.5B | 模型路径 |
| `LLM_MAX_NEW_TOKENS` | 512-2048 | 最大生成长度 |
| `LLM_CACHE_SIZE` | 100 | 缓存大小 |
| **包装层服务** |||
| `WRAPPER_PORT` | 3001 | 服务端口 |
| `WRAPPER_EMBEDDING_SERVICE_URL` | http://localhost:18000 | Embedding服务地址 |
| `WRAPPER_LLM_SERVICE_URL` | http://localhost:18001 | LLM服务地址 |
| `WRAPPER_CACHE_MAX_SIZE` | 1000 | 缓存大小 |
| `WRAPPER_CACHE_TTL` | 3600 | 缓存TTL（秒） |
| `WRAPPER_CIRCUIT_BREAKER_THRESHOLD` | 5 | 熔断阈值 |
| `WRAPPER_CIRCUIT_BREAKER_TIMEOUT` | 60 | 熔断恢复时间（秒） |
| **SurrealDB** |||
| `WRAPPER_SURREALDB_URL` | ws://localhost:18002/rpc | 连接地址 |
| `WRAPPER_SURREALDB_NAMESPACE` | memory_ns | 命名空间 |
| `WRAPPER_SURREALDB_DATABASE` | memory_db | 数据库名 |
| `WRAPPER_SURREALDB_POOL_SIZE` | 10 | 连接池大小 |

### 10.2 错误码参考

| 错误码 | 说明 | HTTP状态码 |
|--------|------|------------|
| `EMPTY_INPUT` | 输入不能为空 | 400 |
| `TEXT_TOO_LONG` | 文本长度超限 | 400 |
| `INVALID_MODEL` | 无效的模型名称 | 400 |
| `RATE_LIMIT_EXCEEDED` | 超出速率限制 | 429 |
| `SERVICE_UNAVAILABLE` | 服务不可用 | 503 |
| `CIRCUIT_BREAKER_OPEN` | 熔断器打开 | 503 |
| `INTERNAL_ERROR` | 内部服务器错误 | 500 |

### 10.3 性能基准

| 端点 | 预期延迟 | P99延迟 | QPS |
|------|----------|---------|-----|
| `/v1/embeddings` (单条) | < 100ms | < 200ms | > 10 |
| `/v1/embeddings` (批量10条) | < 500ms | < 1000ms | > 5 |
| `/v1/chat/completions` | < 1000ms | < 2000ms | > 5 |
| `/api/v1/memories` | < 200ms | < 500ms | > 10 |
| `/api/v1/memories/search` | < 100ms | < 300ms | > 20 |

---

## 文档变更历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| 1.0.0 | 2026-03-04 | 初始版本 |

---

## 联系方式

如有问题或建议，请联系：

- **Email**: support@example.com
- **GitHub**: https://github.com/example/embedding-service
- **文档**: https://docs.example.com

---

**文档结束**
