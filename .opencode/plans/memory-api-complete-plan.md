# 记忆管理 API 完整实施计划

**计划时间**: 2026-03-03  
**计划人**: Kiro AI Assistant  
**版本**: 2.0 (Complete)  
**状态**: 待用户确认后开始实施

---

## 一、项目概述

### 1.1 目标

为包装层服务添加记忆管理功能，实现：
1. **健康检查** (`/api/health`) - 统一监控所有服务状态
2. **记忆上传** (`/api/upload`) - 批量转换向量并存储到知识图谱
3. **记忆搜索** (`/api/search`) - 向量搜索和关键词搜索

### 1.2 技术栈确认

**现有依赖**:
- FastAPI 0.109.0
- httpx 0.26.0
- pydantic 2.9+
- structlog 24.1.0
- prometheus-client 0.19.0

**新增依赖**:
- `surrealdb>=0.3.0` - SurrealDB Python SDK (支持异步)
- 无需numpy（SurrealDB原生支持向量）

### 1.3 SurrealDB 技术确认

**向量搜索能力** ✅:
- 原生支持向量存储和搜索
- HNSW索引（近似最近邻）
- 多种距离函数：EUCLIDEAN, COSINE, MANHATTAN
- KNN查询：`WHERE embedding <|k|> $query_vector`

**混合搜索** ✅:
- 支持全文搜索 + 向量搜索
- RRF算法融合结果：`search::rrf()`

**图数据库特性** ✅:
- 支持记录链接和关系
- 适合构建知识图谱

---

## 二、数据模型设计

### 2.1 SurrealDB 表结构

#### Memory 表（记忆）

```surql
-- 定义 Memory 表
DEFINE TABLE memory SCHEMAFULL;

-- 定义字段
DEFINE FIELD content ON memory TYPE string;
DEFINE FIELD embedding ON memory TYPE array;
DEFINE FIELD metadata ON memory TYPE object;
DEFINE FIELD entities ON memory TYPE array;
DEFINE FIELD created_at ON memory TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON memory TYPE datetime DEFAULT time::now();

-- 定义向量索引（HNSW）
-- 假设向量维度为768（需要根据实际embedding模型确认）
DEFINE INDEX memory_embedding_idx ON memory 
  FIELDS embedding 
  HNSW DIMENSION 768 
  DIST COSINE;

-- 定义全文索引（用于混合搜索）
DEFINE ANALYZER memory_analyzer TOKENIZERS class FILTERS lowercase, ascii;
DEFINE INDEX memory_content_idx ON memory 
  FIELDS content 
  FULLTEXT ANALYZER memory_analyzer BM25;
```

**字段说明**:
- `content`: 记忆文本内容
- `embedding`: 向量表示（由Embedding Service生成）
- `metadata`: 元数据（类型、来源、标签等）
- `entities`: 提取的实体列表
- `created_at`: 创建时间
- `updated_at`: 更新时间

#### Entity 表（实体）

```surql
-- 定义 Entity 表
DEFINE TABLE entity SCHEMAFULL;

-- 定义字段
DEFINE FIELD name ON entity TYPE string;
DEFINE FIELD type ON entity TYPE string;
DEFINE FIELD attributes ON entity TYPE object;
DEFINE FIELD created_at ON entity TYPE datetime DEFAULT time::now();

-- 定义唯一索引
DEFINE INDEX entity_name_type_idx ON entity FIELDS name, type UNIQUE;
```

**字段说明**:
- `name`: 实体名称
- `type`: 实体类型（person, place, concept等）
- `attributes`: 实体属性集合
- `created_at`: 创建时间

#### Relation 表（关系 - 知识图谱边）

```surql
-- 定义 Relation 表（边表）
DEFINE TABLE relation SCHEMAFULL;

-- 定义字段
DEFINE FIELD in ON relation TYPE record(entity);
DEFINE FIELD out ON relation TYPE record(entity);
DEFINE FIELD type ON relation TYPE string;
DEFINE FIELD properties ON relation TYPE object;
DEFINE FIELD created_at ON relation TYPE datetime DEFAULT time::now();
```

**字段说明**:
- `in`: 源实体
- `out`: 目标实体
- `type`: 关系类型（RELATED_TO, SIMILAR_TO等）
- `properties`: 关系属性（如相似度分数）
- `created_at`: 创建时间

### 2.2 数据模型示例

```json
{
  "memory:uuid1": {
    "content": "Python是一种高级编程语言",
    "embedding": [0.1, 0.2, ..., 0.768],
    "metadata": {
      "type": "knowledge",
      "source": "user_input",
      "tags": ["programming", "python"]
    },
    "entities": ["entity:python", "entity:programming_language"],
    "created_at": "2026-03-03T17:00:00Z"
  },
  "entity:python": {
    "name": "Python",
    "type": "programming_language",
    "attributes": {
      "paradigm": "multi-paradigm",
      "typing": "dynamic"
    }
  }
}
```

---

## 三、API 接口设计

### 3.1 健康检查接口

**端点**: `GET /api/health`

**功能**: 返回包装层、Embedding服务、LLM服务、SurrealDB的健康状态

**响应示例**:
```json
{
  "status": "healthy",
  "timestamp": "2026-03-03T17:00:00Z",
  "services": {
    "wrapper": {
      "status": "healthy",
      "uptime": 3600
    },
    "embedding": {
      "status": "healthy",
      "url": "http://localhost:18000",
      "response_time_ms": 50
    },
    "llm": {
      "status": "healthy",
      "url": "http://localhost:18001",
      "response_time_ms": 100
    },
    "surrealdb": {
      "status": "healthy",
      "url": "ws://localhost:8000",
      "response_time_ms": 10
    }
  }
}
```

**状态码**:
- 200: 所有服务健康
- 503: 至少一个服务不健康

### 3.2 记忆上传接口

**端点**: `POST /api/upload`

**功能**: 批量上传记忆，自动生成向量并存储到知识图谱

**请求体**:
```json
{
  "memories": [
    {
      "content": "Python是一种高级编程语言",
      "metadata": {
        "type": "knowledge",
        "source": "user_input",
        "tags": ["programming", "python"]
      },
      "entities": [
        {
          "name": "Python",
          "type": "programming_language",
          "attributes": {
            "paradigm": "multi-paradigm"
          }
        }
      ]
    }
  ],
  "batch_size": 10
}
```

**响应示例**:
```json
{
  "success": true,
  "uploaded": 1,
  "failed": 0,
  "results": [
    {
      "id": "memory:uuid1",
      "status": "success"
    }
  ],
  "processing_time_ms": 500
}
```

**状态码**:
- 200: 上传成功
- 400: 请求参数错误
- 500: 服务器错误

### 3.3 记忆搜索接口

**端点**: `POST /api/search`

**功能**: 搜索记忆，支持向量搜索、关键词搜索、混合搜索

**请求体**:
```json
{
  "query": "什么是Python",
  "mode": "hybrid",
  "limit": 10,
  "threshold": 0.7,
  "filters": {
    "tags": ["programming"]
  }
}
```

**参数说明**:
- `query`: 搜索查询
- `mode`: 搜索模式（vector/keyword/hybrid）
- `limit`: 返回结果数量
- `threshold`: 相似度阈值（0-1）
- `filters`: 过滤条件

**响应示例**:
```json
{
  "success": true,
  "results": [
    {
      "id": "memory:uuid1",
      "content": "Python是一种高级编程语言",
      "score": 0.95,
      "metadata": {
        "type": "knowledge",
        "tags": ["programming", "python"]
      },
      "entities": ["entity:python"]
    }
  ],
  "total": 1,
  "processing_time_ms": 100
}
```

**状态码**:
- 200: 搜索成功
- 400: 请求参数错误
- 500: 服务器错误

---

## 四、核心模块设计

### 4.1 SurrealDBClient 类

**文件**: `wrapper-service/src/utils/surrealdb_client.py`

**职责**: 封装SurrealDB连接和操作

**核心方法**:
```python
class SurrealDBClient:
    async def connect(self) -> None
    async def disconnect(self) -> None
    async def health_check(self) -> dict
    async def create_memory(self, memory: dict) -> str
    async def create_entity(self, entity: dict) -> str
    async def create_relation(self, relation: dict) -> str
    async def search_by_vector(
        self, 
        embedding: list[float], 
        limit: int, 
        threshold: float,
        filters: dict = None
    ) -> list[dict]
    async def search_by_keyword(
        self, 
        query: str, 
        limit: int,
        filters: dict = None
    ) -> list[dict]
    async def hybrid_search(
        self, 
        query: str,
        embedding: list[float],
        limit: int,
        threshold: float,
        filters: dict = None
    ) -> list[dict]
```

### 4.2 MemoryManager 类

**文件**: `wrapper-service/src/utils/memory_manager.py`

**职责**: 记忆管理核心逻辑

**核心方法**:
```python
class MemoryManager:
    def __init__(
        self, 
        surrealdb_client: SurrealDBClient,
        embedding_service_url: str,
        http_client: httpx.AsyncClient
    )
    
    async def upload_memories(
        self, 
        memories: list[dict], 
        batch_size: int = 10
    ) -> dict
    
    async def search_memories(
        self, 
        query: str, 
        mode: str,
        limit: int,
        threshold: float,
        filters: dict = None
    ) -> list[dict]
    
    async def _batch_embed(
        self, 
        texts: list[str], 
        batch_size: int
    ) -> list[list[float]]
    
    async def _build_knowledge_graph(
        self, 
        memory_id: str,
        entities: list[dict]
    ) -> None
```

---

## 五、实施清单

### Phase 1: 基础设施（2-3小时）

- [ ] 1.1 添加SurrealDB依赖到requirements.txt
- [ ] 1.2 更新config.py添加SurrealDB配置
- [ ] 1.3 创建SurrealDBClient类
- [ ] 1.4 编写SurrealDB连接测试
- [ ] 1.5 初始化数据库表结构（SQL脚本）

### Phase 2: MemoryManager核心模块（4-6小时）

- [ ] 2.1 创建MemoryManager类框架
- [ ] 2.2 实现批量向量转换逻辑
- [ ] 2.3 实现记忆上传逻辑
- [ ] 2.4 实现向量搜索逻辑
- [ ] 2.5 实现混合搜索逻辑
- [ ] 2.6 实现知识图谱构建逻辑
- [ ] 2.7 编写单元测试

### Phase 3: API端点实现（3-4小时）

- [ ] 3.1 实现/api/health端点
- [ ] 3.2 实现/api/upload端点
- [ ] 3.3 实现/api/search端点
- [ ] 3.4 添加请求验证（Pydantic模型）
- [ ] 3.5 添加错误处理
- [ ] 3.6 添加日志记录

### Phase 4: 测试和优化（2-3小时）

- [ ] 4.1 端到端测试
- [ ] 4.2 性能测试
- [ ] 4.3 错误场景测试
- [ ] 4.4 文档更新
- [ ] 4.5 代码审查和优化

**总计**: 11-16小时

---

## 六、关键决策点（需要用户确认）

### 6.1 SurrealDB部署

**问题**: SurrealDB如何部署？

**选项**:
1. **本地嵌入式** (推荐快速开发)
   - 使用 `mem://` 或 `file://` 协议
   - 无需额外安装
   - 数据存储在本地文件
   
2. **本地服务器**
   - 使用Docker运行SurrealDB
   - 使用 `ws://localhost:8000` 连接
   - 需要配置认证

3. **远程服务**
   - 使用SurrealDB Cloud或自建服务器
   - 需要配置URL和认证

**建议**: 开发阶段使用本地嵌入式，生产环境使用服务器模式

### 6.2 向量维度

**问题**: Embedding模型的向量维度是多少？

**需要确认**:
- 查看Embedding Service的输出
- 确认向量维度（常见：384, 768, 1024, 1536）
- 更新HNSW索引的DIMENSION参数

**建议**: 先测试获取一个embedding，确认维度

### 6.3 知识图谱复杂度

**问题**: 是否需要实现实体关系和SIMILAR_TO关系？

**选项**:
1. **简化版** (推荐MVP)
   - 只存储记忆和向量
   - 不构建实体和关系
   - 快速实现

2. **完整版**
   - 存储实体、属性、关系
   - 构建知识图谱
   - 支持图查询

**建议**: 先实现简化版，后续迭代添加图功能

### 6.4 搜索模式默认值

**问题**: 默认使用哪种搜索模式？

**选项**:
- `vector`: 纯向量搜索（快速，语义相关）
- `keyword`: 纯关键词搜索（精确匹配）
- `hybrid`: 混合搜索（最佳效果，稍慢）

**建议**: 默认使用 `hybrid` 模式

---

## 七、配置示例

### 7.1 环境变量

```bash
# SurrealDB配置
export WRAPPER_SURREALDB_URL=ws://localhost:8000
export WRAPPER_SURREALDB_NAMESPACE=embedding_service
export WRAPPER_SURREALDB_DATABASE=memories
export WRAPPER_SURREALDB_USERNAME=root
export WRAPPER_SURREALDB_PASSWORD=root

# 记忆管理配置
export WRAPPER_MEMORY_BATCH_SIZE=10
export WRAPPER_MEMORY_SEARCH_LIMIT=10
export WRAPPER_MEMORY_SIMILARITY_THRESHOLD=0.7
```

### 7.2 config.py 更新

```python
class Settings(BaseSettings):
    # ... 现有配置 ...
    
    # SurrealDB配置
    surrealdb_url: str = "ws://localhost:8000"
    surrealdb_namespace: str = "embedding_service"
    surrealdb_database: str = "memories"
    surrealdb_username: str = "root"
    surrealdb_password: str = "root"
    
    # 记忆管理配置
    memory_batch_size: int = 10
    memory_search_limit: int = 10
    memory_similarity_threshold: float = 0.7
```

---

## 八、风险和缓解

### 8.1 技术风险

**风险1**: SurrealDB Python SDK不稳定
- **缓解**: 使用稳定版本（>=0.3.0），添加错误处理

**风险2**: 向量搜索性能问题
- **缓解**: 使用HNSW索引，限制返回结果数量

**风险3**: 批量embedding超时
- **缓解**: 分批处理，添加超时控制

### 8.2 数据风险

**风险1**: 向量维度不匹配
- **缓解**: 在上传前验证向量维度

**风险2**: 数据库连接失败
- **缓解**: 添加重试机制和熔断器

---

## 九、下一步行动

### 9.1 立即确认

请用户确认以下关键决策：
1. ✅ SurrealDB部署方式（建议：本地嵌入式）
2. ✅ 向量维度（需要测试确认）
3. ✅ 知识图谱复杂度（建议：简化版MVP）
4. ✅ 默认搜索模式（建议：hybrid）

### 9.2 确认后开始实施

1. 创建feature分支
2. 按照实施清单逐步实现
3. 每个Phase完成后进行测试
4. 最终集成测试和文档更新

---

**计划状态**: Complete - 等待用户确认关键决策  
**预计工作量**: 11-16小时  
**建议开始时间**: 用户确认后立即开始
