# 记忆管理 API - 功能设计文档 v2.0

**文档版本**: 2.0  
**创建时间**: 2026-03-03  
**状态**: 已优化（基于第一轮评审）  
**设计人**: Kiro AI Assistant

**变更说明**:
- 添加记忆详情查询功能（F9）
- 添加实体查询功能（F10）
- 添加关系查询功能（F11）
- 明确实体去重策略
- 明确SIMILAR_TO创建策略（按需创建）
- 调整性能目标（更现实）
- 统一API路径规范（/api/v1/）
- 定义统一错误响应格式
- 添加分页支持

---

## 一、功能概述

### 1.1 产品定位

记忆管理系统是包装层服务的核心功能模块，为用户提供智能记忆存储、检索和知识图谱构建能力。系统通过向量化技术和图数据库，实现语义级别的记忆管理。

### 1.2 核心价值

1. **智能存储**: 自动将文本记忆转换为向量表示，支持语义搜索
2. **知识图谱**: 自动提取实体和关系，构建结构化知识网络
3. **混合搜索**: 结合向量搜索和关键词搜索，提供最佳检索效果
4. **统一监控**: 集成健康检查，实时监控所有服务状态

### 1.3 目标用户

- AI应用开发者
- 知识管理系统构建者
- 需要语义搜索能力的应用

---

## 二、功能需求

### 2.1 功能列表

| 功能ID | 功能名称 | 优先级 | 描述 | 变更 |
|--------|----------|--------|------|------|
| F1 | 健康检查 | P0 | 监控所有服务健康状态 | - |
| F2 | 记忆上传 | P0 | 批量上传记忆并生成向量 | - |
| F3 | 记忆搜索 | P0 | 多模式搜索记忆 | 添加分页 |
| F4 | 实体管理 | P0 | 自动提取和管理实体 | 明确去重策略 |
| F5 | 关系构建 | P0 | 构建实体间关系 | 明确创建规则 |
| F6 | 知识图谱查询 | P0 | 查询实体关系网络 | P1→P0 |
| F7 | 记忆更新 | P1 | 更新已存储的记忆 | - |
| F8 | 记忆删除 | P1 | 删除指定记忆 | - |
| F9 | 记忆详情查询 | P0 | 根据ID获取记忆详情 | 新增 |
| F10 | 实体查询 | P1 | 查询实体列表 | 新增 |
| F11 | 关系查询 | P1 | 查询实体关系 | 新增 |
| F12 | 相似记忆关联 | P1 | 按需创建SIMILAR_TO关系 | 新增 |

### 2.2 核心用户故事

#### US-1: 健康检查

**作为** 系统管理员  
**我想要** 查看所有服务的健康状态  
**以便** 及时发现和处理服务故障

**验收标准**:
- [ ] 能够查看包装层服务状态
- [ ] 能够查看Embedding服务状态
- [ ] 能够查看LLM服务状态
- [ ] 能够查看SurrealDB状态
- [ ] 响应时间 < 100ms
- [ ] 显示每个服务的响应时间

#### US-2: 批量上传记忆

**作为** 应用开发者  
**我想要** 批量上传多条记忆  
**以便** 快速构建知识库

**验收标准**:
- [ ] 支持一次上传多条记忆（最多100条）
- [ ] 自动生成1024维向量表示
- [ ] 自动提取实体（用户提供）
- [ ] 自动构建实体关系（CONTAINS, RELATED_TO）
- [ ] 实体自动去重（name+type唯一）
- [ ] 返回上传结果统计
- [ ] 支持部分失败处理
- [ ] 批量处理时间 < 5s（10条记忆）

#### US-3: 混合搜索记忆

**作为** 应用用户  
**我想要** 通过自然语言查询相关记忆  
**以便** 快速找到需要的信息

**验收标准**:
- [ ] 支持3种搜索模式（vector/keyword/hybrid）
- [ ] 默认使用hybrid模式
- [ ] 返回语义和关键词综合相关的记忆
- [ ] 按RRF分数排序
- [ ] 支持相似度阈值过滤（默认0.7）
- [ ] 支持结果数量限制（默认10，最大100）
- [ ] 支持分页（offset/limit）
- [ ] 搜索响应时间 < 800ms

#### US-4: 记忆详情查询

**作为** 应用开发者  
**我想要** 根据ID获取记忆的完整信息  
**以便** 展示记忆详情和关联实体

**验收标准**:
- [ ] 支持按ID查询记忆
- [ ] 返回完整的记忆信息（包含向量）
- [ ] 返回关联的实体列表
- [ ] 响应时间 < 100ms

#### US-5: 实体和关系查询

**作为** 知识管理员  
**我想要** 查询实体和关系  
**以便** 了解知识图谱结构

**验收标准**:
- [ ] 支持查询实体列表（分页）
- [ ] 支持按类型过滤实体
- [ ] 支持查询实体的关系
- [ ] 支持图遍历（N度关系）
- [ ] 响应时间 < 500ms

---

## 三、API接口设计

### 3.1 API路径规范

**统一前缀**: `/api/v1/`

**资源命名**:
- 记忆: `/memories`
- 实体: `/entities`
- 关系: `/relations`

### 3.2 健康检查接口

**端点**: `GET /api/v1/health`

**功能**: 返回所有服务的健康状态

**响应示例**:
```json
{
  "status": "healthy",
  "timestamp": "2026-03-03T17:00:00Z",
  "services": {
    "wrapper": {
      "status": "healthy",
      "uptime_seconds": 3600
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
- 503: 至少一个核心服务不健康

### 3.3 记忆上传接口

**端点**: `POST /api/v1/memories`

**功能**: 批量上传记忆，自动生成向量并构建知识图谱

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
            "paradigm": "multi-paradigm",
            "typing": "dynamic"
          }
        }
      ]
    }
  ],
  "options": {
    "batch_size": 10,
    "create_similar_relations": false
  }
}
```

**参数说明**:
- `memories`: 记忆列表（1-100条）
- `options.batch_size`: 批处理大小（默认10）
- `options.create_similar_relations`: 是否创建SIMILAR_TO关系（默认false）

**响应示例**:
```json
{
  "success": true,
  "uploaded": 1,
  "failed": 0,
  "results": [
    {
      "id": "memory:uuid1",
      "status": "success",
      "entities_created": 1,
      "relations_created": 0
    }
  ],
  "processing_time_ms": 500
}
```

**状态码**:
- 200: 上传成功（包括部分失败）
- 400: 请求参数错误
- 500: 服务器错误

### 3.4 记忆搜索接口

**端点**: `POST /api/v1/memories/search`

**功能**: 搜索记忆，支持向量、关键词、混合搜索

**请求体**:
```json
{
  "query": "什么是Python",
  "mode": "hybrid",
  "limit": 10,
  "offset": 0,
  "threshold": 0.7,
  "filters": {
    "tags": ["programming"],
    "type": "knowledge",
    "created_after": "2026-01-01T00:00:00Z"
  }
}
```

**参数说明**:
- `query`: 搜索查询（必填）
- `mode`: 搜索模式（vector/keyword/hybrid，默认hybrid）
- `limit`: 返回数量（1-100，默认10）
- `offset`: 偏移量（默认0）
- `threshold`: 相似度阈值（0.0-1.0，默认0.7）
- `filters`: 过滤条件（可选）

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
      "entities": ["entity:python"],
      "created_at": "2026-03-03T17:00:00Z"
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0,
  "processing_time_ms": 100
}
```

**状态码**:
- 200: 搜索成功
- 400: 请求参数错误
- 500: 服务器错误

### 3.5 记忆详情查询接口

**端点**: `GET /api/v1/memories/{id}`

**功能**: 根据ID获取记忆详情

**路径参数**:
- `id`: 记忆ID（如 memory:uuid1）

**响应示例**:
```json
{
  "success": true,
  "memory": {
    "id": "memory:uuid1",
    "content": "Python是一种高级编程语言",
    "embedding": [0.1, 0.2, ..., 1024维],
    "metadata": {
      "type": "knowledge",
      "tags": ["programming", "python"]
    },
    "entities": [
      {
        "id": "entity:python",
        "name": "Python",
        "type": "programming_language"
      }
    ],
    "created_at": "2026-03-03T17:00:00Z",
    "updated_at": "2026-03-03T17:00:00Z"
  }
}
```

**状态码**:
- 200: 查询成功
- 404: 记忆不存在
- 500: 服务器错误

### 3.6 实体查询接口

**端点**: `GET /api/v1/entities`

**功能**: 查询实体列表

**查询参数**:
- `type`: 实体类型过滤（可选）
- `limit`: 返回数量（默认10，最大100）
- `offset`: 偏移量（默认0）

**响应示例**:
```json
{
  "success": true,
  "entities": [
    {
      "id": "entity:python",
      "name": "Python",
      "type": "programming_language",
      "attributes": {
        "paradigm": "multi-paradigm"
      },
      "memory_count": 5,
      "created_at": "2026-03-03T17:00:00Z"
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

**状态码**:
- 200: 查询成功
- 400: 请求参数错误
- 500: 服务器错误

### 3.7 关系查询接口

**端点**: `GET /api/v1/entities/{id}/relations`

**功能**: 查询实体的关系

**路径参数**:
- `id`: 实体ID

**查询参数**:
- `type`: 关系类型过滤（可选）
- `depth`: 遍历深度（1-3，默认1）
- `limit`: 返回数量（默认10，最大100）

**响应示例**:
```json
{
  "success": true,
  "entity": {
    "id": "entity:python",
    "name": "Python"
  },
  "relations": [
    {
      "id": "relation:uuid1",
      "type": "RELATED_TO",
      "target": {
        "id": "entity:programming",
        "name": "Programming"
      },
      "properties": {
        "co_occurrence_count": 3
      }
    }
  ],
  "total": 1
}
```

**状态码**:
- 200: 查询成功
- 404: 实体不存在
- 500: 服务器错误

### 3.8 相似记忆关联接口

**端点**: `POST /api/v1/memories/{id}/similar`

**功能**: 为指定记忆创建SIMILAR_TO关系

**路径参数**:
- `id`: 记忆ID

**请求体**:
```json
{
  "threshold": 0.9,
  "limit": 10
}
```

**响应示例**:
```json
{
  "success": true,
  "memory_id": "memory:uuid1",
  "similar_memories": [
    {
      "id": "memory:uuid2",
      "similarity": 0.95
    }
  ],
  "relations_created": 1
}
```

**状态码**:
- 200: 创建成功
- 404: 记忆不存在
- 500: 服务器错误

### 3.9 统一错误响应格式

**错误响应结构**:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "错误描述",
    "details": {
      "field": "具体信息"
    },
    "timestamp": "2026-03-03T17:00:00Z"
  }
}
```

**错误码定义**:
- `INVALID_INPUT`: 输入参数错误
- `EMBEDDING_SERVICE_TIMEOUT`: Embedding服务超时
- `EMBEDDING_SERVICE_ERROR`: Embedding服务错误
- `DATABASE_ERROR`: 数据库错误
- `MEMORY_NOT_FOUND`: 记忆不存在
- `ENTITY_NOT_FOUND`: 实体不存在
- `VECTOR_DIMENSION_MISMATCH`: 向量维度不匹配
- `INTERNAL_ERROR`: 内部错误

---

## 四、功能详细设计

### 4.1 健康检查功能（F1）

#### 4.1.1 检查逻辑

**并行检查**:
```python
async def health_check():
    results = await asyncio.gather(
        check_wrapper_service(),
        check_embedding_service(),
        check_llm_service(),
        check_surrealdb(),
        return_exceptions=True
    )
    return aggregate_results(results)
```

**状态判断**:
- `healthy`: 所有服务正常
- `degraded`: LLM服务异常（非核心）
- `unhealthy`: Embedding或SurrealDB异常

#### 4.1.2 超时控制

- 单个服务检查超时: 5s
- 总体超时: 10s

### 4.2 记忆上传功能（F2）

#### 4.2.1 处理流程

```
1. 输入验证
   - 记忆数量: 1-100
   - 内容长度: 1-10000字符
   - 实体格式验证

2. 批量处理（batch_size=10）
   For each batch:
     a. 提取文本内容
     b. 调用Embedding服务（批量）
     c. 验证向量维度（1024）
     d. 开始事务
     e. 存储记忆
     f. 处理实体（去重+合并）
     g. 创建关系（CONTAINS, RELATED_TO）
     h. 提交事务
     i. 记录结果

3. 汇总结果
   - 成功数量
   - 失败数量
   - 详细结果列表
```

#### 4.2.2 实体去重策略

**唯一性约束**: `(name, type)` 组合唯一

**冲突处理**:
```python
if entity_exists(name, type):
    # 合并attributes
    existing_attrs = get_entity_attributes(name, type)
    merged_attrs = {**existing_attrs, **new_attrs}
    update_entity_attributes(name, type, merged_attrs)
    return existing_entity_id
else:
    # 创建新实体
    return create_entity(name, type, attrs)
```

#### 4.2.3 关系创建规则

**CONTAINS关系** (记忆→实体):
- 自动创建
- 每个记忆的每个实体创建一条

**RELATED_TO关系** (实体→实体):
- 同一记忆中的实体两两创建
- 去重: 避免重复创建
- properties:
  - `memory_id`: 来源记忆ID
  - `co_occurrence_count`: 共现次数（累加）

**示例**:
```
记忆: "Python和Java都是编程语言"
实体: [Python, Java, 编程语言]

创建关系:
- memory:1 -CONTAINS-> entity:python
- memory:1 -CONTAINS-> entity:java
- memory:1 -CONTAINS-> entity:programming_language
- entity:python -RELATED_TO-> entity:java (co_occurrence_count=1)
- entity:python -RELATED_TO-> entity:programming_language (co_occurrence_count=1)
- entity:java -RELATED_TO-> entity:programming_language (co_occurrence_count=1)
```

#### 4.2.4 事务处理

**使用SurrealDB事务确保原子性**:
```python
async with db.transaction() as tx:
    memory_id = await tx.create_memory(memory_data)
    entity_ids = await tx.process_entities(entities)
    await tx.create_relations(memory_id, entity_ids)
    await tx.commit()
```

**失败回滚**:
- 如果任何步骤失败，回滚整个事务
- 记录失败原因
- 继续处理下一条记忆

### 4.3 记忆搜索功能（F3）

#### 4.3.1 搜索模式实现

**向量搜索** (vector):
```surql
LET $query_vec = [0.1, 0.2, ..., 1024];
SELECT id, content, metadata, 
       vector::distance::knn() AS score
FROM memory
WHERE embedding <|$limit|> $query_vec
  AND score >= $threshold
ORDER BY score DESC;
```

**关键词搜索** (keyword):
```surql
SELECT id, content, metadata,
       search::score(1) AS score
FROM memory
WHERE content @1@ $query
ORDER BY score DESC
LIMIT $limit;
```

**混合搜索** (hybrid):
```surql
-- 向量搜索
LET $vs = SELECT id FROM memory 
          WHERE embedding <|$limit*2|> $query_vec;

-- 关键词搜索
LET $ft = SELECT id, search::score(1) AS score 
          FROM memory
          WHERE content @1@ $query 
          ORDER BY score DESC 
          LIMIT $limit*2;

-- RRF融合
search::rrf([$vs, $ft], $limit, 60);
```

**RRF算法**:
- 公式: `score = Σ(1 / (k + rank_i))`
- k值: 60（SurrealDB默认）
- 融合权重: 向量和关键词各占50%

#### 4.3.2 过滤条件

**支持的过滤字段**:
```surql
WHERE embedding <|$limit|> $query_vec
  AND metadata.tags CONTAINS $tag
  AND metadata.type = $type
  AND metadata.source = $source
  AND created_at >= $created_after
  AND created_at <= $created_before
```

#### 4.3.3 分页实现

```surql
SELECT * FROM memory
WHERE ...
ORDER BY score DESC
LIMIT $limit
START $offset;
```

**响应中包含分页信息**:
- `total`: 总结果数
- `limit`: 当前页大小
- `offset`: 当前偏移量

---

## 五、数据模型（优化版）

### 5.1 Memory表

```surql
DEFINE TABLE memory SCHEMAFULL;

DEFINE FIELD content ON memory TYPE string 
  ASSERT string::len($value) >= 1 AND string::len($value) <= 10000;
DEFINE FIELD embedding ON memory TYPE array<float> 
  ASSERT array::len($value) = 1024;
DEFINE FIELD metadata ON memory TYPE object;
DEFINE FIELD metadata.type ON memory TYPE string;
DEFINE FIELD metadata.source ON memory TYPE string;
DEFINE FIELD metadata.tags ON memory TYPE array<string>;
DEFINE FIELD entities ON memory TYPE array<record<entity>>;
DEFINE FIELD created_at ON memory TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON memory TYPE datetime DEFAULT time::now();

-- 向量索引
DEFINE INDEX memory_embedding_idx ON memory 
  FIELDS embedding 
  HNSW DIMENSION 1024 DIST COSINE;

-- 全文索引
DEFINE ANALYZER memory_analyzer TOKENIZERS class FILTERS lowercase, ascii;
DEFINE INDEX memory_content_idx ON memory 
  FIELDS content 
  FULLTEXT ANALYZER memory_analyzer BM25;

-- 元数据索引
DEFINE INDEX memory_type_idx ON memory FIELDS metadata.type;
DEFINE INDEX memory_created_idx ON memory FIELDS created_at;
```

### 5.2 Entity表

```surql
DEFINE TABLE entity SCHEMAFULL;

DEFINE FIELD name ON entity TYPE string 
  ASSERT string::len($value) >= 1 AND string::len($value) <= 100;
DEFINE FIELD type ON entity TYPE string;
DEFINE FIELD attributes ON entity TYPE object;
DEFINE FIELD created_at ON entity TYPE datetime DEFAULT time::now();

-- 唯一索引（去重）
DEFINE INDEX entity_unique_idx ON entity FIELDS name, type UNIQUE;

-- 类型索引
DEFINE INDEX entity_type_idx ON entity FIELDS type;
```

### 5.3 Relation表

```surql
DEFINE TABLE relation SCHEMAFULL;

DEFINE FIELD in ON relation TYPE record;
DEFINE FIELD out ON relation TYPE record;
DEFINE FIELD type ON relation TYPE string;
DEFINE FIELD properties ON relation TYPE object;
DEFINE FIELD created_at ON relation TYPE datetime DEFAULT time::now();

-- 关系类型索引
DEFINE INDEX relation_type_idx ON relation FIELDS type;

-- 复合索引（去重）
DEFINE INDEX relation_unique_idx ON relation FIELDS in, out, type UNIQUE;
```

---

## 六、性能要求（调整版）

### 6.1 响应时间目标

| 操作 | 目标时间 | 最大时间 | 变更 |
|------|----------|----------|------|
| 健康检查 | < 100ms | < 500ms | - |
| 上传单条记忆 | < 500ms | < 2s | - |
| 上传10条记忆 | < 3s | < 5s | 最大时间: 10s→5s |
| 向量搜索 | < 300ms | < 1s | - |
| 关键词搜索 | < 100ms | < 500ms | - |
| 混合搜索 | < 800ms | < 2s | 目标: 500ms→800ms |
| 记忆详情查询 | < 100ms | < 500ms | 新增 |
| 实体查询 | < 200ms | < 1s | 新增 |
| 关系查询 | < 500ms | < 2s | 新增 |

**调整说明**:
- 混合搜索目标时间从500ms调整为800ms（更现实）
- 批量上传最大时间从10s调整为5s（更严格）

### 6.2 吞吐量目标

| 操作 | 目标QPS | 备注 |
|------|---------|------|
| 健康检查 | 100 | 轻量级操作 |
| 记忆上传 | 10 | 受Embedding服务限制 |
| 记忆搜索 | 50 | 受SurrealDB性能限制 |
| 记忆详情查询 | 100 | 简单查询 |

### 6.3 数据规模目标

| 指标 | 初期目标 | 长期目标 |
|------|----------|----------|
| 记忆总数 | 10,000 | 1,000,000 |
| 实体总数 | 5,000 | 100,000 |
| 关系总数 | 20,000 | 5,000,000 |
| 向量维度 | 1024 | 1024 |

---

**文档状态**: v2.0 - 第一部分完成  
**下一部分**: 安全要求、可扩展性、验收标准

---

## 七、安全要求

### 7.1 输入验证

**记忆内容**:
- 长度: 1-10000字符
- 禁止: SQL注入特殊字符（通过参数化查询防护）

**实体**:
- 名称长度: 1-100字符
- 类型: 预定义枚举或自定义
- 数量限制: 每条记忆最多20个实体

**批量上传**:
- 数量限制: 1-100条记忆
- 总大小限制: 10MB

**搜索参数**:
- limit: 1-100
- offset: >= 0
- threshold: 0.0-1.0

**向量维度**:
- 严格验证: 必须为1024维
- 类型验证: 必须为float数组

### 7.2 速率限制

| 端点 | 限制 | 时间窗口 |
|------|------|----------|
| POST /api/v1/memories | 10 req | 1分钟 |
| POST /api/v1/memories/search | 60 req | 1分钟 |
| GET /api/v1/memories/{id} | 100 req | 1分钟 |
| GET /api/v1/health | 无限制 | - |

**超限响应**:
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "请求过于频繁，请稍后再试",
    "details": {
      "retry_after_seconds": 60
    }
  }
}
```

### 7.3 错误信息安全

**不暴露**:
- 数据库内部结构
- 服务器路径和配置
- 内部实现细节
- 敏感数据

**错误信息示例**:
- ✅ "记忆不存在"
- ❌ "SELECT * FROM memory WHERE id='xxx' 返回空结果"

### 7.4 资源保护

**超时控制**:
- Embedding服务: 5s
- SurrealDB查询: 10s
- 总体请求: 30s

**连接池限制**:
- HTTP连接池: 最大100连接
- SurrealDB连接池: 最大50连接

**熔断器保护**:
- 失败阈值: 5次/10s
- 熔断时长: 30s
- 半开状态: 允许1个请求测试

---

## 八、可扩展性设计

### 8.1 未来功能规划

**Q1 2026**:
- [ ] F13: 自动实体提取（NER模型集成）
- [ ] F14: 记忆更新功能（PUT /api/v1/memories/{id}）
- [ ] F15: 记忆删除功能（DELETE /api/v1/memories/{id}）

**Q2 2026**:
- [ ] F16: 关系推理（基于图算法）
- [ ] F17: 实体重要性计算（PageRank）
- [ ] F18: 知识图谱可视化API

**Q3 2026**:
- [ ] F19: 多模态记忆（图片、音频）
- [ ] F20: 记忆版本管理
- [ ] F21: 协作编辑

**Q4 2026**:
- [ ] F22: 分布式部署支持
- [ ] F23: 多租户隔离
- [ ] F24: 高级分析和报表

### 8.2 架构扩展点

**水平扩展**:
- 包装层服务: 无状态，可多实例部署
- SurrealDB: 支持集群模式（TiKV后端）
- Embedding服务: 可部署多实例+负载均衡

**垂直扩展**:
- 增加服务器资源（CPU、内存）
- 优化HNSW索引参数
- 使用更大的向量维度

**缓存策略**:
- 热门查询结果缓存（Redis）
- 实体信息缓存（内存LRU）
- 向量缓存（避免重复计算）

### 8.3 性能优化方向

**索引优化**:
- 调整HNSW参数（M, EFC）
- 添加复合索引（metadata字段）
- 定期重建索引

**批处理优化**:
- 动态调整batch_size
- 并行处理多个批次
- 使用异步I/O

**查询优化**:
- 查询结果缓存
- 预计算热门查询
- 使用物化视图

---

## 九、验收标准

### 9.1 功能验收

**P0功能**（必须完成）:
- [ ] F1: 健康检查 - 所有验收标准通过
- [ ] F2: 记忆上传 - 所有验收标准通过
- [ ] F3: 记忆搜索 - 所有验收标准通过
- [ ] F4: 实体管理 - 实体去重正常工作
- [ ] F5: 关系构建 - CONTAINS和RELATED_TO正确创建
- [ ] F6: 知识图谱查询 - 图遍历正常工作
- [ ] F9: 记忆详情查询 - 按ID查询正常

**P1功能**（可延后）:
- [ ] F7: 记忆更新
- [ ] F8: 记忆删除
- [ ] F10: 实体查询
- [ ] F11: 关系查询
- [ ] F12: 相似记忆关联

### 9.2 性能验收

**响应时间**:
- [ ] 健康检查 < 100ms（平均）
- [ ] 上传10条记忆 < 5s（平均）
- [ ] 混合搜索 < 800ms（平均）
- [ ] 所有操作最大响应时间不超过目标2倍

**吞吐量**:
- [ ] 健康检查 >= 100 QPS
- [ ] 记忆上传 >= 10 QPS
- [ ] 记忆搜索 >= 50 QPS

**数据规模**:
- [ ] 支持10,000条记忆
- [ ] 搜索性能不随数据量线性下降
- [ ] HNSW索引正常工作

### 9.3 质量验收

**测试覆盖**:
- [ ] 单元测试覆盖率 >= 80%
- [ ] 集成测试覆盖所有API端点
- [ ] 性能测试覆盖关键场景
- [ ] 错误场景测试完整

**代码质量**:
- [ ] 代码符合PEP 8规范
- [ ] 所有函数有类型注解
- [ ] 关键逻辑有注释
- [ ] 无明显代码重复

**文档完整**:
- [ ] API文档完整准确
- [ ] 部署文档清晰
- [ ] 配置说明详细
- [ ] 故障排查指南

---

## 十、风险和依赖

### 10.1 技术风险

| 风险 | 影响 | 概率 | 缓解措施 | 责任人 |
|------|------|------|----------|--------|
| SurrealDB性能不足 | 高 | 中 | 性能测试，优化索引，考虑替代方案 | 开发团队 |
| Embedding服务超时 | 中 | 中 | 增加超时控制，批处理，熔断器 | 开发团队 |
| 向量维度不匹配 | 高 | 低 | 严格输入验证，配置检查 | 开发团队 |
| 知识图谱复杂度过高 | 中 | 中 | 限制关系数量，异步处理 | 开发团队 |
| 内存占用过大 | 中 | 低 | 监控内存，优化缓存策略 | 运维团队 |

### 10.2 依赖服务

| 服务 | 依赖程度 | 可用性要求 | 降级方案 | SLA |
|------|----------|------------|----------|-----|
| Embedding服务 | 高 | 99% | 熔断器保护，返回错误 | 2s响应 |
| SurrealDB | 高 | 99.9% | 无（核心依赖） | 100ms查询 |
| LLM服务 | 低 | 95% | 不影响记忆功能 | 5s响应 |

### 10.3 数据风险

**数据一致性**:
- 风险: 事务失败导致数据不一致
- 缓解: 使用SurrealDB事务，失败回滚

**数据丢失**:
- 风险: 数据库故障导致数据丢失
- 缓解: 定期备份，使用持久化存储

**数据隔离**:
- 风险: 多用户数据混淆
- 缓解: 添加租户ID，查询时过滤

---

## 十一、实施计划

### 11.1 开发阶段

**Phase 1: 基础设施（2-3小时）**:
- [ ] 添加SurrealDB依赖
- [ ] 创建SurrealDBClient类
- [ ] 更新配置管理
- [ ] 初始化数据库表结构
- [ ] 编写连接测试

**Phase 2: 核心模块（4-6小时）**:
- [ ] 创建MemoryManager类
- [ ] 实现批量向量转换
- [ ] 实现实体去重逻辑
- [ ] 实现关系创建逻辑
- [ ] 实现搜索逻辑（3种模式）
- [ ] 编写单元测试

**Phase 3: API端点（3-4小时）**:
- [ ] 实现健康检查端点
- [ ] 实现记忆上传端点
- [ ] 实现记忆搜索端点
- [ ] 实现记忆详情端点
- [ ] 添加请求验证
- [ ] 添加错误处理

**Phase 4: 测试和优化（2-3小时）**:
- [ ] 端到端测试
- [ ] 性能测试
- [ ] 错误场景测试
- [ ] 文档更新
- [ ] 代码审查

**总计**: 11-16小时

### 11.2 测试计划

**单元测试**:
- SurrealDBClient: 连接、查询、事务
- MemoryManager: 上传、搜索、实体处理
- API端点: 请求验证、响应格式

**集成测试**:
- 端到端上传流程
- 端到端搜索流程
- 实体去重验证
- 关系创建验证

**性能测试**:
- 1,000条记忆搜索性能
- 10,000条记忆搜索性能
- 批量上传性能
- 并发请求测试

**错误场景测试**:
- Embedding服务超时
- SurrealDB连接失败
- 向量维度不匹配
- 输入参数错误

---

## 十二、总结

### 12.1 文档变更总结

**v2.0相比v1.0的主要变更**:

1. **功能补充**（4个新功能）:
   - F9: 记忆详情查询
   - F10: 实体查询
   - F11: 关系查询
   - F12: 相似记忆关联

2. **逻辑优化**:
   - 明确实体去重策略（name+type唯一）
   - 明确SIMILAR_TO创建策略（按需创建）
   - 明确RELATED_TO创建规则（两两创建+共现计数）
   - 添加事务处理确保数据一致性

3. **API规范**:
   - 统一路径前缀（/api/v1/）
   - 统一错误响应格式
   - 添加分页支持
   - 添加速率限制

4. **性能调整**:
   - 混合搜索目标: 500ms → 800ms
   - 批量上传最大时间: 10s → 5s

5. **安全增强**:
   - 添加输入验证规则
   - 添加速率限制
   - 添加资源保护机制

### 12.2 关键设计决策

1. **实体去重**: 使用(name, type)唯一约束，冲突时合并attributes
2. **SIMILAR_TO关系**: 按需创建（独立API），避免上传时性能影响
3. **搜索模式**: 默认hybrid，提供最佳效果
4. **API版本**: 使用/api/v1/前缀，支持未来版本演进
5. **事务处理**: 使用SurrealDB事务确保数据一致性

### 12.3 待确认事项

**已确认**:
- ✅ SurrealDB部署方式: 本地嵌入式（文件模式）
- ✅ 向量维度: 1024
- ✅ 知识图谱复杂度: 完整版
- ✅ 默认搜索模式: hybrid

**无需确认**:
- 所有设计决策已在评审中优化
- 可以直接进入技术设计阶段

### 12.4 下一步

1. **进行第二轮评审**: 技术可行性评审
2. **创建技术设计文档**: 详细的技术实现方案
3. **创建API规格文档**: OpenAPI规范
4. **创建测试计划文档**: 详细的测试用例

---

**文档状态**: v2.0 - 完整版  
**评审状态**: 待第二轮评审（技术可行性）  
**下一文档**: 技术设计文档 v1.0
