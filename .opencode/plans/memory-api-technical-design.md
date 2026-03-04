# 记忆管理 API - 技术设计文档

**文档版本**: 1.0  
**创建时间**: 2026-03-03  
**状态**: 待实施  
**设计人**: Kiro AI Assistant

**基于**:
- 功能设计文档 v2.0
- 评审报告（2轮）
- 向量维度验证报告

---

## 一、技术架构

### 1.1 技术栈

**已确认**:
- Python 3.11+
- FastAPI 0.109.0
- httpx 0.26.0（异步HTTP客户端）
- pydantic 2.9+（数据验证）
- structlog 24.1.0（日志）
- prometheus-client 0.19.0（监控）

**新增**:
- surrealdb>=0.3.0（SurrealDB Python SDK）

**向量配置**:
- 向量维度: 1024（已验证）
- 距离函数: COSINE
- 索引类型: HNSW

### 1.2 模块结构

```
wrapper-service/
├── src/
│   ├── main.py                      # FastAPI应用（已存在）
│   ├── config.py                    # 配置管理（需更新）
│   └── utils/
│       ├── surrealdb_client.py      # 新增：SurrealDB客户端
│       ├── memory_manager.py        # 新增：记忆管理器
│       ├── http_pool.py             # 已存在：HTTP连接池
│       ├── cache.py                 # 已存在：缓存
│       ├── circuit_breaker.py       # 已存在：熔断器
│       ├── logging.py               # 已存在：日志
│       ├── metrics.py               # 已存在：监控
│       └── exceptions.py            # 已存在：异常
├── requirements.txt                 # 需更新
└── data/
    └── memory.db                    # 新增：SurrealDB数据文件
```

---

## 二、核心类设计

### 2.1 SurrealDBClient 类

**文件**: `wrapper-service/src/utils/surrealdb_client.py`

**职责**: 封装SurrealDB连接和基本操作

**类定义**:
```python
from typing import Optional
from surrealdb import AsyncSurreal
import structlog

logger = structlog.get_logger()

class SurrealDBClient:
    """SurrealDB客户端封装"""
    
    def __init__(
        self,
        url: str,
        namespace: str,
        database: str,
        username: str,
        password: str
    ):
        self.url = url
        self.namespace = namespace
        self.database = database
        self.username = username
        self.password = password
        self.db: Optional[AsyncSurreal] = None
    
    async def connect(self) -> None:
        """连接到SurrealDB"""
        self.db = AsyncSurreal(self.url)
        await self.db.connect()
        await self.db.use(self.namespace, self.database)
        await self.db.signin({"username": self.username, "password": self.password})
        logger.info("SurrealDB连接成功", url=self.url)
    
    async def disconnect(self) -> None:
        """断开连接"""
        if self.db:
            await self.db.close()
            logger.info("SurrealDB连接已关闭")
    
    async def health_check(self) -> dict:
        """健康检查"""
        try:
            result = await self.db.query("SELECT * FROM memory LIMIT 1")
            return {"status": "healthy", "response_time_ms": 10}
        except Exception as e:
            logger.error("SurrealDB健康检查失败", error=str(e))
            return {"status": "unhealthy", "error": str(e)}
    
    async def create_memory(self, memory: dict) -> str:
        """创建记忆"""
        result = await self.db.create("memory", memory)
        return result[0]["id"]
    
    async def get_memory(self, memory_id: str) -> Optional[dict]:
        """获取记忆"""
        result = await self.db.select(memory_id)
        return result[0] if result else None
    
    async def search_by_vector(
        self,
        embedding: list[float],
        limit: int,
        threshold: float,
        filters: Optional[dict] = None
    ) -> list[dict]:
        """向量搜索"""
        query = """
            SELECT id, content, metadata, 
                   vector::distance::knn() AS score
            FROM memory
            WHERE embedding <|$limit|> $embedding
              AND score >= $threshold
        """
        
        # 添加过滤条件
        if filters:
            if "tags" in filters:
                query += " AND metadata.tags CONTAINS $tag"
            if "type" in filters:
                query += " AND metadata.type = $type"
        
        query += " ORDER BY score DESC"
        
        params = {
            "embedding": embedding,
            "limit": limit,
            "threshold": threshold,
            **filters if filters else {}
        }
        
        result = await self.db.query(query, params)
        return result[0] if result else []
    
    async def search_by_keyword(
        self,
        query: str,
        limit: int,
        filters: Optional[dict] = None
    ) -> list[dict]:
        """关键词搜索"""
        sql = """
            SELECT id, content, metadata,
                   search::score(1) AS score
            FROM memory
            WHERE content @1@ $query
        """
        
        if filters:
            if "tags" in filters:
                sql += " AND metadata.tags CONTAINS $tag"
            if "type" in filters:
                sql += " AND metadata.type = $type"
        
        sql += " ORDER BY score DESC LIMIT $limit"
        
        params = {"query": query, "limit": limit, **filters if filters else {}}
        result = await self.db.query(sql, params)
        return result[0] if result else []
    
    async def hybrid_search(
        self,
        query: str,
        embedding: list[float],
        limit: int,
        threshold: float,
        filters: Optional[dict] = None
    ) -> list[dict]:
        """混合搜索（RRF融合）"""
        sql = """
            LET $query_vec = $embedding;
            LET $vs = SELECT id FROM memory 
                      WHERE embedding <|$limit_x2|> $query_vec;
            LET $ft = SELECT id, search::score(1) AS score 
                      FROM memory
                      WHERE content @1@ $query 
                      ORDER BY score DESC 
                      LIMIT $limit_x2;
            RETURN search::rrf([$vs, $ft], $limit, 60);
        """
        
        params = {
            "embedding": embedding,
            "query": query,
            "limit": limit,
            "limit_x2": limit * 2,
            "threshold": threshold
        }
        
        result = await self.db.query(sql, params)
        return result[0] if result else []
    
    async def process_entity(self, entity: dict) -> str:
        """处理实体（去重+合并）"""
        sql = """
            LET $existing = (SELECT * FROM entity 
                             WHERE name = $name AND type = $type);
            IF $existing {
                UPDATE $existing[0].id 
                SET attributes = object::merge(attributes, $new_attrs)
                RETURN id;
            } ELSE {
                CREATE entity CONTENT {
                    name: $name,
                    type: $type,
                    attributes: $new_attrs
                }
                RETURN id;
            };
        """
        
        params = {
            "name": entity["name"],
            "type": entity["type"],
            "new_attrs": entity.get("attributes", {})
        }
        
        result = await self.db.query(sql, params)
        return result[0][0]["id"]
    
    async def create_relation(self, relation: dict) -> str:
        """创建关系（去重）"""
        sql = """
            LET $existing = (SELECT * FROM relation 
                             WHERE in = $in AND out = $out AND type = $type);
            IF $existing {
                UPDATE $existing[0].id 
                SET properties.co_occurrence_count += 1
                RETURN id;
            } ELSE {
                CREATE relation CONTENT {
                    in: $in,
                    out: $out,
                    type: $type,
                    properties: { co_occurrence_count: 1 }
                }
                RETURN id;
            };
        """
        
        result = await self.db.query(sql, relation)
        return result[0][0]["id"]
```

**预计代码量**: 200-250行

---

## 三、MemoryManager 类

**文件**: `wrapper-service/src/utils/memory_manager.py`

**职责**: 记忆管理核心逻辑

**类定义**:
```python
from typing import Optional
import httpx
import structlog
from .surrealdb_client import SurrealDBClient

logger = structlog.get_logger()

class MemoryManager:
    """记忆管理器"""
    
    def __init__(
        self,
        surrealdb_client: SurrealDBClient,
        embedding_service_url: str,
        http_client: httpx.AsyncClient,
        vector_dimension: int = 1024
    ):
        self.db = surrealdb_client
        self.embedding_url = embedding_service_url
        self.http_client = http_client
        self.vector_dimension = vector_dimension
    
    async def upload_memories(
        self,
        memories: list[dict],
        batch_size: int = 10
    ) -> dict:
        """批量上传记忆"""
        results = []
        uploaded = 0
        failed = 0
        
        # 分批处理
        for i in range(0, len(memories), batch_size):
            batch = memories[i:i+batch_size]
            
            try:
                # 1. 提取文本内容
                texts = [m["content"] for m in batch]
                
                # 2. 批量生成向量
                embeddings = await self._batch_embed(texts, batch_size)
                
                # 3. 处理每条记忆
                for memory, embedding in zip(batch, embeddings):
                    try:
                        # 验证向量维度
                        if len(embedding) != self.vector_dimension:
                            raise ValueError(
                                f"向量维度不匹配: 期望{self.vector_dimension}，"
                                f"实际{len(embedding)}"
                            )
                        
                        # 处理实体
                        entity_ids = []
                        if "entities" in memory:
                            entity_ids = await self._process_entities(
                                memory["entities"]
                            )
                        
                        # 创建记忆
                        memory_data = {
                            "content": memory["content"],
                            "embedding": embedding,
                            "metadata": memory.get("metadata", {}),
                            "entities": entity_ids
                        }
                        memory_id = await self.db.create_memory(memory_data)
                        
                        # 创建关系
                        if entity_ids:
                            await self._create_relations(memory_id, entity_ids)
                        
                        results.append({
                            "id": memory_id,
                            "status": "success",
                            "entities_created": len(entity_ids)
                        })
                        uploaded += 1
                        
                    except Exception as e:
                        logger.error("记忆处理失败", error=str(e))
                        results.append({
                            "status": "failed",
                            "error": str(e)
                        })
                        failed += 1
                        
            except Exception as e:
                logger.error("批次处理失败", error=str(e))
                failed += len(batch)
                for _ in batch:
                    results.append({
                        "status": "failed",
                        "error": str(e)
                    })
        
        return {
            "success": True,
            "uploaded": uploaded,
            "failed": failed,
            "results": results
        }
    
    async def search_memories(
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 10,
        offset: int = 0,
        threshold: float = 0.7,
        filters: Optional[dict] = None
    ) -> list[dict]:
        """搜索记忆"""
        
        if mode == "vector":
            # 向量搜索
            embedding = await self._generate_embedding(query)
            results = await self.db.search_by_vector(
                embedding, limit, threshold, filters
            )
        
        elif mode == "keyword":
            # 关键词搜索
            results = await self.db.search_by_keyword(query, limit, filters)
        
        elif mode == "hybrid":
            # 混合搜索
            embedding = await self._generate_embedding(query)
            results = await self.db.hybrid_search(
                query, embedding, limit, threshold, filters
            )
        
        else:
            raise ValueError(f"不支持的搜索模式: {mode}")
        
        # 应用分页
        return results[offset:offset+limit]
    
    async def _batch_embed(
        self,
        texts: list[str],
        batch_size: int
    ) -> list[list[float]]:
        """批量生成向量"""
        response = await self.http_client.post(
            f"{self.embedding_url}/v1/embeddings",
            json={
                "input": texts,
                "model": "Qwen3-Embedding-0.6B"
            },
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]
    
    async def _generate_embedding(self, text: str) -> list[float]:
        """生成单个向量"""
        embeddings = await self._batch_embed([text], 1)
        return embeddings[0]
    
    async def _process_entities(self, entities: list[dict]) -> list[str]:
        """处理实体列表"""
        entity_ids = []
        for entity in entities:
            entity_id = await self.db.process_entity(entity)
            entity_ids.append(entity_id)
        return entity_ids
    
    async def _create_relations(
        self,
        memory_id: str,
        entity_ids: list[str]
    ) -> None:
        """创建关系"""
        # 1. 创建CONTAINS关系（记忆→实体）
        for entity_id in entity_ids:
            await self.db.create_relation({
                "in": memory_id,
                "out": entity_id,
                "type": "CONTAINS"
            })
        
        # 2. 创建RELATED_TO关系（实体→实体）
        for i, entity_id1 in enumerate(entity_ids):
            for entity_id2 in entity_ids[i+1:]:
                await self.db.create_relation({
                    "in": entity_id1,
                    "out": entity_id2,
                    "type": "RELATED_TO"
                })
```

**预计代码量**: 300-400行

---

## 四、数据库初始化脚本

**文件**: `wrapper-service/scripts/init_surrealdb.surql`

```surql
-- ==================== Memory 表 ====================
DEFINE TABLE memory SCHEMAFULL;

-- 字段定义
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

-- 向量索引（HNSW）
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

-- ==================== Entity 表 ====================
DEFINE TABLE entity SCHEMAFULL;

-- 字段定义
DEFINE FIELD name ON entity TYPE string 
  ASSERT string::len($value) >= 1 AND string::len($value) <= 100;

DEFINE FIELD type ON entity TYPE string;
DEFINE FIELD attributes ON entity TYPE object;
DEFINE FIELD created_at ON entity TYPE datetime DEFAULT time::now();

-- 唯一索引（去重）
DEFINE INDEX entity_unique_idx ON entity FIELDS name, type UNIQUE;

-- 类型索引
DEFINE INDEX entity_type_idx ON entity FIELDS type;

-- ==================== Relation 表 ====================
DEFINE TABLE relation SCHEMAFULL;

-- 字段定义
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

## 五、配置更新

**文件**: `wrapper-service/src/config.py`

**添加配置**:
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # ... 现有配置 ...
    
    # SurrealDB配置
    surrealdb_url: str = "file://data/memory.db"
    surrealdb_namespace: str = "embedding_service"
    surrealdb_database: str = "memories"
    surrealdb_username: str = "root"
    surrealdb_password: str = "root"
    
    # 记忆管理配置
    vector_dimension: int = 1024
    memory_batch_size: int = 10
    memory_search_limit: int = 10
    memory_similarity_threshold: float = 0.7
    
    class Config:
        env_prefix = "WRAPPER_"
        env_file = ".env"
```

---

**文档状态**: 第一部分完成  
**下一部分**: API端点实现、部署配置

---

## 六、API端点实现

### 6.1 健康检查端点

**文件**: `wrapper-service/src/main.py`

**添加代码**:
```python
@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    import time
    
    # 检查包装层
    wrapper_status = {
        "status": "healthy",
        "uptime_seconds": int(time.time() - app.state.start_time)
    }
    
    # 检查Embedding服务
    embedding_status = await check_service(
        config.embedding_service_url + "/health"
    )
    
    # 检查LLM服务
    llm_status = await check_service(
        config.llm_service_url + "/health"
    )
    
    # 检查SurrealDB
    surrealdb_status = await app.state.surrealdb.health_check()
    
    # 判断整体状态
    all_healthy = all([
        wrapper_status["status"] == "healthy",
        embedding_status["status"] == "healthy",
        surrealdb_status["status"] == "healthy"
    ])
    
    status_code = 200 if all_healthy else 503
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if all_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "services": {
                "wrapper": wrapper_status,
                "embedding": embedding_status,
                "llm": llm_status,
                "surrealdb": surrealdb_status
            }
        }
    )

async def check_service(url: str) -> dict:
    """检查服务健康状态"""
    try:
        start = time.time()
        response = await app.state.http_client.get(url, timeout=5.0)
        response_time = (time.time() - start) * 1000
        
        return {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "url": url,
            "response_time_ms": round(response_time, 2)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "url": url,
            "error": str(e)
        }
```

### 6.2 记忆上传端点

**添加Pydantic模型**:
```python
from pydantic import BaseModel, Field
from typing import Optional

class EntityInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    type: str
    attributes: Optional[dict] = {}

class MemoryInput(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)
    metadata: Optional[dict] = {}
    entities: Optional[list[EntityInput]] = []

class UploadRequest(BaseModel):
    memories: list[MemoryInput] = Field(..., min_items=1, max_items=100)
    options: Optional[dict] = {"batch_size": 10}
```

**添加端点**:
```python
@app.post("/api/v1/memories")
async def upload_memories(request: UploadRequest):
    """批量上传记忆"""
    try:
        result = await app.state.memory_manager.upload_memories(
            memories=[m.dict() for m in request.memories],
            batch_size=request.options.get("batch_size", 10)
        )
        
        return {
            "success": True,
            "uploaded": result["uploaded"],
            "failed": result["failed"],
            "results": result["results"],
            "processing_time_ms": result.get("processing_time_ms", 0)
        }
        
    except Exception as e:
        logger.error("记忆上传失败", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
```

### 6.3 记忆搜索端点

**添加Pydantic模型**:
```python
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: str = Field("hybrid", pattern="^(vector|keyword|hybrid)$")
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0)
    threshold: float = Field(0.7, ge=0.0, le=1.0)
    filters: Optional[dict] = None
```

**添加端点**:
```python
@app.post("/api/v1/memories/search")
async def search_memories(request: SearchRequest):
    """搜索记忆"""
    try:
        results = await app.state.memory_manager.search_memories(
            query=request.query,
            mode=request.mode,
            limit=request.limit,
            offset=request.offset,
            threshold=request.threshold,
            filters=request.filters
        )
        
        return {
            "success": True,
            "results": results,
            "total": len(results),
            "limit": request.limit,
            "offset": request.offset
        }
        
    except Exception as e:
        logger.error("记忆搜索失败", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
```

### 6.4 记忆详情端点

**添加端点**:
```python
@app.get("/api/v1/memories/{memory_id}")
async def get_memory(memory_id: str):
    """获取记忆详情"""
    try:
        memory = await app.state.surrealdb.get_memory(memory_id)
        
        if not memory:
            raise HTTPException(status_code=404, detail="记忆不存在")
        
        return {
            "success": True,
            "memory": memory
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("记忆查询失败", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
```

### 6.5 应用启动和关闭

**更新main.py**:
```python
@app.on_event("startup")
async def startup_event():
    """应用启动"""
    app.state.start_time = time.time()
    
    # 初始化HTTP客户端
    app.state.http_client = httpx.AsyncClient()
    
    # 初始化SurrealDB客户端
    app.state.surrealdb = SurrealDBClient(
        url=config.surrealdb_url,
        namespace=config.surrealdb_namespace,
        database=config.surrealdb_database,
        username=config.surrealdb_username,
        password=config.surrealdb_password
    )
    await app.state.surrealdb.connect()
    
    # 初始化MemoryManager
    app.state.memory_manager = MemoryManager(
        surrealdb_client=app.state.surrealdb,
        embedding_service_url=config.embedding_service_url,
        http_client=app.state.http_client,
        vector_dimension=config.vector_dimension
    )
    
    logger.info("应用启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭"""
    await app.state.surrealdb.disconnect()
    await app.state.http_client.aclose()
    logger.info("应用关闭完成")
```

---

## 七、部署配置

### 7.1 环境变量

**文件**: `wrapper-service/.env.example`

```bash
# 服务配置
WRAPPER_PORT=3001
WRAPPER_LOG_LEVEL=INFO

# 后端服务
WRAPPER_EMBEDDING_SERVICE_URL=http://localhost:18000
WRAPPER_LLM_SERVICE_URL=http://localhost:18001

# SurrealDB配置
WRAPPER_SURREALDB_URL=file://data/memory.db
WRAPPER_SURREALDB_NAMESPACE=embedding_service
WRAPPER_SURREALDB_DATABASE=memories
WRAPPER_SURREALDB_USERNAME=root
WRAPPER_SURREALDB_PASSWORD=root

# 记忆管理配置
WRAPPER_VECTOR_DIMENSION=1024
WRAPPER_MEMORY_BATCH_SIZE=10
WRAPPER_MEMORY_SEARCH_LIMIT=10
WRAPPER_MEMORY_SIMILARITY_THRESHOLD=0.7

# 缓存配置
WRAPPER_CACHE_MAX_SIZE=1000
WRAPPER_CACHE_TTL=3600
```

### 7.2 依赖更新

**文件**: `wrapper-service/requirements.txt`

**添加**:
```txt
surrealdb>=0.3.0
```

### 7.3 数据库初始化

**创建初始化脚本**: `wrapper-service/scripts/init_db.py`

```python
import asyncio
from surrealdb import AsyncSurreal

async def init_database():
    """初始化数据库"""
    db = AsyncSurreal("file://data/memory.db")
    await db.connect()
    await db.use("embedding_service", "memories")
    await db.signin({"username": "root", "password": "root"})
    
    # 读取SQL脚本
    with open("scripts/init_surrealdb.surql", "r") as f:
        sql = f.read()
    
    # 执行初始化
    await db.query(sql)
    
    print("数据库初始化完成")
    await db.close()

if __name__ == "__main__":
    asyncio.run(init_database())
```

### 7.4 启动脚本

**文件**: `wrapper-service/scripts/start.sh`

```bash
#!/bin/bash

# 检查数据目录
mkdir -p data

# 初始化数据库（首次运行）
if [ ! -f "data/memory.db" ]; then
    echo "初始化数据库..."
    python scripts/init_db.py
fi

# 启动服务
echo "启动包装层服务..."
python -m src.main
```

---

## 八、测试计划

### 8.1 单元测试

**文件**: `wrapper-service/tests/test_surrealdb_client.py`

```python
import pytest
from src.utils.surrealdb_client import SurrealDBClient

@pytest.mark.asyncio
async def test_connect():
    """测试连接"""
    client = SurrealDBClient(
        url="mem://",
        namespace="test",
        database="test",
        username="root",
        password="root"
    )
    await client.connect()
    assert client.db is not None
    await client.disconnect()

@pytest.mark.asyncio
async def test_create_memory():
    """测试创建记忆"""
    client = Su
