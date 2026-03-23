# Embedding Service (后端记忆服务) 详细设计文档

**版本**: v2.3.1
**日期**: 2026-03-23
**状态**: 设计阶段
**适用**: 后端记忆服务开发

---

## 目录

1. [架构概述](#1-架构概述)
2. [核心模块设计](#2-核心模块设计)
3. [API 设计](#3-api-设计)
4. [数据存储设计](#4-数据存储设计)
5. [搜索架构](#5-搜索架构)
6. [同步与冲突解决](#6-同步与冲突解决)
7. [与插件端的交互](#7-与插件端的交互)
8. [性能优化](#8-性能优化)
9. [开发指南](#9-开发指南)
10. [部署指南](#10-部署指南)

---

## 1. 架构概述

### 1.1 设计目标

- **高性能**: 支持向量搜索(HNSW) + 全文搜索(BM25) + 图遍历的混合查询
- **高可靠**: 立即处理上传请求，失败时返回明确错误
- **可扩展**: 支持多租户、水平扩展
- **一致性**: 与插件端数据格式保持一致

### 1.2 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      客户端层                                │
│         ┌─────────────────────────────────────┐            │
│         │      OpenCode Memory CLI/Plugin     │            │
│         │         (Node.js / Bun)             │            │
│         └──────────────┬──────────────────────┘            │
└────────────────────────┼────────────────────────────────────┘
                         │ HTTP/WebSocket
┌────────────────────────┼────────────────────────────────────┐
│                      API 层 (FastAPI)                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Routers                                            │   │
│  │  ├── memories.py      # 记忆 CRUD                   │   │
│  │  ├── search.py        # 搜索接口                    │   │
│  │  ├── sync.py          # 同步接口                    │   │
│  │  ├── relations.py     # 图关系                      │   │
│  │  └── health.py        # 健康检查                    │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
├────────────────────────┼────────────────────────────────────┤
│                      服务层                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │  MemoryManager│ │SearchService │ │  SyncService        │   │
│  │             │ │             │ │                     │   │
│  │ - CRUD      │ │ - Hybrid    │ │ - Immediate         │   │
│  │ - Validation│ │   Search    │ │   Upload            │   │
│  │ - Deduplicat│ │ - RRF Merge │ │ - Conflict          │   │
│  │   ion       │ │             │ │   Resolution        │   │
│  └──────┬──────┘ └──────┬──────┘ └──────────┬──────────┘   │
│         │               │                   │               │
│         └───────────────┼───────────────────┘               │
│                         │                                   │
├─────────────────────────┼───────────────────────────────────┤
│                      存储层                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │  SurrealDB  │ │ Meilisearch │ │  Embedding Service  │   │
│  │             │ │             │ │                     │   │
│  │ - Graph     │ │ - Full Text │ │ - Qwen3             │   │
│  │ - Vector    │ │   Search    │ │   (1024-dim)        │   │
│  │   (HNSW)    │ │ - CJK       │ │                     │   │
│  │ - Relations │ │   Tokenizer │ │                     │   │
│  │ - Schema    │ │             │ │                     │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 技术栈

| 组件 | 技术 | 版本 | 说明 |
|------|------|------|------|
| API 框架 | FastAPI | 0.100+ | 异步高性能 |
| 图数据库 | SurrealDB | 3.0+ | 向量+图原生支持 |
| 全文搜索 | Meilisearch | 1.4+ | CJK 分词支持 |
| 嵌入模型 | Qwen3-Embedding | 0.6B | 1024 维向量 |
| LLM | MiniCPM4 | 0.5B | 可选对话功能 |
| 缓存 | aiocache | - | LRU 缓存 |
| 追踪 | OpenTelemetry | - | 分布式追踪 |

---

## 2. 核心模块设计

### 2.1 MemoryManager

**职责**: 记忆的生命周期管理、去重、关系维护。

```python
# wrapper/src/utils/memory_manager.py

class MemoryManager:
    """
    记忆管理器 - 核心服务类
    
    职责:
    1. 记忆的 CRUD 操作
    2. 智能去重（内容哈希 + 语义相似度）
    3. 批量操作优化
    4. 图关系管理
    5. 多租户隔离
    """
    
    def __init__(
        self,
        db: AsyncSurreal,
        embedding_service_url: str,
        meili_client: Optional[MeilisearchClient] = None,
    ):
        self._db = db
        self._embedding_url = embedding_service_url
        self._meili = meili_client
        
        # 去重阈值配置
        self._dedup_thresholds = {
            "preference": 0.88,
            "decision": 0.90,
            "long-term": 0.93,
            "general": 0.95,
            "daily": 1.0,
        }
    
    async def upload_memories(
        self,
        memories: List[Dict],
        tenant_id: str = "default",
    ) -> Dict:
        """
        批量上传记忆
        
        流程:
        1. 计算 embedding（批量）
        2. 内容哈希去重
        3. 语义相似度检测
        4. 插入 SurrealDB
        5. 同步到 Meilisearch（如果启用）
        
        返回:
        {
            "success": 成功数,
            "failed": 失败数,
            "duplicates": 重复数,
            "memory_ids": [ID列表],
            "errors": [错误列表]
        }
        """
        # 1. 批量获取 embeddings
        texts = [m["content"] for m in memories]
        embeddings = await self._get_embeddings(texts)
        
        results = {
            "success": 0,
            "failed": 0,
            "duplicates": 0,
            "memory_ids": [],
            "errors": [],
        }
        
        for i, memory in enumerate(memories):
            try:
                # 2. 检查内容哈希重复
                content_hash = hashlib.md5(
                    memory["content"].encode()
                ).hexdigest()
                
                existing = await self._check_hash_duplicate(
                    content_hash, tenant_id
                )
                if existing:
                    results["duplicates"] += 1
                    continue
                
                # 3. 语义相似度检测
                embedding = embeddings[i]
                similar = await self._check_semantic_duplicate(
                    embedding, memory.get("type", "general"), tenant_id
                )
                
                if similar["is_duplicate"]:
                    # 根据策略处理
                    action = await self._decide_duplicate_action(
                        memory, similar["existing"]
                    )
                    if action == "skip":
                        results["duplicates"] += 1
                        continue
                    elif action == "update":
                        await self._update_memory(
                            similar["existing"]["id"], memory, embedding
                        )
                        results["success"] += 1
                        continue
                
                # 4. 插入 SurrealDB
                memory_id = await self._insert_memory(
                    memory, embedding, content_hash, tenant_id
                )
                results["memory_ids"].append(memory_id)
                results["success"] += 1
                
                # 5. 同步到 Meilisearch
                if self._meili:
                    await self._sync_to_meilisearch(memory_id, memory)
                    
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "memory": memory.get("content", "")[:100],
                    "error": str(e),
                })
        
        return results
    
    async def search_memories(
        self,
        query: str,
        mode: str = "hybrid",  # keyword/vector/hybrid
        limit: int = 10,
        tenant_id: str = "default",
        **filters
    ) -> List[Dict]:
        """
        搜索记忆
        
        模式:
        - keyword: Meilisearch 全文搜索
        - vector: SurrealDB 向量搜索（HNSW）
        - hybrid: RRF 融合两者结果
        """
        if mode == "keyword":
            return await self._search_by_keyword(query, limit, tenant_id, **filters)
        elif mode == "vector":
            return await self._search_by_vector(query, limit, tenant_id, **filters)
        else:  # hybrid
            return await self._hybrid_search(query, limit, tenant_id, **filters)
    
    async def _hybrid_search(
        self,
        query: str,
        limit: int,
        tenant_id: str,
        **filters
    ) -> List[Dict]:
        """
        RRF 混合搜索
        
        公式: score = Σ (weight_i / (k + rank_i))
        k = 60 (推荐值)
        """
        # 并行执行两种搜索
        embedding = await self._get_embeddings([query])[0]
        
        vector_task = self._search_by_vector(
            embedding, limit * 2, tenant_id, **filters
        )
        keyword_task = self._search_by_keyword(
            query, limit * 2, tenant_id, **filters
        )
        
        vector_results, keyword_results = await asyncio.gather(
            vector_task, keyword_task
        )
        
        # RRF 融合
        return self._rrf_merge(
            vector_results,
            keyword_results,
            k=60,
            vector_weight=0.7,
            keyword_weight=0.3,
        )[:limit]
    
    def _rrf_merge(
        self,
        vector_results: List[Dict],
        keyword_results: List[Dict],
        k: int,
        vector_weight: float,
        keyword_weight: float,
    ) -> List[Dict]:
        """RRF (Reciprocal Rank Fusion)"""
        scores = {}
        items = {}
        
        # 向量搜索贡献
        for rank, item in enumerate(vector_results):
            doc_id = item["id"]
            scores[doc_id] = vector_weight / (k + rank + 1)
            items[doc_id] = item
        
        # 关键词搜索贡献
        for rank, item in enumerate(keyword_results):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0) + keyword_weight / (k + rank + 1)
            if doc_id not in items:
                items[doc_id] = item
        
        # 排序
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        return [
            {**items[doc_id], "score": round(scores[doc_id], 6)}
            for doc_id in sorted_ids
        ]
    
    # 图关系操作
    async def create_relation(
        self,
        from_id: str,
        to_id: str,
        relationship_type: str,
        weight: float = 0.5,
        tenant_id: str = "default",
        **metadata
    ) -> Dict:
        """创建记忆间的关系边"""
        query = f"""
        RELATE {from_id}->memory_relation->{to_id}
        SET 
            relationship_type = '{relationship_type}',
            weight = {weight},
            tenant_id = '{tenant_id}',
            created_at = time::now()
            {self._build_metadata_set(metadata)}
        """
        result = await self._db.query(query)
        return self._extract_record(result)
    
    async def get_related_memories(
        self,
        memory_id: str,
        depth: int = 1,
        tenant_id: str = "default",
        limit: int = 20,
    ) -> List[Dict]:
        """图遍历查询"""
        # 构建遍历路径
        path = "->memory_relation->memory" * depth
        
        query = f"""
        SELECT {path}.* AS related
        FROM {memory_id}
        WHERE related.tenant_id = '{tenant_id}'
        LIMIT {limit}
        """
        
        result = await self._db.query(query)
        return self._extract_records(result)
```

### 2.2 SyncService

**职责**: 处理同步请求、冲突解决。

```python
# wrapper/src/utils/sync_service.py

class SyncService:
    """
    同步服务
    
    职责:
    1. 处理增量同步请求
    2. 检测冲突
    3. 冲突持久化
    4. 冲突解决
    """
    
    def __init__(self, memory_manager: MemoryManager):
        self._mm = memory_manager
    
    async def incremental_sync(
        self,
        fingerprints: List[Dict],  # [{path, hash, mtime, source_id}]
        tenant_id: str = "default",
    ) -> Dict:
        """
        增量同步
        
        比较客户端指纹和服务端状态，返回:
        - to_upload: 需要上传的文件
        - to_delete: 服务端已删除的文件
        - conflicts: 冲突的文件（同时修改）
        """
        result = {
            "synced": 0,
            "to_upload": [],
            "to_delete": [],
            "conflicts": [],
        }
        
        # 获取服务端指纹
        server_fingerprints = await self._get_server_fingerprints(tenant_id)
        server_map = {f["source_id"]: f for f in server_fingerprints}
        client_map = {f["source_id"]: f for f in fingerprints}
        
        for client_fp in fingerprints:
            source_id = client_fp["source_id"]
            server_fp = server_map.get(source_id)
            
            if not server_fp:
                # 新文件，需要上传
                result["to_upload"].append({
                    "source_id": source_id,
                    "reason": "new",
                    "path": client_fp["path"],
                })
            elif server_fp["hash"] != client_fp["hash"]:
                # 内容不同，检查冲突
                if client_fp["mtime"] > server_fp["mtime"]:
                    # 客户端更新，上传
                    result["to_upload"].append({
                        "source_id": source_id,
                        "reason": "modified",
                        "path": client_fp["path"],
                    })
                else:
                    # 服务端更新或冲突
                    conflict_id = await self._record_conflict(
                        source_id,
                        client_fp["hash"],
                        server_fp["hash"],
                        client_fp["mtime"],
                        server_fp["mtime"],
                        tenant_id,
                    )
                    result["conflicts"].append({
                        "id": conflict_id,
                        "source_id": source_id,
                        "local_hash": client_fp["hash"],
                        "server_hash": server_fp["hash"],
                    })
            else:
                # 已同步
                result["synced"] += 1
        
        # 检查服务端有但客户端没有的文件（已删除）
        for server_fp in server_fingerprints:
            if server_fp["source_id"] not in client_map:
                result["to_delete"].append(server_fp["source_id"])
        
        return result
    
    async def resolve_conflict(
        self,
        conflict_id: str,
        resolution: str,  # use_local/use_remote/keep_both
        tenant_id: str = "default",
    ) -> Dict:
        """
        解决同步冲突
        
        策略:
        - use_local: 用客户端内容覆盖服务端
        - use_remote: 保留服务端，丢弃客户端
        - keep_both: 保留两个版本（本地重命名）
        """
        # 获取冲突详情
        conflict = await self._get_conflict(conflict_id, tenant_id)
        if not conflict:
            raise ValueError(f"Conflict not found: {conflict_id}")
        
        if resolution == "use_local":
            # 需要客户端重新上传
            await self._update_conflict_status(
                conflict_id, "resolved", resolution
            )
            return {
                "conflict_id": conflict_id,
                "resolution": resolution,
                "action": "awaiting_upload",
                "message": "Please upload local version",
            }
        
        elif resolution == "use_remote":
            # 标记为已解决，客户端应该下载服务端版本
            await self._update_conflict_status(
                conflict_id, "resolved", resolution
            )
            return {
                "conflict_id": conflict_id,
                "resolution": resolution,
                "action": "use_remote",
            }
        
        elif resolution == "keep_both":
            # 服务端保留两个版本
            # 实际实现: 重命名本地版本并上传为新记录
            await self._update_conflict_status(
                conflict_id, "resolved", resolution
            )
            return {
                "conflict_id": conflict_id,
                "resolution": resolution,
                "action": "create_new",
                "message": "Local version will be uploaded as new memory",
            }
        
        else:
            raise ValueError(f"Invalid resolution: {resolution}")
```

---

## 3. API 设计

### 3.1 RESTful API

#### 记忆管理

```python
# POST /api/v1/memories
# 批量上传记忆
{
    "memories": [
        {
            "content": "记忆内容",
            "type": "code",  # code/memory/decision/preference
            "language": "javascript",  # 代码类型
            "filePath": "/project/src/auth.js",
            "tags": ["auth", "jwt", "security"],
            "metadata": {
                "fingerprint": "md5-hash",
                "metrics": {
                    "lines": 150,
                    "complexity": 12
                }
            },
            "source_id": "optional-client-id"
        }
    ],
    "tenant_id": "default"
}

# Response
{
    "success": 5,
    "failed": 0,
    "duplicates": 2,
    "memory_ids": ["memory:abc", "memory:def"],
    "errors": []
}
```

#### 搜索

```python
# POST /api/v1/memories/search
{
    "query": "用户认证",
    "mode": "hybrid",  # keyword/vector/hybrid
    "limit": 10,
    "threshold": 0.7,
    "tenant_id": "default",
    "filters": {
        "type": "code",
        "language": "javascript",
        "tags": ["auth"]
    }
}

# Response
{
    "results": [
        {
            "id": "memory:abc",
            "content": "JWT 认证实现...",
            "score": 0.92,
            "metadata": {
                "language": "javascript",
                "filePath": "/src/auth.js"
            }
        }
    ],
    "total": 1,
    "mode": "hybrid"
}
```

#### 图关系

```python
# POST /api/v1/memories/relations
# 创建关系
{
    "from_id": "memory:abc",
    "to_id": "memory:def",
    "relationship_type": "related",  # related/follow_up/elaboration/contradiction/reference/derived_from
    "weight": 0.8,
    "description": "AuthService 使用 validateToken",
    "tenant_id": "default"
}

# POST /api/v1/memories/graph
# 图遍历
{
    "memory_id": "memory:abc",
    "depth": 2,
    "tenant_id": "default"
}
```

#### 同步

```python
# POST /api/v1/sync/incremental
{
    "fingerprints": [
        {
            "path": "/src/auth.js",
            "hash": "md5-hash",
            "mtime": 1712345678000,
            "source_id": "auth.js"
        }
    ],
    "tenant_id": "default"
}

# Response
{
    "synced": 5,
    "to_upload": [{"source_id": "new.js", "reason": "new"}],
    "to_delete": ["old.js"],
    "conflicts": [
        {
            "id": "conflict:xyz",
            "source_id": "auth.js",
            "local_hash": "abc",
            "server_hash": "def"
        }
    ]
}

# POST /api/v1/sync/conflicts/{conflict_id}/resolve
{
    "resolution": "use_local",  # use_local/use_remote/keep_both
    "tenant_id": "default"
}
```

### 3.2 WebSocket (实时推送)

```python
# /ws/memories/live?tenant_id=default

# 连接后实时接收变更通知
{
    "action": "CREATE",  # CREATE/UPDATE/DELETE
    "memory": {
        "id": "memory:abc",
        "content": "...",
        "timestamp": "2026-03-23T10:30:00Z"
    }
}
```

---

## 4. 数据存储设计

### 4.1 SurrealDB Schema

```sql
-- memory 表
DEFINE TABLE memory TYPE NORMAL SCHEMAFULL;

DEFINE FIELD content ON memory TYPE string;
DEFINE FIELD embedding ON memory TYPE array<float>;
DEFINE FIELD tenant_id ON memory TYPE string DEFAULT 'default';
DEFINE FIELD type ON memory TYPE string DEFAULT 'general';
DEFINE FIELD tags ON memory TYPE array<string>;
DEFINE FIELD metadata ON memory TYPE object FLEXIBLE;
DEFINE FIELD content_hash ON memory TYPE string;
DEFINE FIELD source_id ON memory TYPE string;
DEFINE FIELD created_at ON memory TYPE datetime DEFAULT time::now();

-- HNSW 向量索引
DEFINE INDEX memory_embedding_hnsw ON memory
    FIELDS embedding HNSW DIMENSION 1024 DIST COSINE EFC 200 M 16;

-- 全文索引 (BM25 降级路径)
DEFINE ANALYZER memory_analyzer TOKENIZERS blank, class FILTERS lowercase, ngram(2,8);
DEFINE INDEX memory_content_ft ON memory 
    FIELDS content FULLTEXT ANALYZER memory_analyzer BM25;

-- memory_relation 边表
DEFINE TABLE memory_relation TYPE RELATION IN memory OUT memory SCHEMAFULL;

DEFINE FIELD relationship_type ON memory_relation TYPE string DEFAULT 'related';
DEFINE FIELD weight ON memory_relation TYPE float DEFAULT 0.5;
DEFINE FIELD tenant_id ON memory_relation TYPE string DEFAULT 'default';
DEFINE FIELD description ON memory_relation TYPE option<string>;
DEFINE FIELD created_at ON memory_relation TYPE datetime DEFAULT time::now();

-- conflict 表
DEFINE TABLE conflict TYPE NORMAL SCHEMAFULL;

DEFINE FIELD source_id ON conflict TYPE string;
DEFINE FIELD local_hash ON conflict TYPE string;
DEFINE FIELD server_hash ON conflict TYPE string;
DEFINE FIELD tenant_id ON conflict TYPE string DEFAULT 'default';
DEFINE FIELD status ON conflict TYPE string DEFAULT 'pending';
DEFINE FIELD resolution ON conflict TYPE option<string>;
DEFINE FIELD created_at ON conflict TYPE datetime DEFAULT time::now();
DEFINE FIELD resolved_at ON conflict TYPE option<datetime>;
```

### 4.2 Meilisearch 索引配置

```json
{
  "uid": "memories",
  "primaryKey": "id",
  "searchableAttributes": [
    "content",
    "tags"
  ],
  "filterableAttributes": [
    "tenant_id",
    "type",
    "language",
    "tags"
  ],
  "sortableAttributes": [
    "created_at"
  ],
  "rankingRules": [
    "words",
    "typo",
    "proximity",
    "attribute",
    "sort",
    "exactness"
  ],
  "typoTolerance": {
    "enabled": true,
    "minWordSizeForTypos": {
      "oneTypo": 5,
      "twoTypos": 9
    }
  }
}
```

---

## 5. 搜索架构

### 5.1 三层搜索策略

| 层级 | 引擎 | 适用场景 | 复杂度 |
|------|------|---------|--------|
| 1 | Meilisearch | 全文搜索、CJK 分词 | O(log n) |
| 2 | SurrealDB HNSW | 向量相似度搜索 | O(log n) |
| 3 | RRF 融合 | 混合搜索结果 | O(n) |

### 5.2 RRF 算法实现

```python
def reciprocal_rank_fusion(
    vector_results: List[Dict],
    keyword_results: List[Dict],
    k: int = 60,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> List[Dict]:
    """
    RRF (Reciprocal Rank Fusion)
    
    score(d) = Σ (weight_i / (k + rank_i(d)))
    """
    scores = {}
    items = {}
    
    # 向量搜索贡献
    for rank, item in enumerate(vector_results):
        doc_id = item["id"]
        scores[doc_id] = vector_weight / (k + rank + 1)
        items[doc_id] = item
    
    # 关键词搜索贡献
    for rank, item in enumerate(keyword_results):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0) + keyword_weight / (k + rank + 1)
        if doc_id not in items:
            items[doc_id] = item
    
    # 排序
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    return [
        {**items[doc_id], "score": round(scores[doc_id], 6)}
        for doc_id in sorted_ids
    ]
```

---

## 6. 同步与冲突解决

### 6.1 冲突检测

```python
async def detect_conflict(
    client_fp: Dict,  # {path, hash, mtime}
    server_fp: Dict,  # {path, hash, mtime}
) -> Optional[Dict]:
    """
    检测冲突条件:
    1. 同一 source_id
    2. hash 不同（内容不同）
    3. 服务端 mtime >= 客户端 mtime（服务端也修改了）
    """
    if client_fp["hash"] != server_fp["hash"]:
        if server_fp["mtime"] >= client_fp["mtime"]:
            return {
                "type": "conflict",
                "source_id": client_fp["source_id"],
                "client_mtime": client_fp["mtime"],
                "server_mtime": server_fp["mtime"],
            }
    return None
```

### 6.2 冲突解决策略

| 策略 | 行为 | 适用场景 |
|------|------|---------|
| use_local | 客户端覆盖服务端 | 确认本地版本更新 |
| use_remote | 保留服务端 | 服务端版本更权威 |
| keep_both | 保留两个版本 | 无法决定，都保留 |

---

## 7. 与插件端的交互

### 7.1 数据流

```
插件端 (Node.js)
    │
    ├─ 1. 提取特征 (AST)
    │
    ├─ 2. 立即上传 HTTP POST /api/v1/memories
    │
    ├─ 3. 失败则入队重试
    │
    └─ 4. 搜索 HTTP POST /api/v1/memories/search
            │
            ▼
后端 (Python/FastAPI)
    │
    ├─ 5. 计算 embedding (Qwen3)
    │
    ├─ 6. 去重检测
    │
    ├─ 7. 存储 SurrealDB
    │
    ├─ 8. 索引 Meilisearch
    │
    └─ 9. 返回结果
```

### 7.2 数据格式契约

**上传请求**:

```javascript
// 插件端发送
{
  "memories": [
    {
      "content": "代码内容",
      "type": "code",
      "language": "javascript",
      "filePath": "/project/src/auth.js",
      "tags": ["validateToken", "AuthService"],
      "metadata": {
        "fingerprint": "md5",
        "metrics": { "lines": 150 }
      }
    }
  ]
}

// 后端处理
{
  "content": "代码内容",
  "embedding": [0.1, 0.2, ...],  // 1024-dim
  "type": "code",
  "language": "javascript",
  "tags": ["validateToken", "AuthService"],
  "metadata": {
    "filePath": "/project/src/auth.js",
    "fingerprint": "md5",
    "metrics": { "lines": 150 }
  },
  "content_hash": "md5",
  "tenant_id": "default",
  "created_at": "2026-03-23T10:30:00Z"
}
```

---

## 8. 性能优化

### 8.1 批量操作

```python
# 批量获取 embeddings（避免逐个请求）
async def batch_get_embeddings(texts: List[str]) -> List[List[float]]:
    response = await http_client.post(
        f"{embedding_service_url}/v1/embeddings",
        json={"input": texts, "model": "Qwen3-Embedding-0.6B"}
    )
    return [item["embedding"] for item in response.json()["data"]]

# 批量插入 SurrealDB
async def batch_insert_memories(memories: List[Dict]):
    query = "INSERT INTO memory $memories"
    await db.query(query, {"memories": memories})
```

### 8.2 缓存策略

```python
# 查询结果缓存 (5分钟)
@cached(ttl=300)
async def search_memories(query: str, mode: str, tenant_id: str):
    ...

# Embedding 缓存
@cached(ttl=3600)
async def get_embedding(text: str):
    ...
```

### 8.3 连接池

```python
# HTTP 连接池
http_pool = aiohttp.ClientConnectorPool(
    limit=100,
    limit_per_host=20,
)
```

---

## 9. 开发指南

### 9.1 环境设置

```bash
# 克隆项目
git clone https://github.com/longray/embedding-service.git
cd embedding-service

# 安装依赖
uv pip install -e ".[dev]"

# 启动服务
uv run python -m wrapper.src.main

# 运行测试
uv run pytest tests/ -v
```

### 9.2 添加新功能

1. 在 `wrapper/src/utils/` 添加服务类
2. 在 `wrapper/src/main.py` 添加 API 端点
3. 在 `tests/` 添加测试
4. 更新文档

### 9.3 调试

```python
# 开启详细日志
export LOG_LEVEL=DEBUG

# 追踪性能
with tracer.start_as_current_span("operation"):
    result = await operation()
```

---

## 10. 部署指南

### 10.1 Docker Compose

```yaml
version: '3.8'

services:
  embedding-service:
    build: .
    ports:
      - "17999:17999"
    environment:
      - SURREAL_URL=ws://surrealdb:8000
      - MEILI_URL=http://meilisearch:7700
    depends_on:
      - surrealdb
      - meilisearch

  surrealdb:
    image: surrealdb/surrealdb:latest
    command: start --user root --pass root
    volumes:
      - surreal-data:/data

  meilisearch:
    image: getmeili/meilisearch:v1.4
    volumes:
      - meili-data:/meili_data

volumes:
  surreal-data:
  meili-data:
```

### 10.2 环境变量

```bash
# 必需
export SURREAL_URL=ws://localhost:8000
export SURREAL_NS=memory_ns
export SURREAL_DB=memory_db
export EMBEDDING_SERVICE_URL=http://localhost:18000

# 可选
export MEILI_URL=http://localhost:7700
export MEILI_API_KEY=optional
export WRAPPER_AUTH_ENABLED=false
export LOG_LEVEL=INFO
```

---

## 附录

### A. 与插件端的数据契约

| 字段 | 插件端 | 后端 | 说明 |
|------|--------|------|------|
| content | ✅ | ✅ | 记忆内容 |
| type | ✅ | ✅ | 类型: code/memory/... |
| language | ✅ | ✅ | 编程语言 |
| filePath | metadata | metadata | 文件路径 |
| tags | ✅ | ✅ | 标签列表 |
| fingerprint | metadata | content_hash | 内容指纹 |
| metrics | metadata | metadata | 代码指标 |
| embedding | ❌ | ✅ | 向量（后端计算）|

### B. 错误码

| 状态码 | 含义 | 处理 |
|--------|------|------|
| 200 | 成功 | - |
| 400 | 请求错误 | 检查参数 |
| 409 | 重复 | 忽略或更新 |
| 429 | 限流 | 稍后重试 |
| 500 | 服务端错误 | 重试 |
| 503 | 服务不可用 | 稍后重试 |

---

**文档位置**: `embedding-service/docs/BACKEND_DESIGN.md`
