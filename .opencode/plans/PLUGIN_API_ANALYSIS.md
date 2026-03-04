# OpenCode Memory Plugin API 需求分析

**分析时间**: 2026-03-04  
**分析人**: Kiro AI Assistant  
**目标**: 评估我设计的记忆管理API是否能满足opencode-memory-plugin的需求

---

## 一、项目概述

### 1.1 opencode-memory-plugin 简介

**项目类型**: OpenCode记忆插件  
**当前版本**: v1.2.0  
**未来版本**: v2.0（设计阶段）

**核心功能**:
- 8个记忆工具（memory_write, memory_read, memory_search等）
- 支持向量搜索、关键词搜索、混合搜索
- 使用外部嵌入服务（ModelScope API或本地服务）
- 向量维度：1024（Qwen3-Embedding-0.6B）

**v2.0设计目标**:
- 集成SurrealDB作为后端存储
- 通过Wrapper Service API访问后端服务
- 实现语义搜索和记忆上传功能

---

### 1.2 关键发现

**配置信息**（从memory-config.json）:
```json
{
  "network": {
    "wrapperUrl": "http://localhost:3001",  // 包装层服务地址
    "timeoutMs": 5000
  },
  "embedding": {
    "provider": "external",
    "endpoint": "http://localhost:18000/v1/embeddings",  // 嵌入服务地址
    "model": "Qwen/Qwen3-Embedding-0.6B"
  }
}
```

**关键匹配**:
- ✅ 包装层端口：3001（与我的设计一致）
- ✅ 嵌入服务端口：18000（与我的设计一致）
- ✅ 向量维度：1024（与我的设计一致）
- ✅ 嵌入模型：Qwen3-Embedding-0.6B（与我的设计一致）

---

## 二、API需求对比

### 2.1 健康检查接口

#### 插件需求（DESIGN_API.md）

**端点**: `GET /api/health`

**响应格式**:
```json
{
  "status": "ok",
  "timestamp": "2026-03-05T12:00:00Z",
  "latency": 15,
  "services": {
    "wrapper": "healthy",
    "surrealdb": "healthy",
    "embedding": "healthy",
    "allHealthy": true
  }
}
```

#### 我的设计

**端点**: `GET /api/v1/health`

**响应格式**:
```json
{
  "status": "healthy",
  "services": {
    "surrealdb": {"status": "healthy"},
    "embedding": {"status": "healthy"},
    "llm": {"status": "healthy"}
  }
}
```

#### 差异分析

| 项目 | 插件需求 | 我的设计 | 兼容性 |
|------|----------|----------|--------|
| 路径 | `/api/health` | `/api/v1/health` | ❌ 不兼容 |
| 响应字段 | `status`, `timestamp`, `latency`, `services` | `status`, `services` | ⚠️ 部分兼容 |
| services.wrapper | 需要 | 无 | ❌ 缺失 |
| services.surrealdb | 需要 | 有 | ✅ 兼容 |
| services.embedding | 需要 | 有 | ✅ 兼容 |
| services.allHealthy | 需要 | 无 | ❌ 缺失 |
| timestamp | 需要 | 无 | ❌ 缺失 |
| latency | 需要 | 无 | ❌ 缺失 |

**兼容性评分**: 40%（部分兼容，需要调整）

---

### 2.2 语义搜索接口

#### 插件需求（DESIGN_API.md）

**端点**: `POST /api/search`

**请求格式**:
```json
{
  "query": "用户偏好的编码风格",
  "limit": 10,
  "threshold": 0.3,
  "filters": {
    "project_tag": "projectA"
  }
}
```

**响应格式**:
```json
{
  "success": true,
  "query": "用户偏好的编码风格",
  "count": 3,
  "results": [
    {
      "id": "memory:001",
      "content": "用户偏好使用 TypeScript 进行项目开发...",
      "score": 0.92,
      "project_tag": "projectA",
      "source": "MEMORY.md",
      "line": 15,
      "timestamp": "2026-03-05T10:30:00Z"
    }
  ]
}
```

#### 我的设计

**端点**: `GET /api/v1/memories/search`

**请求参数**:
```
query: string
mode: string (vector|keyword|hybrid)
limit: number
threshold: number
```

**响应格式**:
```json
{
  "results": [
    {
      "id": "memory:001",
      "content": "...",
      "metadata": {},
      "score": 0.92
    }
  ],
  "count": 3
}
```

#### 差异分析

| 项目 | 插件需求 | 我的设计 | 兼容性 |
|------|----------|----------|--------|
| 路径 | `/api/search` | `/api/v1/memories/search` | ❌ 不兼容 |
| HTTP方法 | POST | GET | ❌ 不兼容 |
| 请求方式 | Body | Query参数 | ❌ 不兼容 |
| filters参数 | 支持 | 支持 | ✅ 兼容 |
| 响应.success | 需要 | 无 | ❌ 缺失 |
| 响应.query | 需要（回显） | 无 | ❌ 缺失 |
| 结果.source | 需要 | 无 | ❌ 缺失 |
| 结果.line | 需要 | 无 | ❌ 缺失 |
| 结果.project_tag | 需要 | 无 | ❌ 缺失 |
| 结果.timestamp | 需要 | 无 | ❌ 缺失 |

**兼容性评分**: 20%（基本不兼容，需要大幅调整）

---

### 2.3 上传记忆接口

#### 插件需求（DESIGN_API.md）

**端点**: `POST /api/upload`

**请求格式**:
```json
{
  "entries": [
    {
      "id": "local-001",
      "content": "用户偏好使用 TypeScript",
      "type": "preference",
      "tags": ["typescript", "style"],
      "project_tag": "projectA",
      "project_id": "github-org-repo",
      "project_name": "项目 A",
      "timestamp": "2026-03-05T12:00:00Z",
      "classification_confidence": 0.85,
      "classified_at": "2026-03-05T12:05:00Z",
      "metadata": {}
    }
  ]
}
```

**响应格式**:
```json
{
  "success": true,
  "count": 1,
  "ids": ["memory:001"],
  "failed": []
}
```

#### 我的设计

**端点**: `POST /api/v1/memories`

**请求格式**:
```json
{
  "memories": [
    {
      "content": "...",
      "metadata": {},
      "entities": [...]
    }
  ],
  "batch_size": 10
}
```

**响应格式**:
```json
{
  "success": true,
  "uploaded": 1,
  "failed": 0,
  "results": [
    {
      "id": "memory:001",
      "status": "success",
      "entities_created": 0
    }
  ]
}
```

#### 差异分析

| 项目 | 插件需求 | 我的设计 | 兼容性 |
|------|----------|----------|--------|
| 路径 | `/api/upload` | `/api/v1/memories` | ❌ 不兼容 |
| 请求字段名 | `entries` | `memories` | ❌ 不兼容 |
| entry.id | 需要（本地ID） | 无 | ❌ 缺失 |
| entry.type | 需要 | 在metadata中 | ⚠️ 部分兼容 |
| entry.tags | 需要 | 在metadata中 | ⚠️ 部分兼容 |
| entry.project_tag | 需要 | 无 | ❌ 缺失 |
| entry.project_id | 需要 | 无 | ❌ 缺失 |
| entry.project_name | 需要 | 无 | ❌ 缺失 |
| entry.timestamp | 需要 | 无（自动生成） | ⚠️ 部分兼容 |
| entry.classification_confidence | 需要 | 无 | ❌ 缺失 |
| entry.classified_at | 需要 | 无 | ❌ 缺失 |
| 响应.ids | 需要 | 在results中 | ⚠️ 部分兼容 |
| 响应.failed | 需要（数组） | 有（数量） | ⚠️ 部分兼容 |

**兼容性评分**: 30%（部分兼容，需要大幅调整）

---

## 三、总体兼容性评估

### 3.1 兼容性总结

| API端点 | 路径兼容 | 方法兼容 | 数据结构兼容 | 总体评分 |
|---------|----------|----------|--------------|----------|
| 健康检查 | ❌ | ✅ | ⚠️ | 40% |
| 语义搜索 | ❌ | ❌ | ⚠️ | 20% |
| 上传记忆 | ❌ | ✅ | ⚠️ | 30% |

**平均兼容性**: 30%（基本不兼容）

---

**分析文档第一部分完成**  
**下一部分**: 关键问题和调整建议

## 四、关键问题分析
### 4.1 API路径不兼容
**问题**:
- 插件期望：`/api/health`, `/api/search`, `/api/upload`
- 我的设计：`/api/v1/health`, `/api/v1/memories/search`, `/api/v1/memories`
**影响**: 🔴 高（插件无法直接访问API）
**原因**: 
- 我使用了版本化路径（`/api/v1/`）
- 我使用了RESTful命名（`/memories`）
- 插件使用了简化路径
**解决方案**:
1. **方案A（推荐）**：添加路径别名
   - 保留`/api/v1/*`作为主路径
   - 添加`/api/*`作为别名路径
   - 两套路径指向相同的处理器
   ```python
   # 主路径
   @app.get("/api/v1/health")
   @app.post("/api/v1/memories")
   @app.get("/api/v1/memories/search")
   
   # 别名路径（兼容插件）
   @app.get("/api/health")
   @app.post("/api/upload")  # 别名
   @app.post("/api/search")  # 别名
   ```
2. **方案B**：完全改用插件的路径
   - 放弃版本化路径
   - 使用插件期望的路径
   - 简单但失去版本控制
**推荐**: 方案A（保持向后兼容）
---
### 4.2 搜索接口HTTP方法不兼容
**问题**:
- 插件期望：`POST /api/search`（请求体传参）
- 我的设计：`GET /api/v1/memories/search`（查询参数）
**影响**: 🔴 高（插件无法调用搜索API）
**原因**: 
- GET方法适合简单查询
- POST方法适合复杂过滤条件
- 插件需要传递复杂的filters对象
**解决方案**:
1. **改用POST方法**（推荐）
   ```python
   @app.post("/api/v1/memories/search")
   @app.post("/api/search")  # 别名
   async def search_memories(request: SearchRequest):
       # 支持复杂的filters对象
       ...
   ```
2. **同时支持GET和POST**
   ```python
   @app.get("/api/v1/memories/search")
   @app.post("/api/v1/memories/search")
   async def search_memories(...):
       # GET: 简单查询
       # POST: 复杂过滤
       ...
   ```
**推荐**: 改用POST方法（符合RESTful最佳实践）
---
### 4.3 响应数据结构不完整
**问题**: 插件需要的字段在我的设计中缺失
**影响**: 🟡 中（插件功能受限）
#### 健康检查缺失字段
- ❌ `timestamp`: 时间戳
- ❌ `latency`: 延迟（毫秒）
- ❌ `services.wrapper`: 包装层状态
- ❌ `services.allHealthy`: 总体健康标志
**解决方案**:
```python
@app.get("/api/health")
async def health_check(db: SurrealDBClient = Depends(get_db)):
    start_time = time.time()
    surrealdb_health = await db.health_check()
    latency = int((time.time() - start_time) * 1000)
    
    services = {
        "wrapper": "healthy",
        "surrealdb": surrealdb_health["status"],
        "embedding": "healthy",  # 需要实际检查
        "allHealthy": all(s == "healthy" for s in [...])
    }
    
    return {
        "status": "ok" if services["allHealthy"] else "error",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "latency": latency,
        "services": services
    }
```
#### 搜索结果缺失字段
- ❌ `success`: 成功标志
- ❌ `query`: 查询回显
- ❌ `source`: 来源文件
- ❌ `line`: 行号
- ❌ `project_tag`: 项目标签
- ❌ `timestamp`: 时间戳
**解决方案**:
```python
@app.post("/api/search")
async def search_memories(request: SearchRequest, ...):
    results = await manager.search_memories(...)
    
    # 转换为插件期望的格式
    formatted_results = [
        {
            "id": r["id"],
            "content": r["content"],
            "score": r.get("score", 0),
            "project_tag": r["metadata"].get("project_tag", "unclassified"),
            "source": r["metadata"].get("source", "MEMORY.md"),
            "line": r["metadata"].get("line", 0),
            "timestamp": r.get("created_at", "")
        }
        for r in results
    ]
    
    return {
        "success": True,
        "query": request.query,
        "count": len(formatted_results),
        "results": formatted_results
    }
```
#### 上传记忆缺失字段
- ❌ `entry.id`: 本地ID（客户端生成）
- ❌ `entry.project_tag`: 项目标签
- ❌ `entry.project_id`: 项目ID
- ❌ `entry.project_name`: 项目名称
- ❌ `entry.classification_confidence`: 分类置信度
- ❌ `entry.classified_at`: 分类时间
**解决方案**:
```python
@app.post("/api/upload")
async def upload_memories(request: UploadRequest, ...):
    # 接收插件格式
    entries = request.entries
    
    # 转换为内部格式
    memories = [
        {
            "content": e["content"],
            "metadata": {
                "type": e.get("type"),
                "tags": e.get("tags", []),
                "project_tag": e.get("project_tag", "unclassified"),
                "project_id": e.get("project_id"),
                "project_name": e.get("project_name"),
                "classification_confidence": e.get("classification_confidence"),
                "classified_at": e.get("classified_at"),
                "source_id": e.get("id"),  # 保存客户端ID
                **e.get("metadata", {})
            },
            "entities": []  # 插件不使用实体功能
        }
        for e in entries
    ]
    
    result = await manager.upload_memories(memories)
    
    # 转换为插件期望的格式
    return {
        "success": True,
        "count": result["uploaded"],
        "ids": [r["id"] for r in result["results"] if r["status"] == "success"],
        "failed": [r for r in result["results"] if r["status"] == "failed"]
    }
```
---
### 4.4 数据模型差异
**问题**: 插件的数据模型与我的设计有差异
**影响**: 🟡 中（需要数据转换）
**插件的数据模型**:
- 以项目为中心（project_tag, project_id, project_name）
- 支持分类置信度（classification_confidence）
- 保留客户端ID（entry.id）
- 记录来源文件和行号（source, line）
**我的数据模型**:
- 以记忆为中心
- 支持实体和关系（entities, relations）
- 服务端生成ID
- 元数据存储在metadata对象中
**解决方案**:
1. **扩展metadata字段**（推荐）
   - 在metadata中添加插件需要的字段
   - 保持向后兼容
   ```python
   metadata = {
       "type": "preference",
       "tags": ["typescript"],
       "project_tag": "projectA",  # 新增
       "project_id": "github-org-repo",  # 新增
       "project_name": "项目A",  # 新增
       "source": "MEMORY.md",  # 新增
       "line": 15,  # 新增
       "source_id": "local-001",  # 新增（客户端ID）
       "classification_confidence": 0.85,  # 新增
       "classified_at": "2026-03-05T12:05:00Z"  # 新增
   }
   ```
2. **创建专门的插件适配层**
   - 在API层进行数据转换
   - 内部数据模型保持不变
   - 对外提供插件兼容的接口
**推荐**: 方案1（简单直接）
---
## 五、调整建议
### 5.1 必须调整（P0）
**1. 添加API路径别名**
- 工作量：30分钟
- 影响：高
- 优先级：🔴 P0
**2. 搜索接口改用POST方法**
- 工作量：15分钟
- 影响：高
- 优先级：🔴 P0
**3. 扩展响应数据结构**
- 工作量：1小时
- 影响：高
- 优先级：🔴 P0
**总工作量**: 约1.75小时
---
### 5.2 强烈建议（P1）
**1. 扩展metadata字段支持**
- 工作量：30分钟
- 影响：中
- 优先级：🟡 P1
**2. 添加数据转换层**
- 工作量：1小时
- 影响：中
- 优先级：🟡 P1
**总工作量**: 约1.5小时
---
### 5.3 可选优化（P2）
**1. 创建插件专用API文档**
- 工作量：1小时
- 影响：低
- 优先级：🟢 P2
**2. 添加API版本协商**
- 工作量：2小时
- 影响：低
- 优先级：🟢 P2
---
## 六、实施方案
### 6.1 调整后的API设计
**健康检查**:
```python
# 主路径
@app.get("/api/v1/health")
# 插件兼容路径
@app.get("/api/health")
async def health_check(...):
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "latency": latency_ms,
        "services": {
            "wrapper": "healthy",
            "surrealdb": "healthy",
            "embedding": "healthy",
            "allHealthy": True
        }
    }
```
**语义搜索**:
```python
# 主路径
@app.post("/api/v1/memories/search")
# 插件兼容路径
@app.post("/api/search")
async def search_memories(request: SearchRequest, ...):
    return {
        "success": True,
        "query": request.query,
        "count": len(results),
        "results": [
            {
                "id": r["id"],
                "content": r["content"],
                "score": r["score"],
                "project_tag": r["metadata"]["project_tag"],
                "source": r["metadata"]["source"],
                "line": r["metadata"]["line"],
                "timestamp": r["created_at"]
            }
            for r in results
        ]
    }
```
**上传记忆**:
```python
# 主路径
@app.post("/api/v1/memories")
# 插件兼容路径
@app.post("/api/upload")
async def upload_memories(request: UploadRequest, ...):
    return {
        "success": True,
        "count": uploaded_count,
        "ids": [r["id"] for r in results],
        "failed": failed_entries
    }
```
---
### 6.2 实施步骤
**阶段1：核心调整**（1.75小时）
1. 添加API路径别名（30分钟）
2. 搜索接口改用POST（15分钟）
3. 扩展响应数据结构（1小时）
**阶段2：数据模型扩展**（1.5小时）
1. 扩展metadata字段（30分钟）
2. 添加数据转换层（1小时）
**阶段3：测试验证**（1小时）
1. 单元测试（30分钟）
2. 集成测试（30分钟）
**总工作量**: 约4.25小时
---
## 七、最终结论
### 7.1 兼容性评估
**调整前**: 30%（基本不兼容）
**调整后**: 95%（高度兼容）
**关键改进**:
- ✅ API路径完全兼容
- ✅ HTTP方法完全兼容
- ✅ 请求格式完全兼容
- ✅ 响应格式完全兼容
- ✅ 数据模型兼容
---
### 7.2 能否满足插件需求？
**答案**: ✅ **可以满足**（需要调整）
**前提条件**:
1. 实施P0级别调整（1.75小时）
2. 实施P1级别调整（1.5小时）
3. 通过集成测试验证
**调整后的优势**:
- ✅ 完全兼容插件v2.0设计
- ✅ 保持向后兼容（双路径）
- ✅ 支持版本化API
- ✅ 扩展性强
---
### 7.3 建议
**立即执行**:
1. 实施P0级别调整（必须）
2. 更新技术设计文档v2.0
3. 添加插件兼容性说明
**后续优化**:
1. 实施P1级别调整
2. 创建插件集成测试
3. 编写插件对接文档
---
**分析文档完成**  
**结论**: 我的设计可以满足opencode-memory-plugin的需求，但需要进行约4.25小时的调整工作