# Memory Lookup API 技术设计文档

**文档版本**: 1.0  
**最后更新**: 2026-04-09  
**关联任务**: BL-CA-33  
**状态**: 📝 设计中

---

## 1. 架构设计

### 1.1 组件关系

```
┌─────────────────┐
│   Plugin Client │
└────────┬────────┘
         │ GET /api/v1/memories/lookup
         ▼
┌─────────────────┐
│   Lookup Router │ (wrapper/src/routers/lookup.py)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LookupMixin    │ (wrapper/src/utils/memory_manager/lookup.py)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   SurrealDB     │
└─────────────────┘
```

### 1.2 文件结构

```
wrapper/src/
├── routers/
│   └── lookup.py              # 新: API 路由
├── utils/memory_manager/
│   └── lookup.py              # 新: LookupMixin
├── models.py                  # 修改: 添加 LookupRequest/Response
└── main.py                    # 修改: 注册路由

scripts/
└── init_surrealdb.surql       # 修改: 添加索引

docs/
├── product/
│   └── lookup-api-spec.md     # 产品规格书
└── dev/
    └── lookup-api-design.md   # 本文档
```

---

## 2. 数据库设计

### 2.1 索引设计

```sql
-- source_id 唯一索引（如果 source_id 应该是唯一的）
-- 或者普通索引（如果允许重复）
DEFINE INDEX IF NOT EXISTS idx_memory_source_id ON memory FIELDS source_id;

-- content_hash 索引
DEFINE INDEX IF NOT EXISTS idx_memory_content_hash ON memory FIELDS content_hash;

-- file_path + project_id 复合索引
-- 注意: file_path 存储在 metadata 中，需要特殊处理
DEFINE INDEX IF NOT EXISTS idx_memory_file_project ON memory 
    FIELDS (metadata.file_path, project_id);
```

### 2.2 查询策略

#### 查询 1: By source_id
```sql
SELECT * FROM memory 
WHERE source_id = $source_id 
  AND tenant_id = $tenant_id
  AND ($type IS NONE OR type = $type)
ORDER BY created_at DESC
LIMIT $limit
```

#### 查询 2: By content_hash
```sql
SELECT * FROM memory 
WHERE content_hash = $content_hash 
  AND tenant_id = $tenant_id
  AND ($type IS NONE OR type = $type)
ORDER BY created_at DESC
LIMIT $limit
```

#### 查询 3: By file_path + project_id
```sql
SELECT * FROM memory 
WHERE metadata.file_path = $file_path 
  AND project_id = $project_id
  AND tenant_id = $tenant_id
  AND ($type IS NONE OR type = $type)
ORDER BY created_at DESC
LIMIT $limit
```

---

## 3. API 设计

### 3.1 路由定义

```python
@router.get("/api/v1/memories/lookup")
async def lookup_memory(
    source_id: str | None = Query(None),
    hash: str | None = Query(None),
    hash_algorithm: str = Query("md5"),
    file_path: str | None = Query(None),
    project_id: str | None = Query(None),
    type: str | None = Query(None),
    tenant_id: str = Query("default"),
    limit: int = Query(1, ge=1, le=100),
    all: bool = Query(False),
):
    """查询记忆"""
    pass
```

### 3.2 查询优先级逻辑

```python
def determine_query_strategy(params):
    """确定查询策略"""
    if params.source_id:
        return "by_source_id", params
    elif params.hash:
        return "by_hash", params
    elif params.file_path and params.project_id:
        return "by_file_path", params
    else:
        raise ValueError("Insufficient query parameters")
```

### 3.3 响应构建

```python
def build_single_response(record: dict) -> dict:
    """构建单条响应"""
    return {
        "found": True,
        "memory_id": str(record.get("id", "")),
        "source_id": record.get("source_id"),
        "file_path": record.get("metadata", {}).get("file_path"),
        "project_id": record.get("project_id"),
        "type": record.get("type"),
        "content_hash": record.get("content_hash"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
    }

def build_multi_response(records: list) -> dict:
    """构建多条响应"""
    return {
        "found": True,
        "count": len(records),
        "memories": [
            {
                "memory_id": str(r.get("id", "")),
                "source_id": r.get("source_id"),
                "file_path": r.get("metadata", {}).get("file_path"),
                "created_at": r.get("created_at"),
            }
            for r in records
        ],
    }
```

---

## 4. 实现细节

### 4.1 LookupMixin 设计

```python
class LookupMixin:
    """记忆查询功能 Mixin"""

    async def lookup_by_source_id(
        self, 
        source_id: str, 
        tenant_id: str,
        type_filter: str | None = None,
        limit: int = 1,
    ) -> list[dict]:
        """通过 source_id 查询"""
        pass

    async def lookup_by_hash(
        self, 
        content_hash: str, 
        tenant_id: str,
        type_filter: str | None = None,
        limit: int = 1,
    ) -> list[dict]:
        """通过 content_hash 查询"""
        pass

    async def lookup_by_file_path(
        self, 
        file_path: str, 
        project_id: str,
        tenant_id: str,
        type_filter: str | None = None,
        limit: int = 1,
    ) -> list[dict]:
        """通过 file_path + project_id 查询"""
        pass
```

### 4.2 错误处理

```python
class LookupError(Exception):
    """查询错误"""
    pass

class InsufficientParametersError(LookupError):
    """参数不足"""
    pass

class MemoryNotFoundError(LookupError):
    """记忆未找到"""
    pass
```

---

## 5. 测试策略

### 5.1 单元测试

```python
# test_lookup_mixin.py
async def test_lookup_by_source_id():
    """测试 source_id 查询"""
    pass

async def test_lookup_by_hash():
    """测试 hash 查询"""
    pass

async def test_lookup_by_file_path():
    """测试 file_path 查询"""
    pass

async def test_query_priority():
    """测试查询优先级"""
    pass
```

### 5.2 集成测试

```python
# test_lookup_api.py
async def test_lookup_api_success():
    """测试 API 成功响应"""
    pass

async def test_lookup_api_not_found():
    """测试 API 未找到响应"""
    pass

async def test_lookup_api_invalid_params():
    """测试 API 参数错误"""
    pass
```

---

## 6. 性能考虑

### 6.1 索引优化

- source_id: 高选择性，适合索引
- content_hash: 高选择性，适合索引
- file_path + project_id: 复合索引，避免全表扫描

### 6.2 查询优化

- 使用 LIMIT 限制返回数量
- 按 created_at DESC 排序，利用索引
- 避免 SELECT *，只返回必要字段

### 6.3 缓存策略（可选）

- 高频查询结果可缓存（如 source_id 查询）
- TTL: 5-10 分钟
- 缓存键: `lookup:{source_id}:{tenant_id}`

---

## 7. 安全考虑

### 7.1 输入验证

- 验证 source_id 格式（ULID）
- 验证 hash 格式（32位十六进制）
- 验证 limit 范围（1-100）

### 7.2 权限控制

- 必须提供 tenant_id
- 按 tenant_id 隔离数据
- 不返回敏感字段（embedding, raw_content 等）

---

## 8. 部署计划

### 8.1 数据库迁移

```sql
-- 步骤 1: 添加索引
DEFINE INDEX idx_memory_source_id ON memory FIELDS source_id;
DEFINE INDEX idx_memory_content_hash ON memory FIELDS content_hash;
DEFINE INDEX idx_memory_file_project ON memory FIELDS (metadata.file_path, project_id);
```

### 8.2 代码部署

1. 部署新代码
2. 验证 API 可用性
3. 运行测试套件
4. 监控错误率

---

## 9. 相关文档

- [产品规格书](../product/lookup-api-spec.md)
- [BACKLOG](../../BACKLOG.md) - 任务追踪
- [README](../../README.md) - API 文档

---

**文档维护**: 后端团队  
**审核**: 技术负责人
