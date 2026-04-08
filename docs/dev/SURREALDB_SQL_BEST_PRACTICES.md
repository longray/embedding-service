# SurrealDB SQL 开发规范

**版本**: 1.0  
**日期**: 2026-04-08  
**适用版本**: SurrealDB 3.0+  

---

## 1. 概述

本文档规定了 embedding_service 项目中 SurrealDB SQL 的开发规范，旨在：
- 防止 SQL 注入等安全问题
- 统一代码风格，提高可维护性
- 优化查询性能
- 避免常见陷阱

---

## 2. RecordID 处理规范

### 2.1 必须使用 `type::record()` 函数

**正确**:
```python
# 查询单条记录
"SELECT * FROM memory WHERE id = type::record($memory_id)"

# 数组查询
"WHERE in IN array::map($file_ids, |$id| type::record($id))"

# 动态表名和ID
"SELECT * FROM type::record($table_name, $record_id)"
```

**错误**:
```python
# 不要直接拼接字符串
f"SELECT * FROM memory WHERE id = {memory_id}"

# 不要直接传递字符串ID到数组查询
"WHERE in IN $file_ids"  # file_ids 是字符串列表，in 是 RecordID
```

### 2.2 使用辅助函数统一处理

在 `manager.py` 中定义辅助函数：

```python
def _normalize_memory_id(self, memory_id: str) -> str:
    """规范化记忆 ID，确保统一格式"""
    if not memory_id:
        return memory_id
    if ":" in memory_id:
        return memory_id
    return f"memory:{memory_id}"
```

使用示例：
```python
file_id_records = [self._normalize_memory_id(fid) for fid in file_ids.keys()]
```

### 2.3 RecordID 格式验证

对于外部输入的 ID，必须进行格式验证：

```python
def _is_valid_record_id(self, record_id: str) -> bool:
    """验证 RecordID 格式"""
    import re
    return bool(re.match(r'^memory:[a-z0-9]+$', record_id))
```

---

## 3. 参数绑定规范

### 3.1 所有动态值必须使用参数绑定

**正确**:
```python
query = "SELECT * FROM memory WHERE tenant_id = $tenant_id AND type = $type"
result = await self._db_query(query, {"tenant_id": tenant_id, "type": mem_type})
```

**错误**:
```python
# 不要字符串拼接
query = f"SELECT * FROM memory WHERE tenant_id = '{tenant_id}'"
```

### 3.2 RELATE 语句的特殊处理

RELATE 语句的 RecordID 部分无法参数化，需要特殊处理：

```python
# 验证 RecordID 格式
if not self._is_valid_record_id(from_id):
    raise ValidationError(f"Invalid from_id: {from_id}")
if not self._is_valid_record_id(to_id):
    raise ValidationError(f"Invalid to_id: {to_id}")

# RecordID 部分保持拼接（已验证安全）
# SET 子句的值使用参数绑定
set_clauses = [
    "relationship_type = $rel_type",
    "weight = $weight",
]
params = {
    "rel_type": relationship_type,
    "weight": weight,
}

q = f"RELATE {from_ref}->memory_relation->{to_ref} SET {set_str}"
result = await self._db_query(q, params)
```

### 3.3 批量插入的参数绑定

```python
# 单条批量插入
query = "INSERT INTO memory $data"
result = await self._db_query(query, {"data": batch_data})
```

---

## 4. 性能优化规范

### 4.1 使用复合索引

**推荐索引设计**:
```sql
-- 单列索引
DEFINE INDEX memory_tenant ON memory FIELDS tenant_id;
DEFINE INDEX memory_type ON memory FIELDS type;

-- 复合索引（最左前缀原则）
DEFINE INDEX memory_tenant_type_project ON memory FIELDS tenant_id, type, project_id;
```

**查询优化**:
```sql
-- 使用复合索引（tenant_id 在最左）
SELECT * FROM memory 
WHERE tenant_id = $tenant_id  -- 使用索引
  AND type = 'code'
  AND project_id = $project_id;
```

### 4.2 避免深层嵌套查询

**性能差**:
```sql
-- 避免多层嵌套字段过滤
WHERE metadata.code_analysis.complexity.cyclomatic_complexity > 10
```

**优化方案**:
1. 将常用字段提升到顶层
2. 使用复合索引覆盖查询
3. 应用层过滤（查询后过滤）

### 4.3 分批处理大数据量

**批次大小统一配置**:
```python
# SurrealDB 插入批次大小
BATCH_SIZE = 50

# Meilisearch 同步批次大小
MEILI_BATCH_SIZE = 50
```

**分批处理示例**:
```python
for batch_idx in range(total_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(data))
    current_batch = data[start_idx:end_idx]
    
    query = "INSERT INTO memory $data"
    result = await self._db_query(query, {"data": current_batch})
```

### 4.4 使用 LIMIT 限制结果集

```sql
-- 始终使用 LIMIT
SELECT * FROM memory 
WHERE tenant_id = $tenant_id 
LIMIT 100;

-- 分页查询
SELECT * FROM memory 
WHERE tenant_id = $tenant_id 
LIMIT 100 START 200;
```

---

## 5. 常见陷阱和解决方案

### 5.1 RecordID 格式不匹配

**问题**: 查询时 RecordID 格式不一致导致匹配失败

**原因**: 
- 字符串 ID: `"memory:abc123"`
- RecordID 对象: `RecordID(table_name='memory', record_id='abc123')`

**解决**: 始终使用 `type::record()` 转换

### 5.2 嵌套字段无法使用索引

**问题**: `metadata.file_path` 等嵌套字段无法使用索引

**解决**:
1. 添加复合索引覆盖常用查询
2. 将关键字段提升到顶层（需要数据迁移）
3. 应用层缓存热点数据

### 5.3 大批量操作超时

**问题**: 一次性处理大量数据导致超时

**解决**:
1. 分批处理（50条/批）
2. 异步化改造（后台任务）
3. 增加 HTTP 超时时间

### 5.4 RELATE 语句 SQL 注入

**问题**: RELATE 语句的 RecordID 部分可能被注入

**解决**:
1. 严格验证 RecordID 格式
2. 只允许特定字符: `a-z0-9`
3. 使用正则表达式验证

---

## 6. 代码审查清单

提交 SQL 相关代码前，请确认：

- [ ] 所有动态值使用参数绑定 `$param`
- [ ] RecordID 使用 `type::record()` 函数
- [ ] 外部输入的 ID 经过格式验证
- [ ] 查询使用 LIMIT 限制结果集
- [ ] 大批量操作使用分批处理
- [ ] 新增查询有对应的索引支持
- [ ] 无字符串拼接 SQL（除已验证的 RecordID）

---

## 7. 示例代码

### 7.1 标准查询模板

```python
async def get_memory_by_id(self, memory_id: str, tenant_id: str) -> dict | None:
    """根据 ID 获取记忆"""
    query = """
        SELECT * FROM memory 
        WHERE id = type::record($memory_id) 
          AND tenant_id = $tenant_id
        LIMIT 1
    """
    result = await self._db_query(query, {
        "memory_id": memory_id,
        "tenant_id": tenant_id
    })
    records = self._extract_records(result)
    return records[0] if records else None
```

### 7.2 批量插入模板

```python
async def batch_insert_memories(self, memories: list[dict], tenant_id: str) -> list[str]:
    """批量插入记忆"""
    BATCH_SIZE = 50
    memory_ids = []
    
    for i in range(0, len(memories), BATCH_SIZE):
        batch = memories[i:i + BATCH_SIZE]
        query = "INSERT INTO memory $data"
        result = await self._db_query(query, {"data": batch})
        records = self._extract_records(result)
        memory_ids.extend([str(r.get("id")) for r in records])
    
    return memory_ids
```

### 7.3 图关系创建模板

```python
async def create_relation(
    self, 
    from_id: str, 
    to_id: str, 
    rel_type: str,
    tenant_id: str
) -> dict:
    """创建图关系"""
    # 验证 ID 格式
    if not self._is_valid_record_id(from_id):
        raise ValidationError(f"Invalid from_id: {from_id}")
    if not self._is_valid_record_id(to_id):
        raise ValidationError(f"Invalid to_id: {to_id}")
    
    # 参数化 SET 子句
    set_clauses = ["relationship_type = $rel_type"]
    params = {"rel_type": rel_type, "tenant_id": tenant_id}
    
    q = f"RELATE {from_id}->memory_relation->{to_id} SET {', '.join(set_clauses)}"
    result = await self._db_query(q, params)
    
    return self._extract_records(result)[0]
```

---

## 8. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0 | 2026-04-08 | 初始版本，基于 BL-CA-OPT-01~06 实践经验 |

---

## 9. 参考资源

- [SurrealDB 官方文档](https://surrealdb.com/docs)
- [SurrealQL 参考](https://surrealdb.com/docs/surrealql)
- [项目 SQL 审计报告](../SURREALDB_SQL_AUDIT_REPORT.md)
