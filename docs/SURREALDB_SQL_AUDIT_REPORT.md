# SurrealDB SQL 查询审计报告

**日期**: 2026-04-08  
**版本**: SurrealDB 3.0  
**审计范围**: embedding_service/wrapper/src  
**审计人员**: AI Assistant

---

## 执行摘要

| 指标 | 结果 |
|------|------|
| **总查询数** | 53+ 个 SQL 语句 |
| **问题查询** | 8 个需要优化 |
| **高风险** | 2 个（SQL 注入风险） |
| **中风险** | 4 个（性能问题） |
| **低风险** | 2 个（最佳实践） |

**总体评价**: ⚠️ **需要改进** - 部分查询存在性能和安全隐患

---

## 1. 查询分类统计

### 1.1 按操作类型

| 操作 | 数量 | 占比 |
|------|------|------|
| SELECT | 28 | 53% |
| UPDATE | 8 | 15% |
| INSERT | 3 | 6% |
| DELETE | 4 | 8% |
| RELATE | 5 | 9% |
| CREATE/DEFINE | 5 | 9% |

### 1.2 按复杂度

| 复杂度 | 数量 | 说明 |
|--------|------|------|
| 简单 | 35 | 单表查询，简单 WHERE |
| 中等 | 12 | 嵌套字段查询，数组操作 |
| 复杂 | 6 | 图遍历，批量操作，聚合 |

---

## 2. 发现的问题

### 2.1 🔴 高风险问题

#### 问题 1: SQL 注入风险（字符串拼接）

**位置**: `relations.py:84`

```python
# 当前代码（有风险）
q = (
    f"RELATE {from_ref}->memory_relation->{to_ref} "
    f"SET {set_str}"
)
result = await self._db_query(q, {"tenant_id": effective_tenant_id})
```

**风险**: `from_ref`, `to_ref`, `set_str` 直接拼接到 SQL 中，如果包含恶意字符会导致 SQL 注入。

**修复建议**:
```python
# 使用参数绑定
q = """
    RELATE type::record($from_id)->memory_relation->type::record($to_id)
    SET relationship_type = $rel_type, weight = $weight
"""
result = await self._db_query(q, {
    "from_id": from_ref,
    "to_id": to_ref,
    "rel_type": relationship_type,
    "weight": weight
})
```

---

#### 问题 2: 不安全的字符串转义

**位置**: `relations.py:77-80`

```python
# 当前代码
if description:
    safe_desc = self._sanitize_query(description)
    set_clauses.append(f"description = '{safe_desc}'")
if metadata:
    set_clauses.append(f"metadata = {json.dumps(metadata)}")
```

**风险**: 自定义的 `_sanitize_query` 可能无法覆盖所有注入场景。

**修复建议**: 使用参数绑定代替字符串拼接。

---

### 2.2 🟡 中风险问题

#### 问题 3: RecordID 格式不一致

**位置**: 多个文件

**问题描述**: 项目中存在多种 RecordID 处理方式：

```python
# 方式 1: 直接字符串拼接（stubs.py:381）
file_id_records = [f"memory:{fid.split(':')[1] if ':' in fid else fid}" for fid in file_ids.keys()]

# 方式 2: type::record() 函数（stubs.py:392）
AND in IN array::map($file_ids, |$id| type::record($id))

# 方式 3: 直接传递 RecordID 对象（relations.py:84）
f"RELATE {from_ref}->memory_relation->{to_ref}"
```

**风险**: 格式不一致导致查询失败或性能问题。

**修复建议**: 统一使用 `type::record()` 函数：
```python
# 统一方式
"SELECT * FROM memory WHERE id = type::record($memory_id)"
```

---

#### 问题 4: 缺少索引的查询

**位置**: `crud.py:204`

```sql
SELECT id, metadata FROM memory 
WHERE type = 'code' 
  AND project_id = $project_id 
  AND metadata.file_path = $file_path 
  AND tenant_id = $tenant_id 
LIMIT 1
```

**问题**: `metadata.file_path` 是嵌套字段，无法使用索引。

**修复建议**: 将 `file_path` 提升到顶层字段：
```sql
-- 修改 schema
DEFINE FIELD file_path ON memory TYPE option<string>;

-- 修改查询
SELECT id FROM memory 
WHERE type = 'code' 
  AND project_id = $project_id 
  AND file_path = $file_path
```

---

#### 问题 5: 复杂的嵌套字段查询

**位置**: `stubs.py:201-206`

```sql
SELECT
    id AS memory_id,
    metadata.file_path AS file_path,
    metadata.code_analysis.complexity.cyclomatic_complexity AS complexity,
    metadata.code_analysis.complexity.function_count AS function_count,
    metadata.code_analysis.complexity.class_count AS class_count,
    metadata.code_analysis.imports AS imports
FROM memory
WHERE type = 'code'
    AND project_id = $project_id
    AND tenant_id = $tenant_id
    AND metadata.file_path IS NOT NONE
    AND metadata.code_analysis IS NOT NONE
```

**问题**: 多层嵌套字段查询性能差，且 `IS NOT NONE` 条件无法使用索引。

**修复建议**: 扁平化数据结构或使用 SurrealDB 的 `->` 操作符。

---

#### 问题 6: 批量插入未使用参数绑定

**位置**: `crud.py:301`

```python
query = "INSERT INTO memory $data"
```

**问题**: `$data` 是批量数据，如果数据量大可能导致性能问题。

**修复建议**: 使用分批插入：
```python
# 分批处理，每批 100 条
for batch in chunks(data, 100):
    query = "INSERT INTO memory $batch"
    await self._db_query(query, {"batch": batch})
```

---

### 2.3 🟢 低风险问题

#### 问题 7: 缺少 LIMIT 的查询

**位置**: `sync.py:193`

```sql
SELECT id FROM memory WHERE source_id = $source_id AND tenant_id = $tenant_id LIMIT 1
```

**评价**: ✅ 已使用 LIMIT，但其他查询可能缺少。

---

#### 问题 8: 未使用 EXPLAIN 分析查询

**问题**: 项目中没有使用 `EXPLAIN` 分析查询性能。

**修复建议**: 定期运行：
```sql
EXPLAIN SELECT * FROM memory WHERE tenant_id = 'default';
```

---

## 3. 最佳实践对比

### 3.1 ✅ 好的实践

| 实践 | 位置 | 说明 |
|------|------|------|
| 参数绑定 | `crud.py:204` | 使用 `$param` 绑定参数 |
| type::record() | `stubs.py:392` | 正确使用 RecordID 转换 |
| LIMIT 限制 | `crud.py:204` | 查询使用 LIMIT |
| 事务处理 | `crud.py:300` | 批量操作使用事务 |

### 3.2 ❌ 反模式

| 反模式 | 位置 | 风险 |
|--------|------|------|
| 字符串拼接 SQL | `relations.py:84` | SQL 注入 |
| 嵌套字段查询 | `stubs.py:201` | 性能差 |
| 不一致的 ID 处理 | 多处 | 维护困难 |
| 缺少错误处理 | 多处 | 异常未捕获 |

---

## 4. SurrealDB 3.0 新特性使用建议

### 4.1 推荐使用的特性

#### 1. `type::record()` 函数
```sql
-- 推荐
SELECT * FROM memory WHERE id = type::record($id)

-- 不推荐
SELECT * FROM memory WHERE id = $id
```

#### 2. `array::map()` 函数
```sql
-- 推荐
WHERE in IN array::map($file_ids, |$id| type::record($id))

-- 不推荐（手动转换）
WHERE in IN $file_ids
```

#### 3. `ONLY` 关键字
```sql
-- 推荐（返回单个对象而非数组）
SELECT * FROM ONLY memory:xxx

-- 不推荐
SELECT * FROM memory:xxx
```

#### 4. `FETCH` 关联记录
```sql
-- 推荐（自动获取关联记录）
SELECT *, ->purchased->product.* AS products FROM customer FETCH products
```

---

## 5. 性能优化建议

### 5.1 索引优化

当前索引（来自 schema）：
```sql
DEFINE INDEX memory_embedding_hnsw ON memory FIELDS embedding HNSW...
DEFINE INDEX memory_content_ft ON memory FIELDS content FULLTEXT...
DEFINE INDEX memory_tenant ON memory FIELDS tenant_id
DEFINE INDEX memory_type ON memory FIELDS type
DEFINE INDEX memory_project ON memory FIELDS project_id
```

**建议添加**:
```sql
-- 复合索引（常用查询组合）
DEFINE INDEX memory_tenant_type_project ON memory FIELDS tenant_id, type, project_id;

-- code_analysis 相关索引
DEFINE INDEX memory_code_analysis ON memory FIELDS metadata.code_analysis.language;
```

### 5.2 查询重写

**优化前**:
```sql
SELECT * FROM memory 
WHERE tenant_id = $tenant_id 
  AND type = 'code'
  AND metadata.file_path = $file_path
```

**优化后**:
```sql
SELECT * FROM memory 
WHERE tenant_id = $tenant_id 
  AND type = 'code'
  AND file_path = $file_path  -- 提升到顶层字段
```

---

## 6. 修复优先级

| 优先级 | 问题 | 影响 | 工作量 |
|--------|------|------|--------|
| P0 | SQL 注入风险 | 安全 | 2h |
| P1 | RecordID 格式统一 | 稳定性 | 4h |
| P1 | 嵌套字段优化 | 性能 | 8h |
| P2 | 批量插入优化 | 性能 | 2h |
| P2 | 添加 EXPLAIN 分析 | 维护 | 1h |

---

## 7. 结论

### 7.1 总体评价

**当前状态**: ⚠️ **需要改进**

- 基本功能正常，但存在安全和性能隐患
- SQL 注入风险需要立即修复
- RecordID 处理不一致导致维护困难
- 部分查询性能可以优化

### 7.2 建议行动

1. **立即修复**（本周）:
   - [ ] 修复 SQL 注入风险（relations.py）
   - [ ] 统一 RecordID 处理方式

2. **短期优化**（本月）:
   - [ ] 优化嵌套字段查询
   - [ ] 添加复合索引
   - [ ] 批量插入分批处理

3. **长期改进**（下季度）:
   - [ ] 建立 SQL 查询规范
   - [ ] 添加自动化 SQL 审计
   - [ ] 定期进行 EXPLAIN 分析

---

## 8. 参考资源

- [SurrealDB 官方文档](https://surrealdb.com/docs)
- [SurrealQL 参考](https://surrealdb.com/docs/surrealql)
- [性能优化指南](https://surrealdb.com/learn/fundamentals/performance)
- [Ten Tips for SurrealDB Queries](https://surrealdb.com/blog/ten-tips-and-tricks-for-your-surrealdb-queries)

---

**报告生成时间**: 2026-04-08  
**报告版本**: v1.0  
**下次审计**: 建议 1 个月后
