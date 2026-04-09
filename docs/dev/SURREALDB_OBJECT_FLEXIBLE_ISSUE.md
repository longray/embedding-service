# SurrealDB 3.0 object FLEXIBLE 字段问题总结

## 问题描述

在 SurrealDB 3.0 中使用 `TYPE object FLEXIBLE` 字段存储 Python dict 时遇到兼容性问题。

## 错误现象

### 1. 使用 `CREATE CONTENT` + 参数绑定
```python
query = """
    CREATE audit_log CONTENT {
        details: $details  # $details 是 Python dict
    }
"""
params = {"details": {"content_length": 100, "language": "python"}}
result = await db.query(query, params)
```

**错误**: `Found field 'details.content_length', but no such field exists for table 'audit_log'`

### 2. 使用 `CREATE SET` + 参数绑定
```python
query = "CREATE audit_log SET details = $details"
params = {"details": {"content_length": 100}}
result = await db.query(query, params)
```

**错误**: 同样的错误信息

## 根本原因

### 1. 文档与实际行为不符

官方文档声称 `TYPE object FLEXIBLE` 允许任意嵌套字段：

> "Flexible types allow you to have SCHEMALESS functionality on a SCHEMAFULL table."

但实际上 SurrealDB 3.0 仍然要求定义每个子字段。

### 2. Python SDK CBOR 序列化问题

Python dict 通过 CBOR 编码后，SurrealDB 3.0 无法正确识别为 `object FLEXIBLE` 类型。

### 3. Schema 更新陷阱

`DEFINE FIELD IF NOT EXISTS` **不会**覆盖已存在的字段定义：

```sql
-- 第一次定义
DEFINE FIELD details ON audit_log TYPE object FLEXIBLE;

-- 后续尝试更新（不会生效！）
DEFINE FIELD IF NOT EXISTS details ON audit_log TYPE option<string>;
-- 结果：字段仍然是 TYPE object FLEXIBLE
```

## 解决方案

### 方案：使用 `TYPE option<string>` 存储 JSON 字符串

#### 1. Schema 定义

```sql
-- 使用 OVERWRITE 强制更新已存在的字段
DEFINE FIELD OVERWRITE details ON audit_log TYPE option<string>;
```

#### 2. Python 代码

```python
import json

# 存储时：dict -> JSON 字符串
async def log_audit_event(self, details: dict | None = None):
    set_clauses = ["action = $action", "tenant_id = $tenant_id"]
    params = {"action": action, "tenant_id": tenant_id}
    
    if details:
        set_clauses.append("details = $details")
        params["details"] = json.dumps(details)  # 序列化为 JSON 字符串
    
    query = f"CREATE audit_log SET {', '.join(set_clauses)}"
    result = await self._db_query(query, params)

# 读取时：JSON 字符串 -> dict
async def get_audit_log(self, log_id: str):
    result = await self._db_query("SELECT * FROM audit_log WHERE id = $id", {"id": log_id})
    records = self._extract_records(result)
    if records:
        record = records[0]
        if record.get("details"):
            record["details"] = json.loads(record["details"])  # 反序列化
        return record
```

## 验证测试

```python
# 测试创建带 details 的审计日志
response = await client.post("/api/v1/audit/log", json={
    "action": "memory_create",
    "resource_type": "memory",
    "resource_id": "memory:test-123",
    "details": {"content_length": 100, "language": "python"},  # ✅ 成功
    "user_id": "user-001",
    "tenant_id": "default",
})
assert response.status_code == 200
```

## 经验教训

1. **不要完全相信文档**：SurrealDB 3.0 的 `FLEXIBLE` 行为与文档描述有差异
2. **使用 `OVERWRITE` 更新 Schema**：`IF NOT EXISTS` 不会覆盖已有定义
3. **务实选择**：对于动态结构，使用 `string` 类型存储 JSON 比折腾 `object FLEXIBLE` 更可靠
4. **测试验证**：任何 Schema 变更后都要验证实际行为

## 参考链接

- [DEFINE FIELD FLEXIBLE 官方文档](https://surrealdb.com/docs/surrealql/statements/define/field#flexible-data-types)
- [Python SDK 数据类型](https://surrealdb.com/docs/sdk/python/data-types)
- [CREATE CONTENT 文档](https://surrealdb.com/docs/surrealql/statements/create#creating-records-with-content)

## 相关文件

- `scripts/init_surrealdb.surql` - Schema 定义
- `wrapper/src/utils/memory_manager/audit.py` - AuditMixin 实现
- `test_audit_api.py` - API 测试
- `test_audit_comprehensive.py` - 综合测试套件

---

**记录时间**: 2026-04-09  
**SurrealDB 版本**: v3.0.1  
**Python SDK 版本**: 0.11.0  
**状态**: ✅ 已解决
