# SurrealDB Python SDK RELATE 参数绑定限制

**发现日期**: 2026-04-26  
**影响范围**: wrapper/src/utils/memory_manager/relations.py  
**严重程度**: High - 导致图关系创建失败  
**状态**: 已修复

---

## 问题概述

SurrealDB Python SDK 的 `query()` 方法在使用 `RELATE ... SET` 时，无法正确处理参数绑定（`$variable`），导致查询返回 `None` 而不是创建的关系记录。

## 根因分析

### 失败的实现（SET + 参数绑定）

```python
# 失败的实现
let_statements = [
    f"LET $rel_type = '{type}';",
    f"LET $weight = {float(weight)};",
]
set_clauses = ["`type` = $rel_type", "weight = $weight"]
set_str = ", ".join(set_clauses)
q = "\n".join(let_statements) + f"\nRELATE {from_ref}->reference->{to_ref} SET {set_str};"
result = await self._db_query(q)  # 返回 None
```

**问题**: 虽然 `LET` 语句在 SurrealQL 中定义了变量，但在 `RELATE ... SET` 子句中使用 `$variable` 时，Python SDK 无法正确传递这些参数。

### 成功的实现（CONTENT + JSON）

```python
# 成功的实现
content_obj = {
    "type": type,
    "weight": float(weight),
    "tenant_id": effective_tenant_id,
}
content_json = json.dumps(content_obj)
q = f"RELATE {from_ref}->reference->{to_ref} CONTENT {content_json};"
result = await self._db_query(q)  # 返回创建的关系记录
```

**成功原因**: `CONTENT` 子句接受完整的 JSON 对象，避免了参数绑定问题。

## 官方文档与社区验证

### 1. SurrealDB 官方文档

根据官方文档，RELATE 语句支持两种语法：

```surql
RELATE [ ONLY ] @from_record -> @table -> @to_record
    [ CONTENT @value | SET @field = @value ... ]
    [ RETURN ... ]
```

**关键区别**:

- **SET**: 逐字段设置，支持参数绑定 `$variable`
- **CONTENT**: 整体替换，接受 JSON 对象

### 2. Python SDK 特定行为

标准参数绑定适用于 SELECT/CREATE/UPDATE，但 RELATE 有特殊处理。

**文档未明确说明的限制**: `RELATE ... SET` 中的字段值参数绑定在 Python SDK 中行为不一致。

### 3. GitHub Issues

| Issue | 描述 | 状态 |
|-------|------|------|
| #2806 | 边表名不能参数化 | OPEN - 不计划修复 |
| #7167 | 图语法参数化不支持 | OPEN |
| #3369 | type::thing() 在 RELATE 节点位置的问题 | OPEN |
| #155 | 嵌套变量属性需要括号 | 文档问题 |

**官方声明** (2026-03):

> "Parameterisation of graph / arrow syntax is not anticipated in the foreseeable future due to the complexity involved in optimising such queries."

## 代码库中的实际模式

### 两种 RELATE 模式

| 模式 | 位置 | 实现方式 | 状态 |
|------|------|----------|------|
| Pattern A | relations.py:92-106 | RELATE ... CONTENT {json} | 当前修复 |
| Pattern B | reference.py:145-159 | RELATE ... SET $param | 在事务中使用 |

**关键发现**:

- relations.py 中的注释明确说明 SDK 限制
- reference.py 中的 RELATE ... SET 在事务上下文中工作

### 其他参数绑定限制

| 限制 | 影响 | Workaround |
|------|------|------------|
| 边表名不能参数化 | 动态表名 | 使用 INSERT RELATION |
| Record ID 位置参数化 | type::thing() 内联 | 使用括号包裹 |
| 大嵌套对象 CBOR 绑定 | 复杂 dict 被静默丢弃 | 使用 json.dumps() 内联 |
| 多语句查询结果 | 只返回最后一条 | 使用 query_raw() |

## 最佳实践

### 1. RELATE 语句使用指南

```python
# 推荐: 使用 CONTENT + json.dumps()
content_obj = {"type": "follow_up", "weight": 0.8}
content_json = json.dumps(content_obj)
result = await db.query(
    f"RELATE {from_ref}->reference->{to_ref} CONTENT {content_json};"
)

# 替代: 使用 db.insert_relation()（Python SDK 专属）
from surrealdb import RecordID
result = await db.insert_relation("reference", {
    "in": RecordID.parse(from_ref),
    "out": RecordID.parse(to_ref),
    "type": "follow_up",
    "weight": 0.8,
})
```

### 2. 安全注意事项

```python
# 危险: 直接字符串拼接用户输入
user_input = request.json()["type"]
query = f'RELATE a->b CONTENT {{"type": "{user_input}"}}'  # SQL 注入风险

# 安全: 使用白名单验证
allowed_types = {"follow_up", "related", "reference"}
user_type = request.json()["type"]
if user_type not in allowed_types:
    raise ValueError("Invalid type")
content_obj = {"type": user_type}
query = f"RELATE a->b CONTENT {json.dumps(content_obj)}"
```

## 修复历史

### 2026-04-26: 修复 create_relation 实现

**变更文件**: wrapper/src/utils/memory_manager/relations.py

**变更内容**:

- 将 RELATE ... SET 改为 RELATE ... CONTENT
- 使用 json.dumps() 构建内容对象
- 添加注释说明 SDK 限制

**测试结果**: 所有 15 个 TestRelationsAPI 测试通过

### 2026-04-26: 进一步优化为 db.insert_relation()

**变更文件**: wrapper/src/utils/memory_manager/relations.py

**变更内容**:

- 将 RELATE ... CONTENT 改为 db.insert_relation()
- 使用 RecordID.parse() 自动转换 record ID
- 处理 insert_relation 返回列表的格式

**测试结果**: 所有 65 个 wrapper API 测试通过

**最终方案**:

```python
from surrealdb import RecordID as SurrealRecordID

relation_data = {
    "in": SurrealRecordID.parse(from_ref),
    "out": SurrealRecordID.parse(to_ref),
    "type": type,
    "weight": float(weight),
    "tenant_id": effective_tenant_id,
}
if description:
    relation_data["description"] = description
if metadata:
    relation_data["metadata"] = metadata

result = await self._db.insert_relation("reference", relation_data)

if result:
    # insert_relation returns a list with the created record
    record = result[0] if isinstance(result, list) else result
    return {
        "id": str(record.get("id", "")),
        "from": from_ref,
        "to": to_ref,
        "type": type,
        "weight": weight,
    }
```

**优势**:

- 更简洁（10 行 vs 12 行）
- 更安全（无字符串插值）
- 使用 SDK 原生方法
- 自动 RecordID 转换

## 参考链接

- SurrealDB RELATE Statement 文档
- SurrealDB Python SDK 文档
- GitHub Issue #2806, #7167, #3369
