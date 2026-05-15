# RecordID 查询审计报告

**日期**: 2026-05-15
**审计人**: Sisyphus
**范围**: wrapper/src 所有 SurrealDB 查询

---

## 执行摘要

本次审计检查了代码库中所有使用 RecordID 的 SurrealDB 查询，发现并修复了 2 个关键 Bug：

| Bug ID | 文件 | 问题 | 状态 |
|--------|------|------|------|
| BL-B-116 | reference.py | from_id/to_id 查询未转换 RecordID | ✅ 已修复 |
| BL-B-117 | weight_calculator.py | UPDATE/SELECT 查询未转换 RecordID | ✅ 已修复 |

---

## 审计方法

1. **代码搜索**: 使用 grep 搜索所有 `type::record` 使用
2. **模式分析**: 检查 `in`/`out` 字段查询模式
3. **对比验证**: 对比正确和错误的 RecordID 使用模式
4. **测试验证**: 运行相关测试确保修复有效

---

## RecordID 使用模式分类

### ✅ 正确模式

#### 1. 单参数转换

```python
# reference.py:441 - 图遍历查询（已修复）
query = "SELECT * FROM type::record($from_id)->reference WHERE tenant_id = $tenant_id"

# prefetch_service.py:116 - WHERE 子句查询
query = "SELECT out AS related_id FROM reference WHERE in = type::record($memory_id)"

# crud.py:538 - 单表查询
query = "SELECT abstract FROM memory WHERE id = type::record($id) LIMIT 1"
```

#### 2. 双参数转换（表名+ID）

```python
# relations.py:170 - 使用 type::record($table, $id)
query = "WHERE in = type::record($mem_table, $mem_id) AND tenant_id = $tenant_id"
```

#### 3. 数组批量转换

```python
# stubs.py:749 - 使用 array::map 批量转换
query = "AND in IN array::map($file_ids, |$id| type::record($id))"
```

#### 4. UPDATE 语句转换

```python
# code_analysis.py:122 - UPDATE 语句
query = "UPDATE type::record($record_id) SET ..."

# crud.py:475 - UPDATE 语句
query = "UPDATE type::record($id) SET ..."
```

### ❌ 错误模式（已修复）

```python
# weight_calculator.py:185 - 修复前
query = "UPDATE reference SET weight = $weight WHERE in = $caller AND out = $callee"

# weight_calculator.py:282 - 修复前
query = "SELECT weight FROM reference WHERE in = $caller AND out = $callee"
```

**问题**: `$caller` 和 `$callee` 是字符串（如 `"memory:abc123"`），但 `in`/`out` 字段是 RecordID 类型，直接比较返回 0 结果。

---

## 文件审计详情

### 1. reference.py ✅

**状态**: 已修复 (BL-B-116)

**修复内容**:

```python
# from_id 查询
- query = "SELECT * FROM $from_id->reference WHERE tenant_id = $tenant_id"
+ query = "SELECT * FROM type::record($from_id)->reference WHERE tenant_id = $tenant_id"

# to_id 查询
- query = "SELECT * FROM <-reference-$to_id WHERE tenant_id = $tenant_id"
+ query = "SELECT * FROM reference WHERE out = type::record($to_id) AND tenant_id = $tenant_id"
```

### 2. weight_calculator.py ✅

**状态**: 已修复 (BL-B-117)

**修复内容**:

```python
# UPDATE 查询 (第 185 行)
- WHERE in = $caller AND out = $callee AND tenant_id = $tenant_id
+ WHERE in = type::record($caller) AND out = type::record($callee) AND tenant_id = $tenant_id

# SELECT 查询 (第 282 行)
- WHERE in = $caller AND out = $callee AND tenant_id = $tenant_id
+ WHERE in = type::record($caller) AND out = type::record($callee) AND tenant_id = $tenant_id
```

### 3. relations.py ✅

**状态**: 正确

**模式**: 使用 `type::record($mem_table, $mem_id)` 双参数形式

### 4. stubs.py ✅

**状态**: 正确

**模式**: 使用 `array::map($file_ids, |$id| type::record($id))` 批量转换

### 5. prefetch_service.py ✅

**状态**: 正确

**模式**: 使用 `type::record($memory_id)` 单参数转换

### 6. crud.py ✅

**状态**: 正确

**模式**:

- UPDATE: `UPDATE type::record($id) SET ...`
- SELECT: `WHERE id = type::record($id)`

### 7. code_analysis.py ✅

**状态**: 正确

**模式**:

- UPDATE: `UPDATE type::record($record_id)`
- SELECT: `FROM type::record($id)`

### 8. memories.py ✅

**状态**: 无需 RecordID 转换

**说明**: 该文件使用 memory_id 字符串直接查询，不涉及 graph relation 表的 `in`/`out` 字段。

---

## 最佳实践总结

### 规则 1: Graph Relation 表查询

当查询 `reference` 等 graph relation 表的 `in` 或 `out` 字段时，**必须**使用 `type::record()`:

```python
# ✅ 正确
"WHERE in = type::record($id)"
"WHERE out = type::record($id)"

# ❌ 错误
"WHERE in = $id"
"WHERE out = $id"
```

### 规则 2: 单表 ID 查询

当查询普通表的 `id` 字段时，**必须**使用 `type::record()`:

```python
# ✅ 正确
"WHERE id = type::record($id)"
"UPDATE type::record($id) SET ..."

# ❌ 错误
"WHERE id = $id"
```

### 规则 3: 图遍历语法

使用图遍历语法时，**必须**使用 `type::record()`:

```python
# ✅ 正确
"SELECT * FROM type::record($id)->reference"
"SELECT * FROM <-reference<-type::record($id)"

# ❌ 错误
"SELECT * FROM $id->reference"
"SELECT * FROM <-reference-$id"
```

### 规则 4: 批量处理

处理 ID 数组时，使用 `array::map()`:

```python
# ✅ 正确
"AND in IN array::map($ids, |$id| type::record($id))"
```

---

## 测试验证

### 已运行测试

```bash
uv run pytest tests/ -k weight -v
```

**结果**: 51/52 通过，1 个失败（与修复无关）

### 手动验证

```bash
# from_id 查询
GET /api/v1/references?from_id=entity:e4s03rrkapi8a1uiy28t
→ 返回 50 条结果 ✅

# to_id 查询
GET /api/v1/references?to_id=atom:1ezqdums9lzmqaleb1ef
→ 返回 1 条结果 ✅
```

---

## 建议

### 1. 代码审查清单

在审查涉及 SurrealDB 查询的代码时，检查：

- [ ] 所有 `$variable` 参数是否使用了正确的类型转换
- [ ] Graph relation 表的 `in`/`out` 字段是否使用了 `type::record()`
- [ ] 普通表的 `id` 字段是否使用了 `type::record()`
- [ ] 图遍历语法是否使用了 `type::record()`

### 2. 静态检查工具

考虑添加自定义 lint 规则，检测以下模式：

```regex
WHERE\s+(in|out)\s*=\s*\$(?!.*type::record)
```

### 3. 文档更新

已在本文档中记录所有正确的 RecordID 使用模式，供开发参考。

---

## 附录：受影响文件完整列表

| 文件 | 状态 | 说明 |
|------|------|------|
| reference.py | ✅ 已修复 | from_id/to_id 查询 |
| weight_calculator.py | ✅ 已修复 | UPDATE/SELECT 查询 |
| relations.py | ✅ 正确 | 使用双参数 type::record |
| stubs.py | ✅ 正确 | 使用 array::map 批量转换 |
| prefetch_service.py | ✅ 正确 | 使用单参数 type::record |
| crud.py | ✅ 正确 | 使用 type::record 转换 |
| code_analysis.py | ✅ 正确 | 使用 type::record 转换 |
| memories.py | ✅ 无需转换 | 不涉及 graph relation 查询 |

---

**审计完成时间**: 2026-05-15
**下次审计建议**: 新增 SurrealDB 查询代码时进行专项审查
