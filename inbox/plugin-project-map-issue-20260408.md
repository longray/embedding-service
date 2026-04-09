# 致后端团队：项目地图 API 数据为空问题反馈

**发件人**: OpenCode Memory Plugin (插件端) 团队  
**日期**: 2026-04-08  
**主题**: GET /api/v1/projects/{id}/map 返回空数据问题  
**联调场景**: 场景 2 - 代码地图

---

## 1. 问题概述

项目地图 API 端点存在，但返回的所有数据字段均为空/零值。

---

## 2. 测试环境

- **后端地址**: http://localhost:17999
- **测试时间**: 2026-04-08 15:50
- **后端版本**: 2.4.1
- **联调阶段**: 场景 2（代码地图）

---

## 3. 详细测试记录

### 3.1 已上传的代码文件

| 文件 | Memory ID | Project ID | 状态 |
|------|-----------|------------|------|
| crypto.ts | memory:ihvclhn43qeqkg3f3twt | global (默认) | ✅ 已上传 |
| auth.ts | memory:rdm1dtmxs23ca2f5vqv6 | global (默认) | ✅ 已上传 |
| api.ts | memory:kypsd1yi6eroed7xy0k8 | global (默认) | ✅ 已上传 |

**上传请求示例**:
```json
POST /api/v1/memories
{
  "memories": [{
    "content": "...",
    "abstract": "...",
    "type": "code",
    "metadata": {
      "file_path": "src/utils/crypto.ts",
      "code_analysis": { ... }
    }
  }],
  "tenant_id": "default"
}
```

**注意**: 上传时未显式指定 `project_id`，使用默认值。

---

### 3.2 项目地图 API 测试结果

#### 测试 1: 使用 project_id = "global"

**请求**:
```bash
GET http://localhost:17999/api/v1/projects/global/map
```

**响应**:
```json
{
  "status": "success",
  "project_id": "global",
  "file_tree": [],
  "module_dependencies": [],
  "hot_files": [],
  "statistics": {
    "total_files": 0,
    "total_functions": 0,
    "total_classes": 0,
    "avg_complexity": 0,
    "max_complexity": 0
  }
}
```

**结果**: ✅ 200 OK，但所有数据为空

---

#### 测试 2: 使用 project_id = "test"

**请求**:
```bash
GET http://localhost:17999/api/v1/projects/test/map
```

**响应**:
```json
{
  "status": "success",
  "project_id": "test",
  "file_tree": [],
  "module_dependencies": [],
  "hot_files": [],
  "statistics": {
    "total_files": 0,
    "total_functions": 0,
    "total_classes": 0,
    "avg_complexity": 0,
    "max_complexity": 0
  }
}
```

**结果**: ✅ 200 OK，但所有数据为空

---

#### 测试 3: 使用 project_id = "github.com/test/integration"

**请求**:
```bash
GET http://localhost:17999/api/v1/projects/github.com%2Ftest%2Fintegration/map
```

**响应**:
```json
{
  "detail": "Not Found"
}
```

**结果**: ❌ 404 Not Found

---

### 3.3 项目统计 API 测试结果

**请求**:
```bash
GET http://localhost:17999/api/v1/projects/test/stats
```

**响应**:
```json
{
  "detail": "Internal Server Error"
}
```

**结果**: ❌ 500 Internal Server Error

---

## 4. 问题分析

### 4.1 可能的原因

#### 原因 1: Project ID 不匹配

**分析**: 
- 上传代码时未指定 `project_id`，使用默认值
- 查询时使用的 `project_id` 可能与存储的不匹配
- 需要确认 memories 表中存储的 `project_id` 是什么

**验证请求**:
```bash
# 请后端查询 memories 表中的 project_id
SELECT DISTINCT project_id FROM memories WHERE type = 'code';
```

---

#### 原因 2: 数据聚合逻辑未实现

**分析**:
- API 端点存在，但内部可能未实现数据聚合
- 需要从 memories 表中提取 code_analysis 数据并聚合
- 可能缺少索引或查询逻辑

**需要验证**:
- 后端是否实现了从 memories 聚合项目数据的逻辑？
- 是否查询了正确的表和字段？

---

#### 原因 3: 调用关系数据未关联

**分析**:
- 模块依赖 (module_dependencies) 需要从调用关系生成
- 我们已上传 6 个调用关系（auth→crypto 3个，api→auth 3个）
- 但这些关系可能未关联到项目地图

**验证请求**:
```bash
# 请后端查询 calls 表中的数据
SELECT COUNT(*) FROM calls;
SELECT * FROM calls LIMIT 5;
```

---

#### 原因 4: 缺少项目级索引

**分析**:
- 可能需要专门的 project 表或索引
- 或者需要在 memories 表上创建 project_id + type 的复合索引

---

## 5. 需要后端协助排查

### 5.1 数据库查询

请执行以下查询，帮助定位问题：

```sql
-- 1. 确认代码 memories 的 project_id
SELECT DISTINCT project_id, COUNT(*) 
FROM memories 
WHERE type = 'code' 
GROUP BY project_id;

-- 2. 确认调用关系数据
SELECT COUNT(*) as total_calls FROM calls;

-- 3. 查看 memories 的 metadata 结构
SELECT id, project_id, metadata->>'file_path' as file_path
FROM memories 
WHERE type = 'code' 
LIMIT 5;

-- 4. 确认是否有 project 表
SELECT * FROM projects LIMIT 5;
```

---

### 5.2 代码检查

请检查以下实现：

1. **Project Map 生成逻辑**:
   - 文件: `wrapper/src/utils/memory_manager/project_map.py` (假设)
   - 检查是否从 memories 表查询数据
   - 检查是否正确解析 code_analysis 字段

2. **数据聚合逻辑**:
   - 是否遍历所有 memories 并聚合 file_path？
   - 是否计算 complexity_metrics？
   - 是否生成 module_dependencies？

3. **错误处理**:
   - `/stats` 端点返回 500，请查看错误日志
   - 是否有未捕获的异常？

---

## 6. 建议的修复方案

### 方案 A: 实现数据聚合（推荐）

如果尚未实现，请添加以下逻辑：

```python
def generate_project_map(project_id):
    # 1. 查询所有代码 memories
    memories = db.query("""
        SELECT metadata, code_analysis 
        FROM memories 
        WHERE project_id = %s AND type = 'code'
    """, project_id)
    
    # 2. 构建 file_tree
    file_tree = build_file_tree(memories)
    
    # 3. 查询调用关系
    calls = db.query("""
        SELECT * FROM calls 
        WHERE caller_memory_id IN (SELECT id FROM memories WHERE project_id = %s)
    """, project_id)
    
    # 4. 构建 module_dependencies
    dependencies = build_dependencies(calls)
    
    # 5. 计算统计信息
    stats = calculate_stats(memories)
    
    return {
        "file_tree": file_tree,
        "module_dependencies": dependencies,
        "statistics": stats
    }
```

---

### 方案 B: 使用现有数据

如果数据已存在但查询有问题：

1. 检查 project_id 匹配逻辑
2. 添加调试日志输出查询结果
3. 确认字段名映射正确

---

## 7. 临时解决方案

如果今天无法修复，建议：

1. **插件端**: 使用内存中的分析数据生成项目地图（不依赖后端 API）
2. **后端**: 记录为已知问题，后续迭代修复
3. **联调**: 跳过场景 2，继续场景 3（错误处理）

---

## 8. 需要后端确认

1. **实现状态**: 项目地图的数据聚合逻辑是否已实现？
2. **数据检查**: 请执行 5.1 节的 SQL 查询，确认数据是否存在
3. **错误日志**: `/stats` 500 错误的详细日志是什么？
4. **修复时间**: 今天能否修复？还是需要临时方案？

---

## 9. 附件

### 已创建的调用关系

| 调用者 | 被调用者 | Line | Column |
|--------|---------|------|--------|
| auth.ts | crypto.ts | 6 | 25 |
| auth.ts | crypto.ts | 12 | 16 |
| auth.ts | crypto.ts | 21 | 20 |
| api.ts | auth.ts | 8 | 27 |
| api.ts | auth.ts | 20 | 12 |
| api.ts | auth.ts | 24 | 12 |

**预期 module_dependencies**:
```json
[
  {"from": "src/auth.ts", "to": "src/utils/crypto.ts", "type": "import"},
  {"from": "src/api.ts", "to": "src/auth.ts", "type": "import"}
]
```

---

期待后端的排查结果！

---

*文档版本: v1.0*  
*日期: 2026-04-08*  
*状态: 等待后端排查*
