# 后端团队回复：项目地图查询问题已修复

**发件人**: Embedding Service (后端团队)  
**日期**: 2026-04-08  
**主题**: 项目地图查询返回空数据 - 已修复  
**状态**: ✅ 已修复，等待验证

---

## 修复概览

| 问题 | 状态 | 修复时间 |
|------|------|----------|
| 项目地图查询返回空数据 | ✅ **已修复** | 2026-04-08 |

---

## 问题根因

**查询条件过于严格**：

项目地图查询要求 `metadata.code_analysis IS NOT NONE`，但插件端上传的数据没有 `code_analysis` 字段。

```sql
-- 修改前（过于严格）
WHERE tenant_id = $tenant_id
    AND type = 'code'
    AND project_id = $project_id
    AND metadata.file_path IS NOT NONE
    AND metadata.code_analysis IS NOT NONE  -- 此条件导致查询失败

-- 修改后（修复后）
WHERE tenant_id = $tenant_id
    AND type = 'code'
    AND project_id = $project_id
    AND metadata.file_path IS NOT NONE
    -- 移除了 metadata.code_analysis IS NOT NONE 条件
```

---

## 修复详情

### 修改文件

**文件**: `wrapper/src/utils/memory_manager/stubs.py`

**行号**: 199-213

**修改内容**:
```python
# 修改前
files_query = """
    SELECT
        id AS memory_id,
        metadata.file_path AS file_path,
        ...
    FROM memory
    WHERE tenant_id = $tenant_id
        AND type = 'code'
        AND project_id = $project_id
        AND metadata.file_path IS NOT NONE
        AND metadata.code_analysis IS NOT NONE  -- 移除此行
"""

# 修改后
files_query = """
    SELECT
        id AS memory_id,
        metadata.file_path AS file_path,
        ...
    FROM memory
    WHERE tenant_id = $tenant_id
        AND type = 'code'
        AND project_id = $project_id
        AND metadata.file_path IS NOT NONE
"""
```

---

## 验证测试

### 测试场景

**测试 1**: 上传文件（无 code_analysis）
```javascript
POST /api/v1/memories
{
  "memories": [{
    "type": "code",
    "content": "export function helper() { return 1; }",
    "abstract": "Helper function",
    "project_id": "test-project",
    "metadata": {
      "file_path": "src/helper.ts",
      "language": "typescript"
      // 注意：没有 code_analysis
    }
  }]
}
// ✅ 上传成功
```

**测试 2**: 创建调用关系
```javascript
POST /api/v1/calls/batch
{
  "calls": [{"caller_id": "...", "callee_id": "...", "relation_type": "calls"}]
}
// ✅ 创建成功
```

**测试 3**: 查询项目地图
```bash
GET /api/v1/projects/test-project/map

// ✅ 返回结果
{
  "file_tree": [{"name": "src", "children": [...]}],
  "module_dependencies": [{"from": "src/main.ts", "to": "src/helper.ts", "type": "call"}],
  "hot_files": ["src/main.ts", "src/helper.ts"],
  "statistics": {"total_files": 2, ...}
}
```

### 验证结果

| 测试项 | 结果 |
|--------|------|
| 上传文件（无 code_analysis） | ✅ 成功 |
| 创建调用关系 | ✅ 成功 |
| 项目地图查询 | ✅ **成功** |
| file_tree | ✅ 非空 |
| module_dependencies | ✅ 包含调用关系 |
| statistics | ✅ 正确统计 |

---

## 建议验证步骤

请插件端进行以下验证：

```bash
# 1. 上传测试文件（不带 code_analysis）
POST /api/v1/memories
{"memories": [{"type": "code", "project_id": "test-integration-project", ...}]}

# 2. 创建调用关系
POST /api/v1/calls/batch
{"calls": [...]}

# 3. 查询项目地图
GET /api/v1/projects/test-integration-project/map

# 应该返回非空的 file_tree 和 module_dependencies
```

---

## 已知限制

| 限制 | 说明 |
|------|------|
| 复杂度统计 | 无 code_analysis 时，complexity/function_count/class_count 为 null |
| 热点文件 | 按文件路径排序，不依赖复杂度 |

---

## 代码提交

**Commit**: 待提交（修复在本地）  
**文件**: `wrapper/src/utils/memory_manager/stubs.py`  
**变更**: -1 行（移除 `AND metadata.code_analysis IS NOT NONE`）

---

## 下一步

**插件端**: 请验证项目地图查询是否正常工作  
**后端**: 等待验证通过后提交代码

---

**修复完成时间**: 2026-04-08  
**等待插件端验证**: ⏳

---

*文档版本: v1.0*  
*状态: 已发送，等待确认*
