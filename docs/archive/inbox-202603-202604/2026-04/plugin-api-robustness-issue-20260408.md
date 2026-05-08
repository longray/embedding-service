# 致后端团队：API 健壮性问题反馈

**发件人**: OpenCode Memory Plugin (插件端) 团队  
**日期**: 2026-04-08  
**主题**: 上传 API 对字段缺失敏感，导致用户可能失败  
**优先级**: P1 - 影响用户体验

---

## 1. 关键发现

**API 对字段缺失敏感，导致上传成功但数据不可查询**

### 对比测试

| 场景 | 字段 | 上传结果 | 查询结果 | 状态 |
|------|------|---------|---------|------|
| **完整字段** | 包含 overview, code_analysis | ✅ 200 | ✅ 200 | 正常 |
| **缺少 overview** | 无 overview | ✅ 200 | ❌ 404 | **问题** |
| **缺少 code_analysis** | 无 code_analysis | ✅ 200 | ❌ 404 | **问题** |
| **缺少 project_id** | 无 project_id | ✅ 200 | ❌ 404 | **问题** |

---

## 2. 问题分析

### 现象

**请求**（缺少某些字段）:
```json
POST /api/v1/memories
{
  "memories": [{
    "type": "code",
    "content": "export function test() {}",
    "abstract": "Test",
    "metadata": {
      "file_path": "test.ts"
      // ❌ 缺少 code_analysis
    }
    // ❌ 缺少 overview
    // ❌ 缺少 project_id
  }]
}
```

**响应**:
```json
{
  "total": 1,
  "success": 1,
  "memory_ids": ["memory:xxx"]
}
```

**查询**:
```bash
GET /api/v1/memories/memory:xxx
# 返回: {"detail":"记忆不存在"}
```

**矛盾点**:
- 上传返回成功
- 但数据无法查询
- 用户无法感知失败

---

## 3. 根本原因

### 可能原因 1: 字段验证导致写入失败

**流程**:
```
1. API 接收请求
2. 基础验证通过（返回 200）
3. 数据库写入时字段验证失败
4. 事务回滚，但 API 已返回成功
5. 数据不存在
```

### 可能原因 2: 索引构建失败

**流程**:
```
1. 数据写入数据库
2. 构建搜索索引时失败（缺少必需字段）
3. 查询时通过索引查找失败
4. 返回 404
```

### 可能原因 3: 查询条件不匹配

**流程**:
```
1. 数据写入（缺少某些字段）
2. 查询时要求字段必须存在
3. 条件不匹配，返回 404
```

---

## 4. 影响评估

### 用户场景

**场景 1: 简化上传**
```javascript
// 用户只提供基本字段
await uploadMemories([{
  type: 'code',
  content: code,
  abstract: 'Code file'
  // 用户没提供 overview, code_analysis
}]);
// 结果: 上传成功，但查询不到！
```

**场景 2: 快速分析**
```javascript
// 快速分析，不等待完整结果
await uploadMemories([{
  type: 'code',
  content: code,
  metadata: { file_path: path }
  // 没提供 code_analysis
}]);
// 结果: 上传成功，但查询不到！
```

### 影响范围

| 用户类型 | 影响 | 概率 |
|---------|------|------|
| 普通用户 | 高 | 高 |
| 开发者 | 中 | 中 |
| 自动化工具 | 高 | 高 |

---

## 5. 建议修复方案

### 方案 1: 同步验证（推荐）

**修改**: 上传时同步验证所有必需字段

```python
@app.post("/api/v1/memories")
async def upload_memories(memories):
    required_fields = ['content', 'abstract', 'type']
    code_required_fields = ['overview', 'project_id', 'metadata.file_path']
    
    for memory in memories:
        # 验证必需字段
        if memory.get('type') == 'code':
            for field in code_required_fields:
                if not get_nested_field(memory, field):
                    return {
                        "success": 0,
                        "failed": 1,
                        "errors": [f"Missing required field: {field}"]
                    }
        
        # 同步写入并验证
        memory_id = await write_to_db(memory)
        verify = await query_db(memory_id)
        if not verify:
            return {
                "success": 0,
                "failed": 1,
                "errors": ["Write verification failed"]
            }
    
    return {"success": len(memories), "memory_ids": [...]}
```

**优点**:
- 立即反馈错误
- 用户知道哪些字段缺失
- 数据一致性保证

---

### 方案 2: 自动填充默认值

**修改**: 为缺失字段提供默认值

```python
def fill_defaults(memory):
    defaults = {
        'overview': memory.get('abstract', ''),
        'project_id': 'global',
        'metadata.code_analysis': {
            'language': 'unknown',
            'functions': []
        }
    }
    
    for field, default in defaults.items():
        if not get_nested_field(memory, field):
            set_nested_field(memory, field, default)
    
    return memory
```

**优点**:
- 向后兼容
- 用户无需修改代码
- 灵活容错

---

### 方案 3: 分离验证和写入

**修改**: 添加验证端点

```python
@app.post("/api/v1/memories/validate")
async def validate_memories(memories):
    """预验证，不写入"""
    errors = []
    for memory in memories:
        errors.extend(validate_memory(memory))
    return {"valid": len(errors) == 0, "errors": errors}

@app.post("/api/v1/memories")
async def upload_memories(memories):
    """写入，假设已验证"""
    # 同步写入并确认
    ...
```

**优点**:
- 明确分离验证和写入
- 插件端可以先验证再上传

---

## 6. 推荐方案

**首选: 方案 2（自动填充默认值）+ 方案 1（同步验证）**

组合优点:
1. 自动填充保证兼容性
2. 同步验证保证数据一致性
3. 用户获得明确反馈

---

## 7. 临时解决方案

### 插件端立即修复

在插件端添加强制字段检查:

```javascript
function validateMemory(memory) {
  const required = ['content', 'abstract', 'type'];
  const codeRequired = ['overview', 'project_id', 'metadata.file_path'];
  
  if (memory.type === 'code') {
    for (const field of codeRequired) {
      if (!getField(memory, field)) {
        throw new Error(`Missing required field: ${field}`);
      }
    }
  }
}

async function uploadWithValidation(memories) {
  for (const memory of memories) {
    validateMemory(memory);
  }
  return await wrapperClient.uploadMemories(memories);
}
```

---

## 8. 需要后端确认

1. **根本原因**: 是字段验证、索引构建还是查询条件问题？
2. **修复方案**: 采用哪个方案？
3. **修复时间**: 今天能否修复？
4. **临时方案**: 插件端先添加验证是否可行？

---

## 9. 当前联调状态

| 功能 | 状态 | 说明 |
|------|------|------|
| 完整字段上传 | ✅ | 使用完整字段可成功 |
| 简化字段上传 | ❌ | 可能失败 |
| 调用关系 | ⏳ | 依赖完整上传 |
| 项目地图 | ⏳ | 依赖完整上传 |

**建议**: 联调使用完整字段格式，同时修复健壮性问题。

---

**请后端确认根本原因和修复方案！**

---

*文档版本: v1.0*  
*日期: 2026-04-08*  
*优先级: P1*  
*状态: 等待确认*
