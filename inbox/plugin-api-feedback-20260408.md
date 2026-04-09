# 致后端团队：API 联调问题反馈

**发件人**: OpenCode Memory Plugin (插件端) 团队  
**日期**: 2026-04-08  
**主题**: Phase 2 API 实现与文档不符问题反馈  
**联调状态**: 场景 1 已完成（使用替代方案）

---

## 1. 联调完成情况

### ✅ 已完成的测试

| 步骤 | 内容 | 结果 |
|------|------|------|
| 1 | 上传 crypto.ts | ✅ memory:ihvclhn43qeqkg3f3twt |
| 2 | 上传 auth.ts | ✅ memory:rdm1dtmxs23ca2f5vqv6 |
| 3 | 创建调用关系 | ✅ 3 个关系创建成功 |
| 4 | 查询引用 | ✅ 3 个引用 |
| 5 | 查询依赖 | ✅ 3 个依赖 |

**测试代码**:
- crypto.ts: hashPassword, verifyPassword, generateToken
- auth.ts: register, validateUser, login（调用 crypto）

---

## 2. 发现的问题

### 问题 1: 批量创建调用关系 API 不存在

**文档规定**:
```
POST /api/v1/calls/batch
{
  "calls": [{
    "caller_memory_id": "memory:xxx",
    "callee_memory_id": "memory:yyy",
    "line": 42,
    "column": 10
  }]
}
```

**实际尝试**:
```bash
POST /api/v1/calls/batch
# 返回: 404 Not Found
```

**替代方案**（当前使用）:
```bash
POST /api/v1/memories/relations
{
  "from_id": "memory:xxx",
  "to_id": "memory:yyy",
  "relationship_type": "reference",
  "description": "..."
}
```

**差异**:
- ❌ 无法指定 `line` 和 `column`
- ❌ 无法指定 `caller_memory_id` 和 `callee_memory_id`（只能用 from/to）
- ❌ `relationship_type` 不支持 `calls`

---

### 问题 2: 引用/依赖查询 API 不存在

**文档规定**:
```
GET /api/v1/memories/{id}/references
GET /api/v1/memories/{id}/dependencies
```

**实际尝试**:
```bash
GET /api/v1/memories/memory:xxx/references
# 返回: 404 Not Found
```

**替代方案**（当前使用）:
```bash
POST /api/v1/memories/{id}/relations
{
  "direction": "incoming"  // 或 "outgoing"
}
```

**差异**:
- ❌ 不是独立的 `/references` 和 `/dependencies` 端点
- ❌ 需要使用 `direction` 参数区分
- ⚠️ 返回的是通用关系，不是专门的调用关系

---

### 问题 3: relationship_type 不支持 "calls"

**文档规定**:
```json
{
  "relationship_type": "calls"
}
```

**实际错误**:
```json
{
  "error": "Invalid relationship_type: calls. Must be one of {'related', 'elaboration', 'reference', 'derived_from', 'follow_up', 'contradiction'}"
}
```

**当前使用**: `"reference"`

---

### 问题 4: direction 参数值不一致

**文档规定**:
```json
{
  "direction": "in"   // 或 "out"
}
```

**实际错误**:
```json
{
  "error": "Invalid direction: in. Must be outgoing/incoming/both"
}
```

**当前使用**: `"incoming"` / `"outgoing"`

---

## 3. 问题影响

### 功能影响

| 功能 | 预期 | 实际 | 影响 |
|------|------|------|------|
| 调用位置追踪 | 精确到 line/column | 只能描述 | 无法精确定位 |
| 调用关系类型 | 专门的 "calls" | 通用的 "reference" | 语义不明确 |
| 查询接口 | 独立端点 | 通用 relations | 使用不便 |

### 联调进度影响

- ✅ **场景 1**: 已完成（使用替代方案）
- ⚠️ **场景 2**: 代码地图 API 待验证
- ⚠️ **场景 3**: 错误处理待验证

---

## 4. 建议方案

### 方案 A: 按文档实现（推荐）

后端实现文档规定的 API：
1. 实现 `POST /api/v1/calls/batch`
2. 实现 `GET /api/v1/memories/{id}/references`
3. 实现 `GET /api/v1/memories/{id}/dependencies`
4. 支持 `relationship_type: "calls"`
5. 支持 `direction: "in" / "out"`

**优点**: 符合设计规范，功能完整  
**缺点**: 需要后端开发时间

### 方案 B: 更新文档

更新设计文档以匹配实际实现：
1. 使用 `POST /api/v1/memories/relations` 替代 `calls/batch`
2. 使用 `POST /api/v1/memories/{id}/relations` 替代独立端点
3. 使用 `"reference"` 替代 `"calls"`
4. 使用 `"incoming" / "outgoing"` 替代 `"in" / "out"`

**优点**: 快速解决，无需后端改动  
**缺点**: 功能不完整（缺少 line/column）

### 方案 C: 混合方案

1. 后端快速修复：支持 `"calls"` 类型和 `"in" / "out"` 参数
2. 插件端适配：使用现有 `relations` 端点
3. 后续迭代：实现完整的 `calls/batch` 和独立查询端点

**优点**: 平衡开发时间和功能完整性  
**缺点**: 需要双方协调

---

## 5. 需要后端确认的问题

1. **API 实现优先级**: 是否按文档实现 `calls/batch`？
2. **时间计划**: 如需要实现，预计何时完成？
3. **临时方案**: 联调期间是否使用当前替代方案？
4. **line/column 支持**: 是否可以在 relations 中存储调用位置？

---

## 6. 插件端当前状态

### 已实现（使用替代方案）

- ✅ 调用关系提取（Oxc + Tree-sitter）
- ✅ memory_id 缓存
- ✅ 关系创建（使用 `/memories/relations`）
- ✅ 双向查询（使用 `/memories/{id}/relations`）

### 待实现（等待后端确认）

- ⏳ 精确的调用位置追踪（line/column）
- ⏳ 专门的 calls 类型
- ⏳ 独立的引用/依赖查询端点

---

## 7. 下一步建议

### 短期（今天）

1. 后端确认使用哪个方案
2. 如使用方案 B 或 C，更新 API 文档
3. 继续场景 2/3 联调

### 中期（本周）

1. 根据确认的方案调整实现
2. 补充 line/column 支持（如可能）
3. 完整测试所有场景

### 长期（下周）

1. 实现完整的 calls API（如选择方案 A/C）
2. 性能优化
3. 文档同步

---

**请后端团队确认：**
- [ ] 使用哪个方案（A/B/C）？
- [ ] 是否需要调整联调计划？
- [ ] 是否可以在 relations 中支持 line/column？

期待回复，继续推进联调！

---

*文档版本: v1.0*  
*日期: 2026-04-08*  
*状态: 等待后端确认*
