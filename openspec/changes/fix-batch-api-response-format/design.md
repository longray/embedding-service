## Context

当前 batch API 实现与需求文档存在差异：
- 响应字段名不匹配（`success` vs `entities`, `success_count` vs `created`）
- 缺少去重逻辑
- 缺少 entity_id 验证

## Goals / Non-Goals

**Goals:**
- 修改响应格式以匹配需求文档
- 添加去重逻辑
- 添加 entity_id 存在性验证

**Non-Goals:**
- 不修改单条创建 API
- 不改变业务逻辑（仍保持部分成功模式）

## Decisions

### Decision 1: 修改响应模型字段名
**Rationale**: 与需求文档保持一致，便于插件端解析。  
**Change**: `success` → `entities`/`atoms`, `failed` → 合并到 `entities`/`atoms` 的 status 字段, `success_count` → `created`, `failed_count` → `skipped` + `errors`

### Decision 2: 统一响应项格式
**Rationale**: 需求要求每个条目包含 status 字段。  
**Format**: `{id, ..., status: "created"|"skipped"|"error", error?: string}`

### Decision 3: 去重策略
**Entity**: `abstract + type + tenant_id` 组合唯一  
**Atom**: `entity_id + local_id` 组合唯一

## Risks / Trade-offs

**Risk**: 破坏性变更影响现有客户端  
**Mitigation**: 这是新 API，尚未被插件端正式使用

**Risk**: 去重查询增加数据库负载  
**Mitigation**: 单次 batch 最多 100 条，可接受
