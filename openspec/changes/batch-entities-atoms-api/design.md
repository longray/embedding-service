## Context

插件端 `uploadProject` 流程当前需要 227 次 HTTP 调用（43 文件项目），其中 Entity 创建 43 次、Atom 创建 180 次。Reference 已通过 batch API 优化到 4 次。需要为 Entity 和 Atom 添加类似的 batch API。

参考实现：`POST /api/v1/references/batch` (commit 892507e)

## Goals / Non-Goals

**Goals:**
- 实现 `POST /api/v1/entities/batch` 支持最多 100 条/批
- 实现 `POST /api/v1/atoms/batch` 支持最多 100 条/批
- 复用 references/batch 的模式（部分成功、去重、统计响应）
- 减少 uploadProject HTTP 请求数 97%（227 次 → ~7 次）

**Non-Goals:**
- 不修改现有单条创建端点
- 不添加事务原子性（保持部分成功模式）
- 不实现异步/队列处理（保持同步响应）

## Decisions

### Decision 1: 复用 references/batch 代码模式
**Rationale**: 保持一致性，减少维护成本。  
**Implementation**: 参考 `wrapper/src/routers/reference.py` 中的 `create_references_batch` 实现。

### Decision 2: Entity 去重策略
**Rationale**: 避免重复创建相同内容的 Entity。  
**Strategy**: 使用 `abstract + type + tenant_id` 组合判定重复。

### Decision 3: Atom 去重策略
**Rationale**: 同一 Entity 内 local_id 应唯一。  
**Strategy**: 使用 `entity_id + local_id` 组合判定重复。

### Decision 4: 部分成功模式
**Rationale**: 与 references/batch 保持一致，最大化成功率。  
**Behavior**: 某条失败不影响其他，失败条目放入 errors 数组。

## Risks / Trade-offs

**Risk**: Batch 请求处理时间较长可能导致超时  
**Mitigation**: 限制 100 条/批，保持同步处理

**Risk**: 部分成功模式可能导致数据不一致  
**Mitigation**: 客户端需检查响应中的 errors 数组并处理失败项

**Trade-off**: 去重查询增加数据库负载  
**Acceptance**: 单次 batch 最多 100 次查询，可接受
