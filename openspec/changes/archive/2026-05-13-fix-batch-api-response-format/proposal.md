## Why

验证发现 batch entities/atoms API 的响应格式与需求文档不匹配：
1. 当前返回 `success`/`failed` 数组，需求要求 `entities`/`atoms` 数组
2. 当前使用 `success_count`/`failed_count`，需求使用 `created`/`skipped`/`errors`
3. 缺少去重逻辑（Entity: abstract+type+tenant_id, Atom: entity_id+local_id）
4. Atom batch 缺少 entity_id 存在性验证

需要修复以符合插件端期望的 API 契约。

## What Changes

- **修改** `BatchEntityResponse` 模型：字段名从 `success`/`failed`/`success_count`/`failed_count` 改为 `entities`/`created`/`skipped`/`errors`
- **修改** `BatchAtomResponse` 模型：字段名从 `success`/`failed`/`success_count`/`failed_count` 改为 `atoms`/`created`/`skipped`/`errors`
- **新增** Entity batch 去重逻辑：相同 `abstract + type + tenant_id` 视为重复
- **新增** Atom batch 去重逻辑：相同 `entity_id + local_id` 视为重复
- **新增** Atom batch entity_id 存在性验证

## Capabilities

### Modified Capabilities
- `batch-entity-create`: 修改响应格式，添加去重逻辑
- `batch-atom-create`: 修改响应格式，添加去重和 entity_id 验证

## Impact

- **API 层**: `wrapper/src/routers/entity.py` 和 `wrapper/src/routers/atom.py` 的 batch 端点
- **响应格式**: 破坏性变更（字段名改变）
- **插件端**: 需要同步更新以使用新的响应字段名
