## Why

插件端 `uploadProject` 流程中，Entity 和 Atom 创建占 81% 耗时（223 次 HTTP 调用）。Reference 已通过 batch API 优化，但 Entity 和 Atom 仍需逐条创建，成为性能瓶颈。需要添加批量创建 API 将 HTTP 请求从 227 次降至 ~7 次，总耗时从 ~117s 降至 ~15-20s。

## What Changes

- **新增** `POST /api/v1/entities/batch` 端点：批量创建 Entity（最多 100 条/批）
- **新增** `POST /api/v1/atoms/batch` 端点：批量创建 Atom（最多 100 条/批）
- **模式复用**：与现有 `POST /api/v1/references/batch` 保持一致（部分成功、去重、统计响应）
- **向后兼容**：不修改现有单条创建端点

## Capabilities

### New Capabilities
- `batch-entity-create`: 批量创建 Entity，支持去重（abstract + type + tenant_id）和部分成功模式
- `batch-atom-create`: 批量创建 Atom，支持去重（entity_id + local_id）和部分成功模式

### Modified Capabilities
- 无（仅新增 API，不修改现有功能）

## Impact

- **API 层**: `wrapper/src/routers/entity.py` 和 `wrapper/src/routers/atom.py` 新增 batch 端点
- **性能**: uploadProject 总耗时减少 85%（~117s → ~15-20s）
- **插件端**: 可检测 batch API 可用性并自动降级到逐条创建
