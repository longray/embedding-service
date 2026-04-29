## 1. Atom 字段扩展（atom.py）

- [ ] 1.1 在 `AtomCreateRequest` 添加 6 个新字段：tags, heading_level, parent_id, order, aliases, entity_id
- [ ] 1.2 在 `AtomUpdateRequest` 添加对应新字段（全部 optional）
- [ ] 1.3 在 `AtomResponse` 添加对应新字段（含默认值）
- [ ] 1.4 在 `ATOM_VALID_TYPES` 中添加 "chapter", "section" 类型
- [ ] 1.5 更新 `create_atom` 端点的 atom_data 构建逻辑
- [ ] 1.6 更新 `batch_create_atoms` 端点的 atom_data 构建逻辑
- [ ] 1.7 添加 heading_level 范围验证（1-6）
- [ ] 1.8 编写单元测试验证新字段创建/更新/响应

## 2. 统一搜索端点（search.py）

- [ ] 2.1 创建 `SearchRequest` Pydantic 模型（query, mode, scope, types, atom_types, limit, level, tenant_id）
- [ ] 2.2 创建 `SearchResponse`、`EntityResult`、`AtomResult` Pydantic 模型
- [ ] 2.3 实现 `POST /api/v1/search` 路由
- [ ] 2.4 实现 Entity 搜索逻辑（调用现有 Meilisearch/SurrealDB）
- [ ] 2.5 实现 Atom 搜索逻辑（SurrealDB WHERE LIKE 查询）
- [ ] 2.6 实现混合结果合并排序（按 score 降序）
- [ ] 2.7 在 `main.py` 中注册 search router（如需修改）
- [ ] 2.8 编写搜索端点单元测试（mock SurrealDB/Meilisearch）

## 3. SurrealDB Schema 更新

- [ ] 3.1 在 `init_surrealdb.surql` atom 表添加 6 个新字段定义
- [ ] 3.2 添加 3 个新索引：idx_atom_parent, idx_atom_order, idx_atom_entity
- [ ] 3.3 验证 schema 幂等性（IF NOT EXISTS）

## 4. 验证与测试

- [ ] 4.1 运行现有测试套件确认无回归
- [ ] 4.2 手动测试 Atom CRUD 新字段
- [ ] 4.3 手动测试统一搜索端点
- [ ] 4.4 验证向后兼容性（旧 Atom 无新字段仍正常工作）
