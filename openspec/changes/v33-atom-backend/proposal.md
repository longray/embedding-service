## Why

v3.3 Atom 架构将扁平记忆存储升级为层级化知识图谱，插件端开发已启动，后端需要配合完成 Atom 字段扩展、统一搜索端点和数据库 Schema 更新，使后端 API 支持层级化 Atom 树结构和跨 Entity+Atom 搜索能力。

## What Changes

- **Atom 字段扩展**：在 `AtomCreateRequest`、`AtomUpdateRequest`、`AtomResponse` 中添加 6 个 v3.3 新字段（tags, heading_level, parent_id, order, aliases, entity_id）
- **Atom 类型扩展**：在 `ATOM_VALID_TYPES` 中新增 chapter、section 类型，支持知识文档场景
- **新建统一搜索端点**：`POST /api/v1/search`，支持跨 Entity（Meilisearch）和 Atom（SurrealDB）的混合搜索，返回统一排序结果
- **SurrealDB Schema 更新**：`atom` 表新增 6 个字段定义 + 3 个索引（parent_id, order, entity_id）
- **清理死代码**：`atom.py` 中重复的 `list_atoms` 路由（已删除）

## Capabilities

### New Capabilities

- `unified-search`: 跨 Entity + Atom 的统一搜索端点，支持 mode（vector/keyword/hybrid）、scope（all/memory/code/backlog）、types 过滤、分页

### Modified Capabilities

- `atom-crud`: Atom CRUD 端点扩展 6 个新字段，新增 chapter/section 类型支持，所有新字段 optional 保持向后兼容

## Impact

- **文件变更**：`wrapper/src/routers/atom.py`（模型+CRUD）、`wrapper/src/routers/search.py`（新建统一搜索）、`scripts/init_surrealdb.surql`（Schema）
- **API 兼容性**：所有新字段 optional，现有 API 行为不变
- **依赖**：统一搜索依赖 Meilisearch（Entity 全文）+ SurrealDB（Atom 向量/属性）
- **参考文档**：插件端设计 `D:\github\opencode-memory-plugin\docs\v3.3-ATOM-ARCHITECTURE-DESIGN.md`
