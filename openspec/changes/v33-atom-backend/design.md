## Context

v3.3 Atom 架构将记忆系统从扁平存储升级为层级化知识图谱。插件端已完成设计（参考 `D:\github\opencode-memory-plugin\docs\v3.3-ATOM-ARCHITECTURE-DESIGN.md`），后端需要配合完成：

- 当前 `atom` 表只有代码分析相关字段（function/class/interface 等）
- v3.3 新增知识文档场景字段（chapter/section 的层级、排序、标签）
- 当前搜索只有 `/api/v1/memories/search`（仅 memory 表），缺少跨 Entity+Atom 搜索
- SurrealDB schema 在 `scripts/init_surrealdb.surql` L346-378

**约束**：

- 所有新字段必须 optional，现有 API 行为不变
- 统一搜索端点不修改现有 `/api/v1/memories/search`
- SurrealDB 3.0+ 语法（SCHEMAFULL → SCHEMALESS for atom 表，已是无 schema）

## Goals / Non-Goals

**Goals:**

- Atom CRUD 支持 6 个 v3.3 新字段：tags, heading_level, parent_id, order, aliases, entity_id
- Atom 类型扩展支持 chapter、section
- 新建 `POST /api/v1/search` 统一搜索端点，返回混合 Entity + Atom 结果
- SurrealDB atom 表 schema 更新（字段 + 索引）
- 响应时间 < 200ms（1000 条数据）

**Non-Goals:**

- 自动数据迁移（旧 Atom 保持原格式）
- 插件端实现（在 opencode-memory-plugin 仓库）
- Obsidian 导入/导出（Phase 3）
- 循环引用检测（后端不负责，插件端处理）
- memory_read 返回值变更（插件端负责）

## Decisions

### D1: 新建独立搜索端点 vs 扩展现有

**选择**：新建 `POST /api/v1/search`

**理由**：

- 现有 `/api/v1/memories/search` 返回 `MemorySearchResult`，结构固定
- 统一搜索返回 `EntityResult | AtomResult` 联合类型，结构不同
- 独立端点零风险，不破坏现有 API 契约
- 两个端点可独立演进

### D2: Atom 搜索实现方式

**选择**：SurrealDB WHERE 子句 + Meilisearch 全文搜索

**理由**：

- Atom 在 SurrealDB 中存储，字段过滤用 SurrealDB WHERE
- Entity 在 Meilisearch 中有全文索引，搜索走 Meilisearch
- 混合结果按 score 降序排序
- 后续可扩展为 Atom 也索引到 Meilisearch

### D3: 新字段全部 optional

**选择**：所有 6 个新字段默认 None/空

**理由**：

- 旧 Atom 数据不需要迁移
- 代码分析场景的 Atom 不需要这些字段
- 只有知识文档场景（chapter/section）才需要层级信息
- `entity_id` 暂时 optional，未来可改为 required

## Risks / Trade-offs

- **[搜索性能]** Atom 表无全文索引 → 缓解：先用 WHERE LIKE，后续可加 FULLTEXT 索引
- **[atom 表 SCHEMALESS]** 无 schema 验证 → 缓解：Pydantic 模型层做验证
- **[跨 repo 协调]** API 契约变更需同步插件端 → 缓解：通过 mailbox 通知
- **[重复路由]** atom.py 已清理死代码 → 风险已消除
