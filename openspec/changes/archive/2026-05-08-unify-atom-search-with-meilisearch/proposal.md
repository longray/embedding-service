## Why

当前 Atom 搜索使用 SurrealDB 的 `CONTAINS` 操作符进行关键词搜索，该操作符不支持中文分词，导致中文搜索召回率极低（Keyword Recall@5 仅 0.056）。与此同时，Entity 搜索使用 Meilisearch，支持完整的中文分词、代码术语字典和 BM25 评分。

为了统一搜索体验、提升中文搜索质量，并简化架构维护，需要将 Atom 搜索迁移到 Meilisearch，使其与 Entity 搜索完全一致。

## What Changes

- **新增 Atom Meilisearch 索引**：创建 atom 专用的 Meilisearch 文档结构和索引配置
- **扩展 Meilisearch 客户端**：修改 `_build_meili_doc()` 支持 atom 数据模型
- **双写同步机制**：Atom CRUD 操作同步到 Meilisearch（创建、更新、删除）
- **统一搜索路由**：修改 `_search_atoms_by_keyword()` 使用 Meilisearch 而非 SurrealDB CONTAINS
- **数据迁移脚本**：将现有 atom 数据迁移到 Meilisearch
- **ID 转换处理**：处理 `atom:xxx` 格式 ID 在 Meilisearch 中的存储和查询

**BREAKING**: Atom 搜索将完全依赖 Meilisearch，如果 Meilisearch 不可用，Atom 关键词搜索将降级到 SurrealDB BM25（行为变化）。

## Capabilities

### New Capabilities

- `atom-meilisearch-indexing`: Atom 数据 Meilisearch 索引构建和同步机制
- `atom-search-unification`: 统一 Atom 和 Entity 的搜索接口和实现
- `atom-meili-migration`: 现有 Atom 数据迁移到 Meilisearch 的脚本和流程

### Modified Capabilities

- `memory-search`: 修改搜索路由以支持统一的 Atom/Entity 搜索结果融合

## Impact

- **代码文件**: 
  - `wrapper/src/utils/memory_manager/meili_sync.py` - 扩展 `_build_meili_doc()`
  - `wrapper/src/utils/memory_manager/crud.py` - Atom CRUD 双写逻辑
  - `wrapper/src/routers/search.py` - 统一搜索路由
  - `wrapper/src/utils/meili_client.py` - 索引配置扩展
  
- **API 变化**: Atom 搜索响应格式保持不变，但搜索质量显著提升

- **依赖**: 需要 Meilisearch 服务可用

- **数据**: 新增 Meilisearch 索引存储 atom 数据（冗余存储）

- **性能**: Atom 关键词搜索性能提升（Meilisearch 优于 SurrealDB CONTAINS）

- **部署**: 需要运行迁移脚本将现有 atom 数据同步到 Meilisearch
