# Proposal: Fix SurrealDB Atom BM25 Search

## Why

根据 backend-team 来信分析，当前 Atom 搜索的 SurrealDB 降级路径使用 `CONTAINS` 子字符串匹配，而非 BM25 全文搜索。当 Meilisearch 不可用时，Atom 搜索质量极差，中文几乎不可用。需要修复此问题以提升降级路径的搜索质量。

## What Changes

1. **添加 atom 表 BM25 索引**: 在 `scripts/init_surrealdb.surql` 中为 atom 表添加 BM25 全文搜索索引（content 和 name 字段）
2. **修改搜索逻辑**: 将 `wrapper/src/routers/search.py` 中的 `CONTAINS` 操作符改为 `@@` BM25 操作符
3. **添加相关性评分**: 使用 `search::score()` 函数获取 BM25 相关性评分
4. **更新结果处理**: 使用动态评分替代硬编码的 0.5

## Capabilities

### New Capabilities
- `atom-bm25-search`: Atom 表的 BM25 全文搜索支持，包含索引定义和查询逻辑

### Modified Capabilities
- `unified-search`: 修改 SurrealDB 降级路径的 Atom 关键词搜索实现

## Impact

- **Affected Files**:
  - `scripts/init_surrealdb.surql`: 添加 atom_analyzer 和 BM25 索引
  - `wrapper/src/routers/search.py`: 修改 `_search_atoms_by_keyword()` 函数
- **API Changes**: 无（内部实现优化，API 行为不变）
- **Database Changes**: 需要重新初始化 SurrealDB 或运行 schema 迁移
- **Dependencies**: 依赖 SurrealDB 3.0+ 的 BM25 全文搜索功能
