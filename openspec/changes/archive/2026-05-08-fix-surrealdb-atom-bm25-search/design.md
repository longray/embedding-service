# Design: Fix SurrealDB Atom BM25 Search

## Context

当前 Atom 搜索的 SurrealDB 降级路径使用 `CONTAINS` 子字符串匹配，这导致：
1. 中文搜索质量极差（不支持中文分词）
2. 无相关性评分（硬编码 score: 0.5）
3. 与 memory 表的 BM25 实现不一致

参考 memory 表的实现（`wrapper/src/utils/memory_manager/search.py` 第 193 行），它使用 `@1@` 操作符和 `search::score(1)` 获取 BM25 评分。

## Goals / Non-Goals

**Goals:**
- 为 atom 表添加 BM25 全文搜索索引
- 将 Atom 搜索降级路径从 `CONTAINS` 改为 `@@` 操作符
- 使用 BM25 相关性评分替代硬编码值
- 保持与 memory 表 BM25 实现的一致性

**Non-Goals:**
- 不修改 Meilisearch 优先路径
- 不修改 API 接口契约
- 不添加新的搜索功能

## Decisions

### Decision 1: 使用单独的 atom_analyzer
**Rationale**: 虽然可以复用 memory_analyzer，但使用单独的 analyzer 更灵活，便于未来针对 atom 内容特性调整分词策略。

**Alternative**: 复用 memory_analyzer - 更简单但不够灵活。

### Decision 2: 为 content 和 name 字段都添加 BM25 索引
**Rationale**: Atom 搜索需要同时搜索 content 和 name 字段，两者都需要 BM25 支持。

### Decision 3: 使用 @1@ 和 @2@ 操作符
**Rationale**: SurrealDB 使用 `@<index_number>@` 语法，其中数字对应 FULLTEXT 索引的创建顺序。
- @1@ = idx_atom_content_ft（第一个 FULLTEXT 索引）
- @2@ = idx_atom_name_ft（第二个 FULLTEXT 索引）

### Decision 4: 查询字符串清洗
**Rationale**: 防止 SQL 注入，参考 memory 表的 `_sanitize_query()` 实现。

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Schema 变更需要重新初始化 | 提供迁移脚本或文档说明 |
| @1@/@2@ 索引序号依赖 | 确保索引创建顺序正确，添加注释说明 |
| BM25 比 CONTAINS 慢 | 仅在降级路径使用，Meilisearch 优先路径不受影响 |

## Migration Plan

1. **开发/测试环境**: 直接重新初始化 SurrealDB
   ```bash
   uv run python scripts/reset_all.py
   uv run python scripts/init_database.py
   ```

2. **生产环境**: 需要手动添加索引
   ```sql
   -- 连接到 SurrealDB 执行
   DEFINE ANALYZER IF NOT EXISTS atom_analyzer
       TOKENIZERS class
       FILTERS lowercase, ngram(2,8);
   
   DEFINE INDEX IF NOT EXISTS idx_atom_content_ft ON atom
       FIELDS content FULLTEXT ANALYZER atom_analyzer BM25;
   
   DEFINE INDEX IF NOT EXISTS idx_atom_name_ft ON atom
       FIELDS name FULLTEXT ANALYZER atom_analyzer BM25;
   ```

## Open Questions

- 是否需要为 atom 表的其他字段（如 docstring）也添加 BM25 索引？（当前不需要，未来可扩展）
