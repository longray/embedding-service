# Proposal: Optimize Hybrid Weights and Meilisearch CJK

## Why

根据 backend-team 来信分析，当前搜索实现存在两个问题：
1. **Hybrid 搜索使用固定权重** - 未根据查询语言（中文/英文）调整 RRF 融合权重，导致中文搜索质量不佳
2. **Meilisearch 中文分词配置不准确** - 使用 `zho` 而非 `cmn` 语言代码，且缺少中文标点分隔符

修复后将提升中文搜索准确性和相关性，预期 Keyword Recall@5 从 0.056 提升至 0.3+。

## What Changes

1. **Hybrid 动态权重（P1）**
   - 添加 `_get_hybrid_weights()` 函数，根据查询语言动态调整权重
   - 中文：向量 50%，关键词 50%
   - 英文：向量 60%，关键词 40%
   - 修改 `_search_atoms_hybrid()` 使用动态权重

2. **Meilisearch 中文配置（P2）**
   - 修改 `localizedAttributes` 语言代码：`zho` → `cmn`
   - 添加 `separatorTokens`：中文标点 `、`, `；`, `：`
   - 保持 `nonSeparatorTokens` 不变

## Capabilities

### New Capabilities
- `hybrid-dynamic-weights`: Hybrid 搜索动态权重调整，根据查询语言自动优化向量/关键词权重

### Modified Capabilities
- `meilisearch-cjk-config`: Meilisearch 中文分词配置优化，提升中文搜索准确性

## Impact

- **Affected Files**:
  - `wrapper/src/routers/search.py`: 添加动态权重函数
  - `wrapper/src/utils/meili_client.py`: 修改索引配置
- **API Changes**: 无（内部优化，API 行为不变）
- **Database Changes**: Meilisearch 索引需要重建
- **Dependencies**: 无新增依赖
