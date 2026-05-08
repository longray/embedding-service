# Design: Optimize Hybrid Weights and Meilisearch CJK

## Context

当前搜索实现存在两个关键问题：

1. **Hybrid 搜索固定权重**: 使用硬编码 0.5/0.5，未考虑中英文差异
   - 中文语义鸿沟大，关键词匹配更重要
   - 英文向量语义匹配效果好

2. **Meilisearch 中文配置**: 使用 `zho` 而非 `cmn`，缺少中文标点分隔符
   - `zho` 是宏观语言代码，无法触发 jieba 分词
   - 中文标点未被识别为分隔符

## Goals / Non-Goals

**Goals:**
- 实现 Hybrid 搜索动态权重（中文 50/50，英文 60/40）
- 修复 Meilisearch 中文分词配置（zho→cmn，添加 separatorTokens）
- 提升中文搜索准确性和相关性

**Non-Goals:**
- 不修改 SurrealDB 配置（已在其他变更中完成）
- 不修改 API 接口契约
- 不添加新的搜索模式

## Decisions

### Decision 1: 动态权重作为默认值
**Rationale**: 保持向后兼容，允许配置覆盖
**Alternative**: 强制动态权重 - 不够灵活

### Decision 2: 正则表达式检测中文
**Rationale**: 简单高效，覆盖常用中文字符
**Pattern**: `[\u4e00-\u9fa5]`（基本汉字范围）

### Decision 3: 分阶段实施
**Rationale**: 
- 阶段 1（P1）: Hybrid 动态权重 - 低风险，立即生效
- 阶段 2（P2）: Meilisearch 配置 - 需重建索引，风险较高

### Decision 4: 语言代码 zho→cmn
**Rationale**: Meilisearch Charabia tokenizer 使用 cmn 触发 jieba
**Source**: Meilisearch 官方文档

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| 动态权重可能不适合所有场景 | 保持配置覆盖能力 |
| Meilisearch 索引重建耗时 | 低峰期实施，准备回滚 |
| 正则检测中文不准确 | 使用 Unicode 范围，覆盖 99% 场景 |

## Migration Plan

### 阶段 1: Hybrid 动态权重
1. 添加 `_get_hybrid_weights()` 函数
2. 修改 `_search_atoms_hybrid()`
3. 运行测试验证
4. 部署到生产

### 阶段 2: Meilisearch 配置
1. 备份当前配置
2. 修改 meili_client.py
3. 测试环境验证
4. 低峰期重建索引
5. 验证中文搜索质量

## Open Questions

- 是否需要支持其他语言（日文、韩文）的动态权重？
- 是否需要更复杂的中文检测（如 CJK 扩展字符）？
