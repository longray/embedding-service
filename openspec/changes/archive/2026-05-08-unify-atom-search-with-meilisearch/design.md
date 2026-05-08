## Context

当前 Atom 搜索使用 SurrealDB 的 `CONTAINS` 操作符进行关键词匹配，该操作符不支持中文分词，导致中文搜索召回率极低（Keyword Recall@5 仅 0.056）。

与此同时，Entity 搜索使用 Meilisearch，支持：
- 完整的中文分词（基于 Charabia + Jieba）
- 104 词代码术语字典
- BM25 评分算法
- 同义词支持

为了统一搜索体验并提升中文搜索质量，需要将 Atom 搜索迁移到 Meilisearch。

## Goals / Non-Goals

**Goals:**
- Atom 关键词搜索使用 Meilisearch 而非 SurrealDB CONTAINS
- Atom 搜索支持中文分词和代码术语匹配
- Atom 和 Entity 搜索使用统一的架构和接口
- 现有 Atom 数据迁移到 Meilisearch
- 保持 API 向后兼容

**Non-Goals:**
- 不修改 SurrealDB 的 Atom 存储结构
- 不删除现有的 SurrealDB Atom 数据
- 不修改 Entity 搜索的实现
- 不引入新的搜索模式（保持 vector/keyword/hybrid）

## Decisions

### Decision 1: 扩展 `_build_meili_doc()` 支持 Atom 数据

**选择**: 修改现有的 `MeiliSyncMixin._build_meili_doc()` 方法，添加 `is_atom` 参数支持 Atom 文档构建。

**理由**:
- 复用现有的 Meilisearch 同步基础设施
- 保持代码一致性
- 最小化新增代码

**替代方案**: 创建独立的 `_build_atom_meili_doc()` 方法
- 拒绝原因：会增加代码重复，维护成本高

### Decision 2: Atom 使用独立的 Meilisearch 索引还是共享索引

**选择**: Atom 和 Memory 共享同一个 Meilisearch 索引，通过 `doc_type` 字段区分。

**理由**:
- 简化索引管理
- 支持跨类型搜索（如果需要）
- 减少 Meilisearch 资源占用

**替代方案**: 创建独立的 `atoms` 索引
- 拒绝原因：增加运维复杂度，需要额外配置

### Decision 3: ID 格式处理

**选择**: Meilisearch 文档 ID 使用 `atom_xxx` 格式（下划线替换冒号），保留 `surreal_id` 字段存储原始 ID。

**理由**:
- Meilisearch 文档 ID 不能包含冒号
- 与现有 Memory 文档 ID 格式保持一致（`memory_xxx`）

### Decision 4: 双写策略

**选择**: Atom CRUD 操作同步双写到 SurrealDB 和 Meilisearch。

**理由**:
- 保持数据一致性
- 支持事务回滚
- 现有 Memory 同步机制已验证

**替代方案**: 异步队列处理
- 拒绝原因：增加架构复杂度，当前数据量不需要

### Decision 5: 降级策略

**选择**: 如果 Meilisearch 不可用，Atom 关键词搜索降级到 SurrealDB BM25（而非 CONTAINS）。

**理由**:
- 保持搜索可用性
- BM25 优于 CONTAINS
- 与 Entity 搜索降级策略一致

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| **数据不一致** | 高 | 双写机制 + 迁移脚本支持重复运行 |
| **Meilisearch 存储增长** | 中 | Atom 数量通常有限（每个 Entity 5-20 个） |
| **搜索性能下降** | 低 | Meilisearch 性能优于 SurrealDB CONTAINS |
| **迁移时间** | 中 | 批量处理 + 进度报告 + 支持断点续传 |
| **ID 格式变化** | 低 | 保留 surreal_id 字段，API 返回格式不变 |

**Trade-offs**:
- **数据冗余**: Atom 数据存储在 SurrealDB + Meilisearch，换取搜索性能
- **复杂度增加**: 需要维护双写逻辑，但复用现有基础设施
- **迁移成本**: 需要运行迁移脚本，但支持增量迁移

## Migration Plan

### Phase 1: 代码变更（开发环境）
1. 修改 `_build_meili_doc()` 支持 Atom
2. 添加 Atom CRUD 双写逻辑
3. 修改 `_search_atoms_by_keyword()` 使用 Meilisearch
4. 创建迁移脚本

### Phase 2: 测试验证
1. 单元测试：Atom 文档构建
2. 集成测试：双写同步
3. 性能测试：搜索响应时间
4. 回归测试：现有功能

### Phase 3: 数据迁移（生产环境）
1. 部署代码变更（不启用 Meilisearch 搜索）
2. 运行迁移脚本将现有 Atom 数据同步到 Meilisearch
3. 验证数据完整性
4. 启用 Meilisearch 搜索

### Phase 4: 监控和回滚
1. 监控搜索质量和性能
2. 如出现问题，可回滚到 SurrealDB CONTAINS
3. 修复问题后重新启用

**Rollback Strategy**:
- 代码层面：保留原有 `_search_atoms_by_keyword()` 实现，通过配置切换
- 数据层面：SurrealDB 数据保持不变，可随时回退

## Open Questions

1. **Atom 更新频率**: 是否需要优化批量更新场景？
   - 当前假设：Atom 更新频率低，实时双写可接受

2. **Meilisearch 字段选择**: 是否需要索引所有 Atom 字段？
   - 当前决策：索引核心搜索字段（name, content, type, tags）

3. **中文分词优化**: 是否需要为 Atom 配置特定的中文分词策略？
   - 当前决策：复用 Memory 的 `localizedAttributes` 配置

4. **迁移脚本执行时间**: 大量 Atom 数据迁移可能需要较长时间
   - 缓解措施：支持分批执行、断点续传、后台运行
