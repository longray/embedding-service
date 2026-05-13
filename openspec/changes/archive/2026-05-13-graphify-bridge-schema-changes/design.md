## Context

Graphify 产出比 Oxc 更丰富的语义信息，需要后端 schema 支持新的字段和枚举值。

## Goals / Non-Goals

**Goals:**
- 支持 graphify 完整数据导入
- 保持向后兼容
- 不破坏现有 API

**Non-Goals:**
- 不迁移旧数据
- 不删除 depends_on 枚举值
- 不强制耦合 confidence 和 confidence_score

## Decisions

### Decision 1: confidence 和 confidence_score 独立
**Rationale**: 提供灵活性，客户端可以只提供字符串标签或数值，或两者都提供
**Implementation**: 两个字段都是 Optional，不强制校验对应关系

### Decision 2: norm_label 由客户端提供
**Rationale**: 避免后端猜测标准化规则（如是否移除下划线、如何处理特殊字符）
**Implementation**: 可选字段，null 表示未提供

### Decision 3: depends_on 软废弃
**Rationale**: 保持向后兼容，避免破坏现有数据
**Implementation**: 保留枚举值，文档标记 deprecated

### Decision 4: 同步更新 SurrealDB schema 和 Pydantic 模型
**Rationale**: 保持一致性，避免运行时错误
**Implementation**: 先更新 schema，再更新模型，最后更新 API

## Risks / Trade-offs

**Risk**: Schema 变更需要重启 SurrealDB
**Mitigation**: 使用 DEFINE FIELD IF NOT EXISTS，幂等执行

**Risk**: 新增枚举值可能导致旧客户端发送无效类型
**Mitigation**: 后端验证时返回清晰的错误信息
