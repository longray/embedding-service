## Why

插件端集成 graphify 替代自研 Oxc 代码分析器，需要导入 graph.json（2646 nodes + 3699 edges）。当前后端 schema 缺少 graphify 产出的关键字段：
- 关系置信度（confidence/confidence_score）用于区分 AST 提取 vs 启发式推断
- 新的关系类型（method, imports_from）用于类-方法关系和文件级导入
- 标准化名称（norm_label）用于不区分大小写的搜索

## What Changes

- **新增** reference.confidence 字段（EXTRACTED/INFERRED/AMBIGUOUS）
- **新增** reference.confidence_score 字段（0.0-1.0）
- **新增** ReferenceType.method 和 imports_from 枚举值
- **软废弃** ReferenceType.depends_on（保留兼容，新数据使用 imports/imports_from）
- **新增** entity.norm_label 字段（标准化名称）
- **新增** atom.norm_label 字段（标准化名称）

## Capabilities

### New Capabilities
- `graphify-relation-confidence`: 支持关系置信度标记和数值
- `graphify-relation-types`: 支持 method 和 imports_from 关系类型
- `graphify-norm-label`: 支持标准化名称搜索

### Modified Capabilities
- `reference-type-enum`: 新增 method, imports_from，软废弃 depends_on

## Impact

- **Schema**: SurrealDB reference/entity/atom 表结构变更
- **API**: ReferenceCreateRequest, EntityCreateRequest, AtomCreateRequest 新增字段
- **兼容性**: 所有新增字段可选，不影响现有 API
