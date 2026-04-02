# Backlog - Meilisearch 同步与搜索（场景驱动）

> **愿景**: 让用户能够通过关键词快速找到记忆库中的任何内容  
> **当前问题**: Meilisearch 同步失败，keyword/hybrid 搜索返回空结果  
> **根本原因**: ID 格式不兼容 + 架构设计缺失

---

## 用户故事

### US-01: 作为开发者，我希望能通过关键词搜索记忆
**验收标准**:
- 上传记忆后，能通过关键词搜索到
- 支持中文关键词搜索
- 搜索结果包含完整的记忆内容

### US-02: 作为 AI 助手，我希望能检索历史代码片段
**验收标准**:
- 代码记忆能被关键词搜索到
- 支持代码符号搜索（函数名、类名）
- 搜索结果能正确关联到原始记忆

### US-03: 作为系统，我希望能基于代码复杂度筛选记忆
**验收标准**:
- 代码分析字段同步到 Meilisearch
- 支持按复杂度范围过滤
- 支持按编程语言过滤

---

## Backlog 项（按优先级排序）

### BL-MS-00: 修复 ID 格式不兼容（阻塞级）

| 维度 | 内容 |
|------|------|
| **目标** | 解决 SurrealDB ID (`memory:xxx`) 与 Meilisearch ID 格式不兼容问题，使同步功能正常工作，满足 US-01 基本需求 |
| **涉及范围** | `wrapper/src/utils/meili_client.py`<br>① 添加 `_to_meili_id()` 方法：将 `memory:xxx` 转换为 `memory_xxx`<br>② 添加 `_from_meili_id()` 方法：将 `memory_xxx` 还原为 `memory:xxx`<br>③ 在 `add_documents()` 中自动转换 ID<br>④ 在 `search()` 中自动还原 ID |
| **前置依赖** | 无，这是阻塞所有其他任务的基础设施 |
| **完成标准** | ① MeilisearchClient 实现 ID 转换方法并通过单元测试<br>② 上传记忆后，Meilisearch 中可查且 ID 格式正确（`memory_xxx`）<br>③ 搜索返回的 ID 格式正确（`memory:xxx`） |
| **验证方式** | 步骤 1: 上传记忆<br>`curl -X POST http://localhost:17999/api/v1/memories -H "Content-Type: application/json" -d '{"memories":[{"content":"测试","abstract":"摘要","overview":"概览"}]}'`<br><br>步骤 2: 检查 Meilisearch 文档 ID 格式<br>`curl "http://localhost:18003/indexes/memories/documents?limit=1" -H "Authorization: Bearer $MEILI_MASTER_KEY"`<br>期望：`{"id": "memory_xxxxx", ...}`（下划线格式）<br><br>步骤 3: 搜索并检查返回 ID 格式<br>`curl -X POST http://localhost:17999/api/v1/memories/search -H "Content-Type: application/json" -d '{"query":"测试","mode":"keyword"}'`<br>期望：`{"memory_ids": ["memory:xxxxx", ...]}`（冒号格式）<br><br>步骤 4: 运行测试<br>`uv run pytest tests/test_meili_integration.py -v` |
| **工时预估** | 2.5 小时 |
| **技术债务** | `_to_meili_id` 方法缺失，`_from_meili_id` 实现错误，测试与代码不匹配 |
| **架构决策** | 参见 [ADR-001-meilisearch-id-mapping.md](./architecture/ADR-001-meilisearch-id-mapping.md) |

**关键问题日志**:
```
当前错误: Document identifier "memory:kg3bqr1fkqehbr453tw0" is invalid
原因: Meilisearch ID 只能包含 a-zA-Z0-9-_，不能包含冒号
解决方案: memory:xxx → memory_xxx (冒号替换为下划线)
```

---

### BL-MS-01: 完善 _build_meili_doc 字段构建

| 维度 | 内容 |
|------|------|
| **目标** | 确保 _build_meili_doc 构建的文档包含所有搜索必需的字段，满足 US-01/US-02/US-03 的搜索质量需求 |
| **涉及范围** | `wrapper/src/utils/memory_manager.py` 中的 `_build_meili_doc` 方法<br>① 确保 `surreal_id` 字段存在（冗余存储，便于调试）<br>② 确保所有必需字段存在：id, content, content_zh, tenant_id, type, tags, project_id, created_at, source_id, metadata, abstract, overview<br>③ 代码分析字段：code_language, code_complexity, code_function_count, code_class_count, code_analyzer, code_symbols |
| **前置依赖** | **BL-MS-00 必须完成**（ID 格式转换是前提） |
| **完成标准** | ① `_build_meili_doc` 返回的文档包含所有必需字段<br>② 上传代码记忆时，code_* 字段正确填充<br>③ Meilisearch 索引配置与字段匹配（searchableAttributes, filterableAttributes）<br>④ 分层字段 abstract/overview 正确同步 |
| **验证方式** | 步骤 1: 上传代码记忆<br>`curl -X POST http://localhost:17999/api/v1/memories -H "Content-Type: application/json" -d '{"memories":[{"content":"def foo(): pass","abstract":"函数","overview":"Python函数","type":"code","metadata":{"code_analysis":{"language":"python","complexity":{"cyclomatic_complexity":1}}}}]}'`<br><br>步骤 2: 检查 Meilisearch 文档完整性<br>`curl "http://localhost:18003/indexes/memories/documents?limit=1" -H "Authorization: Bearer $MEILI_MASTER_KEY"`<br>期望：包含所有字段（content, content_zh, code_language, code_complexity, abstract, overview 等）<br><br>步骤 3: 代码搜索验证<br>`curl -X POST http://localhost:17999/api/v1/memories/search -H "Content-Type: application/json" -d '{"query":"foo","mode":"keyword"}'`<br>期望：返回匹配的记忆 |
| **工时预估** | 0.5 小时（检查当前实现，可能已完成） |
| **当前状态** | ⚠️ 代码已实现，需验证字段完整性 |

---

### BL-MS-02: 支持代码复杂度筛选

| 维度 | 内容 |
|------|------|
| **目标** | 支持按代码复杂度、语言等属性筛选记忆，满足 US-03 需求 |
| **涉及范围** | `wrapper/src/utils/memory_manager.py`<br>① `_search_by_keyword` 方法支持 code_filter 参数<br>② 将 code_filter 转换为 Meilisearch filter 表达式<br>③ 支持的筛选条件：language, min_complexity, max_complexity |
| **前置依赖** | BL-MS-00, BL-MS-01 完成 |
| **完成标准** | ① 可以通过 code_filter 按语言筛选<br>② 可以通过 code_filter 按复杂度范围筛选<br>③ 组合筛选正常工作（language + min_complexity） |
| **验证方式** | 步骤 1: 上传不同复杂度的代码<br>步骤 2: 按复杂度筛选<br>`curl -X POST http://localhost:17999/api/v1/memories/search -d '{"query":"function","mode":"keyword","code_filter":{"min_complexity":5}}'`<br>期望：只返回复杂度 >= 5 的记忆 |
| **工时预估** | 1.5 小时 |

---

### BL-MS-03: 历史数据迁移（可选）

| 维度 | 内容 |
|------|------|
| **目标** | 将 SurrealDB 中已有的记忆数据同步到 Meilisearch，使历史数据可搜索 |
| **涉及范围** | 新增脚本 `scripts/migrate_to_meilisearch.py` 或 API 端点<br>① 查询 SurrealDB 中所有记忆<br>② 批量构建 Meilisearch 文档<br>③ 批量写入 Meilisearch |
| **前置依赖** | BL-MS-00, BL-MS-01 完成并验证 |
| **完成标准** | ① 提供迁移脚本或 API<br>② 迁移后 Meilisearch 文档数 = SurrealDB 记录数<br>③ 历史数据可通过 keyword 搜索 |
| **验证方式** | 步骤 1: 运行迁移<br>`uv run python scripts/migrate_to_meilisearch.py`<br><br>步骤 2: 检查文档数<br>`curl "http://localhost:18003/indexes/memories/stats"`<br>期望：numberOfDocuments 等于 SurrealDB 记忆数 |
| **工时预估** | 2 小时 |
| **优先级** | P3（可选），如果只需要新数据可搜索则可延后 |

---

### BL-MS-04: 修复测试与代码不匹配

| 维度 | 内容 |
|------|------|
| **目标** | 修复测试文件中的错误期望，使测试能正确验证功能（当前架构：ID 转换在 MeilisearchClient 中，而非 MemoryManager） |
| **涉及范围** | `tests/test_meili_integration.py`<br>① 修正 `TestMeiliIdConversion` 测试：方法已在 `MeilisearchClient` 中实现，而非 `MemoryManager`<br>② 更新 ID 格式期望：`memory:xxx` → `memory_xxx`（下划线格式在 Meilisearch 中）<br>③ 确保测试使用正确的 mock 数据和断言 |
| **前置依赖** | **BL-MS-00 完成**（ID 转换实现位置已确定） |
| **完成标准** | ① `TestMeiliIdConversion` 测试正确调用 `MeilisearchClient` 方法<br>② 所有 Meilisearch 相关测试通过<br>③ 测试正确验证 ID 转换逻辑 |
| **验证方式** | `uv run pytest tests/test_meili_integration.py -v`<br>期望：所有 23 个测试通过 |
| **工时预估** | 1 小时 |
| **关键问题** | 当前测试期望 `_to_meili_id` 是 `MemoryManager` 的静态方法，但架构决策将其放在 `MeilisearchClient` 实例方法中 |

---

## 依赖关系图

```
┌─────────────────────────────────────────────────────────────┐
│ BL-MS-00: ID 格式转换 (P0-阻塞)                              │
│ • 目标：解决同步失败                                          │
│ • 工时：2.5h                                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
┌──────────────────┐ ┌──────────┐ ┌──────────────────┐
│ BL-MS-01: 字段   │ │ BL-MS-04:│ │ BL-MS-02: 代码   │
│ 完善 (P1)        │ │ 修复测试 │ │ 筛选 (P2)        │
│ • 工时：1h       │ │ (P2)     │ │ • 工时：1.5h     │
└──────────────────┘ │ • 工时：1h│ └──────────────────┘
                     └──────────┘
                           │
                           ▼
               ┌────────────────────┐
               │ BL-MS-03: 历史迁移 │
               │ (P3-可选)          │
               │ • 工时：2h         │
               └────────────────────┘
```

**关键路径**: BL-MS-00 → BL-MS-01 → 功能可用（3.5 小时）
**完整路径**: BL-MS-00 → BL-MS-01 → BL-MS-04 → 测试通过（4.5 小时）

---

## 执行建议

### 第一阶段：基础设施（必须）
1. **BL-MS-00** (2.5h) ✅ 已完成 - ID 格式转换正常工作

### 第二阶段：验证与修复
2. **BL-MS-01** (0.5h) ⏳ 待验证 - 检查字段完整性（代码已实现）
3. **BL-MS-04** (1h) ⏳ 待处理 - 修复测试与代码不匹配

### 第三阶段：增强功能（可选）
4. **BL-MS-02** (1.5h) - 代码筛选（如果需要）
5. **BL-MS-03** (2h) - 历史迁移（如果需要）

---

## 当前状态

| Backlog | 状态 | 说明 |
|---------|------|------|
| BL-MS-00 | ✅ 已完成 | ID 格式转换正常工作 |
| BL-MS-01 | ✅ 已完成 | 字段完整，代码记忆可搜索（已验证） |
| BL-MS-02 | ⏳ 待处理 | 代码筛选功能 |
| BL-MS-03 | ⏳ 待处理 | 历史数据迁移（可选） |
| BL-MS-04 | ⏳ 待处理 | 12 个测试失败，需修复 |

---

## 验证清单（整体）

- [x] 上传记忆后 Meilisearch 中有数据
- [x] keyword 搜索返回正确结果
- [x] 搜索结果 ID 格式正确（`memory:xxx`）
- [x] 代码记忆可被搜索（BL-MS-01 已验证）
- [ ] 代码筛选功能正常（需 BL-MS-02 实现）
- [ ] 所有测试通过（需 BL-MS-04 修复）

---

*最后更新: 2026-04-01*
