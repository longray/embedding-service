# Backlog - Meilisearch 同步修复

> **问题**: Memory Search (keyword/hybrid 模式) 返回空结果  
> **根本原因**: `_build_meili_doc` 是 stub 方法，未正确构建 Meilisearch 文档  
> **影响**: Meilisearch `memories` 索引为空，keyword 搜索无法工作

---

## 使用场景

### 场景 1：Keyword 搜索
用户输入关键词 "authentication" → 后端应返回包含该词的记忆  
**当前**: 返回空结果  
**预期**: 返回匹配的记忆列表

### 场景 2：混合搜索 (Hybrid)
结合向量相似度 + 关键词匹配，提高搜索准确性  
**当前**: 只有向量搜索部分工作  
**预期**: 向量 + 关键词融合结果

### 场景 3：代码搜索
搜索代码文件中的函数名、类名  
**依赖**: Meilisearch 中的 `code_symbols` 字段

---

## Backlog 项

### BL-MS-00: 实现 Meilisearch ID 格式转换 [P0-阻塞]

| 字段 | 内容 |
|------|------|
| **目标** | 解决 SurrealDB ID (`memory:xxx`) 与 Meilisearch ID 格式不兼容问题，使同步功能正常工作 |
| **涉及范围** | `wrapper/src/utils/meili_client.py` - 添加 `_to_meili_id` 和 `_from_meili_id` 方法，在 `add_documents` 和 `search` 中自动转换 ID |
| **前置依赖** | 无，这是阻塞其他任务的基础修复 |
| **完成标准** | ① MeilisearchClient 实现 ID 转换方法 ② 上传记忆后 Meilisearch 中可查 ③ 搜索返回正确的 SurrealDB ID 格式 ④ 所有测试通过 |
| **验证方式** | ① 上传记忆 ② `curl "http://localhost:18003/indexes/memories/documents?limit=5"` 返回文档且 ID 格式为 `memory_xxx` ③ keyword 搜索返回 ID 格式为 `memory:xxx` |
| **工时预估** | 2.5 小时 |
| **技术债务** | `_to_meili_id` 方法完全缺失，`_from_meili_id` 实现错误，测试期望静态方法但实际是实例方法 |
| **架构决策** | 参见 [ADR-001-meilisearch-id-mapping.md](./architecture/ADR-001-meilisearch-id-mapping.md) |

**关键问题**:
```
当前错误: Document identifier "memory:kg3bqr1fkqehbr453tw0" is invalid
原因: Meilisearch ID 只能包含 a-zA-Z0-9-_，不能包含 :
解决: memory:xxx → memory_xxx (冒号替换为下划线)
```

---

### BL-CA-10: 修复 _build_meili_doc 方法 [P1]

| 字段 | 内容 |
|------|------|
| **目标** | 完整实现 `_build_meili_doc` 方法，正确构建包含所有必需字段的 Meilisearch 文档，使 keyword/hybrid 搜索正常工作 |
| **涉及范围** | `wrapper/src/utils/memory_manager.py` 中的 `_build_meili_doc` 方法 |
| **前置依赖** | **BL-MS-00 必须完成**（ID 格式转换是前提） |
| **完成标准** | ① `_build_meili_doc` 返回包含所有必需字段的字典 ② 必需字段: `id`, `content`, `content_zh`, `tenant_id`, `type`, `tags`, `project_id`, `created_at`, `source_id`, `metadata`, `code_language`, `code_complexity` 等 ③ 上传记忆后 Meilisearch `memories` 索引有数据 ④ keyword 搜索返回非空结果 |
| **验证方式** | ① 上传测试记忆 ② 检查 Meilisearch `curl "http://localhost:18003/indexes/memories/documents?limit=5"` 有数据 ③ keyword 搜索 `curl -X POST /api/v1/memories/search -d '{"query": "test", "mode": "keyword"}'` 返回结果 ④ 跑 pytest 测试 |
| **工时预估** | 1 小时 |

**必需字段清单** (根据 `meili_client.py` 配置):

```python
# 核心字段
- id: str                    # 记录 ID
- content: str               # 原始内容
- content_zh: str            # 中文内容（用于搜索）
- tenant_id: str             # 租户 ID
- type: str                  # 记忆类型
- tags: list[str]            # 标签
- project_id: str            # 项目 ID
- created_at: str            # 创建时间 ISO 格式
- source_id: str             # 来源 ID
- metadata: dict             # 元数据

# 代码分析字段 (Phase 1)
- code_language: str         # 代码语言
- code_complexity: int       # 复杂度
- code_function_count: int   # 函数数量
- code_class_count: int      # 类数量
- code_analyzer: str         # 分析器类型
- code_symbols: str          # 符号拼接

# 分层内容字段 (v2.4.0)
- content_abstract: str      # L0: 摘要
- content_overview: str      # L1: 概述
```

**实现参考**:
```python
def _build_meili_doc(self, record_id: str, memory_data: dict[str, Any], tenant_id: str) -> dict[str, Any]:
    """构建 Meilisearch 文档"""
    doc = {
        "id": record_id,
        "content": memory_data.get("content", ""),
        "content_zh": memory_data.get("content", ""),  # 简化处理
        "tenant_id": tenant_id,
        "type": memory_data.get("type", "general"),
        "tags": memory_data.get("tags", []),
        "project_id": memory_data.get("project_id", "global"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_id": memory_data.get("source_id", ""),
        "metadata": memory_data.get("metadata", {}),
    }
    
    # 代码分析字段
    metadata = memory_data.get("metadata", {})
    code_analysis = metadata.get("code_analysis", {})
    if code_analysis:
        doc["code_language"] = code_analysis.get("language", "")
        doc["code_complexity"] = code_analysis.get("complexity", {}).get("cyclomatic_complexity", 0)
        doc["code_function_count"] = code_analysis.get("complexity", {}).get("function_count", 0)
        doc["code_class_count"] = code_analysis.get("complexity", {}).get("class_count", 0)
        doc["code_analyzer"] = code_analysis.get("analyzer", "")
        doc["code_symbols"] = build_code_symbols(code_analysis)
    
    return doc
```

---

### BL-CA-11: 添加分层内容字段 (L0/L1/L2)

| 字段 | 内容 |
|------|------|
| **目标** | 在 `_build_meili_doc` 中添加 `content_abstract` (L0) 和 `content_overview` (L1) 字段，支持分层内容返回 |
| **涉及范围** | `wrapper/src/utils/memory_manager.py` `_build_meili_doc` 方法 |
| **前置依赖** | BL-CA-10 完成 |
| **完成标准** | ① 上传记忆时如果包含 `abstract` 和 `overview`，同步到 Meilisearch ② 搜索时可返回分层内容 |
| **验证方式** | ① 上传带 abstract/overview 的记忆 ② 验证 Meilisearch 文档包含这些字段 |

---

### BL-CA-12: 历史数据迁移 (可选)

| 字段 | 内容 |
|------|------|
| **目标** | 将 SurrealDB 中已有的记忆数据同步到 Meilisearch |
| **涉及范围** | 新增 migration 脚本或 API 端点 |
| **前置依赖** | BL-CA-10 完成并验证 |
| **完成标准** | ① 提供脚本/API 将现有记忆批量导入 Meilisearch ② 验证导入后 keyword 搜索能查到历史数据 |
| **验证方式** | ① 运行迁移脚本 ② 检查 Meilisearch 文档数 = SurrealDB 记录数 ③ keyword 搜索返回历史数据 |

---

## 依赖关系

```
BL-MS-00 (ID 格式转换) - P0 阻塞
    ↓
BL-CA-10 (修复 _build_meili_doc) - P1
    ↓
BL-CA-11 (分层内容字段) - P2
    ↓
BL-CA-12 (历史数据迁移) - P3 可选
```

**关键路径**: BL-MS-00 → BL-CA-10 → 功能可用

## 执行建议

1. **立即执行 BL-MS-00**：这是阻塞所有 Meilisearch 功能的根本原因，必须优先解决
2. **随后执行 BL-CA-10**：在 ID 转换基础上完善文档构建
3. **延后 BL-CA-11/12**：功能增强和数据迁移可以延后

---

## 技术要点

### Meilisearch 文档格式

```json
{
  "id": "memory:abc123",
  "content": "console.log('hello')",
  "content_zh": "console.log('hello')",
  "tenant_id": "default",
  "type": "code",
  "tags": ["javascript"],
  "project_id": "my-project",
  "created_at": "2026-03-31T12:00:00Z",
  "source_id": "local-001",
  "metadata": {...},
  "code_language": "javascript",
  "code_complexity": 5,
  "code_function_count": 3,
  "code_symbols": "foo bar baz"
}
```

### 验证命令

```bash
# 检查 Meilisearch 文档数
curl "http://localhost:18003/indexes/memories/stats" \
  -H "Authorization: Bearer RDo25RtbmF8BSyLyOjBgpBOH8XZo1unrbu83Gz_rX4M"

# 查询文档
curl "http://localhost:18003/indexes/memories/documents?limit=5" \
  -H "Authorization: Bearer RDo25RtbmF8BSyLyOjBgpBOH8XZo1unrbu83Gz_rX4M"

# Keyword 搜索
curl -X POST "http://localhost:17999/api/v1/memories/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "mode": "keyword", "tenant_id": "default"}'
```

---

*创建时间: 2026-03-31*
