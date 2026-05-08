# Embedding Service 架构文档

## 概述

Embedding Service 是一个多模态记忆存储和搜索服务，支持文本嵌入、向量搜索、全文搜索和图关系查询。

## Atom Meilisearch 统一搜索架构

### 架构演进

**v3.3 之前**: Atom 搜索完全依赖 SurrealDB BM25
**v3.3 之后**: Atom 搜索优先使用 Meilisearch，降级到 SurrealDB

### 搜索流程

```
Client Request
    |
_search_atoms_by_keyword()
    |
    +-- Meilisearch 可用? --+
    |                       |
   Yes                      No
    |                       |
_search_atoms_by      _search_atoms_by
_keyword_meili()      _keyword_surreal()
    |                       |
Meilisearch CJK       SurrealDB BM25
分词搜索              CONTAINS 搜索
    |                       |
_format_atom_         直接返回
meili_results()       SurrealDB 结果
    |
ID 转换
(atom_xxx -> atom:xxx)
```

### 降级策略

当 Meilisearch 不可用时，自动降级到 SurrealDB:

```python
if state.memory_manager and state.memory_manager._meili:
    try:
        return await _search_atoms_by_keyword_meili(request)
    except Exception as e:
        logger.warning("[AtomSearch] Meilisearch 降级: %s", e)

# 降级到 SurrealDB
return await _search_atoms_by_keyword_surreal(db, request)
```

### ID 格式转换

- SurrealDB ID: `atom:01HQABC123`
- Meilisearch ID: `atom_01HQABC123`

转换规则:
- SurrealDB -> Meilisearch: `replace(":", "_", 1)`
- Meilisearch -> SurrealDB: `replace("_", ":", 1)`

### 双写同步

Atom CRUD 操作自动同步到 Meilisearch:

```python
# 创建时同步
async def _sync_atom_to_meili(atom_id, atom_data, tenant_id):
    meili_doc = state.memory_manager._build_meili_doc(
        atom_id, atom_data, tenant_id, doc_type="atom"
    )
    await meili.add_documents([meili_doc])

# 删除时同步
async def _delete_atom_from_meili(atom_id):
    meili_id = atom_id.replace(":", "_", 1)
    await meili.delete_document(meili_id)
```

### 索引配置

Meilisearch 索引支持 Atom 字段:

```python
"searchableAttributes": [
    "content_zh", "content_search", "code", "content", "name"
]

"filterableAttributes": [
    "tenant_id", "type", "tags",
    "atom_type", "entity_id", "heading_level", "local_id", "doc_type"
]
```

## 数据清空与重置

### 清空 API

**端点**: `DELETE /api/v1/memories/clear`

**流程**:
1. 验证 API Key
2. 清空 Meilisearch 所有文档
3. 清空 SurrealDB 表（memory, atom, entity, reference, conflict）

**安全机制**:
- 先清空 Meilisearch（验证 API Key）
- 成功后清空 SurrealDB
- API Key 错误时 SurrealDB 不受影响

### 重置脚本

**脚本**: `scripts/reset_all.py`

**功能**:
- 清空 SurrealDB 所有数据
- 清空 Meilisearch 所有文档
- 重新初始化 SurrealDB schema
- 重新初始化 Meilisearch 索引

**使用**:
```bash
# 预览模式
uv run python scripts/reset_all.py --dry-run

# 实际执行（跳过确认）
uv run python scripts/reset_all.py --force

# 仅重置特定服务
uv run python scripts/reset_all.py --skip-meili
uv run python scripts/reset_all.py --skip-db
```

## 版本历史

- **v2.9.0** (2026-05-08): Atom Meilisearch 统一搜索
- **v2.8.4** (2026-04-29): Atom Architecture 完整实施
- **v2.8.0** (2026-04-15): PrecomputeService + Stub 端点
- **v2.7.0** (2026-04-08): 多设备同步 + 冲突解决
- **v2.6.0** (2026-03-30): 质量治理 + MemoryManager Mixin 拆分
