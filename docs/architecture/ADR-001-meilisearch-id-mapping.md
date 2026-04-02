# 架构决策记录：Meilisearch ID 格式映射

**决策编号**: ADR-001  
**日期**: 2026-04-01  
**状态**: 部分实施 — ID 转换已实现，测试同步未完成
**影响范围**: wrapper/src/utils/memory_manager.py, wrapper/src/utils/meili_client.py

---

## 背景

在实现 Meilisearch 双写同步时，发现 SurrealDB 和 Meilisearch 的 ID 格式存在根本性不兼容：

| 系统 | ID 格式示例 | 约束 |
|------|------------|------|
| SurrealDB | `memory:abc123` | 使用 `:` 分隔表名和记录 ID |
| Meilisearch | `abc123` | 只能包含 `a-zA-Z0-9-_`，**不能包含 `:`** |

这导致上传记忆时 Meilisearch 报错：
```
Document identifier "memory:kg3bqr1fkqehbr453tw0" is invalid.
A document identifier can be of type integer or string, 
only composed of alphanumeric characters (a-z A-Z 0-9), hyphens (-) and underscores (_)
```

## 决策

### 方案选择

经过评估，选择 **方案 C：Meilisearch 客户端层处理**

在 `MeilisearchClient` 中封装 ID 转换逻辑：
- `add_documents()` 自动将 SurrealDB ID 转换为 Meilisearch ID
- `search()` 返回时自动将 Meilisearch ID 还原为 SurrealDB ID
- `MemoryManager` 不感知 ID 格式差异

### ID 映射规则

```
SurrealDB ID          Meilisearch ID          说明
─────────────────────────────────────────────────────────
memory:abc123    →    memory_abc123     (冒号替换为下划线)
memory:550e-84   →    memory_550e-84    (保留原有连字符)
```

**关键设计**:
1. **保留表名前缀**：`memory:` → `memory_`，便于区分不同类型文档
2. **使用下划线替换冒号**：避免与 SurrealDB 的 `:` 冲突，同时保持可读性
3. **双向可逆**：通过简单的字符串替换即可双向转换

### 接口契约

#### MeilisearchClient 层

```python
class MeilisearchClient:
    def _to_meili_id(self, surreal_id: str) -> str:
        """SurrealDB ID → Meilisearch ID
        
        转换规则:
        - memory:abc123 → memory_abc123
        - 只替换第一个冒号，保留后续字符
        """
        return surreal_id.replace(":", "_", 1)
    
    def _from_meili_id(self, meili_id: str) -> str:
        """Meilisearch ID → SurrealDB ID
        
        转换规则:
        - memory_abc123 → memory:abc123
        - 将第一个下划线还原为冒号
        """
        return meili_id.replace("_", ":", 1)
    
    async def add_documents(self, documents: list[dict]) -> None:
        """添加文档，自动转换 ID"""
        # 转换所有文档的 ID
        converted_docs = []
        for doc in documents:
            converted_doc = doc.copy()
            if "id" in converted_doc:
                converted_doc["id"] = self._to_meili_id(converted_doc["id"])
            converted_docs.append(converted_doc)
        
        # 调用底层 API
        ...
    
    async def search(self, query: str, **options) -> list[dict]:
        """搜索，返回时还原 ID"""
        # 调用底层 API
        results = ...
        
        # 还原所有结果的 ID
        for hit in results:
            if "id" in hit:
                hit["id"] = self._from_meili_id(hit["id"])
        
        return results
```

#### MemoryManager 层

```python
class MemoryManager:
    def _build_meili_doc(self, record_id: str, ...) -> dict:
        """构建 Meilisearch 文档
        
        注意: record_id 保持 SurrealDB 格式 (memory:xxx)
        转换由 MeilisearchClient 自动处理
        """
        return {
            "id": record_id,  # SurrealDB 格式，如 memory:abc123
            "surreal_id": record_id,  # 冗余存储，便于调试
            # ... 其他字段
        }
```

## 影响分析

### 正面影响

1. **职责清晰**：ID 转换逻辑集中在 MeilisearchClient，符合单一职责原则
2. **透明性**：MemoryManager 不需要关心底层存储的 ID 格式
3. **可维护**：所有 ID 相关逻辑在一个类中，便于修改和测试

### 负面影响

1. **性能开销**：每次 add/search 都需要遍历转换所有 ID
2. **复杂性增加**：需要确保所有调用点都经过转换层

### 风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 遗漏调用点 | 中 | 高 | 代码审查 + 全面测试 |
| 现有数据不一致 | 低 | 中 | 检查 Meilisearch 是否为空，必要时清空重建 |
| 测试与实现不匹配 | 中 | 中 | 更新测试用例，确保期望正确的 ID 格式 |

## 实施计划

### 前置依赖

1. 确认 Meilisearch 当前数据状态（是否为空）
2. 备份现有数据（如有）
3. 准备测试数据

### 实施步骤

1. **在 MeilisearchClient 中添加转换方法** (30 min)
   - 实现 `_to_meili_id` 和 `_from_meili_id`
   - 添加单元测试

2. **修改 add_documents 自动转换 ID** (30 min)
   - 在发送前转换所有文档 ID
   - 验证转换正确性

3. **修改 search 返回时还原 ID** (30 min)
   - 在返回前还原所有结果 ID
   - 验证还原正确性

4. **更新 MemoryManager._build_meili_doc** (15 min)
   - 保持使用 SurrealDB ID 格式
   - 添加 `surreal_id` 冗余字段

5. **更新测试用例** (30 min)
   - 修正测试期望的 ID 格式
   - 添加 ID 转换专项测试

6. **验证端到端流程** (15 min)
   - 上传记忆 → 检查 Meilisearch → 搜索验证

**总计**: 约 2.5 小时

## 替代方案评估

### 方案 A：MemoryManager 层处理（已否决）

**描述**: 在 `MemoryManager` 中实现 `_to_meili_id` 和 `_from_meili_id` 静态方法。

```python
class MemoryManager:
    @staticmethod
    def _to_meili_id(surreal_id: str) -> str:
        return surreal_id.replace(":", "_")
    
    @staticmethod
    def _from_meili_id(meili_id: str) -> str:
        return meili_id.replace("_", ":", 1)
```

**否决原因**:
- 需要在多个地方调用转换（上传、搜索、代码分析更新等）
- 容易遗漏调用点
- 职责不清晰（MemoryManager 不应该关心 Meilisearch 的 ID 约束）

### 方案 B：_build_meili_doc 内联处理（已否决）

**描述**: 在 `_build_meili_doc` 中直接转换 ID，不单独抽取方法。

```python
def _build_meili_doc(self, record_id: str, ...):
    return {
        "id": record_id.replace(":", "_"),  # 直接转换
        # ...
    }
```

**否决原因**:
- 重复代码（搜索反向转换仍需处理）
- 无法复用转换逻辑
- 测试困难
- 搜索时容易遗漏反向转换
- 重复代码（搜索反向转换仍需处理）
- 无法复用转换逻辑
- 测试困难

## 验证标准

1. ✅ 上传记忆后 Meilisearch 文档 ID 格式为 `memory_xxx`（下划线）
2. ✅ 搜索返回的 ID 格式为 `memory:xxx`（冒号）
3. ⚠️ 部分测试通过（18/23），TestUploadDualWrite 因 mock 不匹配失败
4. ✅ 新增 ID 转换专项测试（TestMeiliIdConversion 6/6, TestFormatMeiliResults 6/6）

## 实施进度

| 步骤 | 状态 | 说明 |
|------|------|------|
| MeilisearchClient 添加转换方法 | ✅ 已完成 | `_to_meili_id` / `_from_meili_id` |
| add_documents 自动转换 ID | ✅ 已完成 | |
| search 返回时还原 ID | ✅ 已完成 | |
| MemoryManager 双重转换修复 | ✅ 已完成 | `_format_meili_results` 不再重复转换 |
| 更新测试用例 | ⚠️ 部分完成 | ID 转换测试通过，双写测试 mock 需修复 |
| 端到端验证 | ✅ 已完成 | keyword 搜索正常工作 |

## 相关文档

- [BACKLOG-meilisearch-sync-fix-v2.md](../BACKLOG-meilisearch-sync-fix-v2.md)
- tests/test_meili_integration.py

---

**决策人**: Sisyphus (AI Assistant)
**审核人**: [待填写]
**实施状态**: ⚠️ 部分实施 — 核心功能已工作，测试覆盖待完善
