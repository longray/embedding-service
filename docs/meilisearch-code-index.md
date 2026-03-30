# Meilisearch 代码搜索索引扩展方案

> **版本**: v1.0  
> **日期**: 2026-03-31  
> **状态**: 待审核（仅文档，未动代码）  
> **关联**: CODE-ANALYSIS-DESIGN-v1.2.md

---

## 1. 当前索引配置

文件: `wrapper/src/utils/meili_client.py` 第 43-59 行

### 1.1 现有 filterableAttributes

```python
"filterableAttributes": [
    "tenant_id",
    "type",
    "tags",
    "project_id",
    "date",
    "ip_address",
    "email",
    "version",
    "created_at",
    "source_id",
    "code_language",      # ← 已有
    "code_complexity",    # ← 已有
]
```

### 1.2 现有 searchableAttributes

```python
"searchableAttributes": [
    "content_zh",
    "title_zh",
    "tags_zh",
    "content_search",
    "code",
    "content"
]
```

### 1.3 现有 sortableAttributes

```python
"sortableAttributes": [
    "date",
    "created_at",
    "code_complexity"
]
```

---

## 2. 需要扩展的字段

### 2.1 新增 filterableAttributes

| 字段 | 来源路径 | 说明 |
|------|---------|------|
| `code_function_count` | `metadata.code_analysis.complexity_metrics.function_count` | 按函数数量过滤 |
| `code_class_count` | `metadata.code_analysis.complexity_metrics.class_count` | 按类数量过滤 |
| `code_analyzer` | `metadata.code_analysis.analyzer` | 按分析器类型过滤 |

### 2.2 新增 searchableAttributes

| 字段 | 来源路径 | 说明 |
|------|---------|------|
| `code_symbols` | 自定义拼接（见下文） | 符号名全文搜索 |

### 2.3 code_symbols 拼接逻辑

上传代码记忆时，后端自动拼接符号名到 `code_symbols` 字段：

```python
def build_code_symbols(code_analysis: dict) -> str:
    """将代码分析结果中的符号名拼接为可搜索文本"""
    parts = []

    # 函数名
    for func in code_analysis.get("functions", []):
        parts.append(func.get("name", ""))

    # 类名
    for cls in code_analysis.get("classes", []):
        parts.append(cls.get("name", ""))
        for method in cls.get("methods", []):
            parts.append(method if isinstance(method, str) else method.get("name", ""))

    # 接口名
    for iface in code_analysis.get("interfaces", []):
        parts.append(iface.get("name", ""))

    # 导出名
    for exp in code_analysis.get("exports", []):
        name = exp if isinstance(exp, str) else exp.get("name", "")
        parts.append(name)

    return " ".join(p for p in parts if p)
```

---

## 3. 升级后的完整配置

```python
DEFAULT_INDEX_SETTINGS = {
    "searchableAttributes": [
        "content_zh",
        "title_zh",
        "tags_zh",
        "content_search",
        "code",
        "content",
        "code_symbols",           # ← 新增
    ],
    "filterableAttributes": [
        "tenant_id",
        "type",
        "tags",
        "project_id",
        "date",
        "ip_address",
        "email",
        "version",
        "created_at",
        "source_id",
        "code_language",
        "code_complexity",
        "code_function_count",    # ← 新增
        "code_class_count",       # ← 新增
        "code_analyzer",          # ← 新增
    ],
    "sortableAttributes": [
        "date",
        "created_at",
        "code_complexity",
        "code_function_count",    # ← 新增
    ],
    # 其他配置保持不变
    "nonSeparatorTokens": [".", "-", "@", ":", "/", "_"],
    "localizedAttributes": [{"locales": ["zho"], "attributePatterns": ["*_zh"]}],
    "typoTolerance": {"enabled": True, "disableOnAttributes": ["file_path", "version", "email", "ip_address"]},
    "dictionary": [
        # 现有字典内容保持不变...
    ],
}
```

---

## 4. 索引更新方式

### 4.1 热更新（推荐）

Meilisearch 支持 `updateSettings` 热更新，**无需停机**：

```python
# 在 MeilisearchClient 中调用
await client.configure_index(new_settings)
```

**影响**:

- 更新期间搜索正常可用
- 已有数据不会被重新索引
- 新的 filterableAttributes 只对更新后写入的数据生效

### 4.2 历史数据回填

已有代码记忆需要回填新字段：

```python
# 回填脚本逻辑
for doc in index.get_all_documents():
    code_analysis = doc.get("metadata", {}).get("code_analysis", {})
    if code_analysis:
        updates = {}

        # 回填 code_symbols
        updates["code_symbols"] = build_code_symbols(code_analysis)

        # 回填 function_count / class_count
        complexity = code_analysis.get("complexity_metrics", {})
        if "function_count" in complexity:
            updates["code_function_count"] = complexity["function_count"]
        if "class_count" in complexity:
            updates["code_class_count"] = complexity["class_count"]
        if "analyzer" in code_analysis:
            updates["code_analyzer"] = code_analysis["analyzer"]

        if updates:
            index.update_documents([{"id": doc["id"], **updates}])
```

**预计耗时**：

- 10,000 条记忆 → ~3-5 分钟
- 100,000 条记忆 → ~15-20 分钟

### 4.3 更新时机

| 阶段 | 操作 |
|------|------|
| Phase 1 | 热更新 settings + 插件端上传时自动填充新字段 |
| Phase 2 | 回填历史数据（如有） |

---

## 5. 搜索示例

### 5.1 按语言过滤

```json
POST /api/v1/memories/search
{
  "query": "authentication",
  "code_filter": {
    "language": "typescript"
  }
}
```

后端转换为 Meilisearch filter:

```python
filter = 'code_language = "typescript"'
```

### 5.2 按复杂度范围过滤

```json
POST /api/v1/memories/search
{
  "query": "复杂函数",
  "code_filter": {
    "min_complexity": 10,
    "max_complexity": 30
  }
}
```

后端转换为 Meilisearch filter:

```python
filter = 'code_complexity >= 10 AND code_complexity <= 30'
```

### 5.3 按函数数量排序

```json
GET /api/v1/memories/search?query=service&sort=code_function_count:desc
```

（注：当前搜索 API 不支持 sort 参数，Phase 2 可扩展）

---

## 6. 上传时字段映射

插件端上传 → 后端提取 → Meilisearch 索引字段

| 插件端字段 | 后端提取路径 | Meilisearch 索引字段 |
|-----------|------------|---------------------|
| `language` | `metadata.code_analysis.language` | `code_language` |
| `complexity_metrics.cyclomatic` | `metadata.code_analysis.complexity_metrics.cyclomatic` | `code_complexity` |
| `complexity_metrics.function_count` | `metadata.code_analysis.complexity_metrics.function_count` | `code_function_count` |
| `complexity_metrics.class_count` | `metadata.code_analysis.complexity_metrics.class_count` | `code_class_count` |
| `analyzer` | `metadata.code_analysis.analyzer` | `code_analyzer` |
| 所有符号名拼接 | `build_code_symbols()` | `code_symbols` |

**提取逻辑位置**: `memory_manager.py` 的 `upload_memories()` 方法中，写入 Meilisearch 前提取。

---

## 7. 变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `wrapper/src/utils/meili_client.py` | 修改 | DEFAULT_INDEX_SETTINGS 新增 4 个字段 |
| `wrapper/src/utils/memory_manager.py` | 修改 | 上传时提取 code 字段 + 构建 code_symbols |
| `scripts/init_surrealdb.surql` | 无需修改 | 不涉及 |

---

*文档结束 - 等待审核确认后执行代码变更*
