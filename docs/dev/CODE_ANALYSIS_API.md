# 代码分析 API 文档

> **版本**: v1.4  
> **状态**: 实施中  
> **适用**: 后端开发者、插件开发者

---

## 概述

本文档描述代码分析相关的 API 接口，包括代码上传、搜索、引用查询、代码地图等功能。

---

## 基础接口

### 1. 上传代码记忆

自动触发代码分析并存储。

```http
POST /api/v1/memories
Content-Type: application/json
```

**请求体**:
```json
{
  "memories": [
    {
      "type": "code",
      "content": "// 完整代码内容",
      "abstract": "TypeScript 文件：AuthService 类",
      "overview": "用户认证服务核心模块...",
      "tags": ["typescript", "auth", "service"],
      "project_id": "github.com/user/repo",
      "metadata": {
        "file_path": "src/services/auth.ts",
        "file_name": "auth.ts"
      },
      "local_id": "01HXYZ123ABC"
    }
  ],
  "tenant_id": "default"
}
```

**响应**:
```json
{
  "success": true,
  "created": [
    {
      "local_id": "01HXYZ123ABC",
      "source_id": "mem_abc123",
      "status": "created"
    }
  ]
}
```

**代码分析触发**:
- 上传时自动触发 `analyze_memory_code`
- 分析结果存入 `metadata.code_analysis`
- 同步到 Meilisearch 索引

---

### 2. 搜索代码记忆

```http
POST /api/v1/memories/search
Content-Type: application/json
```

**请求体**:
```json
{
  "query": "authentication",
  "type": "code",
  "code_filter": {
    "language": "typescript",
    "min_complexity": 5,
    "max_complexity": 15,
    "min_function_count": 3,
    "max_function_count": 20
  },
  "limit": 10
}
```

**code_filter 字段**（v1.4 完整版）:

| 字段 | 类型 | 说明 |
|------|------|------|
| `language` | string | 编程语言过滤 |
| `min_complexity` | number | 最小圈复杂度 |
| `max_complexity` | number | 最大圈复杂度 |
| `min_function_count` | number | 最小函数数量（v1.4）|
| `max_function_count` | number | 最大函数数量（v1.4）|
| `min_class_count` | number | 最小类数量（v1.4）|
| `max_class_count` | number | 最大类数量（v1.4）|
| `has_exports` | boolean | 是否有导出（v1.4）|
| `analyzer` | string | 分析器类型（v1.4）|

**响应**:
```json
{
  "results": [
    {
      "id": "mem_abc123",
      "content": "...",
      "metadata": {
        "file_path": "src/auth.ts",
        "code_analysis": {
          "language": "typescript",
          "functions": [...],
          "complexity": {...}
        }
      },
      "score": 0.95
    }
  ],
  "total": 10
}
```

---

## 代码分析专用接口（v1.4 新增）

### 3. 手动触发代码分析

```http
POST /api/v1/memories/{id}/analyze/code
```

**说明**:
- 对已有记忆重新进行代码分析
- 更新 `metadata.code_analysis`
- 同步更新 Meilisearch 索引

**响应**:
```json
{
  "success": true,
  "analysis": {
    "language": "typescript",
    "functions": [...],
    "classes": [...],
    "complexity": {...}
  }
}
```

---

### 4. 查询代码引用（v1.4 Phase 2）

查询某函数/类被哪些代码引用。

```http
GET /api/v1/memories/{id}/references
```

**响应**:
```json
{
  "references": [
    {
      "memory_id": "mem_def456",
      "file_path": "src/api.ts",
      "line": 42,
      "caller_function": "login",
      "confidence": 0.95
    }
  ],
  "total": 5
}
```

**实现说明**:
- 查询 `memory_relation` 表中 `relationship_type = "calls"`
- 返回调用该符号的所有位置

---

### 5. 查询代码依赖（v1.4 Phase 2）

查询某文件依赖哪些其他文件。

```http
GET /api/v1/memories/{id}/dependencies
```

**响应**:
```json
{
  "dependencies": [
    {
      "memory_id": "mem_ghi789",
      "file_path": "src/utils/crypto.ts",
      "line": 15,
      "callee_function": "hashPassword",
      "type": "internal"
    }
  ],
  "total": 3
}
```

**依赖类型**:
- `internal`: 项目内部文件
- `external`: 外部依赖（npm/pip/cargo）
- `builtin`: 内置模块

---

### 6. 获取项目代码地图（v1.4 Phase 3）

```http
GET /api/v1/projects/{project_id}/map
```

**响应**:
```json
{
  "project_id": "github.com/user/repo",
  "file_tree": {
    "path": "src/",
    "type": "directory",
    "children": [
      {
        "path": "src/core/",
        "type": "directory",
        "children": [
          {
            "path": "src/core/auth.ts",
            "type": "file",
            "size": 2048,
            "lines": 150,
            "functions": 5,
            "classes": 1,
            "complexity": 8.5
          }
        ]
      }
    ]
  },
  "module_dependencies": [
    {
      "from": "src/core/auth.ts",
      "to": "src/utils/crypto.ts",
      "type": "import"
    }
  ],
  "hot_files": [
    "src/core/auth.ts",
    "src/utils/api.ts"
  ],
  "statistics": {
    "total_files": 45,
    "total_functions": 150,
    "total_classes": 30,
    "avg_complexity": 5.2,
    "max_complexity": 15
  }
}
```

---

### 7. 语义代码搜索（v1.4 Phase 4）

```http
POST /api/v1/memories/search
Content-Type: application/json
```

**请求体**:
```json
{
  "semantic_query": "用户认证逻辑",
  "type": "code",
  "language": "typescript"
}
```

**说明**:
- 使用代码语义向量进行搜索
- 支持自然语言描述
- 与关键词搜索使用 RRF 融合

---

## 数据结构详解

### CodeAnalysisResult

```typescript
interface CodeAnalysisResult {
  // 基础信息
  language: string;              // "typescript", "python", etc.
  analyzer: string;             // "tree-sitter" | "regex"
  analyzed_at: string;           // ISO 8601
  analyzer_version: string;     // "1.4.0"

  // 符号信息
  functions: FunctionSymbol[];
  classes: ClassSymbol[];
  interfaces: InterfaceSymbol[];
  imports: ImportSymbol[];
  exports: ExportSymbol[];
  calls?: CallSymbol[];          // v1.4 新增

  // 复杂度
  complexity_metrics: ComplexityMetrics;

  // 依赖
  dependencies: DependencyInfo;

  // 错误/警告
  errors?: ParseError[];
  warnings?: ParseWarning[];
}
```

### FunctionSymbol（v1.4 完整版）

```typescript
interface FunctionSymbol {
  name: string;
  start_line: number;
  end_line: number;
  params: Array<{ name: string; type?: string }>;
  return_type?: string;        // v1.4 新增
  is_exported: boolean;        // v1.4 新增
  is_async: boolean;           // v1.4 新增
}
```

### ClassSymbol（v1.4 完整版）

```typescript
interface ClassSymbol {
  name: string;
  start_line: number;
  end_line: number;
  methods: string[];           // v1.4 新增
  properties: string[];        // v1.4 新增
}
```

### CallSymbol（v1.4 新增）

```typescript
interface CallSymbol {
  target: string;              // 被调用函数名
  line: number;
  column?: number;
}
```

---

## 错误处理

### 常见错误码

| 状态码 | 错误 | 说明 |
|--------|------|------|
| 400 | 无效请求 | 参数错误或缺失 |
| 404 | 记忆不存在 | 指定的记忆 ID 不存在 |
| 503 | MemoryManager 未初始化 | 服务未就绪 |
| 500 | 分析失败 | 代码解析异常 |

### 降级处理

当解析失败时，自动降级到基础信息：

```json
{
  "status": "fallback",
  "language": "unknown",
  "complexity": {
    "lines_of_code": 100
  },
  "warnings": [
    {
      "type": "degraded",
      "from": "tree-sitter",
      "to": "fallback",
      "reason": "parse_timeout"
    }
  ]
}
```

---

## 实现指南

### 添加新的 code_filter 字段

1. **更新模型** (`models.py`):
```python
class MemorySearchRequest(BaseModel):
    code_filter: dict[str, Any] | None = None
    # 新增字段已在 dict 中支持
```

2. **更新搜索逻辑** (`routers/search.py`):
```python
if "new_field" in request.code_filter:
    filter_parts.append(f'code_new_field = "{value}"')
```

3. **更新 Meilisearch 索引** (`meili_sync.py`):
```python
code_doc["code_new_field"] = analysis.get("new_field")
```

4. **更新索引配置** (`utils/meili_client.py`):
```python
DEFAULT_INDEX_SETTINGS["filterableAttributes"].append("code_new_field")
```

---

## 相关文档

- **设计文档**: `docs/CODE-ANALYSIS-DESIGN-v1.4.md`
- **产品文档**: `docs/product/CODE_ANALYSIS_FEATURES.md`
- **测试文档**: `tests/test_code_analysis.py`, `tests/test_code_analysis_integration.py`

---

*最后更新: 2026-04-07*
