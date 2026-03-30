# API 契约文档 - 代码分析集成

> **版本**: v1.0  
> **日期**: 2026-03-31  
> **状态**: 待审核（仅文档，未动代码）  
> **关联**: CODE-ANALYSIS-DESIGN-v1.2.md

---

## 1. 概述

本文档定义插件端与后端之间的 API 契约，涵盖代码记忆的上传、搜索和生命周期管理。

**基础 URL**: `http://localhost:17999`  
**认证**: 当前未启用（`WRAPPER_AUTH_ENABLED` 默认 false）  
**数据格式**: JSON

---

## 2. 上传代码记忆

### 2.1 端点

```text
POST /api/v1/memories
```

### 2.2 请求体

```json
{
  "memories": [
    {
      "content": "import { Parser } from 'tree-sitter';\n\nexport class CodeAnalyzer {\n  private parser: Parser;\n\n  async analyze(filePath: string): Promise<AnalysisResult> {\n    const code = await fs.readFile(filePath, 'utf-8');\n    return this.parse(code);\n  }\n\n  private parse(code: string): AnalysisResult {\n    const tree = this.parser.parse(code);\n    return { symbols: this.extractSymbols(tree) };\n  }\n}",
      "abstract": "TypeScript 代码分析器模块，含 CodeAnalyzer 类和 analyze/parse 方法",
      "overview": "代码分析器核心模块，基于 Tree-sitter 实现多语言解析。导出 CodeAnalyzer 类，提供 analyze 和 parse 方法用于解析代码文件并提取符号信息。",
      "type": "code",
      "tags": ["typescript", "code-analysis", "tree-sitter", "analyzer"],
      "project_id": "github.com/opencode/opencode-memory-plugin",
      "metadata": {
        "file_path": "src/analyzer.ts",
        "file_name": "analyzer.ts",
        "code_analysis": {
          "language": "typescript",
          "analyzer": "tree-sitter",
          "analyzed_at": "2026-03-31T12:00:00Z",
          "analyzer_version": "1.0.0",
          "functions": [],
          "classes": [
            {
              "name": "CodeAnalyzer",
              "start_line": 3,
              "end_line": 14,
              "methods": ["analyze", "parse"],
              "properties": ["parser"]
            }
          ],
          "interfaces": [],
          "imports": [
            {
              "source": "tree-sitter",
              "imported_names": ["Parser"],
              "is_default": false,
              "is_namespace": false
            }
          ],
          "exports": [
            {
              "name": "CodeAnalyzer",
              "type": "class",
              "is_default": false
            }
          ],
          "dependencies": {
            "internal": [],
            "external": ["tree-sitter"],
            "builtin": ["fs"]
          },
          "complexity_metrics": {
            "cyclomatic": 3,
            "lines_of_code": 14,
            "function_count": 0,
            "class_count": 1,
            "max_function_complexity": 0,
            "average_function_complexity": 0.0
          },
          "errors": [],
          "warnings": []
        }
      },
      "local_id": "01JMZX1BEJJBFSW9G3V5WZHJRN"
    }
  ],
  "tenant_id": "default"
}
```

### 2.3 请求字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `memories` | Array | ✅ | 记忆列表，支持批量上传 |
| `memories[].content` | string | ✅ | 完整代码内容（L2 层） |
| `memories[].abstract` | string | 推荐 | ≤100 字符摘要（L0 层） |
| `memories[].overview` | string | 推荐 | ≤500 字符概览（L1 层） |
| `memories[].type` | string | ✅ | 固定为 `"code"` |
| `memories[].tags` | string[] | 推荐 | 语言+功能标签 |
| `memories[].project_id` | string | ✅ | Git remote URL 或项目名 |
| `memories[].metadata.file_path` | string | ✅ | 文件相对路径（用于 upsert） |
| `memories[].metadata.file_name` | string | 推荐 | 文件名 |
| `memories[].metadata.code_analysis` | object | ✅ | 代码分析结果（见 Schema 文档） |
| `memories[].local_id` | string | 推荐 | 插件端 ULID |
| `tenant_id` | string | 可选 | 租户 ID，默认 `"default"` |

### 2.4 成功响应（200）

```json
{
  "success": true,
  "created": [
    {
      "local_id": "01JMZX1BEJJBFSW9G3V5WZHJRN",
      "source_id": "mem_abc123def456",
      "status": "created"
    }
  ]
}
```

### 2.5 更新响应（200，upsert 场景）

```json
{
  "success": true,
  "updated": [
    {
      "local_id": "01JMZX1BEJJBFSW9G3V5WZHJRN",
      "source_id": "mem_abc123def456",
      "status": "updated",
      "previous_version": "2026-03-31T10:00:00Z"
    }
  ]
}
```

### 2.6 错误响应

| 状态码 | 场景 | 示例 |
|--------|------|------|
| 400 | 数据验证失败 | `{"detail": "content 字段不能为空"}` |
| 500 | 后端内部错误 | `{"detail": "上传失败: SurrealDB connection error"}` |

---

## 3. 搜索代码记忆

### 3.1 端点

```text
POST /api/v1/memories/search
```

### 3.2 基础搜索（所有记忆混合）

```json
{
  "query": "认证功能",
  "mode": "hybrid",
  "limit": 10,
  "threshold": 0.7,
  "level": 1
}
```

返回结果包含 `type: "code"` 和 `type: "general"` 的混合记忆。

### 3.3 仅搜索代码记忆

```json
{
  "query": "authentication",
  "mode": "hybrid",
  "limit": 10,
  "code_filter": {
    "language": "typescript"
  }
}
```

### 3.4 code_filter 支持的参数

| 参数 | 类型 | 说明 | Meilisearch 转换 |
|------|------|------|-----------------|
| `language` | string | 编程语言 | `code_language = "typescript"` |
| `min_complexity` | int | 最小圈复杂度 | `code_complexity >= 5` |
| `max_complexity` | int | 最大圈复杂度 | `code_complexity <= 10` |

**组合过滤**：

```json
{
  "query": "复杂逻辑",
  "code_filter": {
    "language": "python",
    "min_complexity": 10,
    "max_complexity": 30
  }
}
```

转换为 Meilisearch filter:

```text
code_language = "python" AND code_complexity >= 10 AND code_complexity <= 30
```

### 3.5 搜索响应

```json
{
  "results": [
    {
      "id": "mem_abc123def456",
      "content": "// 完整代码内容...",
      "abstract": "TypeScript 代码分析器模块",
      "overview": "代码分析器核心模块...",
      "type": "code",
      "tags": ["typescript", "code-analysis"],
      "project_id": "github.com/opencode/opencode-memory-plugin",
      "metadata": {
        "file_path": "src/analyzer.ts",
        "code_analysis": {
          "language": "typescript",
          "analyzer": "tree-sitter",
          "complexity_metrics": {
            "cyclomatic": 3,
            "function_count": 0,
            "class_count": 1
          }
        }
      },
      "score": 0.92,
      "created_at": "2026-03-31T12:00:00Z"
    }
  ],
  "total": 5,
  "query": "authentication",
  "mode": "hybrid"
}
```

### 3.6 level 参数控制返回内容

| level | 返回字段 | 适用场景 |
|-------|---------|---------|
| 0 | 仅 `abstract` | 快速浏览列表 |
| 1 | `abstract` + `overview` | 了解概要 |
| 2 | 完整内容（默认） | 查看详情 |

---

## 4. Upsert 逻辑

### 4.1 唯一键

代码记忆使用以下组合作为唯一键：

```text
file_path + project_id + tenant_id
```

### 4.2 Upsert 流程

```text
插件端上传代码记忆
  ↓
后端检查是否已存在：
  SELECT id FROM memory
  WHERE metadata.file_path = $file_path
    AND project_id = $project_id
    AND tenant_id = $tenant_id
  ↓
  ├── 存在 → UPDATE（覆盖旧版本）
  │         保留 source_id，更新 content 和 code_analysis
  └── 不存在 → CREATE（新建）
```

### 4.3 冲突处理

- 同一文件多次快速保存：最后一次生效（last-write-wins）
- 旧版本不做归档，直接覆盖
- `updated_at` 时间戳自动更新

---

## 5. 批量上传优化

### 5.1 批量大小

| 场景 | 建议批量大小 | 说明 |
|------|------------|------|
| 实时编辑（单文件） | 1 | 立即上传 |
| 项目初始化 | 10-20 | 批量上传 |
| 大项目（1000+文件） | 50 | 分批上传 |

### 5.2 插件端批量策略

```text
file.edited 事件 → 防抖 300ms → 加入队列
  ↓
队列处理：
  - 攒 2-3 秒
  - 或队列达到 10 个文件
  ↓
批量调用 POST /api/v1/memories
```

---

## 6. type 字段规范

| 值 | 说明 | 使用场景 |
|----|------|---------|
| `"code"` | 代码记忆 | 插件端上传的代码分析结果 |
| `"general"` | 普通记忆 | 默认类型 |
| `"preference"` | 偏好记忆 | 用户偏好设置 |

**代码记忆标识**：`type = "code"` + `metadata.code_analysis` 存在

---

## 7. 错误码汇总

| HTTP 状态码 | 错误场景 | 处理建议 |
|------------|---------|---------|
| 200 | 成功 | - |
| 400 | 数据验证失败 | 检查请求格式 |
| 401 | 未认证 | 检查 API Key（如启用） |
| 403 | 认证失败 | 检查 API Key 是否正确 |
| 500 | 后端内部错误 | 重试或查看后端日志 |
| 502 | Embedding 服务错误 | 检查 Embedding 服务状态 |
| 503 | MemoryManager 未初始化 | 检查后端启动状态 |

---

*文档结束 - 等待审核确认后执行代码变更*
