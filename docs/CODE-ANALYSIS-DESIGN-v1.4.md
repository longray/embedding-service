# OpenCode Memory 代码分析设计方案

> **版本**: v1.4.0  
> **日期**: 2026-04-06  
> **状态**: 实施中（基于 v1.2 修正）  
> **作者**: opencode-memory-plugin + embedding_service  
> **变更**: 补充 v1.2 遗漏字段，调整实施优先级

---

## 执行摘要

本文档是代码分析功能的**实施版设计**，基于 v1.2 最终版进行修正：

1. **补充遗漏字段** — 完整实现 v1.2 承诺但未实现的数据结构
2. **调整触发机制** — 从 `file.edited` 改为 chokidar + OpenCode 事件
3. **明确排除项** — Ruby 语言支持、MCP 协议
4. **修正实施路线** — 基于实际架构约束重新排期

### 核心变更（v1.2 → v1.4）

| 变更项 | v1.2 | v1.4 | 理由 |
|--------|------|------|------|
| **触发机制** | `file.edited` 事件 | chokidar + OpenCode 事件 | 更灵活的文件监听 |
| **协议** | MCP 工具 | opencode tools | 更轻量，无 MCP 依赖 |
| **Ruby 支持** | 计划中 | ❌ 排除 | 用户明确不需要 |
| **数据模型** | 部分字段未实现 | 全部补齐 | 修复实现差距 |
| **实施周期** | 8 周 | 12-17 周 | 补充遗漏字段工作量 |

---

## 1. 架构设计

### 1.1 系统架构图

```text
┌─────────────────────────────────────────────────────────────────┐
│                    OpenCode Memory Plugin                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  文件监听 (chokidar)                                     │   │
│  │  - OpenCode 事件触发                                     │   │
│  │  - 防抖 (300ms)                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  代码分析模块 (code-analyzer/)                           │   │
│  │                                                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │    Oxc      │  │  Tree-sitter │  │  Fallback   │    │   │
│  │  │  (JS/TS)    │  │   (原生绑定)  │  │  (基础信息)  │    │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │   │
│  │         │                │                │             │   │
│  │         └────────────────┼────────────────┘             │   │
│  │                          ▼                                │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │  分析管理器 (AnalysisManager)                       │ │   │
│  │  │  - 队列管理 (max=10)                                │ │   │
│  │  │  - 并发控制 (max=2)                                 │ │   │
│  │  │  - 降级策略                                         │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  │                          │                                │   │
│  │                          ▼                                │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │  上传模块 (Uploader)                               │ │   │
│  │  │  - 敏感信息过滤                                     │ │   │
│  │  │  - 批量上传                                         │ │   │
│  │  │  - Upsert (file_path + project_id)                │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               │ HTTP POST /api/v1/memories
                               │ (批量上传，含 code_analysis)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    embedding_service Backend                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  API 入口 (main.py)                                     │    │
│  │  - POST /api/v1/memories (接收代码记忆)                │    │
│  │  - POST /api/v1/memories/search (+ code_filter)        │    │
│  │  - GET /api/v1/memories/{id}/references (调用关系)     │    │
│  │  - GET /api/v1/memories/{id}/dependencies (依赖查询)   │    │
│  │  - GET /api/v1/projects/{id}/map (代码地图)            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────┐   │
│  │   Meilisearch    │  │   SurrealDB     │  │  Memory FS  │   │
│  │  (搜索 + 过滤)   │  │   (图关系)       │  │  (文件存储)  │   │
│  │                  │  │                  │  │             │   │
│  │ filterable:      │  │  - memories     │  │  - code    │   │
│  │  - type          │  │  - relations   │  │  - timeline │   │
│  │  - code_language │  │    (calls)      │  │             │   │
│  │  - code_complexity│ │                  │  │             │   │
│  │  - code_function_count │            │  │             │   │
│  │  - code_class_count    │            │  │             │   │
│  │  - code_analyzer       │            │  │             │   │
│  └──────────────────┘  └──────────────────┘  └─────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 职责边界

| 功能 | 插件端 | 后端 | 说明 |
|------|--------|------|------|
| **文件监听** | ✅ chokidar | ❌ | OpenCode 事件触发 |
| **AST 解析** | ✅ Tree-sitter 原生 | ✅ Python Tree-sitter | 双方都有 |
| **代码上传** | ❌ | ✅ 已实现 | 后端接收 |
| **代码搜索** | ❌ | ✅ code_filter | 后端查询 |
| **调用关系存储** | ❌ | ✅ memory_relation | 后端图存储 |
| **引用查询** | ❌ | ✅ 新增 API | 后端查询 |
| **代码地图** | ❌ | ✅ 新增 API | 后端聚合 |
| **opencode 工具** | ✅ 工具定义 | ✅ 工具实现 | 轻量集成 |

---

## 2. 数据结构（v1.4 完整版）

### 2.1 CodeAnalysisResult（完整实现）

```typescript
// 插件端 TypeScript 定义
interface CodeAnalysisResult {
  // 基础信息
  language: string;              // 标准化语言名
  analyzer: string;             // "oxc" | "tree-sitter" | "regex"
  analyzed_at: string;           // ISO 8601 时间戳
  analyzer_version: string;     // "1.4.0"

  // 符号信息（统一结构，缺失特性用空数组）
  functions: FunctionSymbol[];   // ✅ v1.4: 完整字段
  classes: ClassSymbol[];        // ✅ v1.4: 完整字段
  interfaces: InterfaceSymbol[]; // ✅ v1.4: 新增实现
  imports: ImportSymbol[];       // ✅ v1.4: 结构化
  exports: ExportSymbol[];       // ✅ v1.4: 结构化

  // 复杂度指标
  complexity_metrics: ComplexityMetrics;  // ✅ v1.4: 完整字段

  // 依赖信息
  dependencies: DependencyInfo;  // ✅ v1.4: 分类实现

  // 调用关系（v1.4 新增）
  calls?: CallSymbol[];          // ✅ v1.4: 函数调用关系

  // 错误与警告
  errors?: ParseError[];         // ✅ v1.4: 降级时填充
  warnings?: ParseWarning[];     // ✅ v1.4: 降级时填充
}

// 函数符号（v1.4 完整版）
interface FunctionSymbol {
  name: string;
  start_line: number;
  end_line: number;
  params: Array<{ name: string; type?: string }>;  // ✅ v1.4: 支持类型
  return_type?: string;                            // ✅ v1.4: 新增
  is_exported: boolean;                            // ✅ v1.4: 新增
  is_async: boolean;                               // ✅ v1.4: 新增
}

// 类符号（v1.4 完整版）
interface ClassSymbol {
  name: string;
  start_line: number;
  end_line: number;
  methods: string[];       // ✅ v1.4: 方法名列表
  properties: string[];    // ✅ v1.4: 属性名列表
}

// 接口符号（v1.4 新增实现）
interface InterfaceSymbol {
  name: string;
  start_line: number;
  end_line: number;
  methods: string[];
  properties: string[];
}

// 调用符号（v1.4 新增）
interface CallSymbol {
  target: string;          // 被调用函数名
  line: number;            // 调用所在行
  column?: number;         // 调用所在列
}

// 导入符号（v1.4 结构化）
interface ImportSymbol {
  source: string;              // 模块路径
  imported_names: string[];    // 导入的名称列表
  line: number;                // ✅ v1.4: 新增
}

// 导出符号（v1.4 结构化）
interface ExportSymbol {
  name: string;
  type: "function" | "class" | "interface" | "variable" | "reexport";
  is_default: boolean;
  line: number;                // ✅ v1.4: 新增
}

// 复杂度指标（v1.4 完整版）
interface ComplexityMetrics {
  cyclomatic: number;              // 圈复杂度
  lines_of_code: number;            // 代码行数
  function_count: number;           // 函数数量
  class_count: number;              // 类数量
  max_function_complexity: number;  // ✅ v1.4: 最复杂函数
  average_function_complexity: number;  // ✅ v1.4: 平均复杂度
}

// 依赖信息（v1.4 分类实现）
interface DependencyInfo {
  internal: string[];   // ✅ v1.4: 内部依赖（相对路径）
  external: string[];   // ✅ v1.4: 外部依赖（npm/pip/cargo）
  builtin: string[];   // ✅ v1.4: 内置模块（node:fs, os 等）
}

// 解析错误（v1.4 降级时填充）
interface ParseError {
  line: number;
  column: number;
  message: string;
  severity: "error";
}

// 解析警告（v1.4 降级时填充）
interface ParseWarning {
  type: "degraded" | "timeout" | "memory_limit" | "large_file";
  from?: string;       // 降级来源
  to?: string;         // 降级目标
  reason: string;
  duration_ms?: number;
}
```

### 2.2 记忆条目格式（无变更）

```typescript
interface CodeMemoryItem {
  content: string;           // L2: 完整代码
  abstract: string;          // L0: ≤100字符摘要
  overview: string;          // L1: ≤500字符概览
  type: "code";              // 固定为 code 类型
  tags: string[];
  project_id: string;        // Git remote URL 或目录名
  metadata: {
    file_path: string;
    file_name: string;
    code_analysis: CodeAnalysisResult;  // v1.4 完整版
  };
  local_id: string;
  source_id?: string;
}
```

### 2.3 语言特性映射表（无变更）

| 特性 | TS | Python | Go | Rust | Java |
|------|-----|--------|-----|------|------|
| functions | ✅ | ✅ | ✅ (func) | ✅ (fn) | ✅ |
| classes | ✅ | ✅ | ❌→[] | ❌→[] | ✅ |
| interfaces | ✅ | ❌→[] | ✅ | ✅ (trait) | ✅ |
| imports | ✅ | ✅ | ✅ | ✅ | ✅ |
| exports | ✅ | ❌→[] | ⚠️ 大写 | ✅ pub | ✅ |
| calls | ✅ | ✅ | ✅ | ✅ | ✅ |  // v1.4 新增

---

## 3. 降级策略（v1.4 实现版）

### 3.1 三级降级决策树（v1.4 实现）

```text
文件保存 (chokidar 触发)
    │
    ▼
文件大小 > 10000 行？
    ├── Yes → 直接返回基础信息（跳过解析）
    │         warnings: [{ type: "large_file", reason: "exceeds 10000 lines" }]
    │
    └── No → 是 JS/TS 文件？
              ├── Yes → 尝试 Oxc 解析
              │         │
              │         ▼
              │    解析耗时 > 200ms 或 内存不足？
              │         ├── Yes → 降级到 Tree-sitter
              │         │         warnings: [{ type: "degraded", from: "oxc", to: "tree-sitter", reason: "timeout" }]
              │         │
              │         └── No → 返回 Oxc 结果 ✅
              │
              └── No → 直接尝试 Tree-sitter
                        │
                        ▼
                  解析耗时 > 500ms 或 异常？
                        ├── Yes → 降级到基础信息
                        │         warnings: [{ type: "degraded", from: "tree-sitter", to: "fallback", reason: "timeout" }]
                        │
                        └── No → 返回 Tree-sitter 结果 ✅

基础信息（保底）
    │
    ▼
返回 { language, file_path, size, lines_of_code, mtime, analyzer: "fallback" }
```

### 3.2 降级阈值配置（v1.4 确认）

| 条件 | 阈值 | 依据 |
|------|------|------|
| 跳过解析 | > 10000 行 | 内存和性能平衡 |
| Oxc 超时 | > 200ms | 正常 26ms，8x 余量 |
| Tree-sitter 超时 | > 500ms | 正常 50-150ms，3x 余量 |
| 内存不足 | < 100MB 可用 | 需要为其他功能留空间 |
| 内存溢出 | > 500MB | 系统安全上限 |

---

## 4. 调用关系与引用追踪（v1.4 新增）

### 4.1 调用关系存储

```typescript
// memory_relation 表新增 relationship_type = "calls"
interface CallRelation {
  from: string;           // memory:func_a (调用者)
  to: string;             // memory:func_b (被调用者)
  relationship_type: "calls";
  weight: number;         // 调用次数或置信度
  metadata: {
    line: number;         // 调用所在行
    column?: number;      // 调用所在列
    file_path: string;    // 调用所在文件
  };
}
```

### 4.2 引用查询 API

```yaml
GET /api/v1/memories/{id}/references

Response:
  references:
    - memory_id: "mem_xxx"
      file_path: "src/utils.ts"
      line: 42
      caller_function: "validateUser"
      confidence: 0.95

GET /api/v1/memories/{id}/dependencies

Response:
  dependencies:
    - memory_id: "mem_yyy"
      file_path: "src/auth.ts"
      line: 15
      callee_function: "hashPassword"
      type: "internal"  # internal/external/builtin
```

---

## 5. 代码地图（v1.4 新增）

### 5.1 项目级代码地图 API

```yaml
GET /api/v1/projects/{id}/map

Response:
  project_id: "github.com/user/repo"
  file_tree:
    - path: "src/"
      type: "directory"
      children:
        - path: "src/core/"
          type: "directory"
          children:
            - path: "src/core/auth.ts"
              type: "file"
              size: 2048
              lines: 150
              functions: 5
              classes: 1
              complexity: 8.5
  module_dependencies:
    - from: "src/core/auth.ts"
      to: "src/utils/crypto.ts"
      type: "import"
  hot_files:              # 基于代码复杂度或修改频率
    - "src/core/auth.ts"
    - "src/utils/api.ts"
  statistics:
    total_files: 45
    total_functions: 150
    total_classes: 30
    avg_complexity: 5.2
    max_complexity: 15
```

---

## 6. 搜索与过滤（v1.4 扩展）

### 6.1 code_filter 扩展（v1.4 完整版）

```typescript
POST /api/v1/memories/search

Request:
  query: "authentication"
  type: "code"                    // 只搜索代码记忆
  code_filter: {
    language: "typescript",        // ✅ 已实现
    min_complexity: 5,             // ✅ 已实现
    max_complexity: 10,            // ✅ 已实现
    // v1.4 新增:
    min_function_count: 3,         // ✅ v1.4
    max_function_count: 20,        // ✅ v1.4
    min_class_count: 1,            // ✅ v1.4
    max_class_count: 10,           // ✅ v1.4
    has_exports: true,             // ✅ v1.4
    analyzer: "tree-sitter",       // ✅ v1.4
    is_async: true,                // ✅ v1.4 (函数级别)
  }
```

### 6.2 Meilisearch 索引字段（v1.4 完整版）

```json
{
  "filterableAttributes": [
    "type",
    "code_language",
    "code_complexity",
    "code_function_count",      // ✅ v1.4
    "code_class_count",         // ✅ v1.4
    "code_analyzer",            // ✅ v1.4
    "code_has_exports",         // ✅ v1.4
    "tags"
  ],
  "searchableAttributes": [
    "content_zh",
    "content_search",
    "code_symbols",
    "code_function_names",      // ✅ v1.4
    "code_class_names"          // ✅ v1.4
  ]
}
```

---

## 7. opencode 工具集成（v1.4 新增）

### 7.1 工具定义

```json
// .opencode/tools/code-analyzer.json
{
  "tools": [
    {
      "name": "code_search",
      "description": "搜索代码记忆，支持自然语言和代码符号",
      "command": "curl -X POST http://localhost:17999/api/v1/tools/code_search",
      "args": ["query", "language", "max_complexity"]
    },
    {
      "name": "code_context",
      "description": "获取代码文件的完整上下文（函数、类、依赖）",
      "command": "curl http://localhost:17999/api/v1/tools/code_context",
      "args": ["file_path", "line_number"]
    },
    {
      "name": "code_impact",
      "description": "分析修改某文件的影响范围",
      "command": "curl -X POST http://localhost:17999/api/v1/tools/code_impact",
      "args": ["file_path"]
    }
  ]
}
```

### 7.2 工具实现

```yaml
POST /api/v1/tools/code_search

Request:
  query: "用户认证逻辑"
  language: "typescript"
  max_complexity: 10

Response:
  results:
    - file_path: "src/auth.ts"
      function: "validateUser"
      line: 42
      relevance: 0.95
```

---

## 8. 实施路线图（v1.4 修正版）

### Phase 1: 数据模型补齐（2-3 周）— P0

**目标**：修复 v1.2 承诺但未实现的数据字段

| 任务 | 编号 | 说明 |
|------|------|------|
| 函数完整字段 | BL-CA-11 | `return_type`, `is_exported`, `is_async` |
| 类成员提取 | BL-CA-12 | `methods`, `properties` |
| 接口定义提取 | BL-CA-13 | 新增 `_extract_interfaces` |
| 导入结构化 | BL-CA-14 | `source`, `imported_names` |
| 导出结构化 | BL-CA-15 | `name`, `type`, `is_default` |
| 依赖分类 | BL-CA-16 | `internal/external/builtin` |
| 复杂度指标完整 | BL-CA-17 | `max_function_complexity`, `average_function_complexity` |
| code_filter 扩展 | BL-CA-18 | 支持 `function_count`, `class_count` 等 |

### Phase 2: 调用关系与引用追踪（3-4 周）— P1

**目标**：实现跨文件符号引用

| 任务 | 编号 | 说明 |
|------|------|------|
| 函数调用提取 | BL-CA-19 | 新增 `_extract_calls` |
| 调用关系存储 | BL-CA-20 | `memory_relation` 新增 `calls` 类型 |
| 引用查询 API | BL-CA-21 | `GET /memories/{id}/references` |
| 依赖查询 API | BL-CA-22 | `GET /memories/{id}/dependencies` |

### Phase 3: 项目级代码地图（2-3 周）— P1

**目标**：项目结构可视化

| 任务 | 编号 | 说明 |
|------|------|------|
| 项目统计聚合 | BL-CA-23 | 按 `project_id` 聚合 |
| 代码地图 API | BL-CA-24 | `GET /projects/{id}/map` |
| 热点文件标记 | BL-CA-25 | 基于复杂度统计 |

### Phase 4: 语义代码搜索（3-4 周）— P2

**目标**：自然语言搜索代码意图

| 任务 | 编号 | 说明 |
|------|------|------|
| 代码预处理 | BL-CA-26 | 去除注释/空行，提取关键结构 |
| 代码专用 embedding | BL-CA-27 | 函数签名单独 embed |
| 语义搜索 API | BL-CA-28 | `semantic_query` 参数 |
| 混合搜索优化 | BL-CA-29 | 关键词 + 语义 RRF 融合 |

### Phase 5: opencode 工具集成（2-3 周）— P3

**目标**：作为 opencode 工具暴露

| 任务 | 编号 | 说明 |
|------|------|------|
| 工具定义 | BL-CA-30 | `.opencode/tools/code-analyzer.json` |
| code_search 工具 | BL-CA-31 | 代码搜索接口 |
| code_context 工具 | BL-CA-32 | 获取代码上下文 |
| code_impact 工具 | BL-CA-33 | 变更影响分析 |

---

## 9. 与 GitNexus 对比（v1.4 预期）

| 功能 | GitNexus | OpenCode Memory v1.4 | 差距 |
|------|---------|---------------------|------|
| **触发** | file.edited | chokidar + OpenCode | ✅ 类似 |
| **协议** | MCP | opencode tools | ✅ 更轻量 |
| **图数据库** | LadybugDB (Cypher) | SurrealDB (关系表) | ⚠️ 简化 |
| **预计算** | 索引时全量 | 按需实时 | ⚠️ 延迟更低 |
| **调用图** | 完整 | Phase 2 实现 | ⚠️ 3-4 周后 |
| **代码地图** | 完整 | Phase 3 实现 | ⚠️ 5-7 周后 |
| **语义搜索** | 完整 | Phase 4 实现 | ⚠️ 8-11 周后 |
| **MCP 工具** | 16 个 | 3-4 个 | ⚠️ 聚焦核心 |

---

## 10. 附录

### A. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-03-28 | 初始版本 |
| v1.1 | 2026-03-29 | Review 修正，增加 Phase 0 |
| v1.2 | 2026-03-31 | 最终版，双方深度讨论对齐 |
| **v1.4** | **2026-04-06** | **实施版：补充遗漏字段，调整触发机制，排除 Ruby/MCP** |

### B. 排除项（v1.4 明确）

- ❌ **Ruby 语言支持** — 用户明确不需要
- ❌ **MCP 协议** — 使用更轻量的 opencode tools
- ❌ **Tree-sitter WASM** — 当前原生绑定已工作，无需迁移

### C. 关键修复（v1.2 → v1.4）

| 问题 | v1.2 状态 | v1.4 修复 |
|------|----------|----------|
| `FunctionSymbol.return_type` | ❌ 缺失 | ✅ BL-CA-11 |
| `FunctionSymbol.is_exported/is_async` | ❌ 缺失 | ✅ BL-CA-11 |
| `ClassSymbol.methods/properties` | ❌ 缺失 | ✅ BL-CA-12 |
| `InterfaceSymbol` 提取 | ❌ 未实现 | ✅ BL-CA-13 |
| `ImportSymbol` 结构化 | ⚠️ 原始字符串 | ✅ BL-CA-14 |
| `ExportSymbol` 结构化 | ⚠️ 原始字符串 | ✅ BL-CA-15 |
| `DependencyInfo` 分类 | ❌ 扁平列表 | ✅ BL-CA-16 |
| `code_filter` 扩展 | ⚠️ 3 字段 | ✅ BL-CA-18 |
| `errors/warnings` 填充 | ❌ 空壳 | ✅ 降级时填充 |

---

**文档结束**

---

*版本: v1.4.0*  
*日期: 2026-04-06*  
*状态: 实施中*
