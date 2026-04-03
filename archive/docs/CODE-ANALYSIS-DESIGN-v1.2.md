# OpenCode Memory 代码分析设计方案

> **版本**: v1.2.0  
> **日期**: 2026-03-31  
> **状态**: 最终版（基于双方深度讨论）  
> **作者**: opencode-memory-plugin + embedding_service 协同设计

---

## 执行摘要

本文档是插件端（opencode-memory-plugin）与后端（embedding_service）协同设计的**最终版代码分析方案**。经过多轮深度讨论，双方已在所有关键技术点上达成一致。

### 核心决策（v1.2 最终确认）

| 决策项 | 方案 | 关键理由 |
|--------|------|----------|
| **架构模式** | 插件 + CLI 混合 | 插件自动触发，CLI 用于 CI/CD |
| **解析器** | Tree-sitter WASM | 50+ 语言支持，增量解析 |
| **JS/TS 优化** | Oxc (Phase 2) | 26ms 超高速，比 Tree-sitter 快 2-3x |
| **触发方式** | `file.edited` 事件 | 无缝集成，用户无感知 |
| **降级策略** | 三级降级 + 明确阈值 | Oxc → Tree-sitter → 基础信息 |
| **多语言策略** | 统一扁平结构 | 所有语言统一，空数组填充缺失特性 |
| **实时性能** | 防抖+队列+并发控制 | 300ms防抖 + 并发2 + 队列10 |
| **错误处理** | Phase 1 全静默 | 只记录 errors/warnings，不打扰用户 |
| **搜索集成** | 双模式搜索 | 统一搜索 + code_filter 过滤 |

### 关键数据（最终确认）

| 指标 | 值 | 说明 |
|------|-----|------|
| **5000行解析** | <200ms | Tree-sitter WASM |
| **10000行处理** | 跳过详细解析 | 只记录基础信息 |
| **防抖延迟** | 300ms | 防止连续保存重复解析 |
| **最大并发** | 2 | 避免 CPU 过载 |
| **队列上限** | 10 | 防止内存堆积 |
| **Oxc 超时** | 200ms | 超过则降级 Tree-sitter |
| **Tree-sitter 超时** | 500ms | 超过则降级基础信息 |
| **内存上限** | 500MB | 超过则降级 |
| **支持语言** | P0: 5种 | JS/TS/Python/Go/Rust/Java |

### 实施周期（8周）

| 阶段 | 时间 | 内容 |
|------|------|------|
| **Phase 0** | Week 1 | 技术验证 + 设计文档最终化 |
| **Phase 1** | Week 2-3 | 核心功能：解析 + 触发 + 上传 |
| **Phase 2** | Week 4-6 | 增强功能：Oxc + CLI + 增量同步 |
| **Phase 3** | Week 7-8 | 联调测试 + 性能优化 |

---

## 1. 架构设计

### 1.1 系统架构图

```text
┌─────────────────────────────────────────────────────────────────┐
│                    OpenCode Memory Plugin                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  代码分析模块 (code-analyzer/)                           │   │
│  │                                                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │    Oxc      │  │  Tree-sitter │  │  Fallback   │    │   │
│  │  │  (JS/TS)    │  │   (通用)      │  │  (基础信息)  │    │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │   │
│  │         │                │                │             │   │
│  │         └────────────────┼────────────────┘             │   │
│  │                          ▼                                │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │  分析管理器 (AnalysisManager)                       │ │   │
│  │  │  - 防抖 (300ms)                                     │ │   │
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
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  OpenCode 事件监听                                      │   │
│  │  - file.edited (文件保存时触发)                        │   │
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
│  │  - POST /api/v1/sync/* (Phase 2)                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────┐   │
│  │   Meilisearch    │  │   SurrealDB     │  │  Memory FS  │   │
│  │  (搜索 + 过滤)   │  │   (图关系)       │  │  (文件存储)  │   │
│  │                  │  │                  │  │             │   │
│  │ filterable:      │  │  - memories     │  │  - code    │   │
│  │  - type          │  │  - relations   │  │  - timeline │   │
│  │  - code_language │  │                  │  │             │   │
│  │  - code_complexity│ │                  │  │             │   │
│  │  - code_function_count │            │  │             │   │
│  │  - code_class_count  │              │  │             │   │
│  └──────────────────┘  └──────────────────┘  └─────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 职责边界

| 功能 | 插件端 | 后端 | 说明 |
|------|--------|------|------|
| **AST 解析** | ✅ Tree-sitter WASM / Oxc | ✅ Python Tree-sitter | 双方都有，场景不同 |
| **实时文件监听** | ✅ file.edited | ❌ | 仅插件端 |
| **批量项目分析** | ✅ CLI 工具 | ❌ | 仅插件端 |
| **代码上传 API** | ❌ | ✅ 已实现 | 后端接收 |
| **代码搜索过滤** | ❌ | ✅ code_filter | 后端查询 |
| **增量同步 API** | ✅ 指纹计算 | ❌ (Phase 2) | Phase 1 全量 |
| **跨文件关系** | ❌ | ❌ (Phase 2) | 后续版本 |

---

## 2. 数据结构

### 2.1 CodeAnalysisResult（Phase 1 最终版）

```typescript
// 插件端 TypeScript 定义
interface CodeAnalysisResult {
  // 基础信息
  language: string;              // 标准化语言名: "typescript", "python", "go", "rust", "java"
  analyzer: string;             // 分析器: "oxc" | "tree-sitter" | "fallback"
  analyzed_at: string;           // ISO 8601 时间戳
  analyzer_version: string;     // 分析器版本

  // 符号信息（统一结构，缺失特性用空数组）
  functions: FunctionSymbol[];
  classes: ClassSymbol[];
  interfaces: InterfaceSymbol[];  // Python/Go/Rust → []
  
  imports: ImportSymbol[];
  exports: ExportSymbol[];

  // 复杂度指标
  complexity_metrics: ComplexityMetrics;

  // 依赖信息
  dependencies: DependencyInfo;

  // 错误与警告（Phase 1 静默，但上传后端）
  errors?: ParseError[];
  warnings?: ParseWarning[];
}

// 函数符号
interface FunctionSymbol {
  name: string;
  start_line: number;
  end_line: number;
  params: Array<{ name: string; type?: string }>;
  return_type?: string;
  is_exported: boolean;
  is_async: boolean;
}

// 类符号
interface ClassSymbol {
  name: string;
  start_line: number;
  end_line: number;
  methods: string[];       // 方法名列表
  properties: string[];    // 属性名列表
}

// 接口符号
interface InterfaceSymbol {
  name: string;
  start_line: number;
  end_line: number;
  methods: string[];
  properties: string[];
}

// 导入符号
interface ImportSymbol {
  source: string;              // 模块路径
  imported_names: string[];    // 导入的名称
}

// 导出符号
interface ExportSymbol {
  name: string;
  type: "function" | "class" | "interface" | "variable" | "reexport";
  is_default: boolean;
}

// 复杂度指标
interface ComplexityMetrics {
  cyclomatic: number;              // 圈复杂度
  lines_of_code: number;            // 代码行数
  function_count: number;           // 函数数量
  class_count: number;              // 类数量
  max_function_complexity: number;  // 最复杂函数的圈复杂度
  average_function_complexity: number;
}

// 依赖信息
interface DependencyInfo {
  internal: string[];   // 内部依赖（相对路径）
  external: string[];   // 外部依赖（npm/pip/cargo）
  builtin: string[];   // 内置模块（node:fs, os 等）
}

// 解析错误
interface ParseError {
  line: number;
  column: number;
  message: string;
  severity: "error";
}

// 解析警告
interface ParseWarning {
  type: "degraded" | "timeout" | "memory_limit" | "large_file";
  from?: string;       // 降级来源
  to?: string;         // 降级目标
  reason: string;
  duration_ms?: number;
}
```

### 2.2 记忆条目格式

```typescript
interface CodeMemoryItem {
  // L0/L1/L2 分层内容
  content: string;           // L2: 完整代码
  abstract: string;          // L0: ≤100字符摘要
  overview: string;          // L1: ≤500字符概览

  // 基础字段
  type: "code";              // 固定为 code 类型
  tags: string[];            // ["typescript", "function", "api"]
  project_id: string;        // Git remote URL 或目录名

  // 元数据
  metadata: {
    file_path: string;       // 相对路径
    file_name: string;       // 文件名
    code_analysis: CodeAnalysisResult;
  };

  // 同步字段
  local_id: string;          // 插件端 ULID
  source_id?: string;        // 后端返回的 ID
}
```

### 2.3 语言特性映射表

| 特性 | TS | Python | Go | Rust | Java |
|------|-----|--------|-----|------|------|
| functions | ✅ | ✅ | ✅ (func) | ✅ (fn) | ✅ |
| classes | ✅ | ✅ | ❌→[] | ❌→[] | ✅ |
| interfaces | ✅ | ❌→[] | ✅ | ✅ (trait) | ✅ |
| imports | ✅ | ✅ | ✅ | ✅ | ✅ |
| exports | ✅ | ❌→[] | ⚠️ 大写 | ✅ pub | ✅ |

---

## 3. 降级策略

### 3.1 三级降级决策树

```text
文件保存 (file.edited)
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
               │    解析耗时 > 200ms 或 system available memory < 100MB？
              │         ├── Yes → 降级到 Tree-sitter
              │         │         warnings: [{ type: "degraded", from: "oxc", to: "tree-sitter", reason: "timeout" }]
              │         │
              │         └── No → 返回 Oxc 结果 ✅
              │
              └── No → 直接尝试 Tree-sitter
                        │
                        ▼
                  解析耗时 > 500ms 或 内存 > 500MB 或 异常？
                        ├── Yes → 降级到基础信息
                        │         warnings: [{ type: "degraded", from: "tree-sitter", to: "fallback", reason: "timeout" }]
                        │
                        └── No → 返回 Tree-sitter 结果 ✅

基础信息（保底）
    │
    ▼
返回 { language, file_path, size, lines_of_code, mtime }
```

### 3.2 降级阈值配置

| 条件 | 阈值 | 依据 |
|------|------|------|
| 跳过解析 | > 10000 行 | 内存和性能平衡 |
| Oxc 超时 | > 200ms | 正常 26ms，8x 余量 |
| Tree-sitter 超时 | > 500ms | 正常 50-150ms，3x 余量 |
| 内存不足 | < 100MB 可用 | 需要为其他功能留空间 |
| 内存溢出 | > 500MB | 系统安全上限 |

---

## 4. 实时性能优化

### 4.1 配置参数

```typescript
const ANALYSIS_CONFIG = {
  // 防抖配置
  debounceMs: 300,           // 保存后 300ms 开始解析
  
  // 并发控制
  maxConcurrent: 2,          // 最多 2 个文件同时解析
  
  // 队列管理
  maxQueueSize: 10,          // 队列上限，超出丢弃最旧的
  queueTimeoutMs: 5000,      // 队列中文件最多等待 5 秒
  
  // 单文件超时
  fileTimeoutMs: 500,        // 单文件解析超时 500ms
  
  // 大文件阈值
  largeFileThreshold: 5000,  // 5000-10000 行简化解析
  skipFileThreshold: 10000,  // >10000 行跳过详细解析
  
  // 批量上传
  batchDelayMs: 2000,        // 攒 2 秒后批量上传
  batchMaxSize: 10,          // 单次最多上传 10 个文件
};
```

### 4.2 队列优先级

```typescript
enum Priority {
  ACTIVE_FILE = 3,     // 当前正在编辑的文件（最高优先级）
  OPEN_FILE = 2,       // 打开但未激活的文件
  BACKGROUND = 1,     // 后台批量分析（最低优先级）
}
```

### 4.3 批量上传策略

```typescript
// 攒够 batchMaxSize 或等待 batchDelayMs 后批量上传
async function batchUpload(analyses: CodeAnalysisResult[]) {
  await axios.post('/api/v1/memories', {
    memories: analyses.map(a => ({
      type: 'code',
      content: a.content,
      abstract: a.abstract,
      overview: a.overview,
      metadata: { code_analysis: a }
    })),
    tenant_id: 'default'
  });
}
```

---

## 5. 错误处理

### 5.1 错误等级（Phase 1）

| 等级 | 触发条件 | 用户感知 | 处理方式 |
|------|---------|---------|---------|
| **INFO** | 解析成功 | 无 | 静默 |
| **WARN** | 降级触发 | 无 | 静默，记录到 warnings |
| **ERROR** | 解析失败 | 无 | 静默，记录到 errors + 降级 |
| **CRITICAL** | 插件崩溃 | 无（Phase 1） | 静默，Phase 2+ 考虑通知 |

### 5.2 Phase 1 vs Phase 2+

| 阶段 | 错误处理 | 状态指示 |
|------|---------|---------|
| **Phase 1** | 全静默，只记录日志 | 无 |
| **Phase 2+** | 降级时状态栏黄色 | 右下角显示状态 |

---

## 6. 搜索与记忆集成

### 6.1 双模式搜索

**模式1：统一搜索（memory_search）**

```typescript
// 所有记忆混合搜索（包括代码）
memory_search(query: "认证功能")
// 返回：代码文件 + 对话记录（按相关性排序）
```

**模式2：代码过滤搜索**

```typescript
// 搜索时添加 code_filter
POST /api/v1/memories/search
{
  query: "authentication",
  type: "code",                    // 只搜索代码记忆
  code_filter: {
    language: "typescript",
    min_complexity: 5,
    max_complexity: 10
  }
}
```

### 6.2 Meilisearch 索引字段

```json
{
  "filterableAttributes": [
    "type",
    "code_language",
    "code_complexity",
    "code_function_count",
    "code_class_count",
    "code_analyzer",
    "tags"
  ],
  "searchableAttributes": [
    "content_zh",
    "content_search",
    "code_symbols"
  ]
}
```

### 6.3 代码记忆与对话记忆关联

通过以下方式关联：

1. **file_path**: 同一文件的记忆
2. **tags**: 共同标签
3. **时间窗口**: 同时间段的对话+代码

---

## 7. 隐私与安全

### 7.1 敏感信息过滤

插件端上传前扫描代码内容，过滤敏感模式：

```typescript
const SENSITIVE_PATTERNS = [
  /password\s*[:=]\s*["'][^"']{4,}["']/i,
  /api[_-]?key\s*[:=]\s*["'][^"']{4,}["']/i,
  /secret\s*[:=]\s*["'][^"']{4,}["']/i,
  /token\s*[:=]\s*["'][^"']{4,}["']/i,
  /private[_-]?key\s*[:=]\s*/i,
];

function shouldSkipFile(filePath: string, content: string): boolean {
  // 跳过敏感文件
  if (/\.env(\.local)?$/i.test(filePath)) return true;
  
  // 扫描敏感模式
  for (const pattern of SENSITIVE_PATTERNS) {
    if (pattern.test(content)) return true;
  }
  
  return false;
}
```

### 7.2 默认排除文件

```typescript
const EXCLUDED_PATTERNS = [
  /\.env$/,
  /\.env\.\w+$/,
  /\.git\//,
  /node_modules\//,
  /\.DS_Store$/,
  /config.*\.json$/,
  /\.pem$/,
  /\.key$/,
];
```

---

## 8. 代码记忆生命周期

### 8.1 Upsert 策略

同一 `file_path` + `project_id` 的代码记忆只保留最新版本：

```typescript
// 上传时自动更新（Upsert）
POST /api/v1/memories
{
  memories: [{
    type: "code",
    project_id: "github.com/user/repo",
    metadata: {
      file_path: "src/index.ts",
      code_analysis: { ... }
    }
  }]
}

// 后端逻辑：
// 1. 查找是否存在 file_path + project_id 的记录
// 2. 存在 → 更新（UPDATE）
// 3. 不存在 → 创建（INSERT）
```

### 8.2 项目标识生成

```typescript
function getProjectId(): string {
  // 1. 优先使用 Git remote URL
  const remoteUrl = execSync('git remote get-url origin')
    .toString()
    .trim()
    .replace('.git$', '')
    .replace('git@github.com:', 'github.com/');
  
  if (remoteUrl) return remoteUrl;
  
  // 2. 使用项目目录名
  const dirName = path.basename(process.cwd());
  if (dirName) return dirName;
  
  // 3. 默认值
  return "unknown";
}
```

---

## 9. API 契约

### 9.1 上传代码记忆

```yaml
POST /api/v1/memories

Request:
  memories:
    - content: "// 完整代码内容"
      abstract: "TypeScript 文件：CodeAnalyzer 类"
      overview: "代码分析器核心模块..."
      type: "code"
      tags: ["typescript", "code-analysis"]
      project_id: "github.com/user/repo"
      metadata:
        file_path: "src/analyzer.ts"
        file_name: "analyzer.ts"
        code_analysis:
          language: "typescript"
          analyzer: "tree-sitter"
          analyzed_at: "2026-03-31T12:00:00Z"
          analyzer_version: "1.0.0"
          functions: [...]
          classes: [...]
          complexity_metrics: {...}
          dependencies: {...}
      local_id: "01HXYZ123ABC"

  tenant_id: "default"

Response (200):
  success: true
  created:
    - local_id: "01HXYZ123ABC"
      source_id: "mem_abc123"
      status: "created"  # 或 "updated"
```

### 9.2 搜索代码记忆

```yaml
POST /api/v1/memories/search

Request:
  query: "authentication"
  type: "code"                    # 可选，不指定则搜索所有类型
  code_filter:                    # 可选，代码过滤
    language: "typescript"
    min_complexity: 5
    max_complexity: 10

Response (200):
  results: [...]
  total: 10
```

---

## 10. 实施路线图

### Phase 0: 技术验证（Week 1）

| 任务 | 负责人 | 交付物 |
|------|--------|--------|
| Bun + Tree-sitter WASM 验证 | 插件端 | 验证报告 |
| 设计文档最终化 | 双方 | v1.2 文档 |
| 后端 Schema 设计 | 后端 | schema-upgrade.md |
| Meilisearch 索引设计 | 后端 | meilisearch-index.md |

### Phase 1: 核心功能（Week 2-3）

| 任务 | 负责人 | 交付物 |
|------|--------|--------|
| Tree-sitter WASM 集成 | 插件端 | 可用分析器 |
| file.edited 事件监听 | 插件端 | 自动触发 |
| 代码上传 API 对接 | 插件端 | 上传功能 |
| code_filter 搜索 | 后端 | 可用过滤 |

### Phase 2: 增强功能（Week 4-6）

| 任务 | 负责人 | 交付物 |
|------|--------|--------|
| Oxc 集成（JS/TS） | 插件端 | 性能优化 |
| CLI 工具 | 插件端 | 批量分析 |
| 增量同步 API | 后端 | fingerprint API |
| 降级策略完善 | 插件端 | 鲁棒性提升 |

### Phase 3: 联调测试（Week 7-8）

| 任务 | 负责人 |
|------|--------|
| 端到端测试 | 双方 |
| 性能基准测试 | 双方 |
| 文档完善 | 双方 |

---

## 附录

### A. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-03-28 | 初始版本 |
| v1.1 | 2026-03-29 | Review 修正，增加 Phase 0 |
| v1.2 | 2026-03-31 | 最终版，双方深度讨论对齐 |

### B. 术语表

| 术语 | 定义 |
|------|------|
| **AST** | Abstract Syntax Tree，抽象语法树 |
| **WASM** | WebAssembly，浏览器/Node.js 中运行 |
| **指纹** | 文件内容的 SHA-256 hash，用于检测变更 |
| **Upsert** | Update or Insert，更新或插入 |
| **Debounce** | 防抖，延迟执行以合并多次触发 |
| **Oxc** | Rust 编写的超高速 JS/TS 解析器 |

### C. 参考文献

- Tree-sitter 官方文档: https://tree-sitter.github.io
- Oxc 官方仓库: https://github.com/oxc-project/oxc
- Meilisearch 文档: https://www.meilisearch.com/docs

---

**文档结束**

---

*版本: v1.2.0*  
*日期: 2026-03-31*  
*状态: 最终版（双方对齐）*
