# OpenCode Memory 代码分析设计方案

> **版本**: v1.2.0  
> **日期**: 2026-03-29  
> **状态**: 已修正（基于 Review 结果）  
> **作者**: AI Assistant

---

## 执行摘要

本文文档汇总了插件端（opencode-memory-plugin）代码分析能力的统一设计方案。采用 **OpenCode 插件 + CLI 混合架构**，解决代码分析的多语言支持、实时分析和批量处理问题。

> ⚠️ **重要修正**（Review 后）：
>
> - 增加 **Phase 0 技术验证**（1周）
> - 总周期从 **6周延长到12周**
> - 每个 Phase 增加缓冲时间和风险缓解

### 核心决策

| 决策项 | 方案 | 关键理由 |
|--------|------|----------|

| **架构模式** | 插件 + CLI 混合 | 插件自动触发，CLI 用于 CI/CD |

| **解析器** | Tree-sitter WASM | 50+ 语言支持，增量解析 |

| **JS/TS 优化** | Oxc (Phase 4) | 26ms 超高速，比 Tree-sitter 快 2-3x |

| **触发方式** | `file.edited` 事件 | 无缝集成，用户无感知 |

| **降级策略** | 三级降级 | Oxc → Tree-sitter → 基础信息 |

| **多语言策略** | 优先级分层 | P0: 5种，P1: 5种，P2: 其他 |

### 关键数据（Review 后修正）

| 指标 | 原数据 | 修正后 | 说明 |

|------|--------|--------|------|

| **5000行解析** | 50-150ms | **<200ms** | 放宽标准，更现实 |

| **实时编辑响应** | <10ms | **<50ms** | Tree-sitter 增量 |

| **支持语言** | 50+ | **P0: 5种** | JS/TS/Python/Go/Rust/Java |

| **实施周期** | 6周 | **12周** | 增加缓冲和验证 |

| **大文件阈值** | 5000行 | **10000行** | 超过则简化分析 |

| **内存上限** | 300MB | **500MB** | 更安全 |

### 主要风险（Review 识别）

| 风险 | 影响 | 概率 | 缓解措施 |

|------|------|------|----------|

| **Bun + Tree-sitter 不兼容** | 🔴 高 | 中 | Phase 0 验证，Node.js fallback |

| **性能不达预期** | 🟡 中 | 中 | 放宽标准，降级策略 |

| **后端集成复杂** | 🟡 中 | 低 | 提前沟通，向后兼容 |

| **实施延期** | 🟡 中 | 高 | 周期延长至12周 |

---

## 背景与目标

### 问题背景

当前系统存在**代码分析能力缺失**的问题：

1. **OpenCode 无内置代码分析**：OpenCode 提供 LSP 支持（30+ 语言），但**仅限 IDE 功能**（跳转、补全），**无法导出 AST**

2. **多语言需求**：项目涉及 JavaScript/TypeScript、Python、Go、Rust 等多种语言

3. **实时分析需求**：用户编辑代码时需要实时分析并同步到后端

4. **批量分析需求**：项目初始化时需要批量分析所有文件

### 融合目标

设计一个**统一的代码分析方案**，实现：

- ✅ 多语言 AST 解析（50+ 语言）

- ✅ 实时分析（文件编辑时自动触发）

- ✅ 批量分析（项目初始化）

- ✅ 增量同步（只上传变更）

- ✅ 多设备协同（后端聚合存储）

---

## 技术选型决策

### 为什么不用 OpenCode 内置 LSP？

**OpenCode 内置 LSP 能力**：

- ✅ 支持 30+ 语言（Python、Java、Go、Rust、JS/TS 等）

- ✅ 提供 IDE 功能（定义跳转、引用查找、代码补全）

**但是**：

- ❌ **无法导出 AST**：LSP 只提供 IDE 功能，不暴露语法树

- ❌ **无法代码分析**：无法提取符号、计算复杂度、分析依赖

**结论**：OpenCode LSP **不能用于代码分析**，必须引入外部解析器。

### 为什么不用各语言原生解析器？

**各语言原生解析器**：

- Python: `ast` (标准库)

- Java: JavaParser

- Go: `go/ast` (标准库)

- Rust: Syn

- JS/TS: Oxc / SWC

**问题**：

1. **集成复杂度爆炸**：需要安装 N 种语言运行时

2. **API 不统一**：每种解析器不同接口

3. **维护成本极高**：N 套代码，N 倍维护成本

4. **增量解析缺失**：大部分原生解析器不支持增量

**结论**：各语言原生解析器 **不适合统一架构**。

### 为什么用 Tree-sitter？

**Tree-sitter 优势**：

- ✅ **50+ 语言支持**：JavaScript、Python、Go、Rust、Java、C/C++ 等

- ✅ **统一 API**：所有语言相同接口

- ✅ **增量解析**：支持实时编辑，<10ms 响应

- ✅ **错误容忍**：可解析无效代码

- ✅ **WASM 支持**：可在浏览器/Node.js 运行

- ✅ **编辑器标准**：VS Code、Neovim、Atom 都采用

**性能对比**:

| 工具 | 5000行解析 | 内存占用 | 增量解析 |

|------|-----------|---------|---------|

| **Tree-sitter** | 50-150ms | 100-300MB | ✅ 支持 |

| **Oxc** | 26ms | 11-69MB | ❌ 不支持 |

| **SWC** | 84ms | ~50MB | ❌ 不支持 |

**结论**：Tree-sitter **是最佳统一解析器**。

### 为什么 Phase 2 引入 Oxc？

**Oxc 优势**（仅 JS/TS）：

- ✅ **超高速**：26ms（比 Tree-sitter 快 2-3x）

- ✅ **体积小**：~2 MB（比 SWC 小 20x）

- ✅ **功能丰富**：Parser + Linter + Formatter

**使用策略**：

- **Phase 1**：Tree-sitter（所有语言，统一架构）

- **Phase 2**：Oxc（仅 JS/TS，性能优化）

```typescript

// Phase 2: Oxc 优化（仅 JS/TS）

const analyzer = isJavaScript(filePath)

  ? new OxcAnalyzer()      // 26ms，更快

  : new TreeSitterAnalyzer(); // 50-150ms，支持多语言

### 架构图

```text

┌─────────────────────────────────────────────────────────────┐

│  OpenCode 插件（.opencode/plugins/code-analyzer.ts）        │

├─────────────────────────────────────────────────────────────┤

│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │

│  │ Tree-sitter │  │ Tree-sitter │  │   Oxc       │         │

│  │ WASM        │  │ WASM        │  │  (Phase 2)  │         │

│  │ (Python/Go) │  │ WASM        │  │  (JS/TS)    │         │

│  │             │  │ (Rust/Java) │  │             │         │

│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │

│         │                │                │                 │

│         └────────────────┼────────────────┘                 │

│                          ▼                                   │

│  ┌─────────────────────────────────────────────────────┐   │

│  │  本地指纹缓存 (SQLite)                              │   │

│  │  - file_path → {hash, mtime, symbols_hash}        │   │

│  └─────────────────────────────────────────────────────┘   │

│                          │                                   │

│                          ▼                                   │

│  ┌─────────────────────────────────────────────────────┐   │

│  │  自定义工具（code_analyze, code_search）            │   │

│  └─────────────────────────────────────────────────────┘   │

└─────────────────────────────────────────────────────────────┘

            │                              │

            │ file.edited 事件             │ 用户调用

            │ （自动触发）                  │ （手动触发）

            ▼                              ▼

┌─────────────────────────────────────────────────────────────┐

│  CLI 工具（opencode-code-analyzer）                        │

│  ├── Tree-sitter Node.js API                               │

│  ├── 批量分析                                              │

│  └── 脚本化集成                                            │

└─────────────────────────────────────────────────────────────┘

            │

            │ 增量同步（只传变更）

            ▼

┌─────────────────────────────────────────────────────────────┐

│  后端（embedding_service）                                 │

├─────────────────────────────────────────────────────────────┤

│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │

│  │ 向量数据库  │  │ 图数据库    │  │ 记忆存储    │         │

│  │ (Meilisearch)│  │ (SurrealDB) │  │ (文件系统)  │         │

│  └─────────────┘  └─────────────┘  └─────────────┘         │

│         │              │              │                    │

│         └──────────────┼──────────────┘                    │

│                        ▼                                   │

│  ┌─────────────────────────────────────────────────────┐   │

│  │  全局语义搜索 + 跨文件关系图                          │   │

│  └─────────────────────────────────────────────────────┘   │

└─────────────────────────────────────────────────────────────┘

### 决策 2: 插件触发方式

#### 选项对比

| 方式 | 触发时机 | 实时性 | 用户体验 |

|------|---------|--------|---------|

| **file.edited** | 文件保存时 | ✅ 高 | ✅ 无感知 |

| **file.watcher.updated** | 文件变化时 | ✅ 高 | ✅ 无感知 |

| **手动工具调用** | 用户说"分析" | ❌ 低 | ❌ 需记忆命令 |

| **定时轮询** | 每 30 秒 | ❌ 低 | ❌ 延迟高 |

#### 决策

**方案**：**`file.edited` 事件**（文件保存时触发）

```typescript

export const CodeAnalyzerPlugin: Plugin = async (ctx) => {

  return {

    // 文件编辑时自动分析

    "file.edited": async ({ filePath }) => {

      if (!isCodeFile(filePath)) return;

            const result = await analyzeFile(filePath);

      await uploadToBackend(filePath, result);

    },

        // 文件监听（增量）

    "file.watcher.updated": async ({ filePath }) => {

      // 指纹对比，只上传变更

      await syncIncremental(filePath);

    }

  };

};

### 记忆条目格式（扩展）

```typescript

// 上传记忆的完整结构

interface CodeMemoryItem {

  // L0/L1/L2 分层内容

  content: string;            // L2: 完整代码

  abstract: string;           // L0: ≤100字符摘要

  overview: string;           // L1: ≤500字符概览

    // 基础字段

  type: "code";               // 固定为 code 类型

  tags: string[];             // ["typescript", "function", "api"]

  project_id: string;         // 项目标识

    // 元数据（包含代码分析结果）

  metadata: {

    code_analysis: CodeAnalysisResult;  // 插件端分析结果

        // 额外元数据

    file_path: string;        // 文件路径（相对）

    file_name: string;        // 文件名

    repo_name?: string;       // 仓库名

  };

    // 同步字段

  local_id: string;           // 插件端 ULID

  source_id?: string;         // 后端返回的 ID

}

#### 增量同步（新增）

```yaml

POST /api/v1/sync/code-fingerprints

Request:

  fingerprints:

    - path: "src/index.js"

      hash: "abc123..."

      mtime: 1712345678

      size: 1024

      symbols_hash: "def456..."

Response:

  to_upload:        # 需要上传（新增/修改）

    - path: "src/index.js"

      reason: "modified"

      server_mtime: 1712340000

  unchanged:        # 未变更

    - path: "src/utils.js"

  conflicts:        # 冲突（多设备编辑）

    - path: "src/auth.js"

      local_mtime: 1712345678

      server_mtime: 1712345600

### 命令列表

```bash

# 分析单个文件

opencode-code-analyzer analyze src/index.ts

# 分析整个项目

opencode-code-analyzer analyze --project .

# 指定语言（强制使用 Tree-sitter）

opencode-code-analyzer analyze --language python src/script.py

# 实时监听

opencode-code-analyzer watch src/

# 代码搜索

opencode-code-analyzer search "function auth"

# 代码复杂度报告

opencode-code-analyzer metrics src/

# 生成分析报告

opencode-code-analyzer report --output report.json

# 显示帮助

opencode-code-analyzer --help

---

## 错误处理和降级策略

> ⚠️ **Review 新增**: 基于 Review 结果，添加详细的错误处理策略

### 三级降级机制

#### 1. 一级解析：Oxc（仅 JS/TS）

- **使用条件**: JavaScript/TypeScript 文件

- **预期性能**: 26ms（5000行）

- **失败条件**:

  - 语法错误（非标准 JS/TS）

  - 内存不足（< 100MB 可用）

  - 超时（> 500ms）

- **失败动作**: 自动降级到 Tree-sitter

#### 2. 二级解析：Tree-sitter（所有语言）

- **使用条件**: 所有编程语言

- **预期性能**: 50-150ms（5000行）

- **失败条件**:

  - 语法错误（严重语法错误）

  - 内存溢出（> 500MB）

  - 文件损坏或编码错误

- **失败动作**: 自动降级到基础信息

#### 3. 三级解析：基础文件信息（保底）

- **使用条件**: 所有解析器失败

- **返回信息**:

  - 文件路径

  - 文件大小

  - 行数

  - 修改时间

  - 文件类型

- **用途**:

  - 不阻塞工作流

  - 至少能搜索到文件

  - 记录错误日志

### 大文件处理策略

| 文件大小 | 处理策略 | 解析深度 |

|---------|---------|---------|

| < 1000 行 | 完整解析 | 100% |

| 1000-5000 行 | 完整解析 + 性能监控 | 100% |

| 5000-10000 行 | 简化解析（只提取函数签名） | 50% |

| > 10000 行 | 跳过详细分析，只记录基础信息 | 0% |

### 错误恢复流程

```text

用户保存文件

    ↓

尝试 Oxc 解析（仅 JS/TS）

    ↓ 失败

尝试 Tree-sitter 解析

    ↓ 失败

返回基础文件信息

    ↓

记录错误日志

    ↓

通知用户（可选）

    ↓

继续工作流（不阻塞）

### 需要后端支持的变更

#### 1. Schema 扩展

```typescript

// 新增字段

interface MemoryItem {

  // ... 现有字段

  metadata: {

    // ... 现有字段

    code_analysis?: CodeAnalysisResult;  // 【新增】

  };

}

#### 3. API 扩展

```yaml

POST /api/v1/memories/search

Request:

  query: "authentication"

  code_filter:              # 【新增】

    language: "typescript"   # 【新增】

    min_complexity: 5       # 【新增】

    max_complexity: 10      # 【新增】

### 依赖配置

```json

// .opencode/package.json

{

  "dependencies": {

    "web-tree-sitter": "^0.22.6",

    "oxc-parser": "^0.121.0"

  }

}

---

## 风险评估与缓解

### 技术风险

| 风险 | 影响 | 概率 | 缓解措施 |

|------|------|------|----------|

| **Tree-sitter WASM 性能不达标** | 高 | 中 | Worker 线程 + 超时保护；大文件分块解析 |

| **Bun 与 Tree-sitter 兼容性问题** | 高 | 低 | 测试验证；fallback 到 Node.js |

| **多设备同步冲突** | 中 | 高 | 指纹对比；冲突检测和手动解决 |

| **内存溢出（大文件）** | 高 | 中 | 5000行上限；内存监控；优雅降级 |

### 实施风险

| 风险 | 影响 | 概率 | 缓解措施 |

|------|------|------|----------|

| **开发周期延误** | 中 | 中 | 分阶段交付；MVP 优先 |

| **测试覆盖不足** | 高 | 中 | 单元测试 + 集成测试；性能基准测试 |

| **文档不同步** | 低 | 高 | 文档即代码；自动化检查 |

---

## 附录

### A. 与 MCP 方案对比

| 特性 | MCP 方案 | 插件+CLI 方案 | 评估 |

|------|---------|--------------|------|

| **自动触发** | ❌ 手动 | ✅ `file.edited` | 插件更好 |

| **实时性** | 网络延迟 | ✅ 进程内 | 插件更好 |

| **用户体验** | 说"使用工具" | ✅ 无缝集成 | 插件更好 |

| **CI/CD 支持** | 需配置 MCP | ✅ CLI 直接可用 | CLI 更好 |

| **复杂度** | 中 | 中 | 相当 |

| **维护成本** | 中 | 中 | 相当 |

**结论**：插件+CLI 混合方案 **优于** 纯 MCP 方案

### B. 性能基准

| 指标 | 目标 | 说明 |

|------|------|------|

| **5000行解析** | <150ms (Tree-sitter) / <30ms (Oxc) | WASM + Worker |

| **本地搜索** | <100ms | Trie + BM25 |

| **语义搜索** | <200ms | Embedding |

| **同步延迟** | <1s | 三层混合 |

| **内存占用** | <500MB | 正常操作 |

### C. 术语表

| 术语 | 定义 |

|------|------|

| **AST** | Abstract Syntax Tree，抽象语法树 |

| **WASM** | WebAssembly，浏览器/Node.js 中运行 |

| **指纹** | 文件内容的 hash，用于检测变更 |

| **OpenCode** | 开源 AI 编码代理 |

| **L0/L1/L2** | 分层内容存储（摘要/概览/完整） |

---

## 审核清单

- [ ] 架构设计是否合理？

- [ ] 技术选型是否有数据支撑？

- [ ] 实施路线图是否可行？

- [ ] 风险评估是否全面？

- [ ] 与 MCP 方案对比是否清晰？

- [ ] 是否满足所有需求？

---

**文档结束**

---

*版本: v1.1.0*

*日期: 2026-03-29*

*状态: 待审核*
