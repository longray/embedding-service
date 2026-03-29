# OpenCode Memory 代码分析融合设计方案

**版本**: v1.0.0  
**日期**: 2026-03-29  
**状态**: 设计审核阶段  

---

## 目录

1. [执行摘要](#执行摘要)
2. [背景与目标](#背景与目标)
3. [三方现状分析](#三方现状分析)
4. [统一架构设计](#统一架构设计)
5. [技术决策详情](#技术决策详情)
6. [数据模型与API契约](#数据模型与api契约)
7. [实施路线图](#实施路线图)
8. [风险评估与缓解](#风险评估与缓解)
9. [附录](#附录)

---

## 执行摘要

本文档汇总了插件端（opencode-memory-plugin）、后端（embedding_service）和 GitNexus 参考设计的融合方案，解决代码分析能力的统一架构问题。

### 核心决策

| 决策项 | 方案 | 关键理由 |
|--------|------|----------|
| **职责划分** | 插件端实时分析 + 后端聚合查询 | 用户代码不出本地，后端负责跨文件关系 |
| **Tree-sitter 实现** | WASM + Worker 线程 | 平衡性能和兼容性，5000行文件<150ms |
| **图数据存储** | 全存 SurrealDB 后端 | 本地不存图，简化架构 |
| **多设备同步** | 三层混合策略 | 本地即时+远程实时+兜底轮询，秒级延迟 |

### 关键数据

- **5000行文件解析**: 50-150ms (WASM) / 20-60ms (Native)
- **同步延迟**: <100ms (Live Query) / 即时 (File Watcher) / 30s (轮询兜底)
- **实施周期**: 4-6 周（含测试）

---

## 背景与目标

### 问题背景

当前系统存在**代码分析能力割裂**的问题：

1. **插件端**: 有完整设计文档（10阶段），但**零代码实现**
2. **后端**: B-028/B-029 完成基础代码分析（AST解析、复杂度计算）
3. **参考**: GitNexus 提供完整客户端代码分析方案，但无记忆系统集成

### 融合目标

设计一个**统一的代码分析方案**，实现：

- ✅ 插件端实时分析（AST解析、符号提取）
- ✅ 后端聚合存储（向量搜索、图关系、语义查询）
- ✅ 增量同步（避免重复工作）
- ✅ 多设备协同（秒级延迟感知变更）

---

## 三方现状分析

### 1. GitNexus 参考架构

```
源代码 → Tree-sitter AST → 符号提取 → 依赖解析 → 社区检测 → 流程追踪 → 知识图谱
```

**核心经验**（值得借鉴）：

| 特性 | 实现方式 | 价值 |
|------|----------|------|
| **零服务器架构** | 浏览器 WASM + 本地 CLI | 隐私优先，开箱即用 |
| **预计算智能** | 索引时完成社区检测、流程追踪 | 查询时零延迟 |
| **MCP 标准化** | 统一工具协议 | 多 AI 客户端兼容 |
| **增量更新** | Git 指纹 + 文件指纹 | 只处理变更，秒级更新 |
| **多层索引** | File → Symbol → Community → Process | 分层查询，精准定位 |

**6阶段索引管道**：

| 阶段 | 功能 | 进度 | 技术 |
|------|------|------|------|
| 1. 扫描 | 遍历文件树 | 0-15% | 文件系统 |
| 2. 结构 | 创建 File/Folder 节点 | 15-20% | 图数据库 |
| 3. 解析 | AST解析提取符号 | 20-82% | Tree-sitter |
| 4. 解析 | 解析导入、调用、继承 | 集成 | 语言感知逻辑 |
| 5. 社区 | Leiden算法检测集群 | 82-92% | 图算法 |
| 6. 流程 | 追踪执行流程 | 92-100% | BFS遍历 |

**技术栈**：

- AST解析: Tree-sitter (native/WASM)
- 图数据库: LadybugDB (嵌入式)
- 社区检测: Leiden算法 (graphology)
- 搜索: BM25 + 语义 + RRF
- 嵌入: transformers.js (可选)

### 2. 插件端现状

**已有能力**：

- ✅ 记忆 CRUD（write/read/search）
- ✅ 增量同步（fingerprint-based）
- ✅ 图关系（relate/graph traverse）
- ✅ 本地缓存（SQLite + BM25 + Trie）
- ❌ **尚无代码分析实现**（只有设计文档）

**数据结构**（当前记忆条目）：

```javascript
{
  content: "L2 完整内容",
  abstract: "L0 ≤100字符",
  overview: "L1 ≤500字符",
  type: "general|daily|preference|code",
  metadata: { /* 自定义 */ },
  local_id: "ULID",
  source_id: "后端ID"
}
```

**设计文档规划**（CODE-ANALYSIS-DESIGN.md）：

```
opencode-memory-plugin/
├── lib/
│   ├── code-parser.js          # AST解析器
│   ├── feature-extractor.js    # 代码特征提取
│   ├── dependency-resolver.js  # 依赖解析
│   ├── code-graph.js           # 知识图谱
│   ├── community-detector.js   # 社区检测
│   ├── process-tracer.js       # 流程追踪
│   ├── impact-analyzer.js      # 影响分析
│   ├── code-search.js          # 代码搜索
│   └── incremental-indexer.js  # 增量索引
├── tools/
│   └── code-analysis.js        # MCP工具
└── parsers/                    # 语言解析器配置
```

**MCP工具规划**（4个核心）：

1. `code_search` - 符号搜索
2. `code_context` - 360度符号视图
3. `code_impact` - 影响分析
4. `code_analyze` - 代码分析

### 3. 后端现状

**已有能力**（B-028/B-029 完成）：

```python
# wrapper/src/utils/code_analyzer.py
class CodeAnalyzer:
    """基于 Tree-sitter 的代码分析器"""
    
    async def analyze_code(self, content: str, language: str) -> CodeAnalysisResult:
        # 支持语言: Python, JS/TS, Java, Go, Rust, C/C++, HTML, CSS, SQL
        # 分析内容: 函数、类、导入、导出、注释、复杂度
        pass

# wrapper/src/utils/memory_manager.py
async def analyze_memory_code(self, memory_id: str, persist: bool = True):
    # 分析代码
    # 存储到 metadata.code_analysis
    # 同步到 Meilisearch
    pass
```

**数据存储**：

- **SurrealDB**: 记忆内容 + 图关系（RELATE）
- **Meilisearch**: 全文搜索 + 代码字段过滤（language, complexity）

**API端点**：

- `POST /api/v1/memories/{memory_id}/analyze/code` - 代码分析
- `POST /api/v1/memories` - 批量上传（支持 auto_analyze_code）
- `POST /api/v1/memories/search` - 搜索（支持 code_filter）

**缺失能力**：

- ❌ 调用链/依赖解析（跨文件）
- ❌ 社区检测/流程追踪
- ❌ LLM代码摘要（B-030 规划中）

---

## 统一架构设计

### 核心原则

```
┌─────────────────────────────────────────────────────────────┐
│  原则 1: 插件端实时分析优先（用户代码不出本地）                  │
│  原则 2: 后端负责聚合查询（跨文件关系、语义搜索）                │
│  原则 3: 增量同步避免重复（指纹对比，只传变更）                  │
│  原则 4: 后端不复分析（trust 插件端的分析结果）                 │
└─────────────────────────────────────────────────────────────┘
```

### 职责边界划分

| 功能 | 插件端 | 后端 | 说明 |
|------|--------|------|------|
| **AST 解析** | ✅ | ❌ | Tree-sitter WASM 本地运行 |
| **符号提取** | ✅ | ❌ | 函数、类、接口 |
| **文件级依赖** | ✅ | ❌ | import/require |
| **增量检测** | ✅ | ❌ | 指纹对比 |
| **本地搜索** | ✅ | ❌ | Trie + BM25 |
| **跨文件调用链** | ❌ | ✅ | 需要全局视图 |
| **语义搜索** | ❌ | ✅ | Embedding |
| **持久化存储** | ❌ | ✅ | SurrealDB + Meilisearch |
| **图关系查询** | ❌ | ✅ | RELATE 遍历 |

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│  插件端（VS Code Extension）- 实时、本地、增量                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ AST 解析    │  │ 符号提取    │  │ 依赖解析    │         │
│  │ (WASM)      │  │             │  │ (文件级)    │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         ▼                ▼                ▼                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  本地指纹缓存 (SQLite/JSON)                          │   │
│  │  - file_path → {hash, mtime, size, symbols_hash}    │   │
│  └─────────────────────────────────────────────────────┘   │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  本地搜索引擎 (Trie + BM25)                          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
          │
          │ 增量同步（只传变更）
          ▼
┌─────────────────────────────────────────────────────────────┐
│  后端（embedding_service）- 聚合、查询、持久化                 │
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
```

---

## 技术决策详情

### 决策 1: Tree-sitter 实现方式

#### 问题

5000行文件解析性能如何？WASM vs Native 怎么选？

#### 调研数据

| 指标 | WASM | Native | 差异 |
|------|------|--------|------|
| **5000行解析时间** | 50-150ms | 20-60ms | WASM 慢 1.5-3x |
| **内存占用** | 100-300MB | 50-150MB | WASM 高 2x |
| **安装复杂度** | 低（npm install） | 高（需编译） | WASM 易部署 |
| **兼容性** | 高（纯JS） | 中（依赖环境） | WASM 更稳定 |

**GitHub Issue #1277 参考**：

- 1.6MB JSON 文件解析：1.2s（首次）/ 0.7s（增量）
- AST 内存占用：约 300MB
- 建议：大文件需要增量解析优化

#### 决策

**方案：WASM + Worker 线程（推荐）**

```javascript
// 主线程
const worker = new Worker('./code-parser-worker.js');

async function parseFile(filePath, content) {
  return new Promise((resolve, reject) => {
    // 超时保护
    const timeout = setTimeout(() => {
      reject(new Error('Parse timeout'));
    }, 500); // 500ms 超时
    
    worker.postMessage({ filePath, content });
    
    worker.onmessage = (e) => {
      clearTimeout(timeout);
      if (e.data.success) {
        resolve(e.data.result);
      } else {
        reject(new Error(e.data.error));
      }
    };
  });
}

// Worker 线程 (code-parser-worker.js)
importScripts('tree-sitter.js', 'tree-sitter-javascript.js');

self.onmessage = async (e) => {
  try {
    const parser = new Parser();
    parser.setLanguage(JavaScript);
    
    const tree = parser.parse(e.data.content);
    const symbols = extractSymbols(tree);
    
    self.postMessage({ success: true, result: symbols });
  } catch (error) {
    self.postMessage({ success: false, error: error.message });
  }
};
```

**理由**：

- 不阻塞 UI（Worker 线程）
- 超时保护避免崩溃
- 兼容性好（无需编译）
- 性能可接受（<150ms）

---

### 决策 2: 多设备同步策略

#### 问题

文件在 A 设备修改，B 设备如何秒级感知？

#### 调研数据

| 方案 | 延迟 | 复杂度 | 可靠性 | 适用场景 |
|------|------|--------|--------|----------|
| **SurrealDB Live Query** | <100ms | 低 | 高 | 实时推送 |
| **VS Code File Watcher** | 即时 | 低 | 中 | 本地变更 |
| **轮询（30s）** | 30s | 低 | 高 | 兜底恢复 |
| **WebSocket 自定义** | <100ms | 高 | 中 | 不推荐 |

#### 决策

**方案：三层混合策略**

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: VS Code File Watcher（本地变更）                   │
│  - 延迟：即时                                                │
│  - 触发：用户保存文件时立即同步                               │
│  - 实现：vscode.workspace.createFileSystemWatcher            │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: SurrealDB Live Query（远程变更）                   │
│  - 延迟：<100ms                                             │
│  - 触发：后端数据变更时主动推送                               │
│  - 实现：WebSocket 订阅                                      │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: 轮询兜底（故障恢复）                               │
│  - 延迟：30 秒                                              │
│  - 触发：定时检测                                            │
│  - 实现：setInterval                                         │
└─────────────────────────────────────────────────────────────┘
```

**实现代码**：

```javascript
// Layer 1: 本地文件监听
const watcher = vscode.workspace.createFileSystemWatcher('**/*.{js,ts}');

watcher.onDidChange(async (uri) => {
  const content = await fs.readFile(uri.fsPath, 'utf-8');
  const fingerprint = calculateFingerprint(content);
  
  // 立即同步到后端
  await wrapperClient.uploadMemory({
    content,
    metadata: { 
      file_path: uri.fsPath,
      fingerprint 
    }
  });
});

// Layer 2: 远程变更监听
const ws = new WebSocket('ws://localhost:17999/ws/live');

ws.onmessage = (event) => {
  const { action, data } = JSON.parse(event.data);
  
  if (action === 'UPDATE' && data.type === 'code') {
    // 更新本地缓存
    localCache.update(data.memory_id, data);
    
    // 通知用户
    vscode.window.showInformationMessage(
      `文件 ${data.metadata.file_path} 在其他设备已更新`
    );
  }
};

// Layer 3: 兜底轮询
setInterval(async () => {
  const serverFingerprints = await wrapperClient.getFingerprints();
  const localFingerprints = await localCache.getFingerprints();
  
  const diff = compareFingerprints(localFingerprints, serverFingerprints);
  await syncDifferences(diff);
}, 30000);
```

**理由**：

- 本地变更即时同步（Layer 1）
- 远程变更秒级感知（Layer 2）
- 故障时自动恢复（Layer 3）

---

### 决策 3: 本地图存储

#### 问题

代码关系图存本地还是后端？

#### 决策

**方案：全存 SurrealDB 后端，本地不存图**

**理由**：

- 简化插件端架构
- 多设备共享同一图
- 后端 RELATE 查询能力强
- 本地只需缓存查询结果

**数据流**：

```
插件端分析 → 提取符号 → 上传到后端 → 后端构建图关系
                ↓
         本地只存指纹
                ↓
         查询时调用后端图遍历
```

---

## 数据模型与API契约

### 代码分析结果格式（插件 ↔ 后端通用）

```typescript
// CodeAnalysisResult - 插件端生成，后端存储
interface CodeAnalysisResult {
  // 基础信息
  language: string;           // "typescript", "python", etc.
  analyzed_at: string;        // ISO 8601 timestamp
  analyzer_version: string;   // "1.0.0"
  
  // 符号列表
  functions: Array<{
    name: string;
    start_line: number;
    end_line: number;
    params: string[];
    return_type?: string;
    is_exported: boolean;
    is_async: boolean;
  }>;
  
  classes: Array<{
    name: string;
    start_line: number;
    end_line: number;
    extends?: string;
    implements?: string[];
    methods: string[];
  }>;
  
  imports: string[];          // 导入的模块名
  exports: string[];          // 导出的符号名
  
  // 复杂度指标
  complexity: {
    cyclomatic: number;       // 圈复杂度
    lines_of_code: number;
    function_count: number;
    class_count: number;
  };
  
  // 文件指纹（用于增量同步）
  fingerprint: {
    hash: string;             // MD5(content)
    mtime: number;            // 修改时间戳
    size: number;             // 文件大小
    symbols_hash: string;     // 符号结构的 hash
  };
}
```

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
```

### API 契约

#### 上传记忆（扩展）

```yaml
POST /api/v1/memories

Request:
  memories:
    - content: "完整代码内容"
      abstract: "Function myFunction in src/index.js (lines 10-20)"
      overview: "Function myFunction(param1: string, param2: number): boolean\n- Exported: true\n- Calls: validateInput, processData"
      type: "code"
      tags: ["javascript", "function", "api"]
      metadata:
        code_analysis:
          language: "javascript"
          functions:
            - name: "myFunction"
              start_line: 10
              end_line: 20
              params: ["param1", "param2"]
              is_exported: true
          imports: ["lodash", "express"]
          complexity:
            cyclomatic: 3
            lines_of_code: 50
          fingerprint:
            hash: "abc123..."
            mtime: 1712345678
            size: 1024
            symbols_hash: "def456..."
        file_path: "src/index.js"
        file_name: "index.js"
      local_id: "01HQ..."
      source_id: null
  
  tenant_id: "default"
  auto_analyze_code: false  # 重要：后端不再分析

Response:
  total: 1
  success: 1
  memory_ids: ["mem:abc123"]
```

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
```

#### 代码搜索（扩展）

```yaml
POST /api/v1/memories/search

Request:
  query: "authentication"
  code_filter:
    language: "typescript"
    min_complexity: 5
  mode: "hybrid"
  limit: 10

Response:
  results:
    - id: "mem:abc123"
      content: "..."
      metadata:
        code_analysis:
          language: "typescript"
          functions: [...]
          complexity: {cyclomatic: 7}
      similarity: 0.85
```

---

## 实施路线图

### Phase 1: 插件端基础（2 周）

**目标**: 实现 AST 解析和符号提取

**任务清单**：

- [ ] 添加 Tree-sitter WASM 依赖
- [ ] 实现 `lib/code-parser.js`（AST 解析 + Worker 封装）
- [ ] 实现 `lib/code-analyzer.js`（符号提取）
- [ ] 实现 `lib/code-fingerprint.js`（指纹计算）
- [ ] 编写单元测试

**产出**：

- `lib/code-parser.js`
- `lib/code-analyzer.js`
- `lib/code-fingerprint.js`
- 测试覆盖率 >80%

### Phase 2: 后端适配（1 周）

**目标**: 适配插件端分析结果

**任务清单**：

- [ ] 修改上传 API（接受插件端 analysis 结果）
- [ ] 实现指纹同步 API
- [ ] 启用 SurrealDB Live Query WebSocket
- [ ] 端到端集成测试

**产出**：

- 后端 API 更新
- 集成测试通过

### Phase 3: 同步与 MCP 工具（1 周）

**目标**: 实现多设备同步和 MCP 工具

**任务清单**：

- [ ] 实现 VS Code File Watcher 集成
- [ ] 实现 WebSocket 远程监听
- [ ] 实现兜底轮询
- [ ] 实现 `code_analyze` MCP 工具
- [ ] 实现 `code_search` MCP 工具

**产出**：

- `tools/code-analysis.js`
- 同步功能完整

### Phase 4: 高级功能（后续）

**目标**: 跨文件调用链和社区检测

**任务清单**：

- [ ] 后端实现跨文件依赖解析
- [ ] 后端实现调用链图构建
- [ ] 插件端实现 `code_context` 工具
- [ ] 插件端实现 `code_impact` 工具
- [ ] 社区检测（Leiden 算法）

**产出**：

- 完整代码分析工具集

---

## 风险评估与缓解

### 技术风险

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| **Tree-sitter WASM 性能不达标** | 高 | 中 | Worker 线程 + 超时保护；大文件分块解析 |
| **多设备同步冲突** | 中 | 高 | 三层同步策略；冲突检测和手动解决 |
| **内存溢出（大文件）** | 高 | 中 | 5000行上限；内存监控；优雅降级 |
| **后端 API 变更** | 中 | 低 | 版本化 API；向后兼容 |

### 实施风险

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| **开发周期延误** | 中 | 中 | 分阶段交付；MVP 优先 |
| **测试覆盖不足** | 高 | 中 | 单元测试 + 集成测试；性能基准测试 |
| **文档不同步** | 低 | 高 | 文档即代码；自动化检查 |

---

## 附录

### A. 与 GitNexus 对比

| 特性 | GitNexus | 本方案 | 说明 |
|------|----------|--------|------|
| **架构** | 纯客户端（WASM） | 客户端 + 后端 | 我们持久化 + 语义搜索 |
| **图数据库** | LadybugDB（本地） | SurrealDB（后端） | 后端存储，多设备同步 |
| **社区检测** | Leiden 算法 | 后端可选 | 非核心，延后实现 |
| **流程追踪** | BFS 调用链 | 后端图遍历 | 使用 RELATE |
| **MCP 工具** | 7 个工具 | 4 个核心工具 | 先实现核心功能 |
| **增量更新** | Git + 指纹 | 指纹 + 同步 API | 不依赖 Git |
| **语义搜索** | transformers.js | 后端 Embedding | 更准确，更可控 |

### B. 性能基准

| 指标 | 目标 | 说明 |
|------|------|------|
| **5000行解析** | <150ms | WASM + Worker |
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
| **Live Query** | SurrealDB 的实时数据变更推送 |
| **RELATE** | SurrealDB 的图关系查询语法 |
| **MCP** | Model Context Protocol，AI 工具标准协议 |
| **L0/L1/L2** | 分层内容存储（摘要/概览/完整） |

### D. 参考资料

1. **GitNexus**: https://github.com/abhigyanpatwari/GitNexus
2. **Tree-sitter**: https://tree-sitter.github.io/tree-sitter/
3. **SurrealDB Live Query**: https://surrealdb.com/docs/integration/websocket
4. **VS Code File Watcher**: https://code.visualstudio.com/api/references/vscode-api#FileSystemWatcher

---

**文档结束**

---

## 审核清单

- [ ] 架构设计是否合理？
- [ ] 技术决策是否有数据支撑？
- [ ] 实施路线图是否可行？
- [ ] 风险评估是否全面？
- [ ] 是否需要补充其他内容？

请审核后反馈意见，我将根据反馈进行调整。
