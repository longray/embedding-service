# 致后端团队：代码分析 v1.4 任务对齐函

**日期**: 2026-04-07  
**发件人**: OpenCode Memory Plugin 团队  
**主题**: 代码分析功能 v1.4 实施任务边界对齐

---

## 1. 背景与目标

基于 v1.4 设计文档（`CODE-ANALYSIS-DESIGN-v1.4.md`），我们已完成任务拆分。本文档旨在明确**插件端**与**后端**的职责边界，确保双方对数据结构、API 契约和实施优先级达成一致。

---

## 2. 职责边界确认

### 2.1 插件端职责（已实现/待实现）

| 功能 | 状态 | 说明 |
|------|------|------|
| 文件监听 (chokidar) | ✅ 已实现 | 300ms 防抖，OpenCode 事件触发 |
| AST 解析 (Oxc/Tree-sitter) | ✅ 已实现 | JS/TS 用 Oxc，其他语言用 Tree-sitter |
| 代码分析结果生成 | 🔄 进行中 | 补充 v1.2 遗漏字段（见第 3 节） |
| 批量上传 | ✅ 已实现 | 通过 `memory_write` 批量上传 |
| 隐私过滤 | ✅ 已实现 | 敏感信息检测与脱敏 |

### 2.2 后端职责（待实现）

| 功能 | 优先级 | 依赖 |
|------|--------|------|
| Schema 扩展 (BL-CA-18) | P0 | 插件端数据结构最终确认 |
| 调用关系存储 (BL-CA-20) | P1 | 需 `CallSymbol` 数据 |
| 引用查询 API (BL-CA-21) | P1 | 依赖 BL-CA-20 |
| 依赖查询 API (BL-CA-22) | P1 | 依赖 BL-CA-20 |
| 代码地图 API (BL-CA-24) | P1 | 需项目级聚合能力 |
| 语义代码搜索 (BL-CA-26~29) | P2 | 需 embedding 服务支持 |
| opencode 工具集成 (BL-CA-30~33) | P3 | 需 API 稳定后实施 |

---

## 3. 数据结构对齐

### 3.1 CodeAnalysisResult（v1.4 完整版）

插件端将按以下结构生成代码分析结果，请后端确认 Schema 支持能力：

```typescript
interface CodeAnalysisResult {
  // 基础信息
  language: string;              // 标准化语言名
  analyzer: string;             // "oxc" | "tree-sitter" | "regex"
  analyzed_at: string;           // ISO 8601
  analyzer_version: string;     // "1.4.0"

  // 符号信息
  functions: FunctionSymbol[];   // ✅ 完整字段（含 return_type, is_exported, is_async）
  classes: ClassSymbol[];        // ✅ 完整字段（含 methods, properties）
  interfaces: InterfaceSymbol[]; // ✅ v1.4 新增
  imports: ImportSymbol[];       // ✅ 结构化（source, imported_names, line）
  exports: ExportSymbol[];       // ✅ 结构化（name, type, is_default, line）

  // 复杂度指标
  complexity_metrics: ComplexityMetrics;  // ✅ 完整字段

  // 依赖信息
  dependencies: DependencyInfo;  // ✅ 分类实现（internal/external/builtin）

  // 调用关系（v1.4 新增）
  calls?: CallSymbol[];          // ✅ 函数调用关系

  // 错误与警告
  errors?: ParseError[];         // ✅ 降级时填充
  warnings?: ParseWarning[];     // ✅ 降级时填充
}
```

### 3.2 关键字段说明

**FunctionSymbol 完整字段**（BL-CA-11）：
```typescript
interface FunctionSymbol {
  name: string;
  start_line: number;
  end_line: number;
  params: Array<{ name: string; type?: string }>;  // ✅ 支持类型
  return_type?: string;                            // ✅ v1.4 新增
  is_exported: boolean;                            // ✅ v1.4 新增
  is_async: boolean;                               // ✅ v1.4 新增
}
```

**CallSymbol 新增**（BL-CA-19）：
```typescript
interface CallSymbol {
  target: string;          // 被调用函数名
  line: number;            // 调用所在行
  column?: number;         // 调用所在列
}
```

**DependencyInfo 分类**（BL-CA-16）：
```typescript
interface DependencyInfo {
  internal: string[];   // 相对路径导入（./, ../）
  external: string[];   // npm/pip/cargo 等外部包
  builtin: string[];   // 内置模块（node:fs, os, sys 等）
}
```

---

## 4. API 契约确认

### 4.1 现有 API（已对齐）

```yaml
POST /api/v1/memories
# 插件端通过此接口上传代码分析结果
# 请求体包含 code_analysis 字段（CodeAnalysisResult 类型）
```

### 4.2 待实现 API（需后端确认）

**调用关系存储**：
```yaml
# 方案 A：复用 memory_relation 表，新增 relationship_type = "calls"
POST /api/v1/relations
{
  from: "memory:func_a",
  to: "memory:func_b", 
  relationship_type: "calls",
  weight: 1,
  metadata: { line: 42, column: 10, file_path: "src/auth.ts" }
}

# 方案 B：独立 calls 表（推荐，更灵活）
POST /api/v1/calls
{
  caller_memory_id: "mem_xxx",
  callee_memory_id: "mem_yyy",
  line: 42,
  column: 10,
  file_path: "src/auth.ts"
}
```

**引用查询 API**（BL-CA-21）：
```yaml
GET /api/v1/memories/{id}/references

Response:
  references:
    - memory_id: "mem_xxx"
      file_path: "src/utils.ts"
      line: 42
      caller_function: "validateUser"
      confidence: 0.95
```

**依赖查询 API**（BL-CA-22）：
```yaml
GET /api/v1/memories/{id}/dependencies

Response:
  dependencies:
    - memory_id: "mem_yyy"
      file_path: "src/auth.ts"
      line: 15
      callee_function: "hashPassword"
      type: "internal"  # internal/external/builtin
```

**代码地图 API**（BL-CA-24）：
```yaml
GET /api/v1/projects/{id}/map

Response:
  project_id: "github.com/user/repo"
  file_tree: [...]           # 文件树结构
  module_dependencies: [...] # 模块依赖关系
  hot_files: [...]          # 热点文件（基于复杂度）
  statistics: {...}         # 项目统计
```

### 4.3 待确认问题

1. **调用关系存储方案**：方案 A（复用 relations 表）vs 方案 B（独立 calls 表），后端推荐哪种？
2. **批量上传 calls**：是否支持在 `POST /api/v1/memories` 中嵌套 `calls` 数组，还是需单独接口？
3. **project_id 格式**：使用 Git remote URL（如 `github.com/user/repo`）还是目录名？
4. **Meilisearch 索引字段**：请确认以下字段已添加：
   - `code_function_count`
   - `code_class_count`
   - `code_analyzer`
   - `code_has_exports`

---

## 5. 实施优先级建议

基于插件端进度，建议后端按以下顺序实施：

### Phase 1: Schema 扩展（P0，2 周）
- [ ] BL-CA-18: 扩展 Meilisearch 索引字段
- [ ] 确认 CodeAnalysisResult JSON Schema

### Phase 2: 调用关系（P1，3 周）
- [ ] BL-CA-20: 调用关系存储接口
- [ ] BL-CA-21: 引用查询 API
- [ ] BL-CA-22: 依赖查询 API

### Phase 3: 代码地图（P1，2 周）
- [ ] BL-CA-23: 项目统计聚合
- [ ] BL-CA-24: 代码地图 API
- [ ] BL-CA-25: 热点文件标记

### Phase 4+: 语义搜索与工具（P2/P3）
- 待 Phase 1-3 稳定后实施

---

## 6. 插件端当前状态

### 已完成（v3.0.0）
- ✅ Oxc 解析器集成（JS/TS）
- ✅ Tree-sitter 原生绑定（Python/Go/Rust/Java）
- ✅ 文件监听（chokidar，300ms 防抖）
- ✅ 隐私过滤（敏感信息检测）
- ✅ 批量上传队列
- ✅ CLI 工具（`code-analyzer.cjs`）
- ✅ 基础 FunctionSymbol/ClassSymbol 提取

### 进行中（v1.4）
- 🔄 BL-CA-12: CallSymbol 提取（调用关系）
- 🔄 BL-CA-14: Tree-sitter 多语言增强
- 🔄 BL-CA-16: 文件级质量评分

### 待后端完成后联调
- ⏳ 调用关系存储（需 BL-CA-20）
- ⏳ 引用查询（需 BL-CA-21）
- ⏳ 代码地图（需 BL-CA-24）

---

## 7. 需要后端明确的 5 个问题

1. **调用关系存储方案选择**（方案 A vs B）？
2. **批量上传 calls 的方式**（嵌套 vs 独立接口）？
3. **project_id 格式标准**？
4. **Meilisearch 索引字段是否已就绪**？
5. **Phase 1 预计完成时间**？

---

## 8. 附录

### 参考文档
- 设计文档: `D:\embedding_service\docs\CODE-ANALYSIS-DESIGN-v1.4.md`
- 插件端 Backlog: `D:\github\opencode-memory-plugin\BACKLOG.md` (Scene 9)
- 后端 Backlog: `D:\embedding_service\BACKLOG.md` (Scene 9)

### 联系方式
- 插件端负责人: [待填写]
- 后端负责人: [待填写]
- 对齐会议: 建议本周内安排 30 分钟会议确认 API 契约

---

**期待后端的反馈，让我们共同推进 v1.4 的顺利实施！**

---

*文档版本: v1.0*  
*日期: 2026-04-07*  
*状态: 待后端确认*
