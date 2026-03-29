# 代码分析增强设计文档

> 参考 GitNexus（代码智能引擎）的功能分析，增强 Memory Stack 的代码分析能力。
> GitNexus 仅作为参考，非规范要求。本项目不使用其代码（PolyForm Noncommercial License）。

## 0. 核心定位

**本项目是记忆级代码分析服务**，输入是插件逐条上传的代码片段，不是完整 Git 仓库。

### 与 GitNexus 的根本差异

| 维度 | GitNexus | 本项目 (Memory Stack) |
|------|---------|----------------------|
| 输入 | 完整 Git 仓库 / ZIP | 插件逐条上传的记忆片段 |
| 粒度 | 仓库级（全文件树） | 记忆级（单条代码片段） |
| 解析时机 | `gitnexus analyze` 一次性全量 | 上传时逐条分析 |
| 跨文件关系 | 全仓库 import/call 解析 | 延迟解析（上传时标记，后续关联） |
| 客户端 | CLI + Web UI | OpenCode 插件 (MCP) |

### 本项目不做什么

- 不做 CLI 命令行工具（本项目是服务端 API）
- 不做仓库级扫描（输入是记忆片段，不是完整仓库）
- 不做 Web UI（通过 OpenCode 插件交互）
- 不做 `detect_changes`（后端不访问 git 仓库）
- 不做 `rename`（记忆条目是不可变的历史记录）
- 不做多仓库管理

### 现有能力

| 能力 | 实现位置 | 状态 |
|------|---------|------|
| Tree-sitter AST 解析 (11 语言) | `code_analyzer.py` | ✅ python, javascript, typescript, java, go, rust, c, cpp, html, css, sql |
| 函数/类/导入/注释提取 | `CodeAnalyzer` 类 | ✅ |
| 复杂度度量 | `_calculate_complexity()` | ✅ |
| 依赖项提取 | `_extract_dependencies()` | ✅ |
| SurrealDB 图关系 | `create_relation()` API | ✅ |
| 图遍历查询 | `get_related_memories()` | ✅ |
| 向量搜索 (HNSW) | `_search_by_vector()` | ✅ |
| BM25 + RRF 混合搜索 | Meilisearch + SurrealDB | ✅ |
| Leiden 社区检测 | `cluster_memories_leiden()` | ✅ |
| 代码语言检测 | `_detect_programming_language()` | ✅ 基础版 |
| OpenCode 插件 SDK | `@opencode-ai/plugin` v1.3.3 | ✅ |

### 关键差距

| 差距 | 影响 | 优先级 |
|------|------|--------|
| 代码分析结果未持久化 | 每次查看需重新计算 | **P0** |
| 分析结果未同步到 Meilisearch | 搜索无法按代码属性过滤 | **P1** |
| 无 LLM 代码摘要 | 缺少自然语言描述 | **P2** |
| 无跨文件关系解析 | 无法回答"谁调用了这个函数" | **远期** |

---

## 1. 实施计划

### 依赖关系

```mermaid
graph LR
    B027[B-027 修复设计文档] --> B028[B-028 分析持久化]
    B028 --> B029[B-029 Meilisearch 增强]
    B028 --> B030[B-030 LLM 摘要]
    B028 --> B031[B-031 跨文件关系<br/>远期考虑]
    B029 --> B032[B-032 插件端工具]
    B030 --> B032
```yaml

### Phase A: 代码分析持久化 (B-028)

**场景**: 用户上传 Python 代码，后端自动分析并持久化到 `metadata.code_analysis`。

**后端变更**:

- `analyze_memory_code()`: 分析后将结果写入 `metadata.code_analysis`
- `create_memory()`: 上传时如果 `auto_analyze_code=true`，自动触发分析
- `CodeAnalysisResult`: 新增 `analyzed_at`、`analyzer_version` 字段
- `config.py`: 新增 `code_analysis.auto_analyze` 配置

**元数据结构**:

```python
metadata["code_analysis"] = {
    "language": "python",
    "functions": [
        {"name": "validate_user", "start_line": 15, "end_line": 30, "parameters": ["user_id"]}
    ],
    "classes": [
        {"name": "UserService", "start_line": 5, "end_line": 100}
    ],
    "imports": ["from database import get_connection"],
    "dependencies": ["database", "authlib"],
    "complexity": {
        "lines_of_code": 150,
        "function_count": 5,
        "class_count": 2,
        "nesting_depth": 3,
        "cyclomatic_complexity": 8
    },
    "analyzed_at": "2026-03-29T10:00:00Z",
    "analyzer_version": "1.0.0"
}
```

**降级策略**: 分析失败不影响上传，`metadata.code_analysis` 为 `null`。

### Phase B: Meilisearch 搜索增强 (B-029)

**场景**: 用户搜索 "authentication"，希望只返回 Python 代码中复杂度 > 5 的结果。

**变更**:

- Meilisearch 文档新增字段: `code_language`、`code_function_names`、`code_class_names`、`code_max_complexity`
- 搜索 API 新增 `code_filter` 参数: `{language: "python", min_complexity: 5}`
- 向后兼容：无 `code_filter` 时行为不变

### Phase C: LLM 代码摘要 (B-030)

**场景**: 用户上传 200 行代码，希望看到"该模块实现了用户认证功能"的自然语言摘要。

**后端变更**:

- `config.py`: 新增 `LLMConfig`（`endpoint`、`api_key`、`model_name`）
- `memory_manager.py`: 新增 `_generate_code_summary()` 异步方法
- `metadata["code_summary"]`: `{summary, key_functions, purpose, generated_at, model}`
- 新增 API: `POST /api/v1/memories/{id}/enrich/llm`、`GET /api/v1/memories/{id}/summary`

**LLM 配置**:

- 环境变量: `WRAPPER_LLM_ENDPOINT`、`WRAPPER_LLM_API_KEY`、`WRAPPER_LLM_MODEL`
- 兼容 OpenAI API 格式（支持 Kaggle LLM 或其他兼容服务）
- 异步调用，不阻塞上传

### Phase D: 跨文件关系 (B-031, 远期考虑)

**⚠️ 可行性限制**: 记忆级输入的上传顺序不确定，文件 A 上传时文件 B 可能还不存在。

**策略**: 延迟解析 — 上传时提取 import 语句，标记待关联；手动或定时触发批量关联。

**置信度规则**:

| 场景 | Confidence | Reason |
|------|-----------|--------|
| 同文件引用 | 1.0 | `same-file` |
| import 解析匹配 | 0.85 | `import-resolved` |
| 模糊全局匹配 | 0.5 | `fuzzy-global` |
| 常见名称模糊匹配 | 0.3 | `fuzzy-global-low` |

### Phase E: 插件端工具 (B-032)

**7 个新工具**（基于 `@opencode-ai/plugin` v1.3.3 的 `Hooks.tool` 类型）:

| 工具名 | 对应后端 API | 阶段 |
|-------|---------|------|
| `memory_code_analyze` | `POST /api/v1/memories/{id}/analyze/code` | A |
| `memory_code_search` | `POST /api/v1/memories/search` + `code_filter` | B |
| `memory_code_enrich` | `POST /api/v1/memories/{id}/enrich/llm` | C |
| `memory_code_summary` | `GET /api/v1/memories/{id}/summary` | C |
| `memory_code_graph` | `GET /api/v1/memories/{id}/knowledge-graph` | D |
| `memory_code_impact` | `POST /api/v1/memories/{id}/impact` | D |
| `memory_code_processes` | `GET /api/v1/memories/{id}/processes` | D |

**工具注册示例**（基于 `Hooks.tool` 类型）:

```typescript
import type { Plugin, PluginInput } from "@opencode-ai/plugin";

const plugin: Plugin = async (input: PluginInput) => {
  return {
    tool: {
      memory_code_analyze: {
        description: "Analyze code content in a memory and persist results. Returns language, functions, classes, complexity metrics.",
        parameters: {
          type: "object",
          properties: {
            memory_id: { type: "string", description: "Memory ID to analyze" },
          },
          required: ["memory_id"],
        },
        execute: async (args: { memory_id: string }) => {
          const resp = await fetch(
            `${input.serverUrl.origin}/api/v1/memories/${args.memory_id}/analyze/code?tenant_id=${input.project.id}`,
            { method: "POST", headers: { "Content-Type": "application/json" } }
          );
          if (!resp.ok) throw new Error(`Analysis failed: ${resp.status}`);
          return resp.text();
        },
      },
    },
  };
};

export default plugin;
```text

---

## 2. 后端 API 变更汇总

| 端点 | 方法 | 功能 | 状态 | 阶段 |
|------|------|------|------|------|
| `/api/v1/memories` | POST | 添加 `auto_analyze_code` 选项 | 修改 | A |
| `/api/v1/memories/{id}/analyze/code` | POST | 触发代码分析 + 持久化 | 修改 | A |
| `/api/v1/memories/search` | POST | 添加 `code_filter` 参数 | 修改 | B |
| `/api/v1/memories/{id}/enrich/llm` | POST | LLM 代码摘要 | 新增 | C |
| `/api/v1/memories/{id}/summary` | GET | 获取代码摘要 | 新增 | C |
| `/api/v1/memories/{id}/resolve-relations` | POST | 手动触发关系解析 | 新增 | D |
| `/api/v1/memories/{id}/knowledge-graph` | GET | 获取代码知识图谱 | 新增 | D |

---

## 3. 设计原则

1. **只做当前切片**: 每个 Phase 独立实现、可回滚
2. **后端自动完成**: 上传代码后自动触发分析（不阻塞上传）
3. **插件端轻量**: 只做工具注册和 API 调用，不包含解析逻辑
4. **降级优先**: 分析/LLM 失败不影响核心上传和搜索功能
5. **向后兼容**: 新字段都是 `metadata` 子字段，现有 API 行为不变
6. **与现有搜索协同**: 代码分析增强搜索质量，不替代现有搜索

---

## 4. 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| `memory_manager.py` 编辑易损坏 | git stash + 小范围编辑 + 每步跑测试 |
| 代码分析增加上传延迟 | 异步触发，不阻塞上传响应 |
| LLM 服务不可用 | fallback 到空摘要，不影响上传 |
| Meilisearch schema 变更需要重建索引 | 新字段可空，不强制重建 |

### 回滚策略

每个 Phase 独立可回滚：

- Phase A: 删除 `metadata.code_analysis` 字段处理代码，恢复原始 `analyze_memory_code()`
- Phase B: 移除 Meilisearch 新字段，搜索 API 忽略 `code_filter`
- Phase C: 删除 LLM 配置和 `_generate_code_summary()` 方法

### 测试策略

| Phase | 单元测试 | 集成测试 | 回归测试 |
|-------|---------|---------|---------|
| A | 分析持久化、失败降级、自动触发 | E2E 上传→分析→查询 | 32 同步测试 |
| B | Meilisearch 字段映射、过滤查询 | 搜索带 code_filter | 现有搜索测试 |
| C | LLM 调用 mock、摘要持久化 | E2E 上传→摘要→查询 | Phase A 测试 |
| D | 关系解析、置信度计算 | 上传多文件→关联→图查询 | Phase A+B 测试 |

### 性能预估

| 操作 | 预估延迟 | 说明 |
|------|---------|------|
| Tree-sitter 分析 (< 100 行) | ~10-50ms | CPU 密集，但不涉及网络 |
| Tree-sitter 分析 (1000+ 行) | ~100-500ms | 考虑对超长代码截断 |
| LLM 摘要生成 | ~1-5s | 异步，不阻塞上传 |
| Meilisearch 字段更新 | ~5-10ms | 随正常同步完成 |

### 数据迁移

已有记忆无 `code_analysis` 字段。迁移策略：

- **不强制迁移**: 查询时检测 `metadata.code_analysis` 为 `null` 则视为未分析
- **按需补充**: 用户可手动调用 `POST /api/v1/memories/{id}/analyze/code` 触发分析
- **批量迁移** (可选): 提供脚本扫描所有代码类型记忆，逐条触发分析

---

## 5. 参考资料

- GitNexus — https://github.com/abhigyanpatwari/GitNexus (PolyForm Noncommercial License, 仅供参考)
- Tree-sitter — AST 解析引擎
- SurrealDB 3.0 — 本地图数据库
- Meilisearch — 全文搜索引擎
- Qwen3-Embedding-0.6B — 嵌入模型 (640 维)
- `@opencode-ai/plugin` SDK v1.3.3 — OpenCode 插件框架
