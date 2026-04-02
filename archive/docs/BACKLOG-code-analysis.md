# Backlog - 代码分析集成（Phase 1 后端）

> **关联设计**: `docs/CODE-ANALYSIS-DESIGN-v1.2.md`
> **状态**: 待用户确认后执行
> **约束**: 不提交 git，保留现有未提交改动

---

## 使用场景

### 场景 1：插件端上传代码分析结果

```text
用户在 VSCode 编辑 src/analyzer.ts
  → 插件端 file.edited 事件触发
  → Tree-sitter WASM 解析代码
  → POST /api/v1/memories {type: "code", metadata.code_analysis: {...}}
  → 后端接收、存储到 SurrealDB、索引到 Meilisearch
  → 返回 success
```

**后端需要做的**：接收新格式数据，提取代码字段写入 Meilisearch 平级字段（code_language 等），确保 code_filter 搜索正常。

### 场景 2：搜索代码记忆

```text
用户搜索 "CodeAnalyzer"
  → POST /api/v1/memories/search {query: "CodeAnalyzer", code_filter: {language: "typescript"}}
  → 后端转为 Meilisearch filter: code_language = "typescript"
  → 全文匹配 code_symbols 中的符号名
  → 返回匹配的代码记忆
```

**后端需要做的**：code_filter 支持 language / min_complexity / max_complexity，code_symbols 字段可被搜索。

### 场景 3：同一文件重复保存（Upsert）

```text
用户第二次保存 src/analyzer.ts
  → 插件端再次上传 {file_path: "src/analyzer.ts", project_id: "xxx"}
  → 后端检测到 file_path + project_id + tenant_id 已存在
  → UPDATE 而非 CREATE（避免重复）
```

**后端需要做的**：upload_memories 中对 type="code" 的记忆实现 upsert 逻辑。

> ⚠️ **Upsert 涉及 memory_manager.py，该文件有大量未实现的 stub 方法（已知 LSP 153 错误）。本切片仅添加 upsert 分支，不扩散重构。**

---

## Backlog 项

### BL-CA-01：升级 CodeAnalysisResult dataclass

| 字段 | 内容 |
|------|------|
| **目标** | 扩展 CodeAnalysisResult，新增 interfaces/analyzer/errors/warnings 字段，使 imports/exports/dependencies 兼容新格式，同时不破坏现有 Tree-sitter 解析逻辑 |
| **涉及范围** | `wrapper/src/utils/code_analyzer.py`（第 25-58 行 dataclass + to_metadata_dict） |
| **前置依赖** | 无 |
| **完成标准** | ① CodeAnalysisResult 新增 `interfaces: List[Dict] = field(default_factory=list)`、`analyzer: str = "tree-sitter"`、`errors: List[Dict] = field(default_factory=list)`、`warnings: List[Dict] = field(default_factory=list)` ② `to_metadata_dict()` 输出包含新字段 ③ 现有 `_analyze_with_tree_sitter()` 和 `_analyze_with_regex()` 构造函数时传入 `analyzer="tree-sitter"` 或 `analyzer="regex"` ④ 现有 32 个测试全部通过 |
| **验证方式** | `uv run pytest tests/ -v` 全部通过；`uv run pyright wrapper/src/utils/code_analyzer.py` 零新增错误 |

---

### BL-CA-02：扩展 Meilisearch 索引设置

| 字段 | 内容 |
|------|------|
| **目标** | 在 DEFAULT_INDEX_SETTINGS 中新增 code_function_count / code_class_count / code_analyzer 到 filterableAttributes，新增 code_symbols 到 searchableAttributes，新增 code_function_count 到 sortableAttributes |
| **涉及范围** | `wrapper/src/utils/meili_client.py`（第 43-60 行 DEFAULT_INDEX_SETTINGS） |
| **前置依赖** | 无（与 BL-CA-01 无依赖，可并行） |
| **完成标准** | ① filterableAttributes 包含 `code_function_count`、`code_class_count`、`code_analyzer` ② searchableAttributes 包含 `code_symbols` ③ sortableAttributes 包含 `code_function_count` ④ 索引配置格式正确，`configure_index()` 可正常调用 |
| **验证方式** | `uv run pyright wrapper/src/utils/meili_client.py` 零新增错误；手动启动服务验证 `configure_index()` 不报错 |

---

### BL-CA-03：新增 build_code_symbols 工具函数

| 字段 | 内容 |
|------|------|
| **目标** | 实现工具函数，将 code_analysis 中的符号名（函数名、类名、接口名、导出名）拼接为可搜索的空格分隔文本字符串 |
| **涉及范围** | `wrapper/src/utils/code_analyzer.py`（新增一个模块级函数，约 20 行） |
| **前置依赖** | 无（纯新增函数，不依赖 BL-CA-01 的 dataclass 变更） |
| **完成标准** | ① `build_code_symbols(code_analysis: dict) -> str` 函数存在 ② 输入含 functions/classes/interfaces/exports 时正确提取 name 字段 ③ 输入为空字典时返回空字符串 ④ 输入中 exports 为 `str`（旧格式）和 `dict`（新格式）均兼容 |
| **验证方式** | 编写 5 个 pytest 用例覆盖：空输入、仅函数、仅类、混合符号、旧格式 exports 兼容 |

---

### BL-CA-04：上传时提取代码字段到 Meilisearch

| 字段 | 内容 |
|------|------|
| **目标** | 插件端上传 type="code" 的记忆时，后端自动从 metadata.code_analysis 提取 code_language / code_complexity / code_function_count / code_class_count / code_analyzer / code_symbols 到 Meilisearch 文档的平级字段 |
| **涉及范围** | `wrapper/src/main.py` 的 `upload_memories()` 端点（第 508-541 行），在 `memory_manager.upload_memories()` 调用后、返回 result 前，对 memories 中 type="code" 的项提取字段并写入 Meilisearch |
| **前置依赖** | BL-CA-02（Meilisearch 索引已有新 filterableAttributes）、BL-CA-03（build_code_symbols 函数可用） |
| **完成标准** | ① 上传含 code_analysis 的记忆后，Meilisearch 文档包含 code_language / code_complexity / code_function_count / code_class_count / code_analyzer / code_symbols 平级字段 ② 上传不含 code_analysis 的记忆不受影响 ③ code_symbols 由 build_code_symbols() 生成 |
| **验证方式** | 编写 pytest 用例：mock MeilisearchClient.add_documents，上传含 code_analysis 的记忆，断言写入文档包含正确的平级字段 |

---

### BL-CA-05：code_filter 添加 max_complexity 支持

| 字段 | 内容 |
|------|------|
| **目标** | 搜索 API 的 code_filter 参数新增 max_complexity 过滤，转换为 `code_complexity <= N` Meilisearch 过滤条件 |
| **涉及范围** | `wrapper/src/main.py` 的 `search_memories()` 端点（第 551-558 行），在现有 filter_parts 逻辑中添加 `max_complexity` 分支 |
| **前置依赖** | 无（独立修复，1 行代码） |
| **完成标准** | ① code_filter 含 max_complexity 时生成 `code_complexity <= N` 过滤条件 ② 与 language / min_complexity 组合使用时用 AND 连接 ③ 不含 max_complexity 时行为不变 |
| **验证方式** | 编写 pytest 用例：验证 `{"language": "python", "min_complexity": 5, "max_complexity": 30}` 生成正确的 filter 字符串 |

---

### BL-CA-06：修复 v1.2 设计文档 4 个小问题

| 字段 | 内容 |
|------|------|
| **目标** | 修正 CODE-ANALYSIS-DESIGN-v1.2.md 中发现的 4 个文档错误 |
| **涉及范围** | `docs/CODE-ANALYSIS-DESIGN-v1.2.md`（4 处修改） |
| **前置依赖** | 无 |
| **完成标准** | ① 第 304 行 "memory < 100MB" → "system available memory" ② Section 4.3 batch upload 示例补充 `tenant_id` ③ Section 9.2 search 响应 `hits` → `results` ④ 通过 markdownlint 检查 |
| **验证方式** | `uv run task lint-md` 通过；人工 review 4 处修改 |

---

## 依赖关系

```text
BL-CA-01 (dataclass)  ──┐
BL-CA-02 (meili index) ──┼── BL-CA-04 (上传提取) ── 联调测试
BL-CA-03 (symbols func)─┘
BL-CA-05 (max_complexity) ← 独立，随时可做
BL-CA-06 (doc fixes)     ← 独立，随时可做
```

## 执行顺序建议

1. **BL-CA-06** — 文档修复（最简单，热身）
2. **BL-CA-05** — max_complexity（1 行代码）
3. **BL-CA-01** + **BL-CA-02** + **BL-CA-03** — 并行（互不依赖）
4. **BL-CA-04** — 最后做（依赖 01/02/03）
5. 跑聚焦测试 + 汇报

---

*最后更新: 2026-03-31*
