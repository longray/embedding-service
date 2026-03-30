# Backlog

> 后端任务追踪文档，按优先级排序。已完成任务归档至 backlog_archive.md。

**更新时间**: 2026-03-30

---

## v2.4.2 - 进行中

### BL-1 [P2] Tenant ID 不匹配 #sync #low-priority

**目标**: 解决插件端和后端使用不同 tenant_id 导致的数据隔离问题，确保用户上传的记忆能在正确的租户下查询。

**涉及范围**:
- **后端**: `wrapper/src/config.py`（default_tenant_id 配置）
- **后端**: `wrapper/src/main.py`（API 参数处理）
- **后端**: `wrapper/src/utils/memory_manager.py`（数据隔离逻辑）
- **插件端**: `opencode-memory-plugin` 配置（需插件端配合）

**前置依赖**: 无

**完成标准**:
- [ ] 确认后端已完整支持自定义 tenant_id（✅ 已完成）
- [ ] 插件端在所有 API 调用中传递用户配置的 tenant_id
- [ ] 后端用 `longray` tenant_id 能查询到插件上传的记忆
- [ ] 后端用 `default` tenant_id 查询不到插件上传的记忆

**验证方式**:
```bash
# 1. 插件上传记忆（使用 tenant_id="longray"）
# 2. 后端查询验证
curl "http://localhost:17999/api/v1/memories/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "tenant_id": "longray"}'
# 期望: 能找到记忆

curl "http://localhost:17999/api/v1/memories/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "tenant_id": "default"}'
# 期望: 找不到记忆
```

**状态**: ⏳ 进行中（需插件端配合）

---

### BL-2 [P1] 性能基线建立 #performance

**目标**: 建立当前版本的性能基准数据，为后续优化提供可量化的对比依据，识别性能瓶颈。

**涉及范围**:
- **脚本**: `scripts/benchmark.py`（扩展测试覆盖）
- **文档**: `README.md`（更新性能表格）
- **文档**: `docs/BENCHMARK_RESULTS.md`（新增详细报告）

**前置依赖**: 无

**完成标准**:
- [ ] 运行 benchmark.py 获取当前环境性能数据（至少 5 次迭代）
- [ ] 更新 README.md 性能基线表格（替换旧数据）
- [ ] 识别 3 个关键优化方向（基于数据分析）
- [ ] 记录优化目标和当前基线的对比

**验证方式**:
```bash
# 1. 运行基准测试
uv run python scripts/benchmark.py --iterations 5

# 2. 验证 README.md 已更新
grep -A 10 "性能基线" README.md

# 3. 确认数据合理性（单条上传 < 1000ms，搜索 < 500ms）
```

**状态**: ⏳ 进行中

---

## v2.5.0 - 代码分析增强

> 基于 GitNexus 设计理念，增强 Memory Stack 的代码分析能力。核心定位：**记忆级代码分析**（非仓库级）。

### BL-3 [P1] 修复代码分析增强设计文档 #docs #design

**目标**: 修复 `docs/code-analysis-enhancement.md` 的格式问题，确保文档符合质量门禁标准，为后续开发提供准确参考。

**涉及范围**:
- **文档**: `docs/code-analysis-enhancement.md`（全面审查和修复）

**前置依赖**: 无

**完成标准**:
- [ ] 修复 MD031/MD032 格式问题（代码块前后空行隔离）
- [ ] 修复 mermaid 代码块后悬空 `yaml` 标签问题（第60-68行）
- [ ] 通过 `markdownlint-cli2` 检查无报错
- [ ] 术语统一："插件端"（非"前端"），"代码分析增强"（非"代码智能增强"）

**验证方式**:
```bash
# 1. 运行 Markdown 检查
uvx pre-commit run markdownlint-cli2 --files docs/code-analysis-enhancement.md

# 2. 确认无报错
echo $?
# 期望: 0

# 3. 本地渲染检查
# 使用 VS Code 或浏览器打开文档，确认格式正确
```

**状态**: ✅ 已完成

---

### BL-4 [P1] 代码分析结果持久化（Phase A）#code-analysis

**目标**: 将 `analyze_memory_code()` 的返回结果持久化到记忆的 `metadata.code_analysis` 字段，上传代码记忆时自动触发分析，避免每次查看都重新计算。

**涉及范围**:
- **后端**: `wrapper/src/utils/memory_manager.py`
  - 修改 `create_memory()` 添加自动分析触发逻辑
  - 添加 `_is_code_content()` 方法检测内容类型
  - 确保分析失败降级策略
- **后端**: `wrapper/src/config.py`
  - 修改 `CodeAnalysisConfig.auto_analyze` 默认值为 `True`（或按需启用）
- **后端**: `wrapper/src/main.py`
  - 确保 `POST /api/v1/memories` 支持 `auto_analyze_code` 参数

**前置依赖**: 
- BL-3（设计文档确认后实施更准确）
- 确认 `CodeAnalyzer` 功能正常（已验证 ✅）

**完成标准**:
- [ ] 上传代码记忆时，`auto_analyze_code=true` 自动触发分析并持久化
- [ ] 分析结果写入 `metadata.code_analysis` 字段
- [ ] 分析结果包含：language, functions, classes, imports, complexity, analyzed_at, analyzer_version
- [ ] 分析失败**不影响上传**（记录警告，metadata.code_analysis 为 null）
- [ ] 现有 32 个同步测试全部通过
- [ ] Pyright 类型检查 0 errors

**验证方式**:
```bash
# 1. 单元测试
uv run pytest tests/test_code_analysis_persistence.py -v

# 2. 回归测试
uv run pytest tests/test_phase_b_sync.py -v
# 期望: 32/32 passed

# 3. E2E 测试
curl -X POST http://localhost:17999/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "memories": [{"content": "def test(): pass", "type": "general"}],
    "auto_analyze_code": true
  }'

# 4. 验证分析结果已持久化
curl "http://localhost:17999/api/v1/memories/{memory_id}"

# 5. 类型检查
uv run pyright
# 期望: 0 errors
```

**状态**: ✅ 已完成（已实现）

---

### BL-5 [P2] Meilisearch 代码分析字段索引（Phase B）#code-analysis #meilisearch

**目标**: 将代码分析结果同步到 Meilisearch 索引，支持按代码属性（语言、函数名、复杂度）过滤搜索，实现"找出所有 Python 函数中复杂度大于 5 的代码"。

**涉及范围**:
- **后端**: `wrapper/src/utils/meili_client.py`
  - 修改索引设置，添加 code_analysis 相关字段
- **后端**: `wrapper/src/utils/memory_manager.py`
  - 修改 `_build_meili_doc()` 构建 Meilisearch 文档时添加 code_analysis 字段
  - 修改 `create_memory()` / `_update_memory()` 同步时携带代码分析字段
- **Meilisearch**: 索引 schema 更新（需重新初始化或添加新字段）

**前置依赖**: 
- BL-4（需要先有代码分析数据）
- Meilisearch 服务运行中

**完成标准**:
- [ ] Meilisearch 文档包含 code_analysis 字段（language, function_names, class_names, max_complexity）
- [ ] 支持按 `code_language` 过滤: `filter = "code_language = python"`
- [ ] 支持按 `code_functions` 搜索: 搜索函数名
- [ ] 支持按 `code_complexity` 排序: 按复杂度降序
- [ ] 搜索 API 添加 `code_filter` 参数支持
- [ ] 现有搜索功能不受影响（向后兼容）

**验证方式**:
```bash
# 1. Meilisearch 字段验证
curl -X GET "http://localhost:18003/indexes/memories/settings" \
  -H "Authorization: Bearer masterKey"

# 2. 代码搜索测试
curl -X POST http://localhost:17999/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication", "code_filter": {"language": "python"}}'

# 3. 单元测试
uv run pytest tests/test_code_search.py -v
```

**状态**: 📋 待开始

---

### BL-6 [P2] LLM 代码摘要生成（Phase C）#code-analysis #llm

**目标**: 调用外部 LLM API 为代码记忆生成自然语言摘要，使用户能看到"这个模块实现了用户认证功能"的描述，而非仅函数名列表。

**涉及范围**:
- **后端**: `wrapper/src/config.py`
  - 新增 `LLMConfig` 数据类: endpoint, api_key, model_name, max_tokens
  - 环境变量: `WRAPPER_LLM_ENDPOINT`, `WRAPPER_LLM_API_KEY`, `WRAPPER_LLM_MODEL`
- **后端**: `wrapper/src/utils/memory_manager.py`
  - 新增 `async _generate_code_summary()` 方法: 调用 LLM API 生成摘要
  - 修改 `create_memory()`: 上传代码后异步触发摘要生成（不阻塞上传）
  - `metadata.code_summary` 字段: {summary, key_functions, purpose, generated_at, model}
- **后端**: `wrapper/src/main.py`
  - 新增 `POST /api/v1/memories/{memory_id}/enrich/llm`: 手动触发 LLM 摘要生成
  - 新增 `GET /api/v1/memories/{memory_id}/summary`: 获取代码摘要

**前置依赖**: 
- BL-4（需要先有代码分析结果作为 LLM 输入）
- LLM 服务运行中（端口 18001，独立运行）

**完成标准**:
- [ ] `POST /api/v1/memories` 上传代码后异步触发 LLM 摘要（不阻塞上传响应）
- [ ] LLM 调用失败不影响上传（只记录警告）
- [ ] 摘要结果存入 `metadata.code_summary`: {summary, key_functions, purpose, generated_at, model}
- [ ] 支持 Kaggle LLM 或其他 OpenAI 兼容 API（通过配置切换）
- [ ] `POST /api/v1/memories/{memory_id}/enrich/llm` 支持手动触发
- [ ] `GET /api/v1/memories/{memory_id}/summary` 返回摘要

**验证方式**:
```bash
# 1. 手动触发 LLM 摘要
curl -X POST http://localhost:17999/api/v1/memories/{memory_id}/enrich/llm \
  -H "Content-Type: application/json" \
  -d '{"type": "summary"}'

# 2. 获取摘要
curl http://localhost:17999/api/v1/memories/{memory_id}/summary

# 3. 单元测试（mock LLM 调用）
uv run pytest tests/test_llm_summary.py -v
```

**状态**: ✅ 已完成

---

## 暂缓任务

### BL-7 [P3] 跨文件关系解析（Phase D）#code-analysis #future

**目标**: 解析代码记忆间的 import/call 关系，存入 SurrealDB `relation` 表，支持"谁调用了 validate_user"或"这个函数被哪些文件依赖"的查询。

**涉及范围**:
- **后端**: `wrapper/src/utils/code_analyzer.py`
  - 新增 `resolve_imports()` 方法: 解析 import 语句，提取目标文件路径
  - 新增 `resolve_calls()` 方法: 解析函数调用，提取被调用函数名
- **后端**: `wrapper/src/utils/memory_manager.py`
  - 新增 `async _resolve_code_relations()` 方法: 在已有记忆中搜索匹配的 import/call 目标
  - 延迟解析: 上传时标记 import，定时或手动触发批量关联
  - 关系类型新增: IMPORTS, CALLS, EXTENDS, IMPLEMENTS
  - 关系带 confidence (0.0-1.0) 和 reason 字段
- **后端**: `wrapper/src/main.py`
  - 新增 `POST /api/v1/memories/{memory_id}/resolve-relations`: 手动触发关系解析
  - 新增 `GET /api/v1/memories/{memory_id}/knowledge-graph`: 获取代码知识图谱

**前置依赖**: 
- BL-4（需要先有代码分析结果）
- BL-5（需要 Meilisearch 索引来搜索目标文件）

**完成标准**:
- [ ] 上传代码后自动提取 import 语句和函数调用
- [ ] `POST /api/v1/memories/{memory_id}/resolve-relations` 在已有记忆中搜索匹配目标，建立关系
- [ ] 关系带置信度: 同文件 1.0, import 解析 0.85, 模糊匹配 0.5
- [ ] `GET /api/v1/memories/{memory_id}/knowledge-graph` 返回 N 跳关系图
- [ ] 延迟解析策略: 上传时标记，批量关联时建立关系

**验证方式**:
```bash
# 1. 上传两个有依赖关系的文件，然后解析关系
curl -X POST http://localhost:17999/api/v1/memories/{memory_id}/resolve-relations

# 2. 查看知识图谱
curl http://localhost:17999/api/v1/memories/{memory_id}/knowledge-graph?depth=2

# 3. 单元测试
uv run pytest tests/test_code_relations.py -v
```

**状态**: ⚪ 暂缓（可行性待评估，记忆级输入的上传顺序不确定）

---

### BL-8 [P3] 插件端代码分析工具（Phase E）#code-analysis #plugin

**目标**: 在 OpenCode 插件中注册代码分析相关工具，使 AI Agent 能够通过插件调用后端的代码分析 API。

**涉及范围**:
- **插件端**（TypeScript）:
  - 新增 `memory_code_analyze` 工具: 调用 `POST /api/v1/memories/{id}/analyze/code`
  - 新增 `memory_code_search` 工具: 调用 `POST /api/v1/memories/search` + `code_filter`
  - 新增 `memory_code_enrich` 工具: 调用 `POST /api/v1/memories/{id}/enrich/llm`
  - 新增 `memory_code_summary` 工具: 调用 `GET /api/v1/memories/{id}/summary`
  - 新增 `memory_code_graph` 工具: 调用 `GET /api/v1/memories/{id}/knowledge-graph`
  - 新增 `memory_code_impact` 工具: 调用 `POST /api/v1/memories/{id}/impact`（远期）
  - 新增 `memory_code_processes` 工具: 调用 `GET /api/v1/memories/{id}/processes`（远期）
- **配置**: `memory-config.json`
  - 新增 `code_analysis` 配置节: auto_analyze, auto_enrich_llm

**前置依赖**: 
- BL-4 ~ BL-6（后端 API 就绪后才能注册工具）

**完成标准**:
- [ ] 7 个新工具注册成功，LLM 可通过工具名调用
- [ ] 每个工具有完整的 `description` 和 `parameters` JSON Schema
- [ ] 工具注册代码基于 `@opencode-ai/plugin` v1.3.3 的 `Hooks.tool` 类型
- [ ] `memory-config.json` 包含 `code_analysis` 配置

**验证方式**:
```bash
# 1. 在 OpenCode 中调用工具
memory_code_analyze(memory_id="xxx")

# 2. 验证 code_filter 过滤生效
memory_code_search(query="auth", code_filter={"language": "python"})

# 3. 检查配置格式
cat memory-config.json | jq '.code_analysis'
```

**状态**: ⏳ 待后端 API 就绪

---

## 执行路线图

```
本周（立即执行）
├── BL-2 [P1] 性能基线建立 ─────────────────► 运行脚本 + 更新文档 ────► 1-2小时
│   └── 命令: uv run python scripts/benchmark.py --iterations 5
│
└── BL-3 [P1] 修复设计文档 ─────────────────► Markdown检查 + 修复 ────► 2小时
    └── 命令: uvx pre-commit run markdownlint-cli2 --files docs/code-analysis-enhancement.md

下周（核心开发）
└── BL-4 [P1] 代码分析持久化 ───────────────► 自动触发 + 降级策略 ────► 1-2天
    ├── 修改 create_memory() 添加自动分析
    ├── 添加 _is_code_content() 检测
    └── 确保分析失败不影响上传

下下周（并行开发）
├── BL-5 [P2] Meilisearch字段索引 ──────────► 索引配置 + 搜索过滤 ────► 1-2天
│   ├── 修改 Meilisearch 索引设置
│   ├── 修改 _build_meili_doc()
│   └── 搜索 API 添加 code_filter 参数
│
└── BL-6 [P2] LLM代码摘要 ─────────────────► LLM集成 + 异步触发 ────► 1-2天
    ├── 添加 LLMConfig
    ├── 实现 _generate_code_summary()
    └── 异步调用 LLM 服务

转移/暂缓
├── BL-1 [P2] ─────────────────────────────► 移至插件端 backlog
├── BL-8 [P3] ─────────────────────────────► 移至插件端 backlog
└── BL-7 [P3] ─────────────────────────────► 保持暂缓
```

---

## Backlog 规范

**格式**: `- [ ] BL-{N} [{Priority}] 描述 #标签`

**优先级**: P0 = 紧急, P1 = 重要, P2 = 普通, P3 = 低优先级

**状态**: ⏳ 进行中, 📋 待开始, ⚪ 暂缓

**5 要素**:
1. **目标**: 解决什么问题，达成什么效果
2. **涉及范围**: 修改哪些文件/模块
3. **前置依赖**: 依赖哪些任务/条件
4. **完成标准**: 具体的验收 checklist
5. **验证方式**: 如何测试/验证完成

---

**已完成任务**: 见 [backlog_archive.md](backlog_archive.md)

**历史归档**: v2.4.0 之前的已完成任务已归档至 CHANGELOG.md

---

*最后更新: 2026-03-30*
