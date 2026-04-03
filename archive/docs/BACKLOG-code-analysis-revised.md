# Backlog - 代码分析集成（复核后重新规划）

> **版本**: v2.0  
> **日期**: 2026-04-03  
> **状态**: Phase 1 已完成 5/6，Phase 2 待开始  
> **复核依据**: 实际代码检查 + 测试覆盖分析

---

## 执行摘要

经过详细代码复核，**Phase 1 实际已完成 5/6 项任务**，仅剩余 `max_complexity` 支持未实现。Upsert 逻辑已提前实现，无需在 Phase 2 重复。

### 完成状态总览

| Phase | 任务数 | 已完成 | 进行中 | 未开始 |
|-------|--------|--------|--------|--------|
| Phase 1 | 6 | 5 | 1 | 0 |
| Phase 2 | 3 | 0 | 0 | 3 |
| 技术债务 | 2 | 0 | 0 | 2 |

### 关键发现

- ✅ **BL-CA-08 (Upsert)** 已提前在 Phase 1 实现（crud.py 第 198-219 行）
- ⚠️ **BL-CA-05 (max_complexity)** 是唯一阻塞 Phase 1 完成的代码任务（仅缺 1 行）
- 📋 **测试覆盖不足** - 缺少集成测试验证端到端流程

---

## Phase 1 复核详情

### 已完成 ✅

#### BL-CA-01: 升级 CodeAnalysisResult dataclass

| 字段 | 状态 | 代码位置 |
|------|------|----------|
| `analyzer` | ✅ | code_analyzer.py:45 |
| `interfaces` | ✅ | code_analyzer.py:46 |
| `errors` | ✅ | code_analyzer.py:47 |
| `warnings` | ✅ | code_analyzer.py:48 |
| `to_metadata_dict()` 输出 | ✅ | code_analyzer.py:50-68 |

**验证**: `tests/test_code_analysis.py` 第 55-78 行测试通过

---

#### BL-CA-02: 扩展 Meilisearch 索引设置

| 字段 | 类型 | 代码位置 |
|------|------|----------|
| `code_language` | filterable | meili_client.py:65 |
| `code_complexity` | filterable | meili_client.py:66 |
| `code_function_count` | filterable + sortable | meili_client.py:67,71 |
| `code_class_count` | filterable | meili_client.py:68 |
| `code_analyzer` | filterable | meili_client.py:69 |
| `code_symbols` | searchable | meili_client.py:52 |

**验证**: 索引配置已生效，可通过 Meilisearch API 查询设置

---

#### BL-CA-03: 新增 build_code_symbols 工具函数

| 功能 | 状态 | 代码位置 |
|------|------|----------|
| 函数实现 | ✅ | code_analyzer.py:464-486 |
| 兼容旧格式 exports | ✅ | code_analyzer.py:478-484 |
| 单元测试 | ✅ | test_code_analysis.py:103-149 |

**测试覆盖**: 7 个测试用例（空输入、函数、类、混合符号、新旧格式兼容）

---

#### BL-CA-04: 上传时提取代码字段到 Meilisearch

| 字段 | 提取来源 | 代码位置 |
|------|----------|----------|
| `code_language` | code_analysis.language | meili_sync.py:42 |
| `code_complexity` | complexity.cyclomatic_complexity | meili_sync.py:44 |
| `code_function_count` | complexity.function_count | meili_sync.py:45 |
| `code_class_count` | complexity.class_count | meili_sync.py:46 |
| `code_analyzer` | code_analysis.analyzer | meili_sync.py:47 |
| `code_symbols` | metadata.code_symbols | meili_sync.py:49-50 |

**验证**: meili_sync.py 第 38-50 行已实现，在 upload_memories 流程中自动调用

---

#### BL-CA-08: 代码文件 Upsert 逻辑（已提前实现）

| 功能 | 实现 | 代码位置 |
|------|------|----------|
| 唯一键检测 | ✅ | crud.py:203-209 |
| 更新现有记录 | ✅ | crud.py:214-219 |
| 创建新记录 | ✅ | crud.py:282 及后续 |

**Upsert 逻辑**:
```python
# 按 type='code' + project_id + metadata->file_path + tenant_id 查询
SELECT id, metadata FROM memory 
WHERE type = 'code' 
  AND project_id = $project_id 
  AND metadata->file_path = $file_path 
  AND tenant_id = $tenant_id
```

**注意**: 此任务原计划在 Phase 2，但实际已在 Phase 1 完成

---

### 未完成 ❌

#### BL-CA-05: code_filter 添加 max_complexity 支持

| 字段 | 状态 | 说明 |
|------|------|------|
| `language` | ✅ 已实现 | search.py:21-22 |
| `min_complexity` | ✅ 已实现 | search.py:23-24 |
| `max_complexity` | ❌ **缺失** | 需要添加 |

**当前代码** (search.py:19-26):
```python
if request.code_filter:
    filter_parts = []
    if "language" in request.code_filter:
        filter_parts.append(f'code_language = "{request.code_filter["language"]}"')
    if "min_complexity" in request.code_filter:
        filter_parts.append(f"code_complexity >= {request.code_filter['min_complexity']}")
    if filter_parts:
        filter_expr = " AND ".join(filter_parts)
```

**需要添加**:
```python
if "max_complexity" in request.code_filter:
    filter_parts.append(f"code_complexity <= {request.code_filter['max_complexity']}")
```

**影响**: API 契约文档中定义的 `max_complexity` 参数无法使用

---

#### BL-CA-06: 修复 v1.2 设计文档 4 个小问题

| 问题 | 位置 | 修复内容 |
|------|------|----------|
| 内存描述错误 | v1.2:304 | "memory < 100MB" → "system available memory" |
| 缺少 tenant_id | v1.2:4.3 | batch upload 示例补充 `tenant_id` |
| 响应字段名 | v1.2:9.2 | `hits` → `results` (与后端一致) |
| max_complexity | main.py:555 | 文档标注缺失，代码已实现 |

**状态**: 文档债务，不影响功能，建议 Phase 1 收尾时修复

---

## 重新规划的 Backlog

### Phase 1 收尾（立即执行）

#### BL-CA-05-REV: 添加 max_complexity 支持

| 字段 | 内容 |
|------|------|
| **目标** | 搜索 API 的 code_filter 参数新增 max_complexity 过滤，转换为 `code_complexity <= N` Meilisearch 过滤条件 |
| **涉及范围** | `wrapper/src/routers/search.py` 第 19-26 行，在现有 filter_parts 逻辑中添加 `max_complexity` 分支 |
| **前置依赖** | 无（独立任务） |
| **完成标准** | ① code_filter 含 max_complexity 时生成 `code_complexity <= N` 过滤条件 ② 与 language / min_complexity 组合使用时用 AND 连接 ③ 向后兼容（不含 max_complexity 时行为不变） |
| **验证方式** | 编写 pytest 用例：验证 `{"language": "python", "min_complexity": 5, "max_complexity": 30}` 生成正确的 filter 字符串；手动 curl 测试端到端搜索 |
| **预估工作量** | 15 分钟（1 行代码 + 测试） |

---

#### BL-CA-06-REV: 修复 v1.2 设计文档

| 字段 | 内容 |
|------|------|
| **目标** | 修正 CODE-ANALYSIS-DESIGN-v1.2.md 中发现的 4 个文档错误 |
| **涉及范围** | `archive/docs/CODE-ANALYSIS-DESIGN-v1.2.md`（4 处修改） |
| **前置依赖** | 无 |
| **完成标准** | ① 第 304 行 "memory < 100MB" → "available system memory < 100MB" ② Section 4.3 batch upload 示例补充 `tenant_id: "default"` ③ Section 9.2 search 响应 `hits` → `results` ④ 通过 markdownlint 检查 |
| **验证方式** | `uv run task lint-md` 通过；人工 review 4 处修改 |
| **预估工作量** | 10 分钟 |

---

### Phase 2 任务（下一阶段）

#### BL-CA-07: 实现 POST /api/v1/sync/code-fingerprints API

| 字段 | 内容 |
|------|------|
| **目标** | 实现代码文件专用的增量同步 API，支持基于 path + hash + symbols_hash 的变更检测，返回 changed/unchanged/missing/conflicts 四类结果 |
| **涉及范围** | `wrapper/src/routers/sync.py`（新增端点）、`wrapper/src/models.py`（新增 Pydantic 模型）、`wrapper/src/utils/memory_manager/sync.py`（新增 sync_code_fingerprints 方法） |
| **前置依赖** | Phase 1 全部完成 |
| **完成标准** | ① 新增 `CodeFingerprint`, `CodeSyncRequest`, `CodeSyncResponse` Pydantic 模型 ② 实现 `POST /api/v1/sync/code-fingerprints` 端点 ③ MemoryManager 实现 `sync_code_fingerprints()` 方法 ④ 能按 `type="code"` + `project_id` + `metadata->file_path` 查询 ⑤ 正确比较 hash 和 symbols_hash，分类返回 changed/unchanged/missing ⑥ 检测到 mtime 冲突时归入 conflicts |
| **验证方式** | 编写 pytest 用例：mock 不同场景（全新文件、内容变更、符号变更、未变更、mtime 冲突），验证返回结果分类正确；手动 curl 测试端到端流程 |
| **预估工作量** | 2-3 天 |

**API 契约**:
```yaml
POST /api/v1/sync/code-fingerprints

Request:
  project_id: "github.com/user/repo"
  fingerprints:
    - path: "src/index.js"
      hash: "abc123..."
      symbols_hash: "def456..."
      mtime: 1712345678
      size: 1024

Response:
  changed:     # 需要更新
    - path: "src/index.js"
      reason: "content_modified" | "symbols_modified"
      server_mtime: 1712340000
  unchanged:   # 完全一致
    - path: "src/utils.js"
  missing:     # 服务端没有
    - path: "src/new-file.ts"
  conflicts:   # mtime 冲突
    - path: "src/auth.js"
      local_mtime: 1712345678
      server_mtime: 1712345600
```

---

#### BL-CA-09: 端到端联调测试

| 字段 | 内容 |
|------|------|
| **目标** | 与插件端完成端到端联调，验证代码分析全流程 |
| **涉及范围** | 双方测试环境、API 对接、文档同步 |
| **前置依赖** | BL-CA-05、BL-CA-07 完成，插件端实现完成 |
| **完成标准** | ① 插件端能成功上传代码分析结果 ② code_filter 搜索过滤准确 ③ 增量同步 API 返回正确 ④ 多设备同步无冲突 ⑤ 性能达标（5000行文件 < 200ms） |
| **验证方式** | 双方共同执行测试用例，记录联调日志，确认无阻塞问题 |
| **预估工作量** | 2-3 天（依赖插件端进度） |

---

### 技术债务（建议同步处理）

#### DEBT-01: 补充代码分析集成测试

| 字段 | 内容 |
|------|------|
| **目标** | 补充 Phase 1 功能的集成测试，覆盖端到端流程 |
| **涉及范围** | `tests/test_code_analysis_integration.py`（新建） |
| **前置依赖** | BL-CA-05 完成 |
| **完成标准** | ① 测试上传代码记忆时 Meilisearch 字段正确提取 ② 测试 code_filter 所有参数（language, min_complexity, max_complexity） ③ 测试 Upsert 逻辑（同一 file_path 更新而非新建） ④ 测试搜索返回结果包含 code_analysis 元数据 |
| **验证方式** | `uv run pytest tests/test_code_analysis_integration.py -v` 全部通过 |
| **预估工作量** | 1 天 |

**建议测试用例**:
```python
def test_upload_code_memory_extracts_fields_to_meili()
def test_search_with_code_filter_language()
def test_search_with_code_filter_min_max_complexity()
def test_upsert_code_memory_by_file_path()
def test_search_returns_code_analysis_metadata()
```

---

#### DEBT-02: 更新 API 文档

| 字段 | 内容 |
|------|------|
| **目标** | 更新 API_SPECIFICATION.md，添加代码分析相关端点 |
| **涉及范围** | `docs/API_SPECIFICATION.md` |
| **前置依赖** | BL-CA-05、BL-CA-07 完成 |
| **完成标准** | ① 添加 `POST /api/v1/memories` code 类型示例 ② 添加 `code_filter` 参数说明 ③ 添加 `POST /api/v1/sync/code-fingerprints` 端点文档 |
| **验证方式** | 人工 review，确保与实现一致 |
| **预估工作量** | 半天 |

---

## 执行顺序建议

### 立即执行（本周）

1. **BL-CA-05-REV** - 添加 max_complexity 支持（15分钟）
2. **BL-CA-06-REV** - 修复 v1.2 文档（10分钟）
3. **运行测试** - 验证 Phase 1 全部通过

### 下一阶段（下周开始）

4. **DEBT-01** - 补充集成测试（1天）
5. **BL-CA-07** - 增量同步 API（2-3天）
6. **DEBT-02** - 更新 API 文档（半天）

### 等待插件端

7. **BL-CA-09** - 端到端联调（2-3天，需协调）

---

## 附录

### 代码分析功能文件索引

| 文件 | 行数 | 说明 |
|------|------|------|
| `wrapper/src/utils/code_analyzer.py` | 491 | AST 解析、CodeAnalysisResult、build_code_symbols |
| `wrapper/src/utils/memory_manager/code_analysis.py` | 197 | analyze_memory_code、_generate_code_summary |
| `wrapper/src/utils/memory_manager/meili_sync.py` | 72 | Meilisearch 字段提取 |
| `wrapper/src/utils/memory_manager/crud.py` | 415 | upload_memories（含 Upsert 逻辑） |
| `wrapper/src/routers/search.py` | 41 | 搜索端点（待添加 max_complexity） |
| `wrapper/src/utils/meili_client.py` | 484 | Meilisearch 索引配置 |
| `tests/test_code_analysis.py` | 149 | 单元测试（dataclass、build_code_symbols） |

### 依赖关系图

```
Phase 1 收尾
├── BL-CA-05-REV (max_complexity) ──┐
└── BL-CA-06-REV (文档修复) ────────┼──→ Phase 1 完成
                                    │
技术债务                           │
├── DEBT-01 (集成测试) ─────────────┘
└── DEBT-02 (API 文档)

Phase 2
└── BL-CA-07 (增量同步 API) ──→ BL-CA-09 (联调)
```

---

*最后更新: 2026-04-03*  
*复核人: Sisyphus*  
*状态: 等待用户确认执行 Phase 1 收尾*
