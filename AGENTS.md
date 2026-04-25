# Embedding Service - Agent 指南

## 项目注意事项

### Python 环境管理

**⚠️ 重要：不要删除 Python 虚拟环境**

- PyTorch 体积很大，重新下载浪费流量
- 如果包有问题，使用 `uv` 管理包修复

### 包管理

**使用 uv 管理依赖**：

```bash
# 安装包
uv pip install package_name

# 运行 Python 脚本
uv run python script.py

# 运行测试
uv run pytest tests/
```text
embedding_service/
├── src/                        # Embedding 和 LLM 服务
│   └── qwen3_embedding_service/
│       ├── embedding_service.py # Embedding API (端口 18000)
│       └── llm_service.py       # LLM API (端口 18001)
├── wrapper/                     # 包装层服务 (端口 17999)
│   └── src/
│       ├── main.py             # FastAPI app + lifespan (v2.6.0, ~300行)
│       ├── models.py           # 17 个 Pydantic 模型
│       ├── state.py            # 共享单例（避免循环导入）
│       ├── config.py           # 配置管理 (含 MeilisearchConfig)
│       ├── routers/            # API 路由模块
│       │   ├── health.py       # /health
│       │   ├── embeddings.py   # /v1/embeddings
│       │   ├── memories.py     # /memories CRUD + clear + summary + enrich
│       │   ├── search.py       # /memories/search
│       │   ├── relations.py    # 图关系 CRUD + 遍历
│       │   ├── sync.py         # 同步预览/全量/指纹/冲突
│       │   ├── websocket.py    # WebSocket 实时推送
│       │   └── stubs.py        # 11 个 stub 端点
│       └── utils/
│           ├── memory_manager/ # 记忆管理 Mixin 模式 (10 子模块)
│           ├── meili_client.py # Meilisearch 异步客户端
│           ├── surrealdb_client.py # SurrealDB 客户端
│           ├── code_analyzer.py # 代码分析器
│           ├── cache.py        # LRU 缓存
│           ├── auth.py         # WebSocket 认证
│           ├── exceptions.py   # 统一异常层级
│           └── http_pool.py    # HTTP 连接池
├── scripts/                     # 运维脚本
│   ├── migrate_to_meilisearch.py # SurrealDB → Meilisearch 迁移
│   ├── init_surrealdb.surql     # SurrealDB Schema 初始化
│   └── init_surrealdb_fixed.surql
├── tests/                       # 测试套件
│   ├── test_wrapper_api.py      # 核心 API 测试 (56 个)
│   ├── test_meili_integration.py # Meilisearch 集成测试 (23 个)
│   └── ...                      # 其他测试文件
├── docker-compose.yml           # Docker 一键部署
├── .env.example                 # 环境变量模板
└── pyproject.toml               # 项目配置
```text

## 开发命令

### 本地开发
```bash
# 启动服务
uv run python start_services.py --with-llm

# 运行测试
uv run pytest tests/ -v

# 代码检查
uv run ruff check .
uv run pyright
```

### Docker 开发环境（推荐）

**⚠️ 重要：开发环境支持热重载，无需重建容器**

```bash
# 1. 启动开发环境（首次或配置变更后）
docker-compose -f docker-compose.dev.yml up -d

# 2. 修改 wrapper/src/ 代码后，只需重启容器即可生效（无需重建）
docker-compose -f docker-compose.dev.yml restart wrapper

# 3. 查看日志
docker logs wrapper-service-dev -f

# 4. 停止环境
docker-compose -f docker-compose.dev.yml down
```

**热重载原理**:

- 开发环境使用 `target: development` 构建
- 挂载 `./wrapper/src:/app/wrapper/src` 卷
- 修改本地代码后，容器内立即生效
- **只需重启容器，无需重新构建镜像**
- 开发环境使用 `target: development` 构建
- 挂载 `./wrapper/src:/app/wrapper/src` 卷
- 修改本地代码后，容器内立即生效
- **只需重启容器，无需重新构建镜像**

**生产环境部署**:

```bash
# 生产环境使用 production 目标，需要重新构建
docker-compose build --no-cache
docker-compose up -d
```

## Markdown 编写规范

当 AI Agent 或开发者编写/更新 Markdown 文档时，**必须**遵循以下规范：

1. **MD031 (blanks-around-fences)**：代码块（fenced code block）的**前后必须各有一个空行**。
   - ❌ 错误：`**验证方式**` 紧跟 ` ```bash`
   - ✅ 正确：`**验证方式**` 后空一行，再写 ` ```bash`
   - 关闭 ` ``` ` 后也必须空一行再写其他内容
2. **MD032 (blanks-around-lists)**: 列表（`-` 或 `*` 开头）的**前后必须各有一个空行**。
   - ❌ 错误：文本后直接跟列表
   - ✅ 正确：文本后空一行，再开始列表
3. **MD037 (no-space-in-emphasis)**：强调标记（`**` 或 `_`）内**不能有空格**。
   - ❌ 错误：`_handle_messages` 中的 `_` 被识别为强调标记
   - ✅ 正确：使用反引号包裹代码 `_handle_messages`
4. **MD058 (blanks-around-tables)**：表格的**前后必须各有一个空行**。
   - ❌ 错误：文本后直接跟表格
   - ✅ 正确：文本后空一行，再开始表格
5. **MD040 (code-block-style)**：所有 fenced code block 必须显式声明语言（如 ` ```python`、` ```bash`）。
6. **MD001 (heading-increment)**：标题级别必须逐级递增（`h1` → `h2` → `h3`，不可跳级）。
7. **MD033**：允许使用 `<br>` 等内联 HTML。
8. **验证**：提交前运行 `uvx pre-commit run markdownlint-cli2 --all-files` 确认无新增错误。

**忽略规则**：MD013(行长度) | MD024(重复标题) | MD056/060(表格)

## 工具选择规则

**搜索与重构任务的工具优先级**：

1. **ripgrep (rg)**：文本搜索首选（自动跳过 node_modules，并行搜索）
2. **ast-grep (sg)**：结构重构首选（AST 感知，精准替换）
3. **grep**：仅用于管道过滤或极简环境（无 rg/sg 时）

**禁止**：使用 `grep -r` 进行递归目录搜索。

**复杂场景**：调用 `skill("code-search")` 获取详细策略。

## 技术问题参考

### SurrealDB 相关问题

- **[SurrealDB object FLEXIBLE 字段问题](docs/dev/SURREALDB_OBJECT_FLEXIBLE_ISSUE.md)** - 记录 SurrealDB 3.0 中 `TYPE object FLEXIBLE` 与 Python SDK 的兼容性问题及解决方案

## 最近变更

- **v3.2 架构升级**（当前）：
  - 统一架构v3.2实施完成，服务端口迁移：17999 → 18008
  - SurrealDB 3.0+ 语法升级（`COMPUTED`, `FULLTEXT`, `type::record()`）
  - tree-sitter Query性能优化（3.32x提升），纳入代码分析
  - 预计算服务（PrecomputeService）- AST解析、指纹、符号提取
  - WebSocket实时同步（LIVE SELECT）- 记忆变更实时推送
  - 单租户 + `tenant_id`预留字段（多租户物理隔离暂缓至SDK 2.0 stable）
  - 四层架构（Atom/Entity/Relation/Backlog）保留v2.0设计
  - 代码分析功能：CallSymbol提取、引用查询、代码地图、代码统计
  - 文档整理：归档inbox/（49个临时通信）、v2.7同步文档
- **v2.8.0**（已合入v3.2）：
  - PrecomputeService完善
  - Stub端点实现（11个stub端点）
- **v2.7.1**（已合入v3.2）：
  - SQL查询优化（RecordID统一、分批处理、embedding字段优化）
  - 安全性修复
- **v2.7.0**（已合入v3.2）：
  - 多设备同步（指纹查询/同步预览/全量同步/冲突解决）
  - 测试架构优化
- **v2.6.0 质量治理**：
  - BL-35: `memory_manager.py` 1715行 → Mixin 模式 10 子模块
  - BL-28: 实现 `analyze_memory_code`（CodeAnalyzer 集成）
  - BL-33/34: 修复 pyproject.toml 过时配置 + meilisearch_code/ 9 个类型错误
  - BL-38/39: 移除硬编码 API Key + 清理裸 except
  - BL-D1: 归档 29 个过时文档 + 23 个 JSON 报告
  - CHANGELOG 补充 v2.5.0/v2.6.0 条目
- **v2.4.1 sync_preview conflict 检测修复 + 代码质量修复**：
  - 修复 `get_fingerprints` 返回空导致无法检测冲突
  - B-005-B: SurrealDB 3.0 SDK 结果解析逻辑错误（改用 `_extract_records()`）
  - B-005-C: `get_conflict_detail` 参数化表名语法错误（改用 `type::string(id)`）
  - 修复 `SCHEMA_TARGET_VERSION`: `2.3.0` → `2.4.1`
  - 修复 `app = FastAPI()` 缩进错误（从 lifespan 内移到模块级别）
  - 删除重复 API 定义（`analyze_memory_code`, `cluster_memories_leiden`）
  - 添加 `tree_sitter` 导入类型忽略标记
  - Pyright: 34 errors → 0 errors, 测试 32/32 passed
  - E2E 测试通过：上传 → 查指纹 → 检测冲突 → 解决冲突
- **v2.4.0 API 行为优化**：
  - `/api/v1/sync/incremental` → `/api/v1/sync/preview`（旧路由保留为别名）
  - `SyncFullResponse` 新增 `skipped` 列表（含 `local_id`、`existing_id`、`reason`、`similarity`）
  - conflict resolution 大小写兼容（`USE_LOCAL` / `use_local` 均可）
  - 修复 `test_sync_preview_conflicts` mock 缺少 `create` 的已有 bug
- **v2.3.1 调试清空 API**：新增清空记忆数据的安全机制
  - 端点：`DELETE /api/v1/memories/clear`
  - 认证：`WRAPPER_MEILI_API_KEY` header
  - 安全机制：先清空 Meilisearch（验证 API key），再清空 SurrealDB
  - 使用方法：`curl -X DELETE http://localhost:17999/api/v1/memories/clear -H "WRAPPER_MEILI_API_KEY: your_api_key"`
  - 错误响应：401（缺少 key）、403（key 错误）、500（清空失败）
- **v2.3.0 Polyglot 搜索架构**：Meilisearch 全文搜索 + SurrealDB 向量/图，RRF 混合搜索
- 包装层目录从 `wrapper-service/` 迁移到 `wrapper/`
- 新增 `meili_client.py` 异步 Meilisearch 客户端
- 新增 `migrate_to_meilisearch.py` 数据迁移脚本
- 新增 23 个 Meilisearch 集成单元测试
- 已移除 prometheus_client 依赖及相关监控代码
- 使用 structlog 进行日志记录
- API 认证通过环境变量 `WRAPPER_AUTH_ENABLED` 控制

## Meilisearch 使用指南

### 代码搜索索引配置

**位置**: `meilisearch_code/`
**端口**: 18003
**索引**: code_search_index

### 使用方式

**1. 全文搜索（中文、代码）**

```python
from config import MeiliConfig
index = MeiliConfig.get_index()

# 中文搜索
index.search("用户服务")

# 代码搜索
index.search("UserService")

# 组合搜索
index.search("python fastapi")
```text

**2. 精确匹配（IP、邮箱、版本）**

```python
# IP 地址
index.search("", {"filter": 'ip_address = "192.168.1.100"'})

# 邮箱
index.search("", {"filter": 'email = "developer@example.com"'})

# 版本号
index.search("", {"filter": 'version = "v2.1.0"'})
```

**3. 混合搜索**

```python
# 全文搜索 + 过滤器
index.search("用户", {"filter": 'language = "java" AND status = "active"'})
```text

### 双字段策略

- **精确字段**（file_path, version, email, ip_address）→ 使用 filter
- **搜索字段**（*_zh,*_search）→ 使用全文搜索

### 代码搜索优化

**104词代码术语字典**：

- FastAPI, Python, Meilisearch, SurrealDB, Docker, Kubernetes 等常用代码术语
- 版本号：v2.3.0, v3.0.1
- IP地址：192.168.1.1, 127.0.0.1
- 日期：2026-03-12

**nonSeparatorTokens 配置**：

- `-`, `.`, `/`, `:`, `@`, `_` 不作为分隔符
- 支持代码标识符搜索：`meili_client.py`, `config.surrealdb.url`

**使用示例**：

```python
# 搜索文件名
index.search("meili_client.py")

# 搜索配置项
index.search("config.surrealdb.url")

# 搜索版本号
index.search("v2.3.0")
```

### 管理命令

```bash
cd meilisearch_code

# 初始化索引
uv run python init_index.py

# 运行测试
uv run python test_search.py

# 监控索引
uv run python monitor_index.py
```text

## 清空记忆数据（调试专用）

### API 端点

| 端点 | 方法 | 功能 | 认证 |
|--------|------|------|------|
| `DELETE /api/v1/memories/clear` | 清空所有记忆 | WRAPPER_MEILI_API_KEY |

### 认证说明

**必需 Header**：

```bash
WRAPPER_MEILI_API_KEY: <your_api_key>
```

**API Key 获取方式**：

```bash
# 从环境变量获取
export WRAPPER_MEILI_API_KEY=<your_api_key>

# 或者从 README.md 获取
# WRAPPER_MEILI_API_KEY=${MEILI_MASTER_KEY:-masterKey_change_in_production}
```text

### 安全机制

清空 API 采用**两步验证机制**保护记忆数据：

1. **先清空 Meilisearch**（验证 WRAPPER_MEILI_API_KEY）
   - 如果 API key 正确 → Meilisearch 清空成功
   - 如果 API key 错误 → 返回 403，停止操作

2. **再清空 SurrealDB**
   - 只有 Meilisearch 清空成功后才执行
   - 如果 Meilisearch 清空失败，SurrealDB 不会被清空

### 使用方法

```bash
# 方法 1：正确 key（会清空所有数据）
curl -X DELETE http://localhost:17999/api/v1/memories/clear \
  -H "WRAPPER_MEILI_API_KEY: <your_api_key>"

# 方法 2：错误 key（只返回 403，保护数据）
curl -X DELETE http://localhost:17999/api/v1/memories/clear \
  -H "WRAPPER_MEILI_API_KEY: wrong_key"
```

### 响应示例

**成功** (200)：

```json
{
  "success": true,
  "message": "所有记忆数据已清空"
}
```text

**失败 - 缺少 Key** (401)：

```json
{
  "detail": "Missing WRAPPER_MEILI_API_KEY header"
}
```

**失败 - Key 错误** (403)：

```json
{
  "detail": "Invalid WRAPPER_MEILI_API_KEY"
}
```text

**失败 - 清空失败** (500)：

```json
{
  "detail": "清空失败: ..."
}
```

### 清空脚本

**位置**：`scripts/clear_all_data.py`

**用途**：清空后端所有数据（SurrealDB + Meilisearch）

**使用方法**：

```bash
cd D:/embedding_service
export WRAPPER_MEILI_API_KEY=<your_api_key>
uv run python scripts/clear_all_data.py
```text

**清空流程**：

1. 先清空 Meilisearch（验证 API key）
2. 如果 Meilisearch 清空成功，再清空 SurrealDB
3. 如果 API key 错误，Meilisearch 清空失败，SurrealDB 不被清空

### 注意事项

⚠️ **调试专用**：此接口仅用于调试和测试，生产环境应谨慎使用
⚠️ **数据保护**：API key 验证失败时，数据不会被清空
⚠️ **不可逆操作**：清空后所有记忆数据将被永久删除

## 文档规范与质量门禁

本项目在 `.pre-commit-config.yaml` 中配置了强制性的 Markdown 质量门禁。

### 工具链决策

- **Marksman**：仅作为 LSP 服务器，提供编辑器内的链接跳转、引用查找和自动补全功能（不支持命令行扫描）。
- **Markdownlint-cli2**：作为 CLI 检查工具，通过 `pre-commit` 在代码提交时拦截格式不规范的 Markdown 文档。

### 核心忽略规则说明

我们在 `.markdownlint-cli2.jsonc` 中有意禁用了部分过于严格的规则，以平衡排版自由度和历史遗留问题：

- **MD013 (Line length)**: 禁用。允许长行（尤其是包含 URL、复杂表格时），避免强制截断破坏阅读连贯性。
- **MD024 (Multiple headings)**: 禁用。允许同一文档中出现重复标题（如 CHANGELOG 中的 "Bug Fixes"、BACKLOG 中的 "涉及范围"）。
- **MD033 (Inline HTML)**: 禁用。允许使用 `<br>` 等 HTML 标签来控制复杂表格内的换行排版。
- **MD056 / MD060 (Table styles)**: 禁用。放宽表格列数不匹配和对齐的限制，防止工具生成的复杂表格导致海量无意义报错。

### 编写规范强制要求

当 AI Agent 或开发者更新文档时，**必须**遵循以下规范：

1. **标题层级 (MD001)**: 标题级别必须逐级递增（例如：不允许在 `h1` 后直接跟 `h3`），保持清晰的文档大纲。
2. **代码块语言 (MD040)**: 所有 Fenced code blocks 必须显式声明语言（如 `python`, `bash`, `json`），以便高亮渲染。
3. **空行隔离 (MD031/MD032)**: 列表 (List) 和代码块 (Fenced code blocks) 的前后**必须**包含一个空行，防止解析器混淆。
4. **相对链接**: 文档间的引用必须使用相对路径（如 `docs/ROADMAP.md`），严禁使用硬编码的本地绝对路径。
5. **验证手段**: 提交前使用 `uvx pre-commit run markdownlint-cli2 --all-files` 验证格式是否达标。
