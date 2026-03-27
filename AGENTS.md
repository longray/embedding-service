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
```

## 项目结构

```
embedding_service/
├── src/                        # Embedding 和 LLM 服务
│   └── qwen3_embedding_service/
│       ├── embedding_service.py # Embedding API (端口 18000)
│       └── llm_service.py       # LLM API (端口 18001)
├── wrapper/                     # 包装层服务 (端口 17999)
│   └── src/
│       ├── main.py             # FastAPI 主程序 (v2.4.1)
│       ├── config.py           # 配置管理 (含 MeilisearchConfig)
│       └── utils/
│           ├── memory_manager.py # 记忆管理（双写 + 搜索路由）
│           ├── meili_client.py   # Meilisearch 异步客户端
│           ├── surrealdb_client.py # SurrealDB 客户端
│           ├── cache.py         # LRU 缓存
│           ├── auth.py          # API 认证
│           └── http_pool.py     # HTTP 连接池
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
```

## 开发命令

```bash
# 启动服务
uv run python start_services.py --with-llm

# 运行测试
uv run pytest tests/ -v

# 代码检查
uv run ruff check .
uv run pyright
```

## 最近变更

- **v2.4.1 sync_preview conflict 检测修复**：
  - 修复 `get_fingerprints` 返回空导致无法检测冲突
  - B-005-B: SurrealDB 3.0 SDK 结果解析逻辑错误（改用 `_extract_records()`）
  - B-005-C: `get_conflict_detail` 参数化表名语法错误（改用 `type::string(id)`）
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
```

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
```

### 双字段策略

- **精确字段**（file_path, version, email, ip_address）→ 使用 filter
- **搜索字段**（*_zh, *_search）→ 使用全文搜索

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
```

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
```

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
```

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
```

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
```

**清空流程**：
1. 先清空 Meilisearch（验证 API key）
2. 如果 Meilisearch 清空成功，再清空 SurrealDB
3. 如果 API key 错误，Meilisearch 清空失败，SurrealDB 不被清空

### 注意事项

⚠️ **调试专用**：此接口仅用于调试和测试，生产环境应谨慎使用
⚠️ **数据保护**：API key 验证失败时，数据不会被清空
⚠️ **不可逆操作**：清空后所有记忆数据将被永久删除