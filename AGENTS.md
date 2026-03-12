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
│       ├── main.py             # FastAPI 主程序 (v2.3.0)
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