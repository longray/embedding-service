# 初始化脚本使用指南

本文档介绍 Embedding Service 的初始化脚本，帮助你从零开始设置完整的环境。

## 📋 脚本列表

| 脚本 | 功能 | 用途 |
|------|------|------|
| `init_database.py` | 初始化 SurrealDB 数据库 | 创建表、索引、运行时用户 |
| `init_meilisearch.py` | 初始化 Meilisearch 索引 | 创建索引、配置中文分词、代码术语字典 |
| `init_all.py` | 一键初始化所有服务 | 依次初始化数据库和索引 |

---

## 🚀 快速开始

### 1. 启动服务

```bash
# 使用 docker-compose 启动所有服务
docker-compose up -d surrealdb meilisearch

# 或手动启动
# SurrealDB
surreal start --log trace

# Meilisearch
meilisearch --master-key your_key
```

### 2. 一键初始化

```bash
# 最简单的方式：一键初始化所有服务
uv run python scripts/init_all.py
```

### 3. 验证初始化

```bash
# 仅验证环境，不重新初始化
uv run python scripts/init_all.py --verify-only
```

---

## 📦 数据库初始化

### 单独初始化 SurrealDB

```bash
# 使用默认配置
uv run python scripts/init_database.py

# 使用自定义配置
export SURREAL_URL=ws://localhost:18002
export SURREAL_NS=memory_ns
export SURREAL_DB=memory_db
export SURREAL_USER=root
export SURREAL_PASS=root
uv run python scripts/init_database.py

# 仅验证 schema（不重新初始化）
uv run python scripts/init_database.py --verify-only

# 不创建运行时用户
uv run python scripts/init_database.py --no-runtime-user
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SURREAL_URL` | `ws://localhost:18002` | SurrealDB WebSocket URL |
| `SURREAL_NS` | `memory_ns` | 命名空间 |
| `SURREAL_DB` | `memory_db` | 数据库名 |
| `SURREAL_USER` | `root` | 用户名 |
| `SURREAL_PASS` | `root` | 密码 |
| `CREATE_RUNTIME_USER` | `true` | 是否创建运行时用户 |

### 初始化内容

1. **命名空间和数据库**
   - 创建命名空间（如果不存在）
   - 创建数据库（如果不存在）

2. **表结构**
   - `memory` - 记忆表（支持向量搜索、全文搜索）
   - `memory_relation` - 图关系边表
   - `project` - 项目表
   - `schema_version` - Schema 版本表
   - `conflict` - 冲突表（同步冲突解决）

3. **索引**
   - HNSW 向量索引（`memory_embedding_hnsw`）
   - 全文索引（`memory_content_ft`）
   - 多租户索引（`memory_tenant`）
   - 其他辅助索引

4. **运行时用户**（可选）
   - `runtime_user` - 仅数据操作权限（EDITOR 角色）

---

## 🔍 Meilisearch 初始化

### 单独初始化 Meilisearch

```bash
# 使用默认配置
uv run python scripts/init_meilisearch.py

# 使用自定义配置
export WRAPPER_MEILI_URL=http://localhost:7700
export WRAPPER_MEILI_API_KEY=your_master_key
export WRAPPER_MEILI_INDEX_NAME=memories
uv run python scripts/init_meilisearch.py

# 仅验证索引（不重新初始化）
uv run python scripts/init_meilisearch.py --verify-only
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WRAPPER_MEILI_URL` | `http://localhost:7700` | Meilisearch URL |
| `WRAPPER_MEILI_API_KEY` | `None` | API Key（可选） |
| `WRAPPER_MEILI_INDEX_NAME` | `memories` | 索引名 |
| `WRAPPER_MEILI_TIMEOUT` | `30.0` | 请求超时（秒） |

### 初始化内容

1. **索引创建**
   - 创建主索引（`memories`）
   - 设置主键为 `id`

2. **索引配置**
   - **可搜索字段**：`content_zh`、`content_search`、`code`、`content` 等
   - **可过滤字段**：`tenant_id`、`type`、`tags`、`date`、`version` 等
   - **可排序字段**：`date`、`created_at`
   - **非分隔符**：`.`、`-`、`@`、`:`、`/`、`_`（保持代码标识符完整性）
   - **中文本地化**：`*_zh` 字段使用中文分词
   - **拼写容错**：对精确字段禁用（`file_path`、`version` 等）

3. **代码术语字典**（104词）
   - 版本前缀：`v1`、`v2`、`alpha`、`beta`、`rc` 等
   - 编程语言：`python`、`java`、`javascript`、`typescript` 等
   - 常见命名：`http`、`api`、`www`、`com`、`cn` 等
   - 代码术语：`class`、`interface`、`function`、`method` 等
   - 框架/库：`django`、`flask`、`fastapi`、`react` 等
   - ID 前缀：`ID`、`NO`、`NUM`、`CODE` 等
   - 时间：`2025`、`2026`、`Jan`、`Feb` 等
   - IP 段：`192`、`168`、`172`、`10` 等

---

## 🎯 一键初始化

### 完整初始化流程

```bash
# 1. 启动服务
docker-compose up -d surrealdb meilisearch

# 2. 等待服务就绪（约 10-15 秒）
sleep 15

# 3. 一键初始化
uv run python scripts/init_all.py
```

### 命令行选项

```bash
# 仅验证环境（不重新初始化）
uv run python scripts/init_all.py --verify-only

# 跳过 SurrealDB 初始化
uv run python scripts/init_all.py --skip-db

# 跳过 Meilisearch 初始化
uv run python scripts/init_all.py --skip-meili
```

### 初始化流程

1. **健康检查**
   - 检查 SurrealDB 是否运行
   - 检查 Meilisearch 是否运行

2. **SurrealDB 初始化**（如果 `--skip-db` 未指定）
   - 创建命名空间和数据库
   - 执行 schema 初始化脚本
   - 验证表和索引
   - 创建运行时用户

3. **Meilisearch 初始化**（如果 `--skip-meili` 未指定）
   - 创建索引
   - 配置索引设置
   - 验证索引配置

4. **完成提示**
   - 显示下一步操作（启动服务、运行测试等）

---

## 🔧 高级用法

### 自定义命名空间和数据库

```bash
# 设置自定义命名空间和数据库
export SURREAL_NS=production_ns
export SURREAL_DB=production_db

# 初始化
uv run python scripts/init_database.py
```

### 多环境配置

**开发环境**：
```bash
export SURREAL_URL=ws://localhost:18002
export WRAPPER_MEILI_URL=http://localhost:7700
uv run python scripts/init_all.py
```

**测试环境**：
```bash
export SURREAL_URL=ws://test.example.com:18002
export WRAPPER_MEILI_URL=http://test.example.com:7700
uv run python scripts/init_all.py
```

**生产环境**：
```bash
export SURREAL_URL=ws://prod.example.com:18002
export WRAPPER_MEILI_URL=http://prod.example.com:7700
export WRAPPER_MEILI_API_KEY=$PROD_MEILI_KEY
uv run python scripts/init_all.py
```

### 迁移现有数据

如果你有现有数据需要迁移：

```bash
# 1. 初始化空数据库
uv run python scripts/init_database.py

# 2. 迁移 SurrealDB 数据到 Meilisearch
uv run python scripts/migrate_to_meilisearch.py --batch-size 200
```

---

## 📊 验证初始化

### 验证 SurrealDB

```bash
# 方式 1: 使用脚本验证
uv run python scripts/init_database.py --verify-only

# 方式 2: 使用 SurrealDB CLI
surreal sql --ns memory_ns --db memory_db --query "SELECT * FROM tables"

# 方式 3: 使用 API
curl -X POST http://localhost:18002/sql \
  -H "Content-Type: application/json" \
  -d 'SELECT * FROM tables'
```

### 验证 Meilisearch

```bash
# 方式 1: 使用脚本验证
uv run python scripts/init_meilisearch.py --verify-only

# 方式 2: 使用 Meilisearch API
curl http://localhost:7700/indexes/memories

# 方式 3: 查看索引统计
curl http://localhost:7700/indexes/memories/stats
```

### 验证服务健康

```bash
# SurrealDB 健康检查
curl http://localhost:18002/health

# Meilisearch 健康检查
curl http://localhost:7700/health

# 包装服务健康检查
curl http://localhost:17999/health
```

---

## ⚠️ 故障排查

### 问题 1: SurrealDB 连接失败

**症状**：
```
❌ 连接 SurrealDB 失败: [Errno 111] Connection refused
```

**解决方案**：
```bash
# 1. 检查 SurrealDB 是否运行
ps aux | grep surreal

# 2. 启动 SurrealDB
docker-compose up -d surrealdb

# 3. 检查端口
netstat -an | grep 18002
```

### 问题 2: Meilisearch 连接失败

**症状**：
```
❌ 连接 Meilisearch 失败: [Errno 111] Connection refused
```

**解决方案**：
```bash
# 1. 检查 Meilisearch 是否运行
ps aux | grep meilisearch

# 2. 启动 Meilisearch
docker-compose up -d meilisearch

# 3. 检查端口
netstat -an | grep 7700
```

### 问题 3: Schema 初始化失败

**症状**：
```
❌ Schema 初始化失败: ...
```

**解决方案**：
```bash
# 1. 检查 SurrealDB 版本
surreal version

# 2. 确保版本 >= 2.0（推荐 3.0+）
# 3. 重新运行初始化（幂等操作，可重复执行）
uv run python scripts/init_database.py
```

### 问题 4: Meilisearch 索引创建失败

**症状**：
```
❌ 创建索引失败: ...
```

**解决方案**：
```bash
# 1. 检查 Meilisearch 版本
curl http://localhost:7700/version

# 2. 确保版本 >= 1.0（推荐 1.4+）
# 3. 检查 API Key（如果配置）
export WRAPPER_MEILI_API_KEY=your_key
# 4. 重新运行初始化
uv run python scripts/init_meilisearch.py
```

---

## 📚 相关文档

- [同步冲突解决最佳实践](./SYNC_CONFLICT_RESOLUTION.md)
- [API 规范](./API_SPECIFICATION.md)
- [启动指南](./START_GUIDE.md)
- [架构设计](./architecture/WRAPPER_SERVICE_DESIGN.md)

---

## 📝 总结

| 脚本 | 适用场景 | 优先级 |
|------|----------|--------|
| `init_all.py` | 首次部署、完整环境设置 | ⭐⭐⭐ |
| `init_database.py` | 单独初始化数据库 | ⭐⭐ |
| `init_meilisearch.py` | 单独初始化索引 | ⭐⭐ |

**推荐流程**：
1. 首次部署：使用 `init_all.py` 一键初始化
2. 数据库迁移：使用 `init_database.py` 重新初始化
3. 索引重建：使用 `init_meilisearch.py` 重新配置

---

## 文件位置

D:\embedding_service\scripts\README.md

## 验证

- Markdown 语法正确
- 所有命令可运行
- 链接和引用正确

<!-- OMO_INTERNAL_INITIATOR -->
