# Embedding Service (OpenCode Memory Stack)

版本与路线图

- 当前版本: v2.4.2
- 实施阶段: P0 + P1 + P2 + Phase 3 + Polyglot 搜索架构 + 同步冲突解决 + SQL 注入修复 已完成
- 详细路线见 ROADMAP.md

## 开发状态

**当前版本**: v2.4.2
**实施阶段**: P0 + P1 + P2 + Phase 3 + Polyglot 搜索架构 + 同步冲突解决 + SQL 注入修复 已完成

### 已完成 ✅

- ✅ P0 核心功能（Embedding + LLM + 包装层）
- ✅ P1 增强功能（熔断器、缓存、监控、测试套件）
- ✅ P2 生产就绪（API认证授权、CI/CD、完整文档）
- ✅ P3-1 Docker Compose 一键部署
- ✅ P3-2 HNSW 向量搜索优化
- ✅ Phase 3A 批量 Embedding 性能优化（10x 加速）
- ✅ Phase 3B OpenTelemetry 分布式追踪
- ✅ Phase 3C 安全加固（DB 权限分离 + 运行时凭据）
- ✅ Phase 3D WebSocket 实时推送（LIVE SELECT）
- ✅ Phase 3E Polyglot 搜索架构（Meilisearch 全文搜索 + SurrealDB 向量/图）
- ✅ **v2.4.2** API 稳定性修复（B-010, B-012, B-018, B-019）
- ✅ **v2.4.2** 性能基线建立（scripts/benchmark.py）
- ✅ **v2.4.2** SQL 注入修复（B-024 LIMIT/relationship_type 参数化, B-025 record_id type::record()）
- ✅ **v2.4.2** Bandit 安全扫描标记完成

### P3 优化路线图 🚀

| 优先级 | 功能 | 预期收益 | 状态 |
|--------|------|----------|------|
| P3-1 | Docker Compose | 一键部署 | ✅ 已完成 |
| P3-2 | HNSW向量索引 | 搜索10x加速 | ✅ 已完成 |
| P3-3 | 监控告警 | 自动告警 | ⏳ 待开始 |
| P3-4 | Kubernetes | 云原生部署 | ⏳ 待开始 |
| P3-5 | 审计日志 | 合规审计 | ⏳ 待开始 |

查看 [ROADMAP.md](ROADMAP.md) 了解详细计划。

## API端点

### 最小化包装服务（端口 17999）

| 端点 | 方法 | 功能 | 认证 |
|------|------|------|------|
| `/health` | GET | 健康检查 | 🌍 公开 |
| `/v1/embeddings` | POST | 文本嵌入 + 缓存 | 🌍 公开 |
| `/api/v1/memories` | POST | 批量上传记忆 | 🌍 公开 |
| `/api/v1/memories/search` | POST | 搜索记忆 | 🌍 公开 |
| `/ws/memories/live` | WebSocket | 实时推送记忆变更 | 🔓 可选 |
| `/api/v1/memories/clear` | DELETE | **NEW** 清空所有记忆（调试专用） | 🔐 API Key |
| `/api/v1/hnsw/stats` | GET | **NEW** HNSW 索引统计 | 🌍 公开 |
| `/api/v1/hnsw/optimize` | POST | **NEW** 优化 HNSW 参数 | 🌍 公开 |
| `/api/v1/hnsw/rebuild` | POST | **NEW** 重建 HNSW 索引 | 🌍 公开 |
| `/api/v1/cache/stats` | GET | **NEW** 缓存统计 | 🌍 公开 |
| `/api/v1/cache/clear` | POST | **NEW** 清空缓存 | 🌍 公开 |
| `/api/v1/cache/warmup` | POST | **NEW** 预热缓存 | 🌍 公开 |
| `/api/v1/prefetch/related` | POST | **NEW** 预取关联记忆 | 🌍 公开 |
| `/api/v1/prefetch/popular` | POST | **NEW** 预取热门记忆 | 🌍 公开 |
| `/api/v1/sync/preview` | POST | 同步预览（差异分析） | 🌍 公开 |
| `/api/v1/sync/incremental` | POST | 同步预览（兼容别名） | 🌍 公开 |
| `/api/v1/sync/full` | POST | 全量同步 | 🌍 公开 |
| `/api/v1/sync/fingerprints` | GET | 获取服务端指纹 | 🌍 公开 |
| `/api/v1/sync/conflicts/{id}/resolve` | POST | 解决同步冲突 | 🌍 公开 |

### 完整包装服务（端口 3001）

| 端点 | 方法 | 功能 | 认证 |
|------|------|------|------|
| `/v1/embeddings` | POST | 文本嵌入 | 🔐 read |
| `/v1/chat/completions` | POST | 聊天补全 | 🔐 read |
| `/api/v1/memories` | POST | 上传记忆 | 🔐 write |
| `/api/v1/memories/search` | POST | 搜索记忆 | 🔐 read |
| `/health` | GET | 健康检查 | 🌍 公开 |

🔐 = 需要API Key认证, 🌍 = 公开访问

认证启用方式：

```bash
export WRAPPER_AUTH_ENABLED=true
export WRAPPER_API_KEYS="your_key:read;write"
```text

### WebSocket 实时推送

连接 `/ws/memories/live` 端点接收记忆变更的实时通知。

**连接参数**:

- `tenant_id` (可选): 租户 ID，默认 `default`
- `token` (可选): 认证 token（需配置 `WRAPPER_WEBSOCKET_TOKEN`）

**JavaScript 示例**:

```javascript
const ws = new WebSocket('ws://localhost:17999/ws/memories/live?tenant_id=default&token=your_token');
ws.onmessage = (event) => {
  const { action, result } = JSON.parse(event.data);
  console.log(action, result); // CREATE/UPDATE/DELETE
};
```

**Python 示例**:

```python
import json
from websockets import connect

async with connect('ws://localhost:17999/ws/memories/live?tenant_id=default') as ws:
    async for message in ws:
        data = json.loads(message)
        print(data['action'], data['result'])
```text

**认证配置**:

```bash
# 启用 WebSocket 认证（可选，未配置则允许所有连接）
export WRAPPER_WEBSOCKET_TOKEN=your_secret_token
```

### 核心功能

- ✅ **记忆管理**：SurrealDB 向量存储 + Meilisearch 全文搜索，Polyglot 混合搜索架构
- ✅ **API 认证**：API Key 认证和权限控制
- ✅ **LRU 缓存**：文本嵌入结果缓存
- ✅ **HTTP 连接池**：高效 HTTP 请求
- ✅ **SurrealDB 长期连接**：避免频繁连接开销
- ✅ **CI/CD**：GitHub Actions 自动测试
- ✅ **完整测试套件**：150+ 测试用例
- ✅ **Meilisearch 全文搜索**：CJK 中文分词、日期精确匹配、关键词搜索
- ✅ **Meilisearch 代码搜索优化**：104词代码术语字典、代码标识符搜索、双字段策略

### Meilisearch 代码搜索优化

**优化功能**：

- **104词代码术语字典**：FastAPI, Python, Meilisearch, SurrealDB, Docker, Kubernetes 等常用代码术语
- **代码标识符搜索**：支持 `meili_client.py`, `config.surrealdb.url` 等带点号/下划线的标识符
- **双字段策略**：搜索字段（content_zh, content_search, code）+ 精确匹配字段（date, version, ip_address, email）
- **中文分词优化**：CJK 中文分词，支持中文代码注释和文档搜索
- **nonSeparatorTokens**：`-`, `.`, `/`, `:`, `@`, `_` 不作为分隔符，保持代码标识符完整性

**使用场景**：

- 代码文件名搜索：`meili_client.py`, `memory_manager.py`
- 配置项搜索：`config.surrealdb.url`, `wrapper.meili.enabled`
- 版本号搜索：`v2.3.0`, `v3.0.1`
- IP地址搜索：`192.168.1.1`, `127.0.0.1`
- 日期搜索：`2026-03-12`

**详细文档**：查看 [meilisearch_code/README.md](meilisearch_code/README.md) 了解完整配置和使用指南。

## 技术要求与兼容性

- 保持向后兼容及现有接口
- 认证开关可通过环境变量控制
- 兼容现有文档结构，方便跳转至 ROADMAP.md

## 快速开始

### 前置条件

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) 包管理器
- SurrealDB 3.0+ 运行中
- Meilisearch 1.4+ 运行中（可选，用于全文搜索优化）

### 从零开始初始化

```bash
# 1. 启动数据库和搜索引擎
docker-compose up -d surrealdb meilisearch

# 2. 等待服务就绪（约 10-15 秒）
sleep 15

# 3. 一键初始化所有服务
uv run python scripts/init_all.py

# 详细文档请查看: [scripts/README.md](scripts/README.md)
```text

### 启动最小化包装服务

```bash
# 配置 Meilisearch（可选，不配置则回退到 SurrealDB BM25）
export WRAPPER_MEILI_ENABLED=true
export WRAPPER_MEILI_URL=http://127.0.0.1:7700
export WRAPPER_MEILI_API_KEY=your_master_key

# 启动服务
uv run python -m wrapper.src.main
```

### 调试清空记忆数据（v2.3.1+）

**清空 API 端点**：`DELETE /api/v1/memories/clear`

**安全机制**：

1. 先清空 Meilisearch（验证 `WRAPPER_MEILI_API_KEY`）
2. 如果 Meilisearch 清空成功 → 再清空 SurrealDB
3. 如果 API key 错误 → Meilisearch 清空失败，SurrealDB 不被清空

**使用方法**：

```bash
# 从环境变量 MEILI_MASTER_KEY 获取 API Key
# 默认值：masterKey_change_in_production（生产环境需修改）
export MEILI_MASTER_KEY=<your_api_key>

# 正确的 key（会清空所有数据）
curl -X DELETE http://localhost:17999/api/v1/memories/clear \
  -H "WRAPPER_MEILI_API_KEY: <your_api_key>"

# 错误的 key（只返回 403，保护数据）
curl -X DELETE http://localhost:17999/api/v1/memories/clear \
  -H "WRAPPER_MEILI_API_KEY: wrong_key"
```text

**API Key 配置**：

- 环境变量：`MEILI_MASTER_KEY`
- Docker Compose 配置：见 `docker-compose.yml` 第 106 行
- Wrapper 配置：见 `wrapper/src/config.py` 第 146 行
- 默认值：`masterKey_change_in_production`（生产环境需修改）

**响应示例**：

成功 (200)：

```json
{
  "success": true,
  "message": "所有记忆数据已清空"
}
```

失败 (401 - 缺少 key)：

```json
{
  "detail": "Missing WRAPPER_MEILI_API_KEY header"
}
```text

失败 (403 - key 错误)：

```json
{
  "detail": "Invalid WRAPPER_MEILI_API_KEY"
}
```

失败 (500 - 清空失败)：

```json
{
  "detail": "清空失败: ..."
}
```text

**清空脚本**：

```bash
# 清空后端所有数据（SurrealDB + Meilisearch）
cd D:/embedding_service
export WRAPPER_MEILI_API_KEY={去环境变量找}
uv run python scripts/clear_all_data.py
```

### 数据迁移（首次启用 Meilisearch 时）

```bash
# 将 SurrealDB 现有记忆同步到 Meilisearch（幂等，可重复运行）
export SURREAL_URL=ws://localhost:18002/rpc
export SURREAL_NS=memory_ns
export SURREAL_DB=memory_db
uv run python scripts/migrate_to_meilisearch.py --batch-size 200
```text

### 运行测试

```bash
# 运行核心 API 测试（推荐）
uv run pytest tests/test_wrapper_api.py -v

# 运行 Meilisearch 集成测试
uv run pytest tests/test_meili_integration.py -v

# 运行同步冲突解决测试
uv run pytest tests/test_phase_b_sync.py -v

# 运行所有测试
uv run pytest tests/ -v

# 运行性能基准测试（v2.4.2+）
uv run python scripts/benchmark.py --iterations 5
```

### 代码与文档质量门禁

本项目强制实施严格的质量门禁：

```bash
# 1. 运行 Python 格式化 (Ruff)
uv run ruff format src/

# 2. 运行 Python Lint 检查 (Ruff)
uv run ruff check src/ --fix

# 3. 运行静态类型检查 (Pyright)
uv run pyright src/

# 4. 运行 Markdown 文档检查 (Markdownlint)
uvx pre-commit run markdownlint-cli2 --all-files

# 5. 使用 taskipy 快捷命令
uv run task lint-md
uv run task lint-md-stats
```bash

> **注意**: 建议直接使用 `pre-commit` 自动管理所有 hook：`git commit` 时会自动触发上述所有检查。

### 性能基线（v2.4.2）

运行 `scripts/benchmark.py` 获取当前环境性能数据：

| 操作 | 平均延迟 | P50 | P95 |
|------|----------|-----|-----|
| 单文本 Embedding | 156ms | 160ms | 201ms |
| 向量搜索 | 157ms | 156ms | 180ms |
| 混合搜索 | 21ms | 22ms | 24ms |
| 单条上传 | 542ms | 699ms | 724ms |
| E2E 完整流程 | 883ms | 884ms | 919ms |

**环境**: NVIDIA GTX 1060 6GB, Qwen3-Embedding-0.6B, SurrealDB 3.0 + Meilisearch 1.4

### 同步冲突解决

详细的多设备、多用户、离线编辑同步指南，请查看：

📖 **[同步冲突解决最佳实践](docs/SYNC_CONFLICT_RESOLUTION.md)**

**快速开始**：

```python
import httpx

# 1. 同步预览（分析差异）
response = await httpx.post(
    "http://localhost:17999/api/v1/sync/preview",
    json={
        "fingerprints": [
            {"path": "test.md", "mtime": 1711234567890,
             "hash": "abc123", "source_id": "entry-001"}
        ],
        "tenant_id": "default"
    }
)

# 2. 检查冲突
if response.json()["conflicts"]:
    # 3. 解决冲突（大小写不敏感）
    await httpx.post(
        f"http://localhost:17999/api/v1/sync/conflicts/{conflict_id}/resolve",
        json={"resolution": "use_local", "tenant_id": "default"}
    )
```

## 文件位置

D:\embedding_service\README.md

## 验证

- Markdown 语法正确性检查
- 通过浏览器打开或在 CI 中渲染 README.md

<!-- OMO_INTERNAL_INITIATOR -->
