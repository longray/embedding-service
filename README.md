# Embedding Service (OpenCode Memory Stack)

**当前版本**: v2.9.2

> **版本说明**: v2.9.1 为产品版本号。BACKLOG v3.2/v3.3 为内部规划版本，所有 v3.x 任务已合入 v2.9.x 发布。

## 开发状态

**实施阶段**: P0 + P1 + P2 + Phase 3 + BACKLOG v3.3 已完成

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
- ✅ **v2.6.0** 质量治理（memory_manager Mixin 拆分 + main.py 路由模块化 + 35 个单元测试 + 文档归档对齐）
- ✅ **v2.7.0** 多设备同步（指纹查询/同步预览/全量同步/冲突解决）+ 测试架构优化
- ✅ **v2.7.1** SQL 查询优化（RecordID 统一、分批处理、embedding 字段优化）+ 安全性修复
- ✅ **v2.8.0** PrecomputeService 完善 + Stub 端点实现（BACKLOG v3.3 全部完成）
- ✅ **v2.9.0** Atom Meilisearch 统一搜索（Phase 1-3 完整实施）
- ✅ **v2.9.1** Entity ID 格式统一（支持 memory/entity 双表）
- ✅ **v2.9.1** listEntities Pydantic 验证修复（支持缺失字段）
- ✅ **v2.9.1** Entity/Atom Meilisearch 同步修复
- ✅ **v2.9.2** RecordID 查询 Bug 修复（BL-B-116~119）+ 全代码库审计报告

### P3 优化路线图 🚀

| 优先级 | 功能 | 预期收益 | 状态 |
|--------|------|----------|------|
| P3-1 | Docker Compose | 一键部署 | ✅ 已完成 |
| P3-2 | HNSW向量索引 | 搜索10x加速 | ✅ 已完成 |
| P3-3 | 监控告警 | 自动告警 | ⏳ 待开始 |
| P3-4 | Kubernetes | 云原生部署 | 🚧 基础完成 |
| P3-5 | 审计日志 | 合规审计 | ⏳ 待开始 |
| **v3.3** | **PrecomputeService + Stub 端点** | **代码预计算、聚类、预取** | **✅ 已完成** |

查看 [ROADMAP.md](docs/ROADMAP.md) 了解详细计划。

## API端点

### 最小化包装服务（端口 18008）

| 端点 | 方法 | 功能 | 认证 |
|------|------|------|------|
| `/health` | GET | 健康检查 | 🌍 公开 |
| `/v1/embeddings` | POST | 文本嵌入 + 缓存 | 🌍 公开 |
| `/api/v1/memories` | POST | 批量上传记忆 | 🌍 公开 |
| `/api/v1/memories/search` | POST | 搜索记忆（支持 code_filter 代码过滤） | 🌍 公开 |
| `/api/v1/memories/{id}` | GET | 获取记忆详情（默认不含 embedding） | 🌍 公开 |
| `/api/v1/memories/{id}?include_embedding=true` | GET | 获取记忆详情（含 embedding） | 🌍 公开 |
| `/api/v1/memories/lookup` | GET | 记忆查询（source_id/hash/file_path）(BL-CA-34) ✅ | 🌍 公开 |
| `/api/v1/memories/{id}/summary` | GET | 获取记忆摘要 | 🌍 公开 |
| `/api/v1/memories/{id}/enrich/llm` | POST | LLM 摘要增强 | 🌍 公开 |
| `/api/v1/memories/relations` | POST | 创建图关系 | 🌍 公开 |
| `/api/v1/memories/{id}/relations` | POST | 查询记忆关系 | 🌍 公开 |
| `/api/v1/memories/relations/{id}` | DELETE | 删除图关系 | 🌍 公开 |
| `/api/v1/memories/{id}/graph` | POST | 图遍历 | 🌍 公开 |
| `/api/v1/calls/batch` | POST | 批量创建调用关系 (BL-CA-20) ✅ | 🌍 公开 |
| `/api/v1/memories/{id}/references` | GET | 引用查询 - 谁调用了该符号 (BL-CA-21) ✅ | 🌍 公开 |
| `/api/v1/memories/{id}/dependencies` | GET | 依赖查询 - 该符号依赖谁 (BL-CA-22) ✅ | 🌍 公开 |
| `/api/v1/projects/{id}/map` | GET | 代码地图 (BL-CA-23) ✅ | 🌍 公开 |
| `/api/v1/projects/{id}/stats` | GET | 代码统计 (BL-CA-25) ✅ | 🌍 公开 |
| `/api/v1/memories/clear` | DELETE | 清空所有记忆（调试专用） | 🔐 API Key |
| `/api/v1/access-log` | POST | 上报访问日志 | 🌍 公开 |
| `/ws/memories/live` | WebSocket | 实时推送记忆变更 | 🔓 可选 |
| `/api/v1/hnsw/stats` | GET | HNSW 索引统计 ✅ | 🌍 公开 |
| `/api/v1/hnsw/optimize` | POST | 优化 HNSW 参数 ✅ | 🌍 公开 |
| `/api/v1/hnsw/rebuild` | POST | 重建 HNSW 索引 ✅ | 🌍 公开 |
| `/api/v1/cache/stats` | GET | 缓存统计 ✅ | 🌍 公开 |
| `/api/v1/cache/clear` | POST | 清空缓存 ✅ | 🌍 公开 |
| `/api/v1/cache/warmup` | POST | 预热缓存 ✅ | 🌍 公开 |
| `/api/v1/prefetch/related` | POST | 预取关联记忆 ✅ | 🌍 公开 |
| `/api/v1/prefetch/popular` | POST | 预取热门记忆 ✅ | 🌍 公开 |
| `/api/v1/memories/{id}/analyze/code` | POST | 代码分析 ✅ | 🌍 公开 |
| `/api/v1/memories/cluster/leiden` | POST | Leiden 聚类 ✅ | 🌍 公开 |
| `/api/v1/sync/preview` | POST | 同步预览（差异分析） | 🌍 公开 |
| `/api/v1/sync/incremental` | POST | 同步预览（兼容别名） | 🌍 公开 |
| `/api/v1/sync/full` | POST | 全量同步 | 🌍 公开 |
| `/api/v1/sync/fingerprints` | GET | 获取服务端指纹 | 🌍 公开 |
| `/api/v1/sync/conflicts/{id}/resolve` | POST | 解决同步冲突 | 🌍 公开 |
| `/api/v1/atoms` | POST | 创建 Atom | 🌍 公开 |
| `/api/v1/atoms` | GET | 列出 Atoms（支持 max_level 过滤） | 🌍 公开 |
| `/api/v1/atoms/budget` | POST | 上下文预算管理（BM25 + hierarchy） | 🌍 公开 |
| `/api/v1/entities` | POST | 创建 Entity（支持 atoms 内联创建） | 🌍 公开 |
| `/api/v1/entities` | GET | 列出 Entities | 🌍 公开 |
| `/api/v1/entities/{id}` | GET | 获取 Entity | 🌍 公开 |
| `/api/v1/entities/{id}` | PUT | 更新 Entity | 🌍 公开 |
| `/api/v1/entities/{id}` | DELETE | 删除 Entity | 🌍 公开 |
| `/api/v1/entities/{id}/atoms/{atom_id}` | GET | 跨 Entity Atom 链接 | 🌍 公开 |
| `/api/v1/search` | POST | 统一搜索（scope=atom/entity, max_level） | 🌍 公开 |

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
const ws = new WebSocket('ws://localhost:18008/ws/memories/live?tenant_id=default&token=your_token');
ws.onmessage = (event) => {
  const { action, result } = JSON.parse(event.data);
  console.log(action, result); // CREATE/UPDATE/DELETE
};
```

**Python 示例**:

```python
import json
from websockets import connect

async with connect('ws://localhost:18008/ws/memories/live?tenant_id=default') as ws:
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
- ✅ **完整测试套件**：1000+ 测试用例
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
curl -X DELETE http://localhost:18008/api/v1/memories/clear \
  -H "WRAPPER_MEILI_API_KEY: <your_api_key>"

# 错误的 key（只返回 403，保护数据）
curl -X DELETE http://localhost:18008/api/v1/memories/clear \
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
| 单文本 Embedding | 211.5ms | 176.3ms | 290.9ms |
| 向量搜索 | 102.6ms | 98.3ms | 120.0ms |
| 混合搜索 | 14.1ms | 14.1ms | 14.8ms |
| 单条上传 | 277.6ms | 102.8ms | 633.9ms |
| E2E 完整流程 | 754.3ms | 746.4ms | 777.8ms |

**环境**: NVIDIA GTX 1060 6GB, Qwen3-Embedding-0.6B, SurrealDB 3.0 + Meilisearch 1.4
**测试时间**: 2026-03-30, 3 次迭代

### 同步冲突解决

详细的多设备、多用户、离线编辑同步指南，请查看：

📖 **[同步冲突解决最佳实践](docs/SYNC_CONFLICT_RESOLUTION.md)**

**快速开始**：

```python
import httpx

# 1. 同步预览（分析差异）
response = await httpx.post(
    "http://localhost:18008/api/v1/sync/preview",
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
        f"http://localhost:18008/api/v1/sync/conflicts/{conflict_id}/resolve",
        json={"resolution": "use_local", "tenant_id": "default"}
    )
```

## 贡献指南

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与。

## 许可证

[MIT](LICENSE)
