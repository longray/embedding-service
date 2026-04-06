# 贡献指南

感谢你对 Embedding Service 的兴趣！本文档帮助你快速开始贡献。

## 开发环境设置

### 前置要求

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) 包管理器
- Docker & Docker Compose（可选，用于运行依赖服务）

### 快速开始

```bash
# 1. 克隆仓库
git clone <repository-url>
cd embedding_service

# 2. 安装依赖
uv sync

# 3. 启动依赖服务（SurrealDB + Meilisearch）
docker-compose up -d surrealdb meilisearch

# 4. 启动 Embedding + Wrapper 服务
uv run python start_services.py

# 5. 验证服务状态
uv run python scripts/diagnose.py
```

## 代码规范

### Python 代码

我们使用以下工具保持代码质量：

```bash
# 格式化代码
uv run ruff format .

# 检查代码
uv run ruff check .

# 类型检查
uv run pyright

# 运行测试
uv run pytest tests/ -m "unit or integration" -v
```

### 提交信息规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>: <description>

[optional body]

[optional footer]
```

**类型**：
- `feat`: 新功能
- `fix`: 修复
- `docs`: 文档
- `test`: 测试
- `refactor`: 重构
- `perf`: 性能优化
- `chore`: 杂项

**示例**：
```
feat: 实现 HNSW 索引统计端点

- 添加 get_memory_stats() 方法
- 查询 SurrealDB INFO FOR INDEX
- 返回索引元数据

Closes #123
```

## 测试要求

### 测试分层

| 层级 | 标记 | 运行方式 | 说明 |
|------|------|----------|------|
| unit | `@pytest.mark.unit` | `uv run pytest -m unit` | 纯逻辑，无外部依赖 |
| integration | `@pytest.mark.integration` | `uv run pytest -m integration` | 部分 mock |
| e2e | `@pytest.mark.e2e` | `uv run pytest -m e2e` | 真实服务 |

### 提交前检查

```bash
# 运行 pre-commit 检查
uv run pre-commit run --all-files

# 或手动运行关键检查
uv run ruff check .
uv run pyright
uv run pytest tests/ -m "unit or integration" -q
```

## PR 流程

1. **Fork 仓库**（如果是外部贡献者）
2. **创建分支**：`git checkout -b feature/your-feature-name`
3. **开发并测试**：确保所有检查通过
4. **提交更改**：`git commit -m "feat: your description"`
5. **推送分支**：`git push origin feature/your-feature-name`
6. **创建 PR**：描述变更内容和测试情况

### PR 检查清单

- [ ] 代码通过 ruff 检查
- [ ] 代码通过 pyright 类型检查
- [ ] 新增功能有对应测试
- [ ] 所有测试通过
- [ ] 文档已更新（如需要）
- [ ] CHANGELOG.md 已更新（如需要）

## 文档更新

### Markdown 规范

```bash
# 检查 Markdown 格式
uv run task lint-md
```

**关键规则**：
- 代码块必须指定语言（```python）
- 列表前后需要空行
- 标题前后需要空行

## 常见问题

### 服务启动失败

```bash
# 检查端口占用
netstat -an | grep 1800

# 查看服务日志
docker-compose logs surrealdb
docker-compose logs meilisearch

# 一键诊断
uv run python scripts/diagnose.py
```

### 测试失败

```bash
# 仅运行单元测试（快速）
uv run pytest -m unit -v

# 运行特定测试
uv run pytest tests/test_phase_b_sync.py -v

# 查看详细错误
uv run pytest -v --tb=long
```

## 获取帮助

- 查看 [README.md](README.md) 了解项目概览
- 查看 [docs/START_GUIDE.md](docs/START_GUIDE.md) 了解启动指南
- 查看 [BACKLOG.md](BACKLOG.md) 了解当前任务
- 提交 Issue 讨论新功能或报告问题

## 许可证

本项目采用 [MIT](LICENSE) 许可证。
