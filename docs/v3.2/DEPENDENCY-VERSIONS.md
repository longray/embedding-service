# v3.2 依赖版本锁定

> **版本**: v3.2.0  
> **日期**: 2026-04-10  
> **状态**: 实施版  
> **目标**: 锁定关键依赖版本，确保实施稳定性

---

## 依赖版本对照表

### Python 依赖

| 依赖名称 | 当前文档版本 | 建议锁定版本 | 变更原因 | 状态 |
|----------|--------------|--------------|----------|------|
| **surrealdb** | `>=1.0.0,<2.0.0` | `>=1.0.8,<1.1.0` | 缩小范围，使用稳定版 | 🟡 需更新 |
| **meilisearch** | `>=0.40.0,<0.41.0` | `>=0.40.0,<0.41.0` | 版本正确，保持不变 | ✅ 正确 |
| **websockets** | `>=12.0` | `>=12.0,<13.0` | 添加上限，防止重大变更 | 🟡 需更新 |
| **tree-sitter** | 未指定 | `>=0.25.0,<0.26.0` | 🔼 升级到 0.25.x | 🟡 需更新 |
| **tree-sitter-python** | 未指定 | `>=0.25.0,<0.26.0` | 🔼 同步升级到 0.25.x | 🟡 需添加 |
| **tree-sitter-javascript** | 未指定 | `>=0.25.0,<0.26.0` | 🔼 同步升级到 0.25.x | 🟡 需添加 |
| **tree-sitter-typescript** | 未指定 | `>=0.23.0,<0.24.0` | 🔼 升级到 0.23.x | 🟡 需添加 |
| **tree-sitter-go** | 未指定 | `>=0.25.0,<0.26.0` | 🔼 同步升级到 0.25.x | 🟡 需添加 |
| **tree-sitter-rust** | 未指定 | `>=0.24.0,<0.25.0` | 🔼 升级到 0.24.x | 🟡 需添加 |
| **tree-sitter-java** | 未指定 | `>=0.23.0,<0.24.0` | 🔼 升级到 0.23.x | 🟡 需添加 |

### Node.js 依赖

| 依赖名称 | 当前文档版本 | 建议锁定版本 | 变更原因 | 状态 |
|----------|--------------|--------------|----------|------|
| **ws** | `^8.20.0` | `^8.20.0` | 版本正确，保持不变 | ✅ 正确 |
| **@types/ws** | `^8.5.13` | `^8.5.13` | 版本正确，保持不变 | ✅ 正确 |

---

## 更新后的 pyproject.toml

```toml
[project]
name = "opencode-memory-service"
version = "3.2.0"
description = "OpenCode Memory Service v3.2"
requires-python = ">=3.10"

dependencies = [
    # 现有依赖（保持不变）
    "fastapi>=0.115.0,<0.116.0",
    "uvicorn[standard]>=0.32.0,<0.33.0",
    "pydantic>=2.9.0,<2.10.0",
    "transformers>=4.48.0",
    "torch==2.4.0+cu121",
    
    # 更新依赖（版本锁定）
    "surrealdb>=1.0.8,<1.1.0",           # 🔒 缩小范围，使用稳定版
    "meilisearch>=0.40.0,<0.41.0",       # ✅ 保持不变
    "websockets>=12.0,<13.0",            # 🔒 添加上限
    
    # 新增依赖（版本锁定）🔼 升级到 tree-sitter 0.25.x
    "tree-sitter>=0.25.0,<0.26.0",       # 🔼 核心升级到 0.25.x
    "tree-sitter-python>=0.25.0,<0.26.0", # 🔼 Python 语言包 0.25.x
    "tree-sitter-javascript>=0.25.0,<0.26.0", # 🔼 JavaScript 语言包 0.25.x
    "tree-sitter-typescript>=0.23.0,<0.24.0", # 🔼 TypeScript 语言包 0.23.x
    "tree-sitter-go>=0.25.0,<0.26.0",    # 🔼 Go 语言包 0.25.x
    "tree-sitter-rust>=0.24.0,<0.25.0",  # 🔼 Rust 语言包 0.24.x
    "tree-sitter-java>=0.23.0,<0.24.0",  # 🔼 Java 语言包 0.23.x
    
    # 其他依赖
    "fast-json-patch>=1.32",
    "portalocker>=2.7",
    "psutil>=5.9",
    "aiofiles>=23.0",
]
```

---

## 更新后的 package.json

```json
{
  "dependencies": {
    "ws": "^8.20.0",
    "@types/ws": "^8.5.13"
  }
}
```

**说明**: Node.js 依赖版本已正确，无需变更。

---

## 版本选择说明

### surrealdb (Python)

**选择 `>=1.0.8,<1.1.0` 的原因**:
- 1.0.8 是当前最新稳定版
- 1.0.x 系列 API 稳定
- 避免 1.1.0 可能引入的重大变更
- 与 SurrealDB 3.0+ 服务器兼容

### websockets (Python)

**选择 `>=12.0,<13.0` 的原因**:
- 12.x 系列支持 Python 3.10+
- 与 FastAPI WebSocket 兼容
- 13.0 可能引入 API 变更

### tree-sitter (Python)

**选择 `>=0.25.0,<0.26.0` 的原因**:
- 0.25.x 是最新稳定版本（2025年发布）
- 包含性能优化和 bug 修复
- Python 版本要求 >=3.10（与项目一致）

**语言包版本对齐**:
- tree-sitter 核心: `>=0.25.0,<0.26.0`
- python/javascript/go: `>=0.25.0,<0.26.0` (同步)
- typescript/java: `>=0.23.0,<0.24.0` (稍落后但稳定)
- rust: `>=0.24.0,<0.25.0` (中间版本)

**⚠️ 升级注意事项**:
- 从 0.22.x 升级到 0.25.x 是重大版本变更
- 需要测试 API 兼容性
- 查看 [迁移指南](https://github.com/tree-sitter/py-tree-sitter/blob/master/CHANGELOG.md)

### ws (Node.js)

**保持 `^8.20.0` 的原因**:
- 8.20.0 是 npm 最新稳定版
- 与文档中的版本一致
- 无需变更

---

## 实施步骤

### Step 1: 更新文档

1. 更新 `BACKEND-v3.2-IMPLEMENTATION.md` 第2节"依赖升级"
2. 更新 `UNIFIED-ARCHITECTURE-v3.2.md` 中的依赖版本引用
3. 更新 `BACKEND-v3.2-PRECOMPUTE.md` 中的 tree-sitter 版本说明

### Step 2: 更新实际项目文件（后续开发阶段）

1. 更新 `embedding_service/wrapper/pyproject.toml`
2. 生成 `poetry.lock` 锁定精确版本
3. 更新 `opencode-memory-plugin/package.json`（如需要）

### Step 3: 验证

1. 检查所有文档版本一致性
2. 验证依赖版本在 PyPI/npm 存在
3. 测试依赖安装

---

## 验证方式

### 1. 一致性检查

```bash
# 检查文档中依赖版本一致性
grep -r "surrealdb" docs/v3.2/ --include="*.md"
grep -r "tree-sitter" docs/v3.2/ --include="*.md"
grep -r "websockets" docs/v3.2/ --include="*.md"
```

### 2. 版本存在性验证

```bash
# Python 依赖
pip index versions surrealdb
pip index versions tree-sitter
pip index versions tree-sitter-python

# Node.js 依赖
npm view ws versions
```

### 3. 安装测试

```bash
# Python
cd embedding_service/wrapper
pip install -e ".[dev]"

# Node.js
cd opencode-memory-plugin
npm install
```

---

## 风险评估

| 风险项 | 概率 | 影响 | 缓解措施 |
|--------|------|------|----------|
| tree-sitter 语言包版本不兼容 | 中 | 中 | 升级到 0.25.x 系列，测试兼容性 |
| surrealdb 1.0.8 有未发现问题 | 低 | 高 | 充分测试，准备回滚方案 |
| websockets 12.x 与 FastAPI 不兼容 | 低 | 中 | 提前验证 |

---

## 参考文档

- [SurrealDB Python SDK](https://pypi.org/project/surrealdb/)
- [tree-sitter PyPI](https://pypi.org/project/tree-sitter/)
- [tree-sitter-language-bindings](https://github.com/tree-sitter/py-tree-sitter#language-bindings)
- [ws npm](https://www.npmjs.com/package/ws)
- [websockets PyPI](https://pypi.org/project/websockets/)

---

_文档版本: v3.2.0_  
_更新时间: 2026-04-10_  
_状态: 已锁定依赖版本_
