# OpenCode Memory Service - 产品文档

> **版本**: v3.2.0  
> **面向**: 终端用户、产品经理、运维人员  
> **目标**: 说明产品功能、使用场景、快速入门

---

## 产品概述

OpenCode Memory Service 是一个智能记忆存储与检索服务，支持代码分析、记忆管理和语义搜索。

### 核心功能

| 功能模块 | 说明 | 状态 |
|----------|------|------|
| **Embedding 服务** | 文本向量化（Qwen3-Embedding-0.6B） | ✅ 可用 |
| **记忆存储** | 分层存储（L0/L1/L2） | ✅ 可用 |
| **语义搜索** | 向量搜索 + 全文搜索 + 混合搜索 | ✅ 可用 |
| **WebSocket 实时同步** | 可靠的消息推送（心跳、ACK、DIFF） | ✅ 可用 |
| **代码分析** | AST 解析、关系提取（部分功能） | ⚠️ 预览版 |
| **预计算服务** | 代码预计算、关系构建（开发中） | 🚧 开发中 |

### 使用场景

1. **智能代码助手** - 存储代码片段，支持语义搜索
2. **知识库管理** - 构建个人/团队知识库
3. **实时协作** - WebSocket 实时同步记忆变更
4. **代码分析** - 分析代码结构，提取调用关系

---

## 快速入门

### 启动服务

```bash
# 使用 Docker Compose
docker-compose up -d

# 验证服务
curl http://localhost:18008/health
```

### 基本操作

```bash
# 1. 创建记忆
curl -X POST http://localhost:18008/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "memories": [{
      "content": "这是一个测试记忆",
      "type": "note",
      "tags": ["test"]
    }],
    "tenant_id": "default"
  }'

# 2. 搜索记忆
curl -X POST http://localhost:18008/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "测试",
    "mode": "hybrid",
    "tenant_id": "default"
  }'

# 3. 获取 Embedding
curl -X POST http://localhost:18008/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "测试文本",
    "model": "Qwen3-Embedding-0.6B"
  }'
```

---

## 功能详情

### Embedding 服务

- **模型**: Qwen3-Embedding-0.6B
- **向量维度**: 2048
- **支持语言**: 中文、英文、代码
- **端口**: 18000

### 记忆搜索

| 搜索模式 | 说明 | 适用场景 |
|----------|------|----------|
| **keyword** | 关键词搜索 | 精确匹配 |
| **vector** | 向量搜索 | 语义相似 |
| **hybrid** | 混合搜索 | 综合效果 |

### WebSocket 实时同步

- **端点**: `ws://localhost:18008/ws/memories/live`
- **特性**: 心跳保活、消息确认、断线重连、DIFF 模式
- **使用场景**: 实时同步记忆变更

---

## 限制与已知问题

### 当前限制

1. **代码分析** - 预计算服务核心逻辑开发中，暂不可用
2. **HNSW 索引** - 统计和优化功能为 stub 实现
3. **缓存管理** - 缓存统计和清理功能为 stub 实现

### 计划功能

- [ ] 完整的代码分析流水线
- [ ] 图关系可视化
- [ ] 记忆聚类分析
- [ ] 智能推荐

---

## 支持与反馈

- **文档**: [docs/v3.2/](./v3.2/)
- **问题反馈**: GitHub Issues
- **版本历史**: [CHANGELOG.md](../CHANGELOG.md)

---

_文档版本: v3.2.0_  
_最后更新: 2026-04-15_
