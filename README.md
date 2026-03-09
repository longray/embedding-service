# Embedding Service (OpenCode Memory Stack)

版本与路线图
- 当前版本: v1.0.0
- 实施阶段: P0 + P1 + P2 已完成，进入 P3 优化阶段
- 详细路线见 ROADMAP.md

## 开发状态

**当前版本**: v1.0.0
**实施阶段**: P0 + P1 + P2 已完成，进入 P3 优化阶段

### 已完成 ✅
- ✅ P0 核心功能（Embedding + LLM + 包装层）
- ✅ P1 增强功能（熔断器、缓存、监控、测试套件）
- ✅ P2 生产就绪（API认证授权、CI/CD、完整文档）

### P3 优化路线图 🚀

| 优先级 | 功能 | 预期收益 | 状态 |
|--------|------|----------|------|
| P3-1 | Docker Compose | 一键部署 | ⏳ 待开始 |
| P3-2 | HNSW向量索引 | 搜索10x加速 | ⏳ 待开始 |
| P3-3 | 监控告警 | 自动告警 | ⏳ 待开始 |
| P3-4 | Kubernetes | 云原生部署 | ⏳ 待开始 |
| P3-5 | 审计日志 | 合规审计 | ⏳ 待开始 |

查看 [ROADMAP.md](ROADMAP.md) 了解详细计划。

## API端点

### API端点

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
```

### 核心功能
- ✅ **记忆管理**：SurrealDB向量存储，支持混合搜索
- ✅ **API认证**：API Key认证和权限控制
- ✅ **CI/CD**：GitHub Actions自动测试
- 其他现有核心功能保持向后兼容

## 技术要求与兼容性
- 保持向后兼容及现有接口
- 认证开关可通过环境变量控制
- 兼容现有文档结构，方便跳转至 ROADMAP.md

## 文件位置
D:\embedding_service\README.md

## 验证
- Markdown 语法正确性检查
- 通过浏览器打开或在 CI 中渲染 README.md

<!-- OMO_INTERNAL_INITIATOR -->
