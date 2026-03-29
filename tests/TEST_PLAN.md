# 接口测试计划

## 测试目标

对三个服务的所有API接口进行全面、严格的测试，确保：

- 功能正确性
- 错误处理完善
- 性能符合预期
- 服务依赖关系正确

## 测试范围

### 1. Embedding服务（端口18000）

| 接口 | 测试用例 | 优先级 |
|------|---------|--------|
| POST /v1/embeddings | 正常请求、批量请求、空输入、超长文本、无效参数 | P0 |
| GET /health | 服务状态、GPU信息 | P0 |
| GET /v1/models | 模型列表 | P1 |
| GET /stats | 缓存统计 | P1 |

### 2. LLM服务（端口18001）

| 接口 | 测试用例 | 优先级 |
|------|---------|--------|
| POST /v1/chat/completions | 正常对话、多轮对话、参数调整、无效消息 | P0 |
| POST /generate | 简单生成、缓存测试 | P1 |
| GET /health | 服务状态、GPU信息 | P0 |
| GET /v1/models | 模型列表 | P1 |
| GET /stats | 缓存统计 | P1 |

### 3. 包装层服务（端口3001）

| 接口 | 测试用例 | 优先级 |
|------|---------|--------|
| POST /v1/embeddings | 缓存命中、缓存未命中、熔断器测试 | P0 |
| POST /v1/chat/completions | 熔断器测试、后端故障处理 | P0 |
| GET /health | 健康状态、熔断器状态 | P0 |
| GET /metrics | Prometheus指标 | P1 |

## 测试类型

### 功能测试

- ✅ 正常流程测试
- ✅ 边界条件测试
- ✅ 异常处理测试

### 性能测试

- ✅ 响应时间测试
- ✅ 并发请求测试
- ✅ 缓存性能测试

### 集成测试

- ✅ 服务依赖测试
- ✅ 熔断器功能测试
- ✅ 端到端流程测试

## 测试工具

- **pytest**: 测试框架
- **httpx**: HTTP客户端
- **pytest-asyncio**: 异步测试
- **pytest-timeout**: 超时控制

## 测试文件结构

```text
tests/
├── test_embedding_service.py    # Embedding服务测试
├── test_llm_service.py           # LLM服务测试
├── test_wrapper_service.py       # 包装层服务测试
├── test_integration.py           # 集成测试
├── conftest.py                   # 测试配置和fixtures
└── README.md                     # 测试说明
```

## 执行计划

1. **阶段1**：创建测试框架和配置
2. **阶段2**：实现后端服务测试
3. **阶段3**：实现包装层服务测试
4. **阶段4**：实现集成测试
5. **阶段5**：性能测试和报告
