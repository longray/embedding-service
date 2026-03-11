# 接口测试文档

## 测试概述

本测试套件对 Embedding 服务的所有 API 接口进行全面、严格的端到端测试，包括：
- Embedding 服务（端口 18000）
- LLM 服务（端口 18001）
- 包装层服务（端口 3001）
- **最小化包装服务（端口 17999）** - 新增
- **Meilisearch 集成测试** - v2.3.0 新增

## 测试类型

### 基础功能测试
- 健康检查、API 端点、基本功能

### 扩展测试
- **边界条件测试**：超长文本、空输入、特殊字符、大批量
- **错误处理测试**：无效参数、缺失字段、错误类型
- **熔断器测试**：状态转换、故障恢复
- **缓存测试**：命中/未命中、一致性、隔离
- **性能测试**：并发、响应时间、吞吐量
- **安全测试**：注入攻击、恶意输入、安全防护

## 前置条件

### 1. 安装测试依赖

```bash
uv pip install pytest pytest-asyncio httpx
```

### 2. 启动所有服务

```bash
# 使用统一启动脚本
uv run python start_services.py --with-llm

# 或单独启动最小化包装服务
uv run python -m wrapper.src.main
```

## 运行测试

### 运行所有测试

```bash
uv run pytest tests/ -v
```

### 运行最小化包装服务测试（推荐）

```bash
# 运行核心 API 测试套件（56 个测试）
uv run pytest tests/test_wrapper_api.py -v

# 运行特定测试类
uv run pytest tests/test_wrapper_api.py::TestEmbeddingsCache -v
uv run pytest tests/test_wrapper_api.py::TestMemoriesSearchBasic -v
uv run pytest tests/test_wrapper_api.py::TestEndToEndIntegration -v
```

### 运行特定类型的测试

```bash
# 基础功能测试
uv run pytest tests/test_embedding_service.py tests/test_llm_service.py tests/test_wrapper_service.py -v

# 扩展测试（边界条件和错误处理）
uv run pytest tests/test_embedding_service_extended.py tests/test_llm_service_extended.py tests/test_wrapper_service_extended.py -v

# 性能测试
uv run pytest tests/test_performance.py -v

# 安全测试
uv run pytest tests/test_security.py -v

# 集成测试
uv run pytest tests/test_integration.py -v
```

### 运行特定服务的所有测试

```bash
# Embedding服务（基础+扩展）
uv run pytest tests/test_embedding_service*.py -v

# LLM服务（基础+扩展）
uv run pytest tests/test_llm_service*.py -v

# 包装层服务（基础+扩展）
uv run pytest tests/test_wrapper_service*.py -v
```

### 运行特定测试用例

```bash
uv run pytest tests/test_embedding_service.py::TestEmbeddingService::test_health_check -v
```

## 测试覆盖

### 最小化包装服务（端口 17999）- 新增

**测试文件**: `tests/test_wrapper_api.py`（56 个测试）

| 测试类 | 测试数 | 覆盖内容 |
|--------|--------|----------|
| TestHealthEndpoint | 5 | 健康检查、服务状态、缓存统计 |
| TestEmbeddingsBasic | 3 | 单文本嵌入、默认模型、usage 字段 |
| TestEmbeddingsCache | 5 | 缓存命中/未命中、一致性、性能 |
| TestEmbeddingsBoundary | 4 | 空字符串、特殊字符、超长文本 |
| TestEmbeddingsErrors | 4 | 缺失字段、无效类型、格式错误 |
| TestMemoriesUploadBasic | 4 | 单个/批量上传、ID 格式、元数据 |
| TestMemoriesUploadBoundary | 5 | 空列表、无内容、大批量、特殊字符 |
| TestMemoriesUploadErrors | 3 | 缺失字段、无效类型 |
| TestMemoriesSearchBasic | 5 | 关键词/向量/混合搜索、结果结构 |
| TestMemoriesSearchParams | 4 | limit、threshold、默认值 |
| TestMemoriesSearchBoundary | 4 | 空查询、特殊字符、边界值 |
| TestMemoriesSearchErrors | 4 | 缺失字段、无效模式、超出范围 |
| TestEndToEndIntegration | 2 | 上传→搜索验证、多次上传 |
| TestPerformance | 4 | 响应时间、并发测试 |

**运行方式**:
```bash
uv run pytest tests/test_wrapper_api.py -v
```

### Embedding 服务
**基础测试** (6个):
- ✅ 健康检查
- ✅ 单个文本嵌入
- ✅ 批量文本嵌入
- ✅ 空输入处理
- ✅ 模型列表
- ✅ 统计信息

**扩展测试** (11个):
- ✅ 超长文本（~8000字符）
- ✅ 空字符串
- ✅ 特殊字符（emoji、Unicode）
- ✅ 大批量（50条）
- ✅ 仅空白字符
- ✅ 缺失必需字段
- ✅ 无效数据类型
- ✅ 空数组输入
- ✅ null输入
- ✅ 格式错误的JSON
- ✅ 无效模型名

### LLM服务
**基础测试** (5个):
- ✅ 健康检查
- ✅ 聊天补全
- ✅ 简单生成
- ✅ 模型列表
- ✅ 统计信息

**扩展测试** (13个):
- ✅ 超长消息
- ✅ 空消息内容
- ✅ 特殊字符
- ✅ 多轮对话
- ✅ max_tokens参数
- ✅ 缺失必需字段
- ✅ 无效消息类型
- ✅ 空消息数组
- ✅ 无效role
- ✅ 缺失content字段
- ✅ null消息
- ✅ 无效max_tokens
- ✅ 缺失model字段

### 包装层服务
**基础测试** (4个):
- ✅ 健康检查
- ✅ 嵌入接口（缓存测试）
- ✅ 聊天接口
- ✅ Prometheus指标

**扩展测试** (9个):
- ✅ 熔断器状态验证
- ✅ 熔断器关闭状态
- ✅ 成功请求通过包装层
- ✅ 缓存命中/未命中
- ✅ 缓存隔离（不同输入）
- ✅ 批量输入缓存
- ✅ 缓存统计
- ✅ Prometheus指标端点
- ✅ 请求后指标更新

### 性能测试 (7个)
- ✅ 单个请求响应时间（<2秒）
- ✅ 并发请求（10个）
- ✅ 批量处理（100条）
- ✅ 聊天响应时间（<5秒）
- ✅ 并发聊天（5个）
- ✅ 缓存性能提升
- ✅ 包装层开销（<100ms）

### 安全测试 (10个)
- ✅ SQL注入防护
- ✅ XSS攻击防护
- ✅ 命令注入防护
- ✅ 超大payload处理
- ✅ 深度嵌套JSON
- ✅ null字节处理
- ✅ CORS头
- ✅ Content-Type验证
- ✅ 快速连续请求
- ✅ 速率限制（如果实现）

### Meilisearch 集成测试 (23个) - v2.3.0 新增

**测试文件**: `tests/test_meili_integration.py`

| 测试类 | 测试数 | 覆盖内容 |
|--------|--------|----------|
| TestMeiliClientInit | 3 | 客户端初始化、禁用状态、默认配置 |
| TestMeiliClientIndexManagement | 3 | 索引创建、幂等性、设置配置 |
| TestMeiliClientDocumentOperations | 4 | 文档上传、批量操作、删除 |
| TestMeiliClientSearch | 5 | 关键词搜索、中文搜索、日期搜索、结果格式 |
| TestMemoryManagerMeiliIntegration | 4 | 双写同步、搜索路由、降级回退 |
| TestMeiliSearchRouting | 4 | keyword→Meilisearch、vector→SurrealDB、hybrid→RRF |

**运行方式**:
```bash
# 运行 Meilisearch 集成测试
uv run pytest tests/test_meili_integration.py -v
```

**前置条件**:
- Meilisearch 1.4+ 运行中（默认端口 7700）
- 配置 `WRAPPER_MEILI_ENABLED=true` 环境变量
- 不需要 Meilisearch 也可运行（mock 测试）
### 集成测试 (2个)

## 测试报告

生成HTML测试报告：

```bash
uv pip install pytest-html
uv run pytest tests/ --html=report.html --self-contained-html
```

## 故障排查

### 测试失败：连接被拒绝

**原因**：服务未启动

**解决**：
```bash
uv run python start_services.py --with-llm
```

### 测试超时

**原因**：服务响应慢或未就绪

**解决**：等待服务完全启动（约30-60秒）

## 持续集成

在CI/CD中运行测试：

```bash
# 启动服务（后台）
uv run python start_services.py --with-llm &
sleep 60  # 等待服务就绪

# 运行测试
uv run pytest tests/ -v --tb=short

# 停止服务
pkill -f "python.*embedding_service"
pkill -f "python.*llm_service"
pkill -f "python.*wrapper.*main"
```
