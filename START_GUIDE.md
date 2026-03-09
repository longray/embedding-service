# 服务启动指南

## 🚀 统一启动脚本

使用 `start_services.py` 可以一键启动所有服务。

### 基本用法

```bash
# 方式1：只启动 Embedding + 包装层（推荐）
uv run python start_services.py

# 方式2：启动所有服务（Embedding + LLM + 包装层）
uv run python start_services.py --with-llm

# 方式3：只启动后端服务（测试用）
uv run python start_services.py --no-wrapper
uv run python start_services.py --with-llm --no-wrapper
```

### 启动流程

```
1. 启动 Embedding 服务（必需）
   ├── 端口：18000
   └── 等待就绪（约10-30秒）

2. 启动 LLM 服务（可选）
   ├── 端口：18001
   └── 等待就绪（约10-30秒）

3. 启动包装层服务（推荐）
   ├── 端口：17999
   └── 等待就绪（<5秒）
```

### 服务访问地址

启动成功后，可以通过以下地址访问：

| 服务 | 地址 | 说明 |
|------|------|------|
| **包装层** | http://localhost:17999 | 推荐使用（带缓存、熔断器） |
| Embedding | http://localhost:18000 | 直接访问后端 |
| LLM | http://localhost:18001 | 直接访问后端 |

### 停止服务

按 `Ctrl+C` 停止所有服务。

### 依赖关系

```
包装层服务 (17999)
    ├── 必须依赖：Embedding 服务 (18000)
    └── 可选依赖：LLM 服务 (18001)
```

### 故障排查

**问题1：Embedding服务启动失败**
- 检查端口18000是否被占用
- 检查模型文件是否存在
- 查看错误日志

**问题2：服务未能就绪**
- 首次启动需要下载模型（约1.2GB）
- GPU模式需要更长的启动时间
- 检查健康检查端点：`curl http://localhost:18000/health`

**问题3：包装层服务启动失败**
- 确保后端服务已启动
- 检查端口17999是否被占用
- 检查环境变量配置

## 📝 手动启动（不推荐）

如果需要手动启动各个服务：

```bash
# 终端1：Embedding服务
uv run python src/qwen3_embedding_service/embedding_service.py

# 终端2：LLM服务（可选）
uv run python src/qwen3_embedding_service/llm_service.py

# 终端3：包装层服务
cd wrapper-service
uv run python -m src.main
```

## 🎯 推荐配置

**开发环境**：
```bash
uv run python start_services.py
```
- 只启动必需的服务
- 快速启动，节省资源

**生产环境**：
```bash
uv run python start_services.py --with-llm
```
- 启动所有服务
- 提供完整功能

## ⚙️ 环境变量配置

启动前可以配置以下环境变量：

```bash
# Embedding服务
export EMB_MAX_BATCH_SIZE=256
export EMB_CACHE_SIZE=1000

# LLM服务
export LLM_CACHE_SIZE=100

# 包装层服务
export WRAPPER_PORT=17999
export WRAPPER_CACHE_MAX_SIZE=1000
export WRAPPER_CACHE_TTL=3600
```

详细配置说明见 `wrapper-service/README.md`。
