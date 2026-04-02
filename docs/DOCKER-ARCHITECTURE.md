# Docker 架构设计

> 本文档面向开发者，记录 Docker 化改造的设计决策、技术细节和维护指南。

**版本**: v2.5.0  
**更新日期**: 2026-04-01

---

## 设计原则

1. **不干扰开发环境**：Docker 端口全部 +10000，与本地 bat 脚本服务并行运行
2. **全部服务 Docker 化**：所有服务（含 Embedding）均在 Docker 中运行
3. **环境一致性**：容器内端口保持原定义，仅宿主机映射端口偏移
4. **GPU 全力保障 Embedding**：LLM 暂不启用，全部 GPU 资源分配给 Embedding
5. **启动顺序严格匹配 bat**：surrealdb → embedding → meilisearch → wrapper

---

## 端口策略

### 容器内 vs 宿主机

| 服务 | 容器内端口 | 宿主机端口 | 说明 |
|------|-----------|-----------|------|
| Wrapper | 17999 | 27999 | 容器间通信用 17999 |
| Embedding | 18000 | 28000 | GPU 加速推理 |
| LLM | 18001 | 28001 | 可选，显存不足时关闭 |
| SurrealDB | 18002 | 28002 | 数据持久化到 docker-data/ |
| Meilisearch | 7700 | 28003 | 容器内默认 7700 |

### 为什么容器内保持原端口

- 容器间通过 Docker 网络通信，使用容器名解析（如 `http://embedding:18000`）
- 仅宿主机映射端口偏移，避免修改服务内部配置
- 与 bat 脚本的端口定义保持一致，降低维护成本

---

## 基础镜像选型

### 决策：`nvidia/cuda:12.1.0-runtime-ubuntu22.04`

**为什么不用 `python:3.11-slim`**：

| 对比项 | python:3.11-slim | nvidia/cuda:12.1.0-runtime |
|--------|------------------|---------------------------|
| CUDA Runtime | ❌ 无 | ✅ 内置 |
| GPU 支持 | 需手动安装 CUDA | 开箱即用 |
| 镜像大小 | ~150MB | ~1.5GB |
| Python | 需安装 | 需安装 |

**结论**：GPU 推理服务必须使用 CUDA 基础镜像，否则即使挂载 GPU 也无法调用。

### Python 3.11 安装

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 安装 Python 3.11（与 .venv 版本一致）
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install -y python3.11 python3.11-venv python3.11-dev curl \
    && rm -rf /var/lib/apt/lists/*

# 设置 python3 指向 python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
```

**版本一致性**：当前 `.venv` 使用 Python 3.11，容器内必须保持一致，避免依赖兼容性问题。

---

## 依赖管理策略

### 双源配置

```dockerfile
# 1. 普通包：阿里云 PyPI 镜像
RUN uv pip install --system -e "." \
    --index-url https://mirrors.aliyun.com/pypi/simple/

# 2. PyTorch CUDA：阿里云 pytorch-wheels 镜像
RUN uv pip install --system \
    torch==2.4.0+cu121 \
    torchvision==0.19.0+cu121 \
    torchaudio==2.4.0+cu121 \
    --find-links https://mirrors.aliyun.com/pytorch-wheels/cu121/
```

### 为什么不用 `.venv` 直接挂载

| 方案 | 优点 | 缺点 |
|------|------|------|
| 挂载 .venv | 快速 | Windows wheel 不兼容 Linux |
| 容器内重建 | 环境一致 | 首次构建慢 |

**结论**：`.venv` 是 Windows 环境编译的，无法在 Linux 容器中运行。必须在 Dockerfile 中重建。

### 已验证的包源

| 源 | URL | 状态 |
|-----|-----|------|
| 阿里云 PyPI | `https://mirrors.aliyun.com/pypi/simple/` | ✅ 200 OK |
| 阿里云 PyTorch Wheels | `https://mirrors.aliyun.com/pytorch-wheels/cu121/` | ✅ 200 OK |

### Wrapper 依赖策略

Wrapper 服务采用"依赖在镜像层 + 源码实时映射"的分层策略：

```dockerfile
# Dockerfile.wrapper - 依赖安装在镜像层（构建时）
RUN uv pip install --system "uvicorn[standard]>=0.30" fastapi ...

# docker-compose.yml - 源码实时映射（运行时）
volumes:
  - ./wrapper/src:/app/wrapper/src
```

| 层级 | 内容 | 更新频率 | 策略 |
|------|------|----------|------|
| 依赖 | fastapi, uvicorn, surrealdb 等 | 低 | 镜像层（`RUN pip install`） |
| 源码 | wrapper/src/*.py | 高 | volume mount（实时映射） |
| 脚本 | scripts/*.surql | 中 | volume mount（实时映射） |

**好处**：改源码只需 `docker compose restart wrapper`（或 Live-Reload 自动重载），不需要 `docker compose build`。

---

## GPU 支持原理

### Docker Compose 配置

```yaml
services:
  embedding:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 工作原理

1. `nvidia-container-toolkit` 在容器启动时注入 NVIDIA 驱动库
2. 容器内无需安装 NVIDIA 驱动，只需 CUDA runtime
3. PyTorch 通过 `torch.cuda.is_available()` 检测 GPU

### GTX 1060 注意事项

| 参数 | 值 | 影响 |
|------|-----|------|
| 显存 | 6GB | Embedding (0.6B) ~2GB, LLM (0.5B) ~1.5GB，同时运行可能紧张 |
| 计算能力 | 6.1 (Pascal) | CUDA 12.1 支持，但无 Tensor Core |
| 推荐策略 | 先启用 Embedding，LLM 按需 | 避免 OOM |

---

## 服务通信

### Docker 网络

```yaml
networks:
  embedding-network:
    driver: bridge
```

所有服务加入同一网络，通过容器名互相访问：

```python
# Wrapper 配置示例
WRAPPER_EMBEDDING_SERVICE_URL=http://embedding:18000
WRAPPER_SURREALDB_URL=ws://surrealdb:18002/rpc
WRAPPER_MEILI_URL=http://meilisearch:7700
```

### 启动顺序

```text
surrealdb (无依赖)
    ↓ health check 通过
embedding (depends_on: surrealdb healthy)
    ↓ health check 通过
meilisearch (depends_on: surrealdb healthy)
    ↓ health check 通过
wrapper (depends_on: surrealdb + embedding + meilisearch)
    → 通过 http://embedding:18000 访问 Docker Embedding
```

docker-compose 的 `depends_on` + `condition: service_healthy` 确保严格按此顺序启动。

### 容器间通信

所有服务加入同一 Docker bridge 网络（`embedding-network`），通过容器名互相访问：

```python
# Wrapper 配置（当前生效）
WRAPPER_EMBEDDING_SERVICE_URL=http://embedding:18000
WRAPPER_SURREALDB_URL=ws://surrealdb:18002/rpc
WRAPPER_MEILI_URL=http://meilisearch:7700
```

---

## Live-Reload 开发模式

Wrapper 服务支持 Live-Reload，开发者在宿主机修改源码后容器自动重载。

### 实现方式

```yaml
# docker-compose.yml
wrapper:
  environment:
    - WRAPPER_RELOAD=true
  volumes:
    - ./wrapper/src:/app/wrapper/src
    - ./scripts:/app/scripts
```

```python
# wrapper/src/main.py
reload_enabled = os.getenv("WRAPPER_RELOAD", "false").lower() == "true"
uvicorn.run(
    "wrapper.src.main:app" if reload_enabled else app,
    host=config.host,
    port=config.port,
    reload=reload_enabled,
    reload_dirs=["/app/wrapper/src"] if reload_enabled else None,
)
```

### 设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 文件监控 | watchfiles（纯 Rust） | 比 watchdog 更快、跨平台、无原生依赖 |
| uvicorn 版本 | `uvicorn[standard]>=0.30`（0.42.0） | standard 版包含 watchfiles 和 httptools |
| 传入方式 | import string `"wrapper.src.main:app"` | uvicorn reload 模式要求传入字符串而非 app 对象 |
| 监控目录 | 仅 `/app/wrapper/src` | 排除 `__pycache__`、日志等非源码变化 |
| 控制开关 | `WRAPPER_RELOAD` 环境变量 | 生产环境设为 `false` 或不设置即可关闭 |

### Windows \_\_pycache\_\_ 兼容性

宿主机 Windows 生成的 `.pyc` 文件与 Linux 容器不兼容。解决方案：

1. Python 在容器中运行时自动重新编译 `.py` 文件
2. `reload_dirs` 仅监控 `.py` 文件变化，`__pycache__` 变化不会触发重载
3. 无需手动清理宿主机的 `__pycache__`

### 关闭 Live-Reload

生产部署时关闭热重载，避免不必要的文件监控开销：

```bash
# 方法 1：修改 .env
WRAPPER_RELOAD=false

# 方法 2：从 docker-compose.yml 移除 WRAPPER_RELOAD 环境变量
```

---

## 数据持久化

### Bind Mounts vs Named Volumes

| 方案 | 优点 | 缺点 |
|------|------|------|
| Named Volumes | Docker 管理 | 数据位置不直观 |
| Bind Mounts | 路径明确 | 需手动创建目录 |

**决策**：使用 Bind Mounts，数据统一归拢到 `docker-data/`：

```yaml
volumes:
  - D:/embedding_service/docker-data/surrealdb:/data
  - D:/embedding_service/docker-data/meilisearch:/meili_data
  - D:/embedding_service/docker-data/models:/models
```

---

## 健康检查

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:18000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s  # Embedding 模型加载需要时间
```

**注意**：Embedding 服务的 `start_period` 设为 60s，给模型加载留出时间。

---

## 构建与测试

### 本地调试

```bash
# 构建单个服务
docker-compose build embedding

# 启动并查看日志
docker-compose up embedding

# 进入容器调试
docker exec -it embedding-service bash
```

### GPU 验证

```bash
# 容器内验证 GPU
docker exec embedding-service python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
print(f'Device name: {torch.cuda.get_device_name(0)}')
"
```

### 性能对比

| 指标 | 本地 (.venv) | Docker |
|------|-------------|--------|
| Embedding 延迟 | ~211ms | 待测试 |
| 搜索延迟 | ~102ms | 待测试 |
| 启动时间 | ~30s | ~60s（含模型加载） |

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `docker-compose.yml` | 服务编排（4 服务 + jaeger tracing profile） |
| `Dockerfile.embedding` | Embedding CUDA 镜像（nvidia/cuda:12.1.0） |
| `wrapper/Dockerfile` | Wrapper 镜像（python:3.11-slim + uvicorn[standard]） |
| `Dockerfile.llm` | LLM 服务镜像（暂未启用） |
| `.dockerignore` | 构建上下文排除规则 |
| `docker-start.bat` | Windows 一键启动脚本 |
| `docker-stop.bat` | Windows 一键停止脚本 |
| `.env` | 环境变量 |
| `docker-data/` | 数据持久化目录（surrealdb/ + meilisearch/） |

---

## 迁移路径

```text
阶段 A（调试期）✅ 已完成:
  SurrealDB    → Docker (28002)
  Meilisearch  → Docker (28003)
  Embedding    → 本地 (18000)  ← Docker embedding 不启动（profiles: [gpu]）
  Wrapper      → Docker (27999)，通过 host.docker.internal:18000 访问本地 Embedding
  LLM          → 暂不启用

阶段 B（切换期）✅ 已完成:
  停止本地 Embedding
  启动 Docker Embedding（移除 profiles: [gpu]，默认启用）
  改 .env: WRAPPER_EMBEDDING_SERVICE_URL=http://embedding:18000
  重启 Wrapper

阶段 C（稳定期）← 当前阶段:
  全部服务 Docker 化
  端口保持 +10000（不干扰其他开发环境）
  LLM 服务待算力充足后启用
```

---

*本文档与代码实现保持同步更新*
