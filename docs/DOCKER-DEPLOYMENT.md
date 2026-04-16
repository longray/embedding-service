# Docker 部署指南

> 本文档面向使用者和运维人员，提供 Embedding Service 的 Docker 化部署完整指南。

**版本**: v2.5.0  
**更新日期**: 2026-04-01

---

## 架构概览

本项目将 4 个服务容器化编排，实现一键部署：

| 服务 | 端口 | 功能 |
|------|------|------|
| SurrealDB | 28002 | 向量数据库（记忆存储） |
| Meilisearch | 28003 | 全文搜索引擎（中文分词 + 代码搜索） |
| Embedding | 28000 | 文本向量化（GPU 加速） |
| Wrapper | 27999 | API 网关（统一入口） |

**启动顺序**：SurrealDB → Embedding → Meilisearch → Wrapper

**GPU 策略**：全部 GPU 资源分配给 Embedding，LLM 暂不启用

**数据持久化**：所有数据映射到宿主机 `docker-data/` 目录，容器销毁后数据不丢失。

---

## 前置条件

### 必需软件

| 软件 | 最低版本 | 说明 |
|------|----------|------|
| Docker Desktop for Windows | 4.20+ | 必须启用 WSL2 后端 |
| NVIDIA 显卡驱动 | ≥535 | 支持 CUDA 12.1 |
| NVIDIA Container Toolkit | 最新 | WSL2 内安装，使 Docker 能访问 GPU |
| 磁盘空间 | ≥20GB | 模型 + 数据 + Docker 镜像 |

### 验证 GPU 可用性

```bash
# 1. 检查 NVIDIA 驱动
nvidia-smi

# 2. 验证 Docker 能访问 GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

如果第二步输出与第一步相同的 GPU 信息，说明配置正确。

### WSL2 中安装 NVIDIA Container Toolkit

```bash
# 在 WSL2 终端中执行
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 一键部署

### 快速启动

```bash
# 1. 进入项目目录
cd D:\embedding_service

# 2. 创建数据目录
mkdir -p docker-data\surrealdb docker-data\meilisearch docker-data\models

# 3. 配置环境变量（可选，有默认值）
copy .env.example .env

# 4. 启动所有服务（SurrealDB → Embedding → Meilisearch → Wrapper）
docker-compose up -d

# 5. 查看启动日志
docker-compose logs -f
```

### 验证部署

```bash
# 确认所有服务健康
curl http://localhost:27999/health
# 检查 embedding_service.status 是否为 "healthy"
# 检查 surrealdb.status 是否为 "healthy"
# 检查 meilisearch.status 是否为 "available"

# 单独确认 Docker Embedding 健康（含 GPU 信息）
curl http://localhost:28000/health

# 端到端测试：通过 Wrapper 调用 Embedding
curl -X POST http://localhost:27999/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "hello world"}'
```

---

## 端口与数据目录

### 端口映射

| 服务 | 容器内端口 | 宿主机端口 | 访问地址 |
|------|-----------|-----------|----------|
| Wrapper | 18008 | 27999 | http://localhost:27999 |
| Embedding | 18000 | 28000 | http://localhost:28000 |
| SurrealDB | 18002 | 28002 | ws://localhost:28002/rpc |
| Meilisearch | 7700 | 28003 | http://localhost:28003 |

### 数据目录结构

```text
D:/embedding_service/docker-data/
├── surrealdb/          # SurrealDB 数据（rocksdb 格式）
└── meilisearch/        # Meilisearch 索引数据
```

模型文件通过 volume mount 从宿主机 `src/qwen3_embedding_service/models/` 直接挂载到容器的 `/models/`。

**备份建议**：定期备份 `docker-data/` 目录即可保留所有数据。

---

## 常用操作

### 查看服务状态

```bash
docker-compose ps
```

### 查看日志

```bash
# 所有服务
docker-compose logs -f

# 单个服务
docker-compose logs -f embedding
```

### 重启服务

```bash
docker-compose restart embedding
```

### 停止所有服务

```bash
docker-compose down
```

⚠️ `down` 不会删除数据，数据保留在 `docker-data/` 中。

### 完全清理（删除数据）

```bash
docker-compose down -v
rm -rf docker-data/

---

## 开发模式（Live-Reload）

Wrapper 服务支持 Live-Reload 开发模式，修改源码后自动重载，无需手动重启。

### 启用 Live-Reload

Live-Reload 默认已启用（`docker-compose.yml` 中 `WRAPPER_RELOAD=true`）。

```bash
# 1. 启动服务（首次需要构建镜像）
docker-compose up -d --build wrapper

# 2. 确认 Live-Reload 已启用
docker-compose logs wrapper | grep WatchFiles
# 期望输出: Started reloader process [1] using WatchFiles
# 期望输出: Will watch for changes in these directories: ['/app/wrapper/src']
```

### 工作流程

1. 在宿主机用编辑器修改 `wrapper/src/` 下的 `.py` 文件
2. 保存文件
3. uvicorn 自动检测变化并重载（通常 1-2 秒）
4. 查看 `docker-compose logs -f wrapper` 确认重载成功

### 关闭 Live-Reload（生产模式）

```bash
# 方法 1：修改 .env 文件
WRAPPER_RELOAD=false
docker-compose up -d wrapper

# 方法 2：从 docker-compose.yml 中移除 WRAPPER_RELOAD 环境变量
# 然后重建：docker-compose up -d --build wrapper
```

### Volume Mount 注意事项

| 注意事项 | 说明 |
|----------|------|
| Windows 路径 | Docker Desktop 会自动转换路径，无需手动处理 |
| `__pycache__` | Windows 编译的 `.pyc` 在 Linux 无效，Python 会自动重新编译 |
| 依赖变更 | 修改依赖需要重新构建镜像：`docker-compose build wrapper` |
| 文件权限 | Git Bash 环境下 volume mount 可能遇到权限问题，建议使用 Docker Desktop |

---

## 故障排查

### GPU 不可见

**症状**：Embedding 服务启动后无法使用 GPU，日志显示 `CUDA not available`

**排查步骤**：

```bash
# 1. 检查 NVIDIA Container Toolkit 是否安装
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 2. 检查 docker-compose.yml 中 GPU 配置
grep -A 5 "deploy:" docker-compose.yml

# 3. 检查容器内是否能检测到 GPU
docker exec embedding-service python -c "import torch; print(torch.cuda.is_available())"
```

**解决方案**：

1. 确保 NVIDIA 驱动 ≥535
2. 重新安装 NVIDIA Container Toolkit
3. 重启 Docker Desktop

### 服务启动失败

```bash
# 查看具体错误
docker-compose logs <service-name>

# 常见原因：
# - 端口被占用：netstat -ano | findstr <port>
# - 数据目录权限：确保 docker-data/ 可写
# - 模型下载失败：检查网络连接
```

### 模型首次启动慢

首次启动需要下载模型文件（Embedding ~1.2GB，LLM ~1GB），请耐心等待。

模型下载完成后会缓存到 `docker-data/models/`，后续启动只需几秒。

### Meilisearch 连接失败

Wrapper 服务连接 Meilisearch 失败时，检查：

```bash
# 1. Meilisearch 是否启动
curl http://localhost:28003/health

# 2. API Key 是否正确
grep MEILI_MASTER_KEY .env

# 3. Wrapper 日志中的连接地址
docker-compose logs wrapper | grep meili
```

---

## 环境变量配置

`.env` 文件中的关键配置：

```bash
# Meilisearch Master Key（生产环境必须修改）
MEILI_MASTER_KEY=masterKey_change_in_production

# 认证开关
WRAPPER_AUTH_ENABLED=false

# Live-Reload 开关（开发模式启用，生产环境关闭）
WRAPPER_RELOAD=true

# OpenTelemetry 追踪（可选）
WRAPPER_OTEL_ENABLED=false
```

完整配置参考 `.env.example`。

---

## 与 bat 脚本的对比

| 特性 | bat 脚本 | Docker Compose |
|------|----------|----------------|
| 部署方式 | 手动安装各组件 | 一键启动 |
| 环境隔离 | 无 | 完全隔离 |
| GPU 支持 | 直接使用 | 需配置 NVIDIA Container Toolkit |
| 数据持久化 | 本地目录 | docker-data/ 目录 |
| 端口 | 18008/18000/18002/18003 | +10000（28008/28000/28002/28003） |
| 启动顺序 | 串行等待健康检查 | depends_on + healthcheck 自动保证 |
| 适用场景 | 开发调试 | 生产部署 / 长期运行 |

---

*本文档与 `docker-compose.yml` 保持同步更新*
