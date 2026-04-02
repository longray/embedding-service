# GPU-Enabled Embedding Service 部署完整方案

## 📋 项目概述
部署一个支持NVIDIA GPU的embedding服务，用于高效的文本向量化处理。本方案包含完整的Docker配置、源代码框架和验证步骤。

---

## 🔧 系统要求

### 硬件要求
- NVIDIA GPU (compute capability >= 7.5)
  - 支持的型号: Tesla T4/V100/A100, RTX 2060/3090/4090 等
- 最小8GB GPU显存 (建议16GB+)
- 主机内存8GB+ (建议16GB+)

### 软件要求
- **开发环境**: Docker Desktop 4.29+ (WSL2后端)
- **生产环境**: Docker Engine + NVIDIA Container Runtime
- NVIDIA GPU驱动: 470.0+ 版本
- CUDA工具包: 11.8+ (NVIDIA镜像已包含)

---

## 📁 项目结构

```
embedding-service/
├── Dockerfile                 # 多阶段构建Dockerfile
├── docker-compose.yml         # Docker Compose配置
├── requirements.txt           # Python依赖
├── .dockerignore             # 构建忽略文件
├── src/
│   ├── app.py               # FastAPI应用入口
│   ├── embedding_model.py   # Embedding模型加载/推理
│   ├── config.py            # 配置管理
│   └── utils.py             # 工具函数
├── models/                   # 模型存储目录(volume挂载)
│   └── .gitkeep
└── tests/
    ├── test_api.py          # API测试
    └── test_gpu.py          # GPU功能测试
```

---

## 🐳 Docker配置文件

### 1️⃣ Dockerfile (多阶段构建)

```dockerfile
# syntax=docker/dockerfile:1

# ============ 构建阶段 ============
FROM python:3.11-slim as builder

WORKDIR /tmp

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements并生成wheels
COPY requirements.txt .
RUN pip install --user --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --user --no-cache-dir -r requirements.txt

# ============ 运行阶段 ============
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制wheels
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 复制应用代码
COPY src/ ./src/
COPY config.py ./

# 创建模型目录
RUN mkdir -p /app/models

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    EMBEDDING_MODEL_PATH=/app/models

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# 开放端口
EXPOSE 8000

# 启动应用
CMD ["python3", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2️⃣ docker-compose.yml

```yaml
version: '3.9'

services:
  embedding-service:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: embedding-service
    image: embedding-service:latest
    
    # 端口映射
    ports:
      - "8000:8000"
    
    # GPU配置 (生产环境使用)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1                    # 使用1个GPU
              capabilities: [gpu]
    
    # 环境变量
    environment:
      - PYTHONUNBUFFERED=1
      - CUDA_VISIBLE_DEVICES=0
      - LOG_LEVEL=INFO
      - MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
      - BATCH_SIZE=32
    
    # 数据卷挂载
    volumes:
      - ./models:/app/models:rw              # 模型持久化
      - ./logs:/app/logs:rw                  # 日志持久化
    
    # 重启策略
    restart: unless-stopped
    
    # 资源限制
    mem_limit: 8g
    shm_size: 2gb                           # 共享内存(GPU通信需要)
    
    # 日志配置
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

### 3️⃣ .dockerignore

```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv
.git
.gitignore
.env
.env.local
.DS_Store
*.log
logs/
*.egg-info/
dist/
build/
.pytest_cache/
.vscode/
.idea/
```

---

## 🐍 Python应用代码

### 1️⃣ requirements.txt

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.1.1
sentence-transformers==2.2.2
numpy==1.24.3
pydantic==2.5.0
python-dotenv==1.0.0
requests==2.31.0
pydantic-settings==2.1.0
```

### 2️⃣ src/config.py

```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """应用配置"""
    
    # 应用基本配置
    APP_NAME: str = "Embedding Service"
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL: str = "INFO"
    
    # GPU和模型配置
    CUDA_VISIBLE_DEVICES: Optional[str] = "0"
    EMBEDDING_MODEL_PATH: str = "/app/models"
    MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # 推理配置
    BATCH_SIZE: int = 32
    MAX_SEQUENCE_LENGTH: int = 512
    DEVICE: str = "cuda"  # 自动检测cuda或cpu
    
    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

### 3️⃣ src/embedding_model.py

```python
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging
from .config import settings

logger = logging.getLogger(__name__)

class EmbeddingModelManager:
    """Embedding模型管理器"""
    
    _instance: Optional['EmbeddingModelManager'] = None
    _model: Optional[SentenceTransformer] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self._load_model()
    
    def _load_model(self):
        """加载embedding模型"""
        logger.info(f"Loading model: {settings.MODEL_NAME}")
        
        try:
            self._model = SentenceTransformer(
                settings.MODEL_NAME,
                device=settings.DEVICE,
                cache_folder=settings.EMBEDDING_MODEL_PATH
            )
            logger.info(f"Model loaded successfully on device: {settings.DEVICE}")
            logger.info(f"GPU available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def encode(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """编码文本为embedding向量"""
        if batch_size is None:
            batch_size = settings.BATCH_SIZE
        
        logger.info(f"Encoding {len(texts)} texts with batch_size={batch_size}")
        
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": settings.MODEL_NAME,
            "embedding_dimension": self._model.get_sentence_embedding_dimension(),
            "device": str(self._model.device),
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }

# 单例实例
embedding_manager = EmbeddingModelManager()
```

### 4️⃣ src/app.py

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import logging
import torch
import time

from .config import settings
from .embedding_model import embedding_manager

# 配置日志
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============ 数据模型 ============

class EmbeddingRequest(BaseModel):
    """Embedding请求模型"""
    texts: List[str]
    batch_size: int = settings.BATCH_SIZE
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["Hello world", "This is a test"],
                "batch_size": 32
            }
        }

class EmbeddingResponse(BaseModel):
    """Embedding响应模型"""
    embeddings: List[List[float]]
    dimension: int
    count: int
    processing_time_ms: float

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    gpu_available: bool
    gpu_name: str
    model_loaded: bool

# ============ API端点 ============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return HealthResponse(
        status="healthy",
        gpu_available=torch.cuda.is_available(),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        model_loaded=embedding_manager._model is not None
    )

@app.post("/embed", response_model=EmbeddingResponse)
async def embed(request: EmbeddingRequest):
    """生成embedding向量"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts cannot be empty")
    
    if len(request.texts) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 texts per request")
    
    try:
        start_time = time.time()
        embeddings = embedding_manager.encode(request.texts, request.batch_size)
        processing_time_ms = (time.time() - start_time) * 1000
        
        return EmbeddingResponse(
            embeddings=embeddings,
            dimension=len(embeddings[0]) if embeddings else 0,
            count=len(embeddings),
            processing_time_ms=processing_time_ms
        )
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def model_info():
    """获取模型信息"""
    return embedding_manager.get_model_info()

@app.post("/batch-embed")
async def batch_embed(request: EmbeddingRequest):
    """批量embedding (大文本集)"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts cannot be empty")
    
    try:
        start_time = time.time()
        embeddings = embedding_manager.encode(request.texts, request.batch_size)
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Batch processed: {len(embeddings)} embeddings in {processing_time_ms:.2f}ms")
        
        return {
            "embeddings": embeddings,
            "count": len(embeddings),
            "processing_time_ms": processing_time_ms,
            "throughput_texts_per_second": len(embeddings) / (processing_time_ms / 1000)
        }
    except Exception as e:
        logger.error(f"Batch embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """根端点"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "embed": "/embed",
            "batch_embed": "/batch-embed",
            "model_info": "/model-info"
        }
    }

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info(f"Shutting down {settings.APP_NAME}")
    torch.cuda.empty_cache()
```

### 5️⃣ src/utils.py

```python
import torch
import logging

logger = logging.getLogger(__name__)

def check_gpu_availability() -> dict:
    """检查GPU可用性和信息"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": props.total_memory / 1e9,
                "current_memory_gb": torch.cuda.memory_allocated(i) / 1e9,
                "reserved_memory_gb": torch.cuda.memory_reserved(i) / 1e9
            })
    
    return info

def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")

def log_gpu_memory_usage():
    """记录GPU内存使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            logger.info(f"GPU {i}: Allocated {allocated:.2f}GB, Reserved {reserved:.2f}GB")
```

---

## 🚀 部署步骤

### 本地开发环境 (Docker Desktop)

```bash
# 1. 克隆项目
git clone <your-repo>
cd embedding-service

# 2. 构建镜像
docker build -t embedding-service:latest .

# 3. 运行容器 (开发模式，使用--device)
docker run --device nvidia.com/gpu=all \
  -d \
  --name embedding-service \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e LOG_LEVEL=INFO \
  embedding-service:latest

# 4. 查看日志
docker logs -f embedding-service
```

### 使用Docker Compose (推荐)

```bash
# 1. 启动服务
docker compose up -d

# 2. 查看服务状态
docker compose ps

# 3. 查看日志
docker compose logs -f embedding-service

# 4. 停止服务
docker compose down

# 5. 清理（包括卷）
docker compose down -v
```

### 生产环境 (Docker Engine + NVIDIA Runtime)

```bash
# 1. 安装NVIDIA Container Runtime
# Ubuntu/Debian:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - && \
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list && \
sudo apt-get update && sudo apt-get install -y nvidia-container-runtime

# 2. 构建镜像
docker build -t your-registry/embedding-service:v1.0 .

# 3. 推送到registry
docker push your-registry/embedding-service:v1.0

# 4. 运行容器 (使用--gpus)
docker run --gpus all \
  -d \
  --name embedding-service \
  -p 8000:8000 \
  -v /data/models:/app/models \
  -v /data/logs:/app/logs \
  --restart unless-stopped \
  your-registry/embedding-service:v1.0
```

---

## ✅ 验证和测试

### 1️⃣ 健康检查
```bash
curl http://localhost:8000/health
```

预期响应:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA A100",
  "model_loaded": true
}
```

### 2️⃣ 模型信息
```bash
curl http://localhost:8000/model-info
```

### 3️⃣ 测试Embedding
```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Hello world",
      "This is a test",
      "Embedding service works"
    ],
    "batch_size": 32
  }'
```

预期响应:
```json
{
  "embeddings": [[...], [...], [...]],
  "dimension": 384,
  "count": 3,
  "processing_time_ms": 123.45
}
```

### 4️⃣ 访问API文档
```
http://localhost:8000/docs          # Swagger UI
http://localhost:8000/redoc         # ReDoc
```

### 5️⃣ Python测试脚本

```python
# test_embedding_service.py
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """测试健康检查"""
    resp = requests.get(f"{BASE_URL}/health")
    print("Health Check:", json.dumps(resp.json(), indent=2))

def test_embedding():
    """测试embedding生成"""
    texts = [
        "The quick brown fox",
        "Machine learning with GPU",
        "Deep learning models"
    ]
    
    payload = {
        "texts": texts,
        "batch_size": 32
    }
    
    start = time.time()
    resp = requests.post(f"{BASE_URL}/embed", json=payload)
    elapsed = time.time() - start
    
    result = resp.json()
    print(f"Embedding Test ({elapsed:.3f}s):")
    print(f"- Count: {result['count']}")
    print(f"- Dimension: {result['dimension']}")
    print(f"- Processing time: {result['processing_time_ms']:.2f}ms")
    print(f"- First embedding (10 values): {result['embeddings'][0][:10]}")

def test_model_info():
    """测试模型信息"""
    resp = requests.get(f"{BASE_URL}/model-info")
    print("Model Info:", json.dumps(resp.json(), indent=2))

if __name__ == "__main__":
    print("Testing Embedding Service...\n")
    test_health()
    print()
    test_model_info()
    print()
    test_embedding()
```

运行测试:
```bash
python test_embedding_service.py
```

---

## 🔍 GPU检查和调试

### 检查容器内GPU
```bash
# 进入容器
docker exec -it embedding-service bash

# 查看GPU信息
nvidia-smi

# 运行Python检查
python3 << EOF
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
print(f"Current Device: {torch.cuda.current_device()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
EOF
```

### 检查宿主机GPU
```bash
# Linux/Windows WSL2
nvidia-smi

# Docker Desktop (Mac/Windows)
docker run --device nvidia.com/gpu=all \
  nvcr.io/nvidia/k8s/cuda-sample:nbody \
  nbody -gpu -benchmark
```

### 查看容器资源使用
```bash
# 实时监控
docker stats embedding-service

# 详细信息
docker inspect embedding-service

# 查看日志中的GPU使用
docker logs embedding-service | grep -i gpu
```

---

## 📊 性能优化建议

1. **批处理**
   - 增加 `BATCH_SIZE` 提高吞吐量 (需要足够GPU内存)
   - 典型值: 32-128

2. **GPU内存优化**
   - 减少 `BATCH_SIZE` 如果OOM
   - 启用混合精度 (fp16): 在model.encode中添加normalize_embeddings
   - 使用ONNX Runtime加速推理

3. **多GPU支持**
   - docker-compose中修改 `count: all` 使用所有GPU
   - 使用 `CUDA_VISIBLE_DEVICES=0,1,2` 指定多个GPU

4. **模型优化**
   - 选择更小的模型: `all-MiniLM-L6-v2` vs `all-mpnet-base-v2`
   - 使用量化模型

5. **网络优化**
   - 增加 `uvicorn` worker数量
   - 启用gzip压缩大embeddings响应

---

## 🛠️ 常见问题和解决方案

| 问题 | 解决方案 |
|------|--------|
| `docker: Error response from daemon: could not select device driver ""` | 需要安装NVIDIA Container Runtime |
| GPU未显示在容器中 | 检查 `docker ps` 的DEVICE设置，重启Docker daemon |
| OOM (Out of Memory) | 减少 `BATCH_SIZE`，增加GPU卡内存预留 |
| 模型加载慢 | 第一次会下载，后续使用volume缓存 |
| CPU占用率高 | 增加 `BATCH_SIZE` 充分利用GPU |

---

## 📝 检查清单

- [ ] GPU驱动已安装 (`nvidia-smi` 可运行)
- [ ] Docker Desktop/Engine 已安装
- [ ] NVIDIA Container Runtime 已安装 (生产环境)
- [ ] Docker镜像已成功构建 (`docker build`)
- [ ] 容器可正常启动 (`docker compose up`)
- [ ] API健康检查通过 (`curl /health` 返回200)
- [ ] Embedding生成正常 (`curl -X POST /embed`)
- [ ] GPU内存使用正常 (`docker stats`)
- [ ] 日志无错误 (`docker logs`)
- [ ] 模型已缓存在volume中

---

## 📚 参考文档

- Docker GPU支持: https://docs.docker.com/desktop/features/gpu/
- NVIDIA Container Toolkit: https://github.com/NVIDIA/nvidia-docker
- Sentence Transformers: https://www.sbert.net/
- PyTorch CUDA: https://pytorch.org/
- FastAPI: https://fastapi.tiangolo.com/

---

**更新日期**: 2024年12月
**版本**: 1.0
