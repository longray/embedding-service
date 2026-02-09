import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Literal
from transformers import AutoTokenizer, AutoModel
import time
import logging
from functools import lru_cache
import hashlib

# ==================== 环境初始化 ====================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODEL_CACHE_DIR

print(f"模型缓存目录: {MODEL_CACHE_DIR}")
print(f"目录存在: {os.path.exists(MODEL_CACHE_DIR)}")

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("embedding_service")

# ==================== 设备检测 & 全局配置 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"检测到设备: {DEVICE} | PyTorch: {torch.__version__}")

# 显存自适应配置
def get_safe_batch_size():
    if DEVICE != "cuda":
        return 64
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    return 256 if total_gb >= 12 else 128 if total_gb >= 6 else 64

MAX_BATCH_SIZE = int(os.getenv("EMB_MAX_BATCH_SIZE", get_safe_batch_size()))
MAX_TEXT_LENGTH = 32768
MAX_LENGTH = 2048 if MAX_BATCH_SIZE >= 256 else 4096 if MAX_BATCH_SIZE >= 128 else 8192

logger.info(f"运行配置: 批量上限={MAX_BATCH_SIZE}, 序列长度={MAX_LENGTH}")

if DEVICE == "cuda":
    logger.info(
        f"   GPU: {torch.cuda.get_device_name(0)} | "
        f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB"
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

# ==================== 模型加载 ====================
MODEL_PATH = os.getenv("EMB_MODEL_PATH", os.path.join(MODEL_CACHE_DIR, "Qwen", "Qwen3-Embedding-0___6B"))

try:
    logger.info(f"加载模型: {MODEL_PATH} 到 {DEVICE}...")
    start = time.time()

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE_DIR
    )
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=dtype,
        cache_dir=MODEL_CACHE_DIR,
    )
    model.to(DEVICE)
    model.eval()

    load_time = time.time() - start
    logger.info(f"模型加载成功 ({load_time:.2f}s)")

    with torch.no_grad():
        dummy_input = tokenizer("warmup", return_tensors="pt").to(DEVICE)
        _ = model(**dummy_input)
    logger.info("模型预热完成")

except Exception as e:
    logger.error(f"模型加载失败: {e}", exc_info=True)
    raise


# ==================== Embedding 函数 ====================
def get_embeddings(texts: List[str]) -> np.ndarray:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state

        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()


# ==================== 工具函数 ====================
def hash_text(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


# ==================== 缓存机制 ====================
CACHE_SIZE = int(os.getenv("EMB_CACHE_SIZE", "1000"))

@lru_cache(maxsize=CACHE_SIZE)
def cached_encode(text_hash: str, text: str) -> tuple:
    embedding = get_embeddings([text])[0]
    return tuple(embedding.tolist())


# ==================== Pydantic 模型 ====================
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(...)
    model: str = Field("Qwen3-Embedding-0.6B")
    encoding_format: Literal["float", "base64"] = Field("float")
    dimensions: Optional[int] = Field(None, ge=32, le=1024)
    normalize: bool = Field(True)

    @validator('input')
    def validate_input(cls, v):
        max_batch = int(os.getenv("EMB_MAX_BATCH_SIZE", str(MAX_BATCH_SIZE)))
        max_length = 32768

        if isinstance(v, list):
            if not v:
                raise ValueError("输入列表不能为空")
            if len(v) > max_batch:
                raise ValueError(f"批量大小不能超过 {max_batch}")
            for i, text in enumerate(v):
                if len(text) > max_length:
                    raise ValueError(f"第 {i + 1} 条文本长度超过 {max_length} 字符")
        elif isinstance(v, str):
            if len(v) > max_length:
                raise ValueError(f"文本长度不能超过 {max_length} 字符")
        else:
            raise ValueError("输入必须是字符串或字符串列表")
        return v


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int
    processing_time_ms: float


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingObject]
    model: str = "Qwen3-Embedding-0.6B"
    usage: Usage


# ==================== FastAPI 应用 ====================
app = FastAPI(
    title="Qwen3-Embedding-0.6B API",
    description=f"""
    Qwen3-Embedding-0.6B 本地嵌入服务

    当前配置：
    - 最大批量: {MAX_BATCH_SIZE} 条
    - 最大序列长度: {MAX_LENGTH} tokens

    批量与长度关系：
    - 批量 >= 256 -> 序列长度 2048（保护显存）
    - 批量 >= 128 -> 序列长度 4096（平衡模式）
    - 批量 < 128  -> 序列长度 8192（长文本优先）
    """,
    version="2.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== API 端点 ====================
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    start_time = time.time()
    texts = [request.input] if isinstance(request.input, str) else request.input
    batch_size = len(texts)

    logger.info(f"处理 {batch_size} 条文本 | 维度: {request.dimensions or 1024}")

    try:
        embeddings_list = []
        total_tokens = 0

        for idx, text in enumerate(texts):
            text_hash = hash_text(text)
            total_tokens += len(text.split())

            emb_tuple = cached_encode(text_hash, text)
            embedding = np.array(emb_tuple)

            if request.dimensions and request.dimensions < embedding.shape[0]:
                embedding = embedding[:request.dimensions]

            embeddings_list.append(embedding)

        embeddings = np.stack(embeddings_list)

        data = [
            EmbeddingObject(index=i, embedding=emb.tolist())
            for i, emb in enumerate(embeddings)
        ]

        processing_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"完成 {batch_size} 条 | "
            f"维度: {embeddings.shape[1]} | "
            f"耗时: {processing_time_ms:.2f}ms | "
            f"速度: {batch_size / (processing_time_ms / 1000):.1f} 条/秒"
        )

        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage=Usage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens,
                processing_time_ms=round(processing_time_ms, 2)
            )
        )

    except Exception as e:
        logger.error(f"嵌入生成失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"嵌入生成失败: {str(e)}")


@app.get("/health")
async def health_check():
    gpu_info = {}
    if DEVICE == "cuda":
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2),
            "gpu_memory_used_mb": round(torch.cuda.memory_allocated(0) / 1024 ** 2, 2),
            "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1024 ** 2, 2),
        }

    return {
        "status": "healthy",
        "service": "embedding",
        "version": "2.0.1",
        "device": DEVICE,
        "max_batch_size": MAX_BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "model": MODEL_PATH,
        "cuda_available": torch.cuda.is_available(),
        **gpu_info
    }


@app.get("/v1/models")
async def list_models():
    return {
        "data": [{
            "id": "Qwen3-Embedding-0.6B",
            "object": "model",
            "created": 1700000000,
            "owned_by": "Alibaba",
            "dimensions": 1024,
            "max_batch_size": MAX_BATCH_SIZE,
        }],
        "object": "list"
    }


@app.get("/stats")
async def get_stats():
    cache_info = cached_encode.cache_info()
    total = cache_info.hits + cache_info.misses
    return {
        "service": "embedding",
        "cache": {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "maxsize": cache_info.maxsize,
            "currsize": cache_info.currsize,
            "hit_rate": round(cache_info.hits / total * 100, 2) if total > 0 else 0.0
        },
        "config": {
            "max_batch_size": MAX_BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "device": DEVICE,
        },
        "model_loaded": model is not None
    }


# ==================== 启动信息打印函数 ====================
def print_startup_info(host: str = "0.0.0.0", port: int = 18000):
    """打印服务启动信息（纯ASCII，无颜色代码）"""
    import socket

    # 获取本机 IP
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"

    # 计算显存占用参考值
    if DEVICE == "cuda":
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        if MAX_BATCH_SIZE >= 256:
            est_memory = "3.5-4.5 GB"
        elif MAX_BATCH_SIZE >= 128:
            est_memory = "2.5-3.5 GB"
        else:
            est_memory = "2.0-2.5 GB"
        gpu_info = f"显存占用估算: ~{est_memory} / {total_gb:.1f} GB"
        device_str = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        gpu_info = "显存占用估算: N/A (CPU模式)"
        device_str = "CPU"

    # 当前模式
    if MAX_BATCH_SIZE >= 256:
        mode = "高吞吐模式"
    elif MAX_BATCH_SIZE >= 128:
        mode = "平衡模式"
    else:
        mode = "长文本模式"

    # 构建输出（使用简单ASCII字符，确保CMD对齐）
    lines = [
        "",
        "=" * 62,
        "           Qwen3-Embedding-0.6B API 服务已启动",
        "=" * 62,
        "",
        " 服务端点",
        "  " + "-" * 58,
        f"  本地访问:    http://localhost:{port}",
        f"  局域网访问:  http://{local_ip}:{port}",
        f"  健康检查:    http://localhost:{port}/health",
        f"  API 文档:    http://localhost:{port}/docs",
        "",
        " 运行配置",
        "  " + "-" * 58,
        f"  最大批量:    {MAX_BATCH_SIZE} 条/请求",
        f"  序列长度:    {MAX_LENGTH} tokens",
        f"  设备:        {device_str}",
        f"  {gpu_info}",
        "",
        " 批量与序列长度关系（显存保护机制）",
        "  " + "-" * 58,
        "  +-------------+-------------+-------------------------+",
        "  |  批量大小   |  序列长度   |  适用场景               |",
        "  +-------------+-------------+-------------------------+",
        "  |   >= 256    |    2048     |  高吞吐短文本(如搜索)   |",
        "  |   >= 128    |    4096     |  平衡模式(推荐)         |",
        "  |    < 128    |    8192     |  长文本优先(如文档)     |",
        "  +-------------+-------------+-------------------------+",
        f"  当前: 批量={MAX_BATCH_SIZE}, 长度={MAX_LENGTH} ({mode})",
        "",
        " 环境变量覆盖（可选）",
        "  " + "-" * 58,
        "  set EMB_MAX_BATCH_SIZE=256 && python embedding_service.py  # 高吞吐",
        "  set EMB_MAX_BATCH_SIZE=64  && python embedding_service.py  # 保守",
        "=" * 62,
        "",
    ]

    print("\\n".join(lines))
    logger.info(f"服务启动完成，监听 {host}:{port}")


# ==================== 启动 ====================
if __name__ == "__main__":
    import uvicorn

    # 检查模型缓存
    model_cache_path = os.path.join(MODEL_CACHE_DIR, "Qwen", "Qwen3-Embedding-0___6B")
    if not os.path.exists(model_cache_path):
        logger.warning(f"模型缓存不存在: {model_cache_path}")
        logger.warning("首次启动将自动下载（约 1.2GB），请耐心等待...")

    # 配置
    HOST = "0.0.0.0"
    PORT = 18000

    # 打印启动信息（在 uvicorn 启动前）
    print_startup_info(HOST, PORT)

    # 启动服务 - 修复：直接传 app 对象，不是字符串
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        workers=1,
        log_level="info",
        reload=False
    )

