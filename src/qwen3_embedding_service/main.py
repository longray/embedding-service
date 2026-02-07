import os
import sys
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Literal
from transformers import AutoTokenizer, AutoModel  # ⬅️ 替换 modelscope
import time
import logging
from functools import lru_cache
import hashlib

# ==================== 环境初始化 ====================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT,"models")

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODEL_CACHE_DIR  # ⬅️ 只用 HF_HOME，移除 MODELSCOPE_*

print(f"✅ 模型缓存目录: {MODEL_CACHE_DIR}")
print(f"✅ 目录存在: {os.path.exists(MODEL_CACHE_DIR)}")

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 设备检测 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🚀 检测到设备: {DEVICE} | PyTorch: {torch.__version__}")
if DEVICE == "cuda":
    logger.info(
        f"   GPU: {torch.cuda.get_device_name(0)} | "
        f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB"
    )
    # GTX 1060 优化设置
    torch.backends.cuda.matmul.allow_tf32 = False  # Pascal 不支持 TF32
    torch.backends.cudnn.benchmark = True

# ==================== 模型加载（Transformers 方式）====================
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODEL_CACHE_DIR, "Qwen", "Qwen3-Embedding-0___6B"))

try:
    logger.info(f"⏳ 加载模型: {MODEL_PATH} 到 {DEVICE}...")
    start = time.time()

    # GTX 1060 6GB 必须用半精度，否则显存不够
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE_DIR
    )
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
        cache_dir=MODEL_CACHE_DIR,
        # device_map="auto"  # 可选：自动分配层到 GPU/CPU
    )
    model.to(DEVICE)
    model.eval()

    # # 编译模型加速（PyTorch 2.0+，可选）
    # if hasattr(torch, 'compile') and DEVICE == "cuda":
    #     try:
    #         model = torch.compile(model, mode="reduce-overhead")
    #         logger.info("✅ 模型编译优化已启用")
    #     except Exception as e:
    #         logger.warning(f"⚠️ 模型编译失败: {e}")

    load_time = time.time() - start
    logger.info(f"✅ 模型加载成功 ({load_time:.2f}s)")

    # 预热
    with torch.no_grad():
        dummy_input = tokenizer("warmup", return_tensors="pt").to(DEVICE)
        _ = model(**dummy_input)
    logger.info("🔥 模型预热完成")

except Exception as e:
    logger.error(f"❌ 模型加载失败: {e}", exc_info=True)
    raise


# ==================== Embedding 函数 ====================
def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    批量获取文本嵌入（Mean Pooling + L2 归一化）
    """
    # Tokenize
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192  # Qwen3 支持 32K，但 8K 足够且更快
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        # Forward
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

        # Mean Pooling（考虑 padding）
        attention_mask = inputs['attention_mask']  # [batch, seq_len]
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)  # [batch, hidden_dim]
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        # L2 归一化（必须在 GPU 上做，减少数据传输）
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # 移回 CPU 并转 numpy
        return embeddings.cpu().numpy()


# ==================== 工具函数 ====================
def hash_text(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


# ==================== 缓存机制 ====================
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "1000"))


@lru_cache(maxsize=CACHE_SIZE)
def cached_encode(text_hash: str, text: str) -> tuple:
    """
    LRU 缓存编码结果（返回 tuple 因为 lru_cache 需要可哈希对象）
    """
    embedding = get_embeddings([text])[0]  # [hidden_dim]
    return tuple(embedding.tolist())


# ==================== Pydantic 模型 ====================
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(
        ...,
        description="输入文本（字符串或字符串列表）",
        example="The quick brown fox jumps over the lazy dog"
    )
    model: str = Field("Qwen3-Embedding-0.6B", description="模型标识符")
    encoding_format: Literal["float", "base64"] = Field("float", description="输出格式")
    dimensions: Optional[int] = Field(None, ge=32, le=1024, description="输出维度（32-1024）")
    normalize: bool = Field(True, description="⚠️ 必须为 True！Qwen3 嵌入必须归一化")

    @validator('input')
    def validate_input(cls, v):
        MAX_BATCH_SIZE = 64
        MAX_TEXT_LENGTH = 32768

        if isinstance(v, list):
            if not v:
                raise ValueError("输入列表不能为空")
            if len(v) > MAX_BATCH_SIZE:
                raise ValueError(f"批量大小不能超过 {MAX_BATCH_SIZE}")
            for i, text in enumerate(v):
                if len(text) > MAX_TEXT_LENGTH:
                    raise ValueError(f"第 {i + 1} 条文本长度超过 {MAX_TEXT_LENGTH} 字符")
        elif isinstance(v, str):
            if len(v) > MAX_TEXT_LENGTH:
                raise ValueError(f"文本长度不能超过 {MAX_TEXT_LENGTH} 字符")
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
    description="""
    🚀 高性能本地嵌入服务 | Qwen3-Embedding-0.6B (GPU Accelerated)

    ## 关键特性
    - ✅ **GPU 加速**（GTX 1060 6GB 优化）
    - ✅ **强制 L2 归一化**（相似度计算必需）
    - ✅ **OpenAI 兼容接口**（无缝集成 LangChain）
    - ✅ **本地模型缓存**（models/ 目录，不占用 C 盘）
    - ✅ **LRU 缓存**（重复文本加速 10-50 倍）

    ## 重要提示
    ⚠️ **所有输出已自动归一化**（L2 范数 = 1），可直接用于余弦相似度计算。
    """,
    version="2.0.0",  # ⬅️ 版本升级（移除 modelscope）
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== API 端点 ====================
@app.post(
    "/v1/embeddings",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="生成文本嵌入"
)
async def create_embedding(request: EmbeddingRequest):
    start_time = time.time()
    texts = [request.input] if isinstance(request.input, str) else request.input
    batch_size = len(texts)

    logger.info(f"📥 处理 {batch_size} 条文本 | 维度: {request.dimensions or 1024}")

    try:
        # 批量编码（使用缓存）
        embeddings_list = []
        total_tokens = 0

        for idx, text in enumerate(texts):
            text_hash = hash_text(text)
            total_tokens += len(text.split())  # 简单 token 估算

            # 从缓存获取（返回 tuple，转回 list）
            emb_tuple = cached_encode(text_hash, text)
            embedding = np.array(emb_tuple)

            # 维度裁剪（Qwen3 支持运行时裁剪）
            if request.dimensions and request.dimensions < embedding.shape[0]:
                embedding = embedding[:request.dimensions]

            embeddings_list.append(embedding)

        # 合并为矩阵
        embeddings = np.stack(embeddings_list)  # [batch, dim]

        # 构造响应
        data = [
            EmbeddingObject(index=i, embedding=emb.tolist())
            for i, emb in enumerate(embeddings)
        ]

        processing_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"✅ 完成 {batch_size} 条 | "
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
        logger.error(f"❌ 嵌入生成失败: {str(e)}", exc_info=True)
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
        "version": "2.0.0",
        "device": DEVICE,
        "model": MODEL_PATH,
        "model_cache_dir": MODEL_CACHE_DIR,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
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
            "max_input_tokens": 32768
        }],
        "object": "list"
    }


@app.get("/stats")
async def get_stats():
    cache_info = cached_encode.cache_info()
    total = cache_info.hits + cache_info.misses
    return {
        "cache": {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "maxsize": cache_info.maxsize,
            "currsize": cache_info.currsize,
            "hit_rate": round(cache_info.hits / total * 100, 2) if total > 0 else 0.0
        },
        "device": DEVICE,
        "model": MODEL_PATH,
        "model_loaded": model is not None
    }


# ==================== 启动 ====================
if __name__ == "__main__":
    import uvicorn

    # 检查模型缓存
    model_cache_path = os.path.join(MODEL_CACHE_DIR, "Qwen", "Qwen3-Embedding-0___6B")
    if not os.path.exists(model_cache_path):
        logger.warning(f"⚠️  模型缓存不存在: {model_cache_path}")
        logger.warning("   首次启动将自动下载（约 1.2GB），请耐心等待...")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=18000,
        workers=1,  # GPU 模式必须单 worker
        log_level="info",
        reload=False
    )