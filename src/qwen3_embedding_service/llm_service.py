import os
import sys
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Literal, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import logging
from functools import lru_cache
import hashlib
# 在 llm_service.py 顶部添加（消除视觉干扰）
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*Torch was not compiled with flash attention.*",
    category=UserWarning
)

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
logger = logging.getLogger("llm_service")

# ==================== 设备检测 & 全局配置 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"检测到设备: {DEVICE} | PyTorch: {torch.__version__}")


# 显存自适应配置
def get_safe_config():
    """根据显存返回安全配置（支持低至1GB显存的设备）"""
    if DEVICE != "cuda":
        return {"max_batch_size": 4, "max_new_tokens": 2048, "max_length": 4096}

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

    if total_gb >= 2:
        return {"max_batch_size": 2, "max_new_tokens": 1024, "max_length": 2048}

    # 极低显存：<2GB（如老旧GPU/集成显卡）
    # 此时MiniCPM4-0.5B模型本身约需1.5-2GB（bfloat16），难以安全运行
    else:
        logger.warning(
            f"检测到极低显存({total_gb:.1f}GB)，MiniCPM4-0.5B在GPU上运行风险高，"
            "建议设置环境变量 CUDA_VISIBLE_DEVICES='' 强制使用CPU模式"
        )
        return {"max_batch_size": 1, "max_new_tokens": 512, "max_length": 1024}


SAFE_CONFIG = get_safe_config()
MAX_BATCH_SIZE = int(os.getenv("LLM_MAX_BATCH_SIZE", SAFE_CONFIG["max_batch_size"]))
MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", SAFE_CONFIG["max_new_tokens"]))
MAX_LENGTH = int(os.getenv("LLM_MAX_LENGTH", SAFE_CONFIG["max_length"]))

logger.info(f"运行配置: 批量上限={MAX_BATCH_SIZE}, 最大生成长度={MAX_NEW_TOKENS}, 最大序列长度={MAX_LENGTH}")

if DEVICE == "cuda":
    logger.info(
        f"   GPU: {torch.cuda.get_device_name(0)} | "
        f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB"
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# ==================== 模型加载 ====================
# 支持从 ModelScope 或本地路径加载
MODEL_PATH = os.getenv("LLM_MODEL_PATH", "OpenBMB/MiniCPM4-0.5B")

try:
    logger.info(f"加载模型: {MODEL_PATH} 到 {DEVICE}...")
    start = time.time()

    # MiniCPM4-0.5B 推荐使用 bfloat16
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    logger.info(f"加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE_DIR
    )
    logger.info(f"加载model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=dtype,
        device_map=DEVICE if DEVICE == "cuda" else None,
        cache_dir=MODEL_CACHE_DIR,
    )

    if DEVICE == "cpu":
        model = model.to(DEVICE)
    logger.info(f"执行eval...")
    model.eval()

    load_time = time.time() - start
    logger.info(f"模型加载成功 ({load_time:.2f}s)")

    # 模型预热
    with torch.no_grad():
        dummy_input = tokenizer("你好", return_tensors="pt").to(DEVICE)
        _ = model.generate(**dummy_input, max_new_tokens=10)
    logger.info("模型预热完成")

except Exception as e:
    logger.error(f"模型加载失败: {e}", exc_info=True)
    raise


# ==================== 对话生成函数 ====================
def generate_response(
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.7,
        max_new_tokens: int = None,
        do_sample: bool = True
) -> str:
    """
    使用 MiniCPM4-0.5B 生成对话回复

    Args:
        messages: 对话历史，格式为 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        temperature: 采样温度
        top_p: 核采样参数
        max_new_tokens: 最大生成 token 数
        do_sample: 是否使用采样
    """
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS

    # 应用对话模板
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 编码输入
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(DEVICE)

    # 生成参数
    gen_kwargs = {
        "max_new_tokens": min(max_new_tokens, MAX_NEW_TOKENS),
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # 移除 None 值
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.no_grad():
        model_outputs = model.generate(
            **model_inputs,
            **gen_kwargs
        )

    # 解码输出，去除输入部分
    output_token_ids = [
        model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs['input_ids']))
    ]

    response = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
    return response


# ==================== 工具函数 ====================
def hash_text(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


# ==================== 缓存机制 ====================
CACHE_SIZE = int(os.getenv("LLM_CACHE_SIZE", "100"))


@lru_cache(maxsize=CACHE_SIZE)
def cached_generate(text_hash: str, prompt: str, temperature: float, top_p: float, max_tokens: int) -> tuple:
    """缓存生成结果"""
    messages = [{"role": "user", "content": prompt}]
    response = generate_response(messages, temperature, top_p, max_tokens)
    return (response,)


# ==================== Pydantic 模型 ====================
class Message(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(...)
    content: str = Field(..., min_length=1)


class ChatCompletionRequest(BaseModel):
    model: str = Field("MiniCPM4-0.5B")
    messages: List[Message] = Field(...)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.7, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=8192)
    stream: Optional[bool] = Field(False)
    do_sample: Optional[bool] = Field(True)

    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError("消息列表不能为空")
        if len(v) > 20:
            raise ValueError("消息历史过长，最多支持20轮对话")
        return v


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-local"
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class SimpleGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.7, ge=0.0, le=1.0)
    max_new_tokens: int = Field(512, ge=1, le=MAX_NEW_TOKENS)
    use_cache: bool = Field(True)

    @validator('prompt')
    def validate_prompt(cls, v):
        max_len = 32768
        if len(v) > max_len:
            raise ValueError(f"提示词长度不能超过 {max_len} 字符")
        return v


class GenerateResponse(BaseModel):
    response: str
    model: str
    usage: Dict
    generation_time_ms: float


# ==================== FastAPI 应用 ====================
app = FastAPI(
    title="MiniCPM4-0.5B API",
    description=f"""
    MiniCPM4-0.5B 本地对话生成服务

    当前配置：
    - 最大批量: {MAX_BATCH_SIZE} 条/请求
    - 最大生成长度: {MAX_NEW_TOKENS} tokens
    - 最大序列长度: {MAX_LENGTH} tokens

    模型特性：
    - 参数量: 0.5B (5亿参数)
    - 架构: MiniCPM (基于 Llama 改进)
    - 支持长上下文: 最高 128K (取决于配置)
    - 推理效率: 支持稀疏注意力 InfLLM v2

    提供两种接口：
    1. /v1/chat/completions - 兼容 OpenAI 格式的对话接口
    2. /generate - 简单生成接口（支持缓存）
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== API 端点 ====================
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """OpenAI 兼容的对话补全接口"""
    start_time = time.time()

    logger.info(f"对话请求 | 消息数: {len(request.messages)} | 温度: {request.temperature}")

    try:
        # 转换消息格式
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        # 计算输入 token 数（估算）
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = len(tokenizer.encode(prompt_text))

        # 生成回复
        response_text = generate_response(
            messages=messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens,
            do_sample=request.do_sample
        )

        completion_tokens = len(tokenizer.encode(response_text))
        total_tokens = prompt_tokens + completion_tokens
        generation_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"生成完成 | 输入: {prompt_tokens} tokens | "
            f"输出: {completion_tokens} tokens | "
            f"耗时: {generation_time_ms:.2f}ms"
        )

        return ChatCompletionResponse(
            created=int(time.time()),
            model=request.model,
            choices=[Choice(
                index=0,
                message=Message(role="assistant", content=response_text),
                finish_reason="stop"
            )],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        )

    except Exception as e:
        logger.error(f"生成失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@app.post("/generate", response_model=GenerateResponse)
async def simple_generate(request: SimpleGenerateRequest):
    """简单生成接口（支持缓存）"""
    start_time = time.time()

    logger.info(f"生成请求 | 缓存: {request.use_cache} | 温度: {request.temperature}")

    try:
        # 检查缓存
        if request.use_cache:
            text_hash = hash_text(request.prompt)
            cached_result = cached_generate(
                text_hash,
                request.prompt,
                request.temperature,
                request.top_p,
                request.max_new_tokens
            )
            response_text = cached_result[0]
            from_cache = True
        else:
            messages = [{"role": "user", "content": request.prompt}]
            response_text = generate_response(
                messages=messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_new_tokens=request.max_new_tokens
            )
            from_cache = False

        generation_time_ms = (time.time() - start_time) * 1000
        completion_tokens = len(tokenizer.encode(response_text))

        logger.info(
            f"生成完成 | 缓存: {from_cache} | "
            f"输出: {completion_tokens} tokens | "
            f"耗时: {generation_time_ms:.2f}ms"
        )

        return GenerateResponse(
            response=response_text,
            model="MiniCPM4-0.5B",
            usage={
                "completion_tokens": completion_tokens,
                "from_cache": from_cache
            },
            generation_time_ms=round(generation_time_ms, 2)
        )

    except Exception as e:
        logger.error(f"生成失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


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
        "service": "llm",
        "version": "1.0.0",
        "device": DEVICE,
        "model": MODEL_PATH,
        "max_batch_size": MAX_BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_length": MAX_LENGTH,
        "cuda_available": torch.cuda.is_available(),
        **gpu_info
    }


@app.get("/v1/models")
async def list_models():
    return {
        "data": [{
            "id": "MiniCPM4-0.5B",
            "object": "model",
            "created": 1700000000,
            "owned_by": "OpenBMB",
            "parameters": "0.5B",
            "max_tokens": MAX_NEW_TOKENS,
        }],
        "object": "list"
    }


@app.get("/stats")
async def get_stats():
    cache_info = cached_generate.cache_info()
    total = cache_info.hits + cache_info.misses
    return {
        "service": "llm",
        "cache": {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "maxsize": cache_info.maxsize,
            "currsize": cache_info.currsize,
            "hit_rate": round(cache_info.hits / total * 100, 2) if total > 0 else 0.0
        },
        "config": {
            "max_batch_size": MAX_BATCH_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_length": MAX_LENGTH,
            "device": DEVICE,
        },
        "model_loaded": model is not None
    }


# ==================== 启动信息打印函数 ====================
def print_startup_info(host: str = "0.0.0.0", port: int = 18001):
    """打印服务启动信息"""
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
        est_memory = "1.5-2.0 GB"  # MiniCPM4-0.5B 实测约 1.8GB
        gpu_info = f"显存占用估算: ~{est_memory} / {total_gb:.1f} GB"
        device_str = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        gpu_info = "显存占用估算: N/A (CPU模式)"
        device_str = "CPU"

    # 构建输出
    lines = [
        "",
        "=" * 62,
        "           MiniCPM4-0.5B API 服务已启动",
        "=" * 62,
        "",
        " 服务端点",
        "  " + "-" * 58,
        f"  对话接口:    http://localhost:{port}/v1/chat/completions",
        f"  生成接口:    http://localhost:{port}/generate",
        f"  健康检查:    http://localhost:{port}/health",
        f"  API 文档:    http://localhost:{port}/docs",
        f"  模型列表:    http://localhost:{port}/v1/models",
        f"  统计信息:    http://localhost:{port}/stats",
        "",
        " 局域网访问",
        "  " + "-" * 58,
        f"  基础地址:    http://{local_ip}:{port}",
        "",
        " 接口列表",
        "  " + "-" * 58,
        f"  POST /v1/chat/completions  - OpenAI 兼容对话接口",
        f"  POST /generate             - 简单生成接口（支持缓存）",
        f"  GET  /health               - 健康检查",
        f"  GET  /v1/models            - 模型列表",
        f"  GET  /stats                - 统计信息",
        "",
        " 运行配置",
        "  " + "-" * 58,
        f"  最大批量:    {MAX_BATCH_SIZE} 条/请求",
        f"  最大生成长度: {MAX_NEW_TOKENS} tokens",
        f"  最大序列长度: {MAX_LENGTH} tokens",
        f"  设备:        {device_str}",
        f"  {gpu_info}",
        "",
        " 环境变量覆盖（可选）",
        "  " + "-" * 58,
        "  set LLM_MAX_BATCH_SIZE=8   && python llm_service.py  # 提高并发",
        "  set LLM_MAX_NEW_TOKENS=1024 && python llm_service.py  # 限制生成长度",
        "  set LLM_MODEL_PATH=./models/MiniCPM4-0.5B && python llm_service.py  # 本地路径",
        "=" * 62,
        "",
    ]

    print("\\n".join(lines))
    logger.info(f"服务启动完成，监听 {host}:{port}")

# ==================== 启动 ====================
if __name__ == "__main__":
    import uvicorn

    # 检查模型缓存
    model_cache_path = os.path.join(MODEL_CACHE_DIR, "models--OpenBMB--MiniCPM4-0.5B")
    if not os.path.exists(model_cache_path):
        logger.warning(f"模型缓存:{model_cache_path}")
        logger.warning(f"模型缓存可能不存在，首次启动将自动下载")
        logger.warning("模型大小约 1.2GB，请确保网络连接...")

    # 配置
    HOST = "0.0.0.0"
    PORT = 18001

    # 打印启动信息
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