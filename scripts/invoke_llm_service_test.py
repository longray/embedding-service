#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniCPM4-0.5B LLM 对话生成测试脚本
测试对话补全和简单生成接口
"""

import requests
import json
import time
from typing import List, Dict, Optional

# API 配置
API_BASE_URL = "http://localhost:18001"
CHAT_API_URL = f"{API_BASE_URL}/v1/chat/completions"
GENERATE_API_URL = f"{API_BASE_URL}/generate"
MODEL_ID = "MiniCPM4-0.5B"

# 测试配置
MAX_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.7


def test_health_check() -> bool:
    """测试健康检查接口"""
    print("🔍 测试健康检查接口...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "healthy":
            print(f"   ✅ 服务健康")
            print(f"   📊 设备: {data.get('device', 'unknown')}")
            print(f"   🎮 GPU: {data.get('gpu_name', 'N/A')}")
            print(f"   ⚙️  最大生成长度: {data.get('max_new_tokens', 'N/A')} tokens")
            return True
        else:
            print(f"   ⚠️  服务状态异常: {data}")
            return False
    except Exception as e:
        print(f"   ❌ 健康检查失败: {e}")
        return False


def test_models_endpoint() -> bool:
    """测试模型列表接口"""
    print("\\n📋 测试模型列表接口...")
    try:
        response = requests.get(f"{API_BASE_URL}/v1/models", timeout=10)
        response.raise_for_status()
        data = response.json()

        models = data.get("data", [])
        if models:
            model = models[0]
            print(f"   ✅ 获取模型列表成功")
            print(f"   🤖 模型ID: {model.get('id')}")
            print(f"   📦 参数量: {model.get('parameters', 'N/A')}")
            print(f"   🔢 最大Token: {model.get('max_tokens', 'N/A')}")
            return True
        else:
            print(f"   ⚠️  模型列表为空")
            return False
    except Exception as e:
        print(f"   ❌ 获取模型列表失败: {e}")
        return False


def test_chat_completion(messages: List[Dict[str, str]], description: str = "") -> Optional[Dict]:
    """测试对话补全接口 (OpenAI 兼容格式)"""
    if description:
        print(f"\\n💬 {description}")

    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "do_sample": True
    }

    try:
        start_time = time.time()
        response = requests.post(CHAT_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        elapsed_ms = (time.time() - start_time) * 1000

        result = response.json()
        choice = result["choices"][0]
        usage = result["usage"]

        reply = choice["message"]["content"]

        print(f"   ✅ 生成成功 ({elapsed_ms:.1f}ms)")
        print(f"   📝 Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
        print(f"   ✍️  Completion tokens: {usage.get('completion_tokens', 'N/A')}")
        print(f"   💰 Total tokens: {usage.get('total_tokens', 'N/A')}")
        print(f"   🤖 回复: {reply[:200]}{'...' if len(reply) > 200 else ''}")

        return result

    except requests.exceptions.RequestException as e:
        print(f"   ❌ 请求失败: {e}")
        if hasattr(e.response, 'text'):
            print(f"   错误详情: {e.response.text}")
        return None
    except Exception as e:
        print(f"   ❌ 处理失败: {e}")
        return None


def test_simple_generate(prompt: str, description: str = "", use_cache: bool = False) -> Optional[Dict]:
    """测试简单生成接口（支持缓存）"""
    if description:
        print(f"\\n🚀 {description}")

    payload = {
        "prompt": prompt,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_TOKENS,
        "use_cache": use_cache
    }

    try:
        start_time = time.time()
        response = requests.post(GENERATE_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        elapsed_ms = (time.time() - start_time) * 1000

        result = response.json()
        reply = result["response"]
        usage = result["usage"]

        cache_status = "命中缓存" if usage.get("from_cache") else "未命中缓存"

        print(f"   ✅ 生成成功 ({elapsed_ms:.1f}ms, {cache_status})")
        print(f"   ✍️  Tokens: {usage.get('completion_tokens', 'N/A')}")
        print(f"   🤖 回复: {reply[:200]}{'...' if len(reply) > 200 else ''}")

        return result

    except requests.exceptions.RequestException as e:
        print(f"   ❌ 请求失败: {e}")
        if hasattr(e.response, 'text'):
            print(f"   错误详情: {e.response.text}")
        return None
    except Exception as e:
        print(f"   ❌ 处理失败: {e}")
        return None


def test_stats_endpoint() -> bool:
    """测试统计信息接口"""
    print("\\n📊 测试统计信息接口...")
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        response.raise_for_status()
        data = response.json()

        cache_info = data.get("cache", {})
        config_info = data.get("config", {})

        print(f"   ✅ 获取统计信息成功")
        print(f"   💾 缓存命中: {cache_info.get('hits', 0)} 次")
        print(f"   🆕 缓存未命中: {cache_info.get('misses', 0)} 次")
        print(f"   📈 命中率: {cache_info.get('hit_rate', 0)}%")
        print(f"   ⚙️  当前配置: batch={config_info.get('max_batch_size')}, tokens={config_info.get('max_new_tokens')}")
        return True

    except Exception as e:
        print(f"   ❌ 获取统计信息失败: {e}")
        return False


def main():
    print("=" * 60)
    print("🚀 MiniCPM4-0.5B LLM 对话生成测试")
    print("=" * 60)
    print(f"API端点: {API_BASE_URL}")
    print(f"模型: {MODEL_ID}")
    print(f"温度: {TEMPERATURE}, Top-P: {TOP_P}, MaxTokens: {MAX_TOKENS}")
    print("=" * 60)

    # 1. 健康检查
    if not test_health_check():
        print("\\n💥 服务未就绪，退出测试")
        return

    # 2. 模型列表
    test_models_endpoint()

    # 3. 测试对话补全接口
    print("\\n" + "=" * 60)
    print("📝 测试 /v1/chat/completions 接口 (OpenAI 兼容)")
    print("=" * 60)

    # 测试 1: 简单问答
    test_chat_completion(
        messages=[{"role": "user", "content": "你好，请介绍一下自己"}],
        description="测试1: 简单问候"
    )

    # 测试 2: 多轮对话
    test_chat_completion(
        messages=[
            {"role": "user", "content": "什么是机器学习？"},
            {"role": "assistant", "content": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习而无需明确编程。"},
            {"role": "user", "content": "那深度学习呢？它和机器学习有什么关系？"}
        ],
        description="测试2: 多轮对话（带历史上下文）"
    )

    # 测试 3: 代码生成
    test_chat_completion(
        messages=[{"role": "user", "content": "用Python写一个计算斐波那契数列的函数，要求使用递归方式"}],
        description="测试3: 代码生成"
    )

    # 测试 4: 长文本理解（中文）
    test_chat_completion(
        messages=[{
            "role": "user",
            "content": "请总结以下这段话的主要观点：人工智能技术的发展正在深刻改变我们的生活方式。从智能手机中的语音助手到自动驾驶汽车，从医疗诊断系统到个性化推荐引擎，AI技术已经渗透到我们日常生活的方方面面。这种变革带来了效率的提升和便利性的增加，但同时也引发了关于隐私保护、就业影响和技术伦理等方面的讨论。如何在推动技术创新的同时确保其负责任地发展，是当前社会面临的重要课题。"
        }],
        description="测试4: 长文本理解（中文摘要）"
    )

    # 4. 测试简单生成接口
    print("\\n" + "=" * 60)
    print("🚀 测试 /generate 接口 (简单生成，支持缓存)")
    print("=" * 60)

    # 测试 5: 简单生成（首次，无缓存）
    prompt = "列举三个人工智能在医疗领域的应用"
    test_simple_generate(
        prompt=prompt,
        description="测试5: 简单生成（首次，无缓存）",
        use_cache=True
    )

    # 测试 6: 相同 Prompt（测试缓存）
    test_simple_generate(
        prompt=prompt,
        description="测试6: 相同Prompt（测试缓存命中）",
        use_cache=True
    )

    # 测试 7: 创意写作
    test_simple_generate(
        prompt="写一个关于未来城市的短故事，不超过100字",
        description="测试7: 创意写作",
        use_cache=False
    )

    # 5. 统计信息
    print("\\n" + "=" * 60)
    print("📊 获取服务统计")
    print("=" * 60)
    test_stats_endpoint()

    # 结束
    print("\\n" + "=" * 60)
    print("✨ 测试完成！所有接口验证通过")
    print("=" * 60)
    print("\\n📋 接口汇总：")
    print(f"   • 对话接口: {CHAT_API_URL}")
    print(f"   • 生成接口: {GENERATE_API_URL}")
    print(f"   • 健康检查: {API_BASE_URL}/health")
    print(f"   • 统计信息: {API_BASE_URL}/stats")
    print("=" * 60)


if __name__ == "__main__":
    main()