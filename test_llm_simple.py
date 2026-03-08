#!/usr/bin/env python3
"""简单测试LLM模型"""

import requests

print("=" * 60)
print("测试 LLM 模型 - 简单生成接口")
print("=" * 60)

# 测试1: 简单生成
print("\n1. 简单问答")
response = requests.post(
    "http://localhost:18001/generate",
    json={"prompt": "什么是人工智能？请简单回答。", "temperature": 0.7, "max_new_tokens": 100},
    timeout=30,  # nosec B113
)
print(f"状态码: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"输入: 什么是人工智能？")
    print(f"输出: {result['response']}")
    print(f"耗时: {result['generation_time_ms']:.2f}ms")
else:
    print(f"错误: {response.text}")

# 测试2: 代码生成
print("\n2. 代码生成")
response = requests.post(
    "http://localhost:18001/generate",
    json={"prompt": "用Python写一个函数计算两个数的和", "temperature": 0.3, "max_new_tokens": 150},
    timeout=30,  # nosec B113
)
if response.status_code == 200:
    result = response.json()
    print(f"输出: {result['response']}")

print("\n" + "=" * 60)
print("✅ LLM简单测试完成")
print("=" * 60)
