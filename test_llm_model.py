#!/usr/bin/env python3
"""测试LLM模型效果"""

import requests
import json

print("=" * 60)
print("测试 LLM 模型效果")
print("=" * 60)

# 测试用例
test_cases = [
    {"name": "简单问答", "messages": [{"role": "user", "content": "什么是人工智能？"}], "max_tokens": 100},
    {
        "name": "代码生成",
        "messages": [{"role": "user", "content": "用Python写一个计算斐波那契数列的函数"}],
        "max_tokens": 150,
    },
    {
        "name": "多轮对话",
        "messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
            {"role": "user", "content": "介绍一下机器学习"},
        ],
        "max_tokens": 100,
    },
]

for i, test in enumerate(test_cases, 1):
    print(f"\n{i}. {test['name']}")
    print("-" * 60)

    response = requests.post(
        "http://localhost:18001/v1/chat/completions",
        json={
            "model": "MiniCPM4-0.5B",
            "messages": test["messages"],
            "max_tokens": test["max_tokens"],
            "temperature": 0.7,
        },
        timeout=30,  # nosec B113
    )

    result = response.json()

    print(f"输入: {test['messages'][-1]['content']}")
    print(f"输出: {result['choices'][0]['message']['content']}")
    print(f"Token使用: {result['usage']}")

print("\n" + "=" * 60)
print("✅ LLM模型测试完成")
print("=" * 60)
