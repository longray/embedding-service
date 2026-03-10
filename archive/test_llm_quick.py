#!/usr/bin/env python3
import requests

print("测试LLM模型")
response = requests.post(
    "http://localhost:18001/generate",
    json={"prompt": "什么是人工智能？", "max_new_tokens": 50},
    timeout=30,  # nosec B113
)
print(f"状态: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"输出: {result['response']}")
    print(f"耗时: {result['generation_time_ms']:.0f}ms")
else:
    print(f"错误: {response.text}")
