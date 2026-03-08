#!/usr/bin/env python3
"""测试Embedding模型效果"""

import requests
import json

# 测试文本
test_texts = [
    "人工智能是计算机科学的一个分支",
    "机器学习是AI的核心技术",
    "深度学习使用神经网络",
    "今天天气很好",
    "我喜欢吃苹果",
]

print("=" * 60)
print("测试 Embedding 模型效果")
print("=" * 60)

# 测试单个文本
print("\n1. 单文本嵌入测试")
response = requests.post(
    "http://localhost:18000/v1/embeddings",
    json={"input": test_texts[0], "model": "Qwen3-Embedding-0.6B"},
    timeout=30,  # nosec B113
)
result = response.json()
print(f"输入: {test_texts[0]}")
print(f"向量维度: {len(result['data'][0]['embedding'])}")
print(f"前10维: {result['data'][0]['embedding'][:10]}")

# 测试批量文本
print("\n2. 批量文本嵌入测试")
response = requests.post(
    "http://localhost:18000/v1/embeddings",
    json={"input": test_texts, "model": "Qwen3-Embedding-0.6B"},
    timeout=30,  # nosec B113
)
result = response.json()
print(f"输入数量: {len(test_texts)}")
print(f"输出数量: {len(result['data'])}")

# 计算相似度
print("\n3. 语义相似度测试")
import numpy as np


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


embeddings = [item["embedding"] for item in result["data"]]
print("\n相似度矩阵:")
print("     ", "  ".join([f"文本{i + 1}" for i in range(len(test_texts))]))
for i, text in enumerate(test_texts):
    similarities = [cosine_similarity(embeddings[i], embeddings[j]) for j in range(len(embeddings))]
    print(f"文本{i + 1}: " + "  ".join([f"{s:.3f}" for s in similarities]))

print("\n文本内容:")
for i, text in enumerate(test_texts):
    print(f"文本{i + 1}: {text}")

print("\n" + "=" * 60)
print("✅ Embedding模型测试完成")
print("=" * 60)
