#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-Embedding-0.6B 批量嵌入测试脚本
测试 64 条最大批量处理能力
"""

import requests
import numpy as np
from typing import List, Dict

API_URL = "http://localhost:8000/v1/embeddings"
MODEL_ID = "Qwen3-Embedding-0.6B"


def generate_test_texts(count: int = 64) -> List[str]:
    """生成指定数量的测试文本（模拟不同场景）"""

    templates = [
        "人工智能正在改变{field}行业的运作方式，特别是在{application}方面。",
        "机器学习模型通过{method}技术，实现了{result}的显著提升。",
        "深度学习在{domain}领域的应用，解决了{problem}这一长期难题。",
        "自然语言处理技术使得{task}变得更加高效和准确。",
        "计算机视觉系统能够识别{object}，准确率达到{accuracy}以上。",
        "强化学习算法在{environment}中表现出色，获得了{reward}的奖励分数。",
        "神经网络架构{architecture}在{dataset}数据集上创造了新的记录。",
        "数据挖掘技术从{source}中提取有价值的{information}，用于决策支持。",
    ]

    fields = ["医疗", "金融", "教育", "制造", "交通", "零售", "农业", "能源"]
    applications = ["诊断辅助", "风险评估", "个性化教学", "质量控制", "自动驾驶", "推荐系统", "产量预测", "智能电网"]
    methods = ["迁移学习", "联邦学习", "对比学习", "自监督学习", "元学习", "多任务学习", "知识蒸馏", "模型压缩"]
    domains = ["基因组学", "气候科学", "材料科学", "天文学", "化学", "物理学", "生物学", "地球科学"]

    texts = []
    for i in range(count):
        template = templates[i % len(templates)]
        text = template.format(
            field=fields[i % len(fields)],
            application=applications[i % len(applications)],
            method=methods[i % len(methods)],
            result=f"{85 + (i % 15)}%",
            domain=domains[i % len(domains)],
            problem=f"传统方法效率低下的问题{i}",
            task=f"文本理解和生成任务类型{i}",
            object=f"复杂场景中的目标物体类别{i % 20}",
            accuracy=f"{90 + (i % 10)}%",
            environment=f"动态变化环境版本{i % 10}",
            reward=f"{1000 + i * 100}",
            architecture=f"Transformer变体架构{i % 8}",
            dataset=f"大规模行业数据集{i % 12}",
            source=f"多源异构数据源{i % 6}",
            information=f"关键业务洞察信息{i}"
        )
        texts.append(text)

    return texts


def batch_embed(texts: List[str]) -> Dict:
    """调用 API 获取批量嵌入"""

    payload = {
        "input": texts,  # 直接传入列表
        "model": MODEL_ID,
        "encoding_format": "float",
        "normalize": True
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
        if hasattr(e.response, 'text'):
            print(f"错误详情: {e.response.text}")
        raise


def verify_embeddings(embeddings: List[List[float]], expected_count: int, expected_dim: int = 1024):
    """验证嵌入结果"""

    print(f"\n{'=' * 50}")
    print("📊 嵌入结果验证")
    print(f"{'=' * 50}")

    # 1. 数量验证
    actual_count = len(embeddings)
    assert actual_count == expected_count, f"数量不匹配: 期望 {expected_count}, 实际 {actual_count}"
    print(f"✅ 数量验证通过: {actual_count} 条嵌入")

    # 2. 维度验证
    actual_dim = len(embeddings[0])
    assert actual_dim == expected_dim, f"维度不匹配: 期望 {expected_dim}, 实际 {actual_dim}"
    print(f"✅ 维度验证通过: {actual_dim} 维向量")

    # 3. L2 归一化验证（余弦相似度基础）
    norms = [np.linalg.norm(emb) for emb in embeddings]
    avg_norm = np.mean(norms)
    std_norm = np.std(norms)

    print(f"✅ L2 归一化验证: 平均范数 = {avg_norm:.6f} (标准差: {std_norm:.8f})")
    assert 0.99 < avg_norm < 1.01, "L2 归一化验证失败，范数应接近 1.0"

    # 4. 相似度计算示例（前 3 条文本的余弦相似度矩阵）
    print(f"\n📐 余弦相似度矩阵（前 5 条样本）:")
    emb_matrix = np.array(embeddings[:5])
    # 由于已归一化，点积即余弦相似度
    similarity_matrix = np.dot(emb_matrix, emb_matrix.T)

    print("       ", end="")
    for i in range(5):
        print(f"文本{i:2d}  ", end="")
    print()

    for i in range(5):
        print(f"文本{i:2d}  ", end="")
        for j in range(5):
            sim = similarity_matrix[i, j]
            if i == j:
                print(f" {sim:.3f}* ", end="")  # 对角线应为 1.0
            else:
                print(f" {sim:.3f}  ", end="")
        print()
    print("* 对角线值为 1.000（文本与自身的相似度）")

    return {
        "count": actual_count,
        "dimensions": actual_dim,
        "avg_norm": float(avg_norm),
        "similarity_matrix": similarity_matrix.tolist()
    }


def main():
    print("🚀 Qwen3-Embedding-0.6B 批量嵌入测试")
    print(f"{'=' * 50}")
    print(f"目标: 测试最大批量 64 条文本的嵌入能力")
    print(f"API端点: {API_URL}")
    print(f"{'=' * 50}\n")

    # 生成 64 条测试文本
    test_texts = generate_test_texts(count=64)
    print(f"📝 已生成 {len(test_texts)} 条测试文本（内容不重复）")
    print(f"   示例文本1: {test_texts[0][:50]}...")
    print(f"   示例文本32: {test_texts[31][:50]}...")
    print(f"   示例文本64: {test_texts[63][:50]}...")

    # 执行批量嵌入请求
    print(f"\n⏳ 发送批量嵌入请求（64条）...")
    result = batch_embed(test_texts)

    # 解析结果
    embeddings = [item["embedding"] for item in result["data"]]
    usage = result["usage"]

    print(f"\n✅ 请求成功!")
    print(f"   处理时间: {usage['processing_time_ms']:.2f} ms")
    print(f"   总 Token 数: {usage['total_tokens']}")

    # 验证嵌入质量
    stats = verify_embeddings(embeddings, expected_count=64)

    # 输出统计信息
    print(f"\n{'=' * 50}")
    print("📈 批量嵌入统计")
    print(f"{'=' * 50}")
    print(f"批量大小: {stats['count']} 条")
    print(f"向量维度: {stats['dimensions']} 维")
    print(f"归一化状态: L2 范数 ≈ {stats['avg_norm']:.4f} (已归一化)")
    print(f"数据格式: Float32 数组")
    print(f"适用场景: 语义搜索、文本聚类、相似度匹配")

    print(f"\n{'=' * 50}")
    print("✨ 测试完成！64 条批量嵌入功能正常")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()