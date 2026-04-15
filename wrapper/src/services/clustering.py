"""记忆聚类服务 - 使用基于相似度的社区检测算法

由于 leidenalg 依赖复杂，使用简化的谱聚类 + 连通分量算法实现类似功能。
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ClusteringService:
    """记忆聚类服务

    使用基于向量相似度的社区检测算法对记忆进行聚类。
    """

    def __init__(self):
        self._similarity_threshold = 0.75  # 默认相似度阈值

    async def cluster_memories(
        self,
        memory_ids: list[str],
        embeddings: list[list[float]],
        content_threshold: float = 0.75,
        max_clusters: int = 20,
    ) -> dict[str, Any]:
        """对记忆进行聚类分析

        Args:
            memory_ids: 记忆 ID 列表
            embeddings: 对应的嵌入向量列表
            content_threshold: 内容相似度阈值
            max_clusters: 最大聚类数量

        Returns:
            聚类结果，包含簇信息、成员和中心点
        """
        try:
            if not memory_ids or not embeddings:
                return {
                    "status": "error",
                    "message": "没有提供记忆数据",
                    "clusters": [],
                    "total_memories": 0,
                }

            if len(memory_ids) != len(embeddings):
                return {
                    "status": "error",
                    "message": "记忆 ID 和嵌入向量数量不匹配",
                    "clusters": [],
                    "total_memories": len(memory_ids),
                }

            # 转换为 numpy 数组
            embeddings_array = np.array(embeddings)
            n_memories = len(memory_ids)

            if n_memories < 2:
                return {
                    "status": "success",
                    "message": "记忆数量太少，无法聚类",
                    "clusters": [
                        {
                            "cluster_id": 0,
                            "members": memory_ids,
                            "size": n_memories,
                            "centroid": embeddings[0] if embeddings else [],
                        }
                    ],
                    "total_memories": n_memories,
                    "num_clusters": 1,
                }

            # 计算相似度矩阵
            similarity_matrix = self._compute_similarity_matrix(embeddings_array)

            # 使用连通分量进行聚类
            clusters = self._connected_components_clustering(
                memory_ids, similarity_matrix, content_threshold, max_clusters
            )

            # 计算每个簇的中心点
            for cluster in clusters:
                cluster["centroid"] = self._compute_centroid(embeddings_array, memory_ids, cluster["members"])

            return {
                "status": "success",
                "message": f"成功将 {n_memories} 个记忆聚类为 {len(clusters)} 个簇",
                "clusters": clusters,
                "total_memories": n_memories,
                "num_clusters": len(clusters),
                "similarity_threshold": content_threshold,
            }

        except Exception as e:
            logger.error("[ClusteringService] 聚类失败: %s", e)
            return {
                "status": "error",
                "message": str(e),
                "clusters": [],
                "total_memories": len(memory_ids) if memory_ids else 0,
            }

    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """计算余弦相似度矩阵

        Args:
            embeddings: 嵌入向量数组 (n_samples, n_features)

        Returns:
            相似度矩阵 (n_samples, n_samples)
        """
        # 归一化向量
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 避免除以零
        normalized = embeddings / norms

        # 计算余弦相似度
        similarity = np.dot(normalized, normalized.T)

        # 确保值在 [0, 1] 范围内（处理浮点误差）
        similarity = np.clip(similarity, 0, 1)

        return similarity

    def _connected_components_clustering(
        self,
        memory_ids: list[str],
        similarity_matrix: np.ndarray,
        threshold: float,
        max_clusters: int,
    ) -> list[dict[str, Any]]:
        """基于连通分量的聚类

        将相似度高于阈值的记忆连接，找出连通分量作为簇。

        Args:
            memory_ids: 记忆 ID 列表
            similarity_matrix: 相似度矩阵
            threshold: 相似度阈值
            max_clusters: 最大聚类数

        Returns:
            聚类列表
        """
        n = len(memory_ids)

        # 构建邻接矩阵（相似度 > 阈值）
        adjacency = similarity_matrix > threshold
        np.fill_diagonal(adjacency, True)  # 每个节点与自己连通

        # 使用并查集找出连通分量
        parent = list(range(n))

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # 合并相似的记忆
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j]:
                    union(i, j)

        # 收集连通分量
        components: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            if root not in components:
                components[root] = []
            components[root].append(i)

        # 转换为聚类结果
        clusters = []
        for idx, (root, members) in enumerate(components.items()):
            cluster_memories = [memory_ids[i] for i in members]
            clusters.append(
                {
                    "cluster_id": idx,
                    "members": cluster_memories,
                    "size": len(cluster_memories),
                    "representative": cluster_memories[0] if cluster_memories else None,
                }
            )

        # 按簇大小排序
        clusters.sort(key=lambda x: x["size"], reverse=True)

        # 如果簇数量超过限制，合并小簇
        if len(clusters) > max_clusters:
            # 保留前 max_clusters-1 个大簇
            main_clusters = clusters[: max_clusters - 1]
            small_clusters = clusters[max_clusters - 1 :]

            # 合并小簇为"其他"簇
            other_members = []
            for c in small_clusters:
                other_members.extend(c["members"])

            if other_members:
                main_clusters.append(
                    {
                        "cluster_id": max_clusters - 1,
                        "members": other_members,
                        "size": len(other_members),
                        "representative": other_members[0] if other_members else None,
                        "is_other": True,
                    }
                )

            clusters = main_clusters

        return clusters

    def _compute_centroid(
        self,
        embeddings: np.ndarray,
        memory_ids: list[str],
        cluster_members: list[str],
    ) -> list[float]:
        """计算簇的中心点（平均向量）

        Args:
            embeddings: 所有嵌入向量
            memory_ids: 记忆 ID 列表
            cluster_members: 簇成员 ID 列表

        Returns:
            中心点向量
        """
        # 找到成员在原始列表中的索引
        member_indices = []
        for mid in cluster_members:
            try:
                idx = memory_ids.index(mid)
                member_indices.append(idx)
            except ValueError:
                continue

        if not member_indices:
            return []

        # 计算平均向量
        centroid = np.mean(embeddings[member_indices], axis=0)
        return centroid.tolist()


# 全局服务实例
_clustering_service: ClusteringService | None = None


def get_clustering_service() -> ClusteringService:
    """获取聚类服务实例（单例模式）"""
    global _clustering_service
    if _clustering_service is None:
        _clustering_service = ClusteringService()
    return _clustering_service
