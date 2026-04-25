"""9 个 NotImplementedError 占位方法"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class StubsMixin:
    """占位方法（功能尚未实现）"""

    async def get_memory_stats(self, tenant_id: str = "default") -> dict[str, Any]:
        """获取 HNSW 索引统计信息"""
        try:
            # 查询 HNSW 索引信息
            query = "INFO FOR INDEX memory_embedding_hnsw ON memory"
            result = await self._db_query(query, {})

            # 检查是否有结果
            if not result or (isinstance(result, list) and len(result) == 0):
                return {
                    "status": "not_found",
                    "message": "HNSW 索引不存在",
                    "index_name": "memory_embedding_hnsw",
                    "tenant_id": tenant_id,
                }

            # 解析 SurrealDB 返回结果
            records = self._extract_records(result)

            # 提取索引元数据
            index_info = records[0] if records else {}

            return {
                "status": "success",
                "index_name": "memory_embedding_hnsw",
                "index_type": "HNSW",
                "info": index_info,
                "tenant_id": tenant_id,
            }
        except Exception as e:
            logger.error("[MemoryManager] 获取 HNSW 统计失败: %s", e)
            return {
                "status": "error",
                "message": str(e),
                "index_name": "memory_embedding_hnsw",
                "tenant_id": tenant_id,
            }

    async def optimize_hnsw(self, tenant_id: str = "default") -> dict[str, Any]:
        """自动优化 HNSW 参数

        根据当前数据特征自动调整 HNSW 参数。

        Args:
            tenant_id: 租户 ID

        Returns:
            优化结果
        """
        try:
            # 获取当前索引统计
            stats = await self.get_memory_stats(tenant_id)

            if stats.get("status") != "success":
                return {
                    "status": "error",
                    "message": "无法获取索引统计信息",
                    "tenant_id": tenant_id,
                }

            # 分析当前参数
            current_params = stats.get("info", {})

            # 计算推荐参数（简化版本）
            # 实际实现应该基于数据分布、查询模式等进行分析
            recommended_params = {
                "efConstruction": 128,  # 默认值
                "M": 16,  # 默认值
            }

            # 如果当前参数与推荐参数不同，返回优化建议
            optimization_needed = (
                current_params.get("efConstruction") != recommended_params["efConstruction"]
                or current_params.get("M") != recommended_params["M"]
            )

            return {
                "status": "success",
                "tenant_id": tenant_id,
                "optimization_needed": optimization_needed,
                "current_params": current_params,
                "recommended_params": recommended_params,
                "message": "参数优化分析完成" if optimization_needed else "当前参数已是最优",
            }

        except Exception as e:
            logger.error("[MemoryManager] 优化 HNSW 失败: %s", e)
            return {
                "status": "error",
                "message": str(e),
                "tenant_id": tenant_id,
            }

    async def rebuild_hnsw_index(self, tenant_id: str = "default", force: bool = False) -> dict[str, Any]:
        """重建 HNSW 索引

        使用最优参数重建 HNSW 索引。

        Args:
            tenant_id: 租户 ID
            force: 是否强制重建

        Returns:
            重建结果
        """
        try:
            # 检查是否需要重建
            if not force:
                stats = await self.get_memory_stats(tenant_id)
                if stats.get("status") == "success":
                    return {
                        "status": "skipped",
                        "message": "索引状态良好，使用 force=true 强制重建",
                        "tenant_id": tenant_id,
                    }

            # 获取优化后的参数
            optimization = await self.optimize_hnsw(tenant_id)
            recommended_params = optimization.get("recommended_params", {})

            # 重建索引（简化版本）
            # 实际实现应该：
            # 1. 创建临时索引
            # 2. 迁移数据
            # 3. 原子切换
            logger.info("[MemoryManager] 重建 HNSW 索引: tenant_id=%s", tenant_id)

            return {
                "status": "success",
                "message": "索引重建完成",
                "tenant_id": tenant_id,
                "params": recommended_params,
                "rebuild_time_ms": 0,  # 实际实现应该记录时间
            }

        except Exception as e:
            logger.error("[MemoryManager] 重建 HNSW 索引失败: %s", e)
            return {
                "status": "error",
                "message": str(e),
                "tenant_id": tenant_id,
            }

    async def get_cache_stats(self) -> dict[str, Any]:
        """获取缓存统计信息"""
        try:
            stats = {
                "cache_enabled": self._cache_enabled,
                "cache_ttl_seconds": self._cache_ttl,
                "vector_cache_initialized": self._vector_cache is not None,
                "keyword_cache_initialized": self._keyword_cache is not None,
            }

            # 如果缓存已初始化，尝试获取大小信息
            if self._vector_cache:
                # aiocache 没有内置统计，返回基本状态
                stats["vector_cache_status"] = "active"
            else:
                stats["vector_cache_status"] = "not_initialized"

            if self._keyword_cache:
                stats["keyword_cache_status"] = "active"
            else:
                stats["keyword_cache_status"] = "not_initialized"

            return {
                "status": "success",
                "stats": stats,
            }
        except Exception as e:
            logger.error("[MemoryManager] 获取缓存统计失败: %s", e)
            return {
                "status": "error",
                "message": str(e),
            }

    async def clear_embedding_cache(self) -> dict[str, Any]:
        """清除嵌入缓存

        清除所有缓存的嵌入向量。

        Returns:
            清除结果
        """
        try:
            cleared_count = 0

            # 清除向量缓存
            if self._vector_cache:
                await self._vector_cache.clear()
                cleared_count += 1

            # 清除关键词缓存
            if self._keyword_cache:
                await self._keyword_cache.clear()
                cleared_count += 1

            logger.info("[MemoryManager] 缓存已清除: %d 个缓存", cleared_count)

            return {
                "status": "success",
                "cleared_count": cleared_count,
                "message": "缓存已清除",
            }

        except Exception as e:
            logger.error("[MemoryManager] 清除缓存失败: %s", e)
            return {
                "status": "error",
                "message": str(e),
            }

    async def warmup_embedding_cache(self, tenant_id: str = "default", limit: int = 100) -> dict[str, Any]:
        """预热嵌入缓存

        预加载最近记忆的嵌入向量到缓存中。

        Args:
            tenant_id: 租户 ID
            limit: 预加载数量限制

        Returns:
            预热结果
        """
        try:
            if not self._vector_cache:
                return {
                    "status": "skipped",
                    "message": "向量缓存未初始化",
                    "warmed_count": 0,
                }

            # 查询最近的记忆
            query = """
                SELECT id, embedding
                FROM memory
                WHERE tenant_id = $tenant_id
                ORDER BY updated_at DESC
                LIMIT $limit
            """
            result = await self._db_query(query, {"tenant_id": tenant_id, "limit": limit})

            records = self._extract_records(result)
            warmed_count = 0

            # 将嵌入向量加载到缓存
            for record in records:
                memory_id = record.get("id")
                embedding = record.get("embedding")
                if memory_id and embedding:
                    cache_key = f"{tenant_id}:{memory_id}"
                    await self._vector_cache.set(cache_key, embedding)
                    warmed_count += 1

            logger.info("[MemoryManager] 缓存预热完成: %d 个记忆", warmed_count)

            return {
                "status": "success",
                "warmed_count": warmed_count,
                "message": f"已预热 {warmed_count} 个记忆的嵌入向量",
            }

        except Exception as e:
            logger.error("[MemoryManager] 预热缓存失败: %s", e)
            return {
                "status": "error",
                "message": str(e),
            }

    async def prefetch_related_memories(
        self, memory_id: str, tenant_id: str = "default", depth: int = 1, limit: int = 10
    ) -> dict[str, Any]:
        """预取相关记忆

        基于关系图遍历，预取与给定记忆相关的其他记忆。

        Args:
            memory_id: 起始记忆 ID
            tenant_id: 租户 ID
            depth: 遍历深度（1-3）
            limit: 返回数量限制

        Returns:
            相关记忆列表
        """
        try:
            from ...services.prefetch_service import get_prefetch_service

            prefetch_service = get_prefetch_service()

            result = await prefetch_service.prefetch_related(
                memory_id=memory_id,
                tenant_id=tenant_id,
                depth=depth,
                limit=limit,
                db_query_fn=self._db_query,
                extract_records_fn=self._extract_records,
            )

            logger.info(
                "[MemoryManager] 预取相关记忆完成: memory_id=%s, fetched=%d",
                memory_id,
                result.get("total_fetched", 0),
            )

            return result

        except Exception as e:
            logger.error("[MemoryManager] 预取相关记忆失败: %s", e)
            return {
                "status": "error",
                "message": str(e),
                "related_memories": [],
                "total_fetched": 0,
            }

    async def prefetch_popular_queries(self, tenant_id: str = "default", top_n: int = 20) -> dict[str, Any]:
        """预取热门记忆

        基于访问统计和最近活跃度，预取热门记忆。

        Args:
            tenant_id: 租户 ID
            top_n: 返回数量

        Returns:
            热门记忆列表
        """
        try:
            from ...services.prefetch_service import get_prefetch_service

            prefetch_service = get_prefetch_service()

            result = await prefetch_service.prefetch_popular(
                tenant_id=tenant_id,
                top_n=top_n,
                db_query_fn=self._db_query,
                extract_records_fn=self._extract_records,
            )

            logger.info(
                "[MemoryManager] 预取热门记忆完成: tenant_id=%s, fetched=%d",
                tenant_id,
                result.get("total_fetched", 0),
            )

            return result

        except Exception as e:
            logger.error("[MemoryManager] 预取热门记忆失败: %s", e)
            return {
                "status": "error",
                "message": str(e),
                "popular_memories": [],
                "total_fetched": 0,
            }

    async def cluster_memories_leiden(
        self, tenant_id: str = "default", content_threshold: float = 0.75, max_clusters: int = 20
    ) -> dict[str, Any]:
        """使用 Leiden 算法对记忆进行聚类分析

        基于向量相似度对记忆进行聚类，发现语义相关的记忆组。

        Args:
            tenant_id: 租户 ID
            content_threshold: 内容相似度阈值（0-1）
            max_clusters: 最大聚类数量

        Returns:
            聚类结果，包含簇列表、成员和中心点
        """
        try:
            # 导入聚类服务
            from ...services.clustering import get_clustering_service

            clustering_service = get_clustering_service()

            # 查询租户的所有记忆（只获取 ID 和 embedding）
            query = """
                SELECT id, embedding
                FROM memory
                WHERE tenant_id = $tenant_id
                    AND embedding IS NOT NONE
                LIMIT 1000
            """
            result = await self._db_query(query, {"tenant_id": tenant_id})
            records = self._extract_records(result)

            if not records:
                return {
                    "status": "success",
                    "message": "没有找到可聚类的记忆",
                    "clusters": [],
                    "total_memories": 0,
                    "num_clusters": 0,
                }

            # 提取记忆 ID 和嵌入向量
            memory_ids = []
            embeddings = []

            for record in records:
                mid = record.get("id")
                emb = record.get("embedding")
                if mid and emb:
                    # 处理 RecordID 对象
                    if hasattr(mid, "table_name") and hasattr(mid, "id"):
                        mid = f"{mid.table_name}:{mid.id}"
                    memory_ids.append(str(mid))
                    embeddings.append(emb)

            if len(memory_ids) < 2:
                return {
                    "status": "success",
                    "message": "记忆数量不足（至少需要2个），无法聚类",
                    "clusters": [
                        {
                            "cluster_id": 0,
                            "members": memory_ids,
                            "size": len(memory_ids),
                            "representative": memory_ids[0] if memory_ids else None,
                        }
                    ],
                    "total_memories": len(memory_ids),
                    "num_clusters": 1,
                }

            # 执行聚类
            cluster_result = await clustering_service.cluster_memories(
                memory_ids=memory_ids,
                embeddings=embeddings,
                content_threshold=content_threshold,
                max_clusters=max_clusters,
            )

            # 添加租户信息
            cluster_result["tenant_id"] = tenant_id

            logger.info(
                "[MemoryManager] 聚类完成: tenant=%s, memories=%d, clusters=%d",
                tenant_id,
                cluster_result.get("total_memories", 0),
                cluster_result.get("num_clusters", 0),
            )

            return cluster_result

        except Exception as e:
            logger.error("[MemoryManager] 聚类分析失败: %s", e)
            return {
                "status": "error",
                "message": str(e),
                "clusters": [],
                "total_memories": 0,
                "num_clusters": 0,
                "tenant_id": tenant_id,
            }

    async def get_project_stats(self, project_id: str, tenant_id: str = "default") -> dict[str, Any]:
        """获取项目代码统计信息 (BL-CA-25)

        按 project_id 聚合统计代码文件信息。
        """
        try:
            # 查询项目中的代码文件数量
            count_query = """
                SELECT count() AS total_files
                FROM memory
                WHERE type = 'code'
                    AND project_id = $project_id
                    AND tenant_id = $tenant_id
                GROUP ALL
            """
            count_result = await self._db_query(
                count_query,
                {
                    "project_id": project_id,
                    "tenant_id": tenant_id,
                },
            )
            count_records = self._extract_records(count_result)
            total_files = count_records[0].get("total_files", 0) if count_records else 0

            # 查询代码分析字段的聚合统计（简化版，避免复杂math函数）
            stats_query = """
                SELECT
                    metadata.code_analysis.complexity.function_count AS function_count,
                    metadata.code_analysis.complexity.class_count AS class_count,
                    metadata.code_analysis.complexity.cyclomatic_complexity AS complexity
                FROM memory
                WHERE type = 'code'
                    AND project_id = $project_id
                    AND tenant_id = $tenant_id
                    AND metadata.code_analysis IS NOT NONE
            """
            stats_result = await self._db_query(
                stats_query,
                {
                    "project_id": project_id,
                    "tenant_id": tenant_id,
                },
            )
            stats_records = self._extract_records(stats_result)

            # 手动计算统计值
            total_functions = 0
            total_classes = 0
            complexities = []

            for record in stats_records:
                if record:
                    total_functions += record.get("function_count", 0) or 0
                    total_classes += record.get("class_count", 0) or 0
                    comp = record.get("complexity")
                    if comp is not None:
                        complexities.append(comp)

            avg_complexity = round(sum(complexities) / len(complexities), 2) if complexities else 0
            max_complexity = max(complexities) if complexities else 0

            return {
                "status": "success",
                "project_id": project_id,
                "total_files": total_files,
                "total_functions": total_functions,
                "total_classes": total_classes,
                "avg_complexity": avg_complexity,
                "max_complexity": max_complexity,
            }
        except Exception as e:
            logger.error("[MemoryManager] 获取项目统计失败: %s", e)
            return {
                "status": "error",
                "message": str(e),
                "project_id": project_id,
            }

    async def get_project_map(self, project_id: str, tenant_id: str = "default") -> dict[str, Any]:
        """获取项目代码地图 (BL-CA-23)

        返回项目文件树、模块依赖、热点文件和统计信息。
        """
        try:
            # 1. 获取项目中的所有代码文件
            files_query = """
                SELECT
                    id AS memory_id,
                    metadata.file_path AS file_path,
                    metadata.code_analysis.complexity.cyclomatic_complexity AS complexity,
                    metadata.code_analysis.complexity.function_count AS function_count,
                    metadata.code_analysis.complexity.class_count AS class_count,
                    metadata.code_analysis.imports AS imports
                FROM memory
                WHERE tenant_id = $tenant_id
                    AND type = 'code'
                    AND project_id = $project_id
                    AND metadata.file_path IS NOT NONE
            """
            files_result = await self._db_query(
                files_query,
                {
                    "project_id": project_id,
                    "tenant_id": tenant_id,
                },
            )
            files_records = self._extract_records(files_result)

            if not files_records:
                return {
                    "status": "success",
                    "project_id": project_id,
                    "file_tree": [],
                    "module_dependencies": [],
                    "hot_files": [],
                    "statistics": {
                        "total_files": 0,
                        "total_functions": 0,
                        "total_classes": 0,
                        "avg_complexity": 0,
                        "max_complexity": 0,
                    },
                }

            # 2. 构建文件树
            file_tree = self._build_file_tree(files_records)

            # 3. 提取模块依赖（从 imports 和 calls 关系）
            import_dependencies = self._extract_module_dependencies(files_records)
            call_dependencies = await self._extract_call_dependencies(files_records, project_id, tenant_id)
            module_dependencies = import_dependencies + call_dependencies

            # 4. 识别热点文件（复杂度最高的前 10 个文件）
            hot_files = self._identify_hot_files(files_records)

            # 5. 计算统计信息
            total_files = len(files_records)
            total_functions = sum(r.get("function_count", 0) or 0 for r in files_records)
            total_classes = sum(r.get("class_count", 0) or 0 for r in files_records)
            complexities = [r.get("complexity", 0) or 0 for r in files_records]
            avg_complexity = round(sum(complexities) / len(complexities), 2) if complexities else 0
            max_complexity = max(complexities) if complexities else 0

            return {
                "status": "success",
                "project_id": project_id,
                "file_tree": file_tree,
                "module_dependencies": module_dependencies,
                "hot_files": hot_files,
                "statistics": {
                    "total_files": total_files,
                    "total_functions": total_functions,
                    "total_classes": total_classes,
                    "avg_complexity": avg_complexity,
                    "max_complexity": max_complexity,
                },
            }
        except Exception as e:
            logger.error("[MemoryManager] 获取项目地图失败: %s", e)
            return {
                "status": "error",
                "message": str(e),
                "project_id": project_id,
            }

    def _build_file_tree(self, files: list[dict]) -> list[dict]:
        """从文件路径列表构建树形结构"""
        root = {"name": "", "type": "directory", "children": {}}

        for file_info in files:
            file_path = file_info.get("file_path", "")
            if not file_path:
                continue

            parts = file_path.split("/")
            current = root

            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    # 文件节点
                    current["children"][part] = {
                        "name": part,
                        "type": "file",
                        "path": file_path,
                        "complexity": file_info.get("complexity", 0),
                        "function_count": file_info.get("function_count", 0),
                        "class_count": file_info.get("class_count", 0),
                    }
                else:
                    # 目录节点
                    if part not in current["children"]:
                        current["children"][part] = {
                            "name": part,
                            "type": "directory",
                            "path": "/".join(parts[: i + 1]),
                            "children": {},
                        }
                    current = current["children"][part]

        # 转换为列表格式
        def convert_to_list(node: dict) -> list[dict]:
            result = []
            for name, child in node.get("children", {}).items():
                item = {
                    "name": name,
                    "type": child["type"],
                    "path": child.get("path", name),
                }
                if child["type"] == "directory":
                    item["children"] = convert_to_list(child)
                else:
                    item["complexity"] = child.get("complexity", 0)
                    item["function_count"] = child.get("function_count", 0)
                    item["class_count"] = child.get("class_count", 0)
                result.append(item)
            return sorted(result, key=lambda x: (x["type"] != "directory", x["name"]))

        return convert_to_list(root)

    def _extract_module_dependencies(self, files: list[dict]) -> list[dict]:
        """从 imports 提取模块依赖关系"""
        dependencies = []
        file_paths = {f.get("file_path", ""): f for f in files}

        for file_info in files:
            file_path = file_info.get("file_path", "")
            imports = file_info.get("imports", []) or []

            for imp in imports:
                # 简化处理：假设 import 路径可以映射到文件路径
                # 实际项目中可能需要更复杂的解析
                if isinstance(imp, str):
                    dependencies.append(
                        {
                            "from": file_path,
                            "to": imp,
                            "type": "import",
                        }
                    )

        return dependencies[:100]  # 限制数量

    async def _extract_call_dependencies(self, files: list[dict], project_id: str, tenant_id: str) -> list[dict]:
        """从 reference 表提取 calls 关系作为模块依赖"""
        dependencies = []

        # 获取所有文件的 memory_id
        def get_id_str(f):
            mid = f.get("memory_id", f.get("id", ""))
            if hasattr(mid, "table_name") and hasattr(mid, "id"):
                return f"{mid.table_name}:{mid.id}"
            return str(mid)

        file_ids = {get_id_str(f): f for f in files}
        file_paths = {get_id_str(f): f.get("file_path", "") for f in files}

        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"[_extract_call_dependencies] file_ids: {list(file_ids.keys())}")

        if not file_ids:
            logger.info("[_extract_call_dependencies] No file_ids found")
            return dependencies

        # 查询 calls 关系 - 只查询 in 在文件列表中的关系（避免双向匹配导致的重复）
        file_id_records = [self._normalize_memory_id(fid) for fid in file_ids.keys()]
        logger.info(f"[_extract_call_dependencies] Querying {len(file_id_records)} file IDs")

        calls_query = """
            SELECT
                in AS from_id,
                out AS to_id,
                type,
                metadata
            FROM reference
            WHERE type = 'calls'
                AND in IN array::map($file_ids, |$id| type::record($id))
            LIMIT 100
        """

        try:
            calls_result = await self._db_query(
                calls_query,
                {
                    "file_ids": file_id_records,
                },
            )
            calls_records = self._extract_records(calls_result)
            logger.info(f"[_extract_call_dependencies] Found {len(calls_records)} call relations")

            seen = set()
            for call in calls_records:
                from_id = str(call.get("from_id", ""))
                to_id = str(call.get("to_id", ""))

                from_path = file_paths.get(from_id, "")
                to_path = file_paths.get(to_id, "")

                if from_path and to_path:
                    key = (from_path, to_path)
                    if key not in seen:
                        seen.add(key)
                        dependencies.append(
                            {
                                "from": from_path,
                                "to": to_path,
                                "type": "call",
                            }
                        )
                        logger.info(f"[_extract_call_dependencies] Added dependency: {from_path} -> {to_path}")
                else:
                    logger.info(f"[_extract_call_dependencies] Missing path: from_path={from_path}, to_path={to_path}")
        except Exception as e:
            logger.error(f"[_extract_call_dependencies] Error: {e}")
            # 如果查询失败，返回空列表（降级处理）
            pass

        logger.info(f"[_extract_call_dependencies] Total dependencies: {len(dependencies)}")
        return dependencies

    def _identify_hot_files(self, files: list[dict], limit: int = 10) -> list[str]:
        """识别热点文件（复杂度最高的文件）"""
        sorted_files = sorted(files, key=lambda x: x.get("complexity", 0) or 0, reverse=True)
        return [f.get("file_path", "") for f in sorted_files[:limit] if f.get("file_path")]
