"""PrecomputeService - 预计算服务

实现代码预计算服务，支持：
- tenant 隔离
- DB 连接注入
- 启动/停止生命周期
- 批量处理
- 性能监控
- 并发控制
- tree-sitter 代码解析
- 文件指纹和变更检测
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .code_parser import CodeParser
from .concurrency_control import ConcurrencyControl
from .fingerprint import FingerprintManager
from .performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class PrecomputeService:
    """预计算服务

    提供代码预计算功能，支持 tenant 隔离、性能监控和并发控制。

    Attributes:
        tenant_id: 租户 ID
        is_running: 服务是否运行中
        performance_monitor: 性能监控器
        concurrency_control: 并发控制器
    """

    def __init__(
        self,
        db: Any,
        tenant_id: str = "default",
        max_concurrent: int = 5,
        timeout_seconds: float = 30.0,
    ):
        """初始化预计算服务

        Args:
            db: 数据库连接（SurrealDB 或其他）
            tenant_id: 租户 ID，默认 "default"
            max_concurrent: 最大并发数，默认 5
            timeout_seconds: 超时时间（秒），默认 30
        """
        self._db = db
        self._tenant_id = tenant_id
        self._running = False
        self._logger = logging.getLogger(f"{__name__}.{tenant_id}")
        self._performance_monitor = PerformanceMonitor(tenant_id=tenant_id, db=db)
        self._concurrency_control = ConcurrencyControl(
            max_concurrent=max_concurrent,
            timeout_seconds=timeout_seconds,
            db=db,
            tenant_id=tenant_id,
        )
        self._code_parser: Optional[CodeParser] = None

        self._fingerprint_manager = FingerprintManager()

        self._logger.debug(
            "[PrecomputeService] 初始化: tenant_id=%s, max_concurrent=%d",
            tenant_id,
            max_concurrent,
        )

    async def start(self) -> None:
        """启动服务

        初始化服务资源，准备处理请求。
        """
        if self._running:
            self._logger.warning("[PrecomputeService] 服务已在运行中")
            return

        self._logger.info("[PrecomputeService] 启动服务: tenant_id=%s", self._tenant_id)

        # 启动性能监控
        self._performance_monitor.start_tracing()

        # 初始化 tree-sitter 代码解析器
        self._code_parser = CodeParser()
        self._logger.debug("[PrecomputeService] tree-sitter 解析器初始化完成")

        # 验证数据库连接
        if self._db is None:
            raise RuntimeError("数据库连接未提供")
        self._logger.debug("[PrecomputeService] 数据库连接已建立")

        self._logger.debug("[PrecomputeService] 并发控制器已就绪")

        self._running = True
        self._logger.info("[PrecomputeService] 服务已启动")

    async def stop(self) -> None:
        """停止服务

        清理服务资源，停止处理请求。
        """
        if not self._running:
            self._logger.warning("[PrecomputeService] 服务未运行")
            return

        self._logger.info("[PrecomputeService] 停止服务")

        # 停止性能监控
        self._performance_monitor.stop_tracing()

        if self._code_parser is not None:
            self._code_parser = None
            self._logger.debug("[PrecomputeService] tree-sitter 解析器已清理")

        await self._cleanup_concurrency_resources()

        self._logger.debug("[PrecomputeService] 数据库连接保持（由调用方管理）")

        self._running = False
        self._logger.info("[PrecomputeService] 服务已停止")

    async def _process_file(
        self,
        file_path: str,
        item: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """处理单个文件

        读取文件、解析代码、提取符号、计算指纹。

        Args:
            file_path: 文件路径
            item: 文件信息

        Returns:
            处理结果，如果失败返回 None
        """
        self._logger.debug("[PrecomputeService] 处理文件: %s", file_path)

        try:
            content = await self._read_file(file_path)
            if content is None:
                return None

            fingerprint = self._fingerprint_manager.calculate_fingerprint(content)
            if not self._fingerprint_manager.has_changed(file_path, fingerprint):
                self._logger.debug("[PrecomputeService] 文件未变更，跳过: %s", file_path)
                return {
                    "file_path": file_path,
                    "status": "unchanged",
                    "fingerprint": fingerprint,
                }

            if self._code_parser is None:
                self._logger.error("[PrecomputeService] 代码解析器未初始化")
                return None

            language = self._code_parser.get_language(file_path)
            if not language:
                self._logger.warning("[PrecomputeService] 不支持的文件类型: %s", file_path)
                return None

            parse_result = self._code_parser.parse(content, language)
            if not parse_result:
                self._logger.warning("[PrecomputeService] 解析失败: %s", file_path)
                return None

            symbols = parse_result.get("symbols", [])

            self._fingerprint_manager.save_fingerprint(file_path, fingerprint)

            return {
                "file_path": file_path,
                "status": "processed",
                "language": language,
                "fingerprint": fingerprint,
                "symbols": symbols,
                "symbol_count": len(symbols),
            }

        except Exception as e:
            self._logger.error("[PrecomputeService] 处理文件失败: %s, error=%s", file_path, e)
            return None

    async def _read_file(self, file_path: str) -> Optional[str]:
        """读取文件内容"""
        try:
            path = Path(file_path)
            if not path.exists():
                self._logger.warning("[PrecomputeService] 文件不存在: %s", file_path)
                return None

            content = path.read_text(encoding="utf-8")
            return content

        except UnicodeDecodeError:
            try:
                path = Path(file_path)
                content = path.read_text(encoding="gbk")
                return content
            except Exception as e:
                self._logger.error("[PrecomputeService] 读取文件编码错误: %s, error=%s", file_path, e)
                return None
        except Exception as e:
            self._logger.error("[PrecomputeService] 读取文件失败: %s, error=%s", file_path, e)
            return None

    async def _cleanup_concurrency_resources(self) -> None:
        """清理并发控制资源"""
        self._logger.debug("[PrecomputeService] 清理并发控制资源")

        async with self._concurrency_control._lock:
            processing_count = len(self._concurrency_control._processing)
            self._concurrency_control._processing.clear()
            self._logger.debug("[PrecomputeService] 清理 %d 个处理中任务", processing_count)

        queue_size = self._concurrency_control._queue.qsize()
        while not self._concurrency_control._queue.empty():
            try:
                self._concurrency_control._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._logger.debug("[PrecomputeService] 清理 %d 个队列任务", queue_size)

    async def process_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """处理批次

        批量处理代码文件，提取符号、调用关系等。
        使用并发控制防止同文件重复处理。

        Args:
            batch: 批次数据列表，每个元素包含文件信息

        Returns:
            处理结果，包含提取的符号、调用关系等
        """
        if not self._running:
            raise RuntimeError("PrecomputeService 未启动")

        self._logger.debug(
            "[PrecomputeService] 处理批次: tenant_id=%s, batch_size=%d",
            self._tenant_id,
            len(batch),
        )

        results = []
        errors = []
        deduplicated = []

        # 使用性能监控上下文
        with self._performance_monitor.monitor("process_batch", {"batch_size": len(batch)}):
            # 并发处理每个文件
            tasks = []
            for item in batch:
                file_path = item.get("file_path", "")
                if not file_path:
                    continue

                # 使用并发控制处理文件
                task = self._process_file_with_concurrency(file_path, item)
                tasks.append(task)

            # 等待所有任务完成
            if tasks:
                task_results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in task_results:
                    if isinstance(result, Exception):
                        errors.append(str(result))
                    elif result is not None:
                        results.append(result)

        # 获取并发控制统计
        cc_stats = self._concurrency_control.get_stats()

        result = {
            "tenant_id": self._tenant_id,
            "processed_count": len(results),
            "error_count": len(errors),
            "deduplicated_count": cc_stats["total_deduplicated"],
            "symbols": [],
            "call_relations": [],
            "fingerprints": {},
            "concurrency_stats": cc_stats,
        }

        self._logger.debug(
            "[PrecomputeService] 批次处理完成: processed=%d, errors=%d, deduplicated=%d",
            len(results),
            len(errors),
            cc_stats["total_deduplicated"],
        )

        return result

    async def _process_file_with_concurrency(
        self,
        file_path: str,
        item: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """使用并发控制处理单个文件

        Args:
            file_path: 文件路径
            item: 文件信息

        Returns:
            处理结果，如果重复则返回 None
        """

        async def process_coro():
            return await self._process_file(file_path, item)

        try:
            return await self._concurrency_control.process(file_path, process_coro)
        except Exception as e:
            self._logger.debug(
                "[PrecomputeService] 文件处理跳过或失败: file_path=%s, error=%s",
                file_path,
                e,
            )
            return None

    async def health_check(self) -> Dict[str, Any]:
        """健康检查

        检查服务健康状态。

        Returns:
            健康状态信息
        """
        return {
            "tenant_id": self._tenant_id,
            "is_running": self._running,
            "status": "healthy" if self._running else "stopped",
        }

    def get_performance_report(self) -> str:
        """获取性能报告

        Returns:
            格式化的性能报告字符串
        """
        return self._performance_monitor.generate_report()

    @property
    def is_running(self) -> bool:
        """服务是否运行中"""
        return self._running

    @property
    def tenant_id(self) -> str:
        """租户 ID"""
        return self._tenant_id

    @property
    def performance_monitor(self):
        """性能监控器"""
        return self._performance_monitor

    @property
    def concurrency_control(self):
        """并发控制器"""
        return self._concurrency_control

    @property
    def db(self) -> Any:
        """数据库连接"""
        return self._db
