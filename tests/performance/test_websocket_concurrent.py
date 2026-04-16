"""WebSocket 并发连接性能测试

验证 WebSocket 服务端支持 ≥1000 并发连接。

运行方式：
    uv run python tests/performance/test_websocket_concurrent.py --clients 1000 --duration 60

测试指标：
- 支持 1000+ 并发连接
- 内存使用 < 2GB
- CPU 使用 < 80%
- 无连接丢失
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from websockets.client import connect as ws_connect
from websockets.legacy.client import WebSocketClientProtocol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConnectionStats:
    """连接统计"""

    total: int = 0
    success: int = 0
    failed: int = 0
    disconnected: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """性能指标"""

    duration: float = 0.0
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    connection_stats: ConnectionStats = field(default_factory=ConnectionStats)


class WebSocketLoadClient:
    """WebSocket 负载测试客户端"""

    def __init__(self, client_id: int, url: str):
        self.client_id = client_id
        self.url = url
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.connected = False
        self.messages_received = 0
        self.messages_sent = 0
        self.errors: List[str] = []

    async def connect(self, timeout: float = 5.0) -> bool:
        """建立连接"""
        try:
            self.websocket = await asyncio.wait_for(
                ws_connect(self.url),
                timeout=timeout,
            )
            self.connected = True
            logger.debug("Client %d connected", self.client_id)
            return True
        except Exception as e:
            self.errors.append(str(e))
            logger.error("Client %d connection failed: %s", self.client_id, e)
            return False

    async def send_message(self, message: dict) -> bool:
        """发送消息"""
        if not self.connected or not self.websocket:
            return False

        try:
            await self.websocket.send(json.dumps(message))
            self.messages_sent += 1
            return True
        except Exception as e:
            self.errors.append(str(e))
            return False

    async def receive_message(self, timeout: float = 1.0) -> Optional[dict]:
        """接收消息"""
        if not self.connected or not self.websocket:
            return None

        try:
            message = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=timeout,
            )
            self.messages_received += 1
            return json.loads(message)
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            self.errors.append(str(e))
            return None

    async def heartbeat(self) -> bool:
        """发送心跳"""
        return await self.send_message({"type": "ping", "client_id": self.client_id})

    async def disconnect(self) -> None:
        """断开连接"""
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass
        self.connected = False
        logger.debug("Client %d disconnected", self.client_id)


class PerformanceMonitor:
    """性能监控"""

    def __init__(self):
        self.peak_memory_mb = 0.0
        self.cpu_samples: List[float] = []
        self._monitoring = False
        self._task: Optional[asyncio.Task] = None

    async def start_monitoring(self, interval: float = 1.0) -> None:
        """开始监控"""
        self._monitoring = True
        self._task = asyncio.create_task(self._monitor_loop(interval))

    async def stop_monitoring(self) -> None:
        """停止监控"""
        self._monitoring = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self, interval: float) -> None:
        """监控循环"""
        while self._monitoring:
            try:
                memory_mb = self._get_memory_usage()
                cpu_percent = self._get_cpu_usage()

                self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
                self.cpu_samples.append(cpu_percent)

                await asyncio.sleep(interval)
            except Exception as e:
                logger.error("Monitor error: %s", e)
                await asyncio.sleep(interval)

    def _get_memory_usage(self) -> float:
        """获取内存使用（MB）"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """获取 CPU 使用（%）"""
        try:
            import psutil

            process = psutil.Process()
            return process.cpu_percent(interval=0.1)
        except ImportError:
            return 0.0

    def get_avg_cpu(self) -> float:
        """获取平均 CPU 使用"""
        if not self.cpu_samples:
            return 0.0
        return sum(self.cpu_samples) / len(self.cpu_samples)

    def get_peak_cpu(self) -> float:
        """获取峰值 CPU 使用"""
        if not self.cpu_samples:
            return 0.0
        return max(self.cpu_samples)


class WebSocketConcurrentTest:
    """WebSocket 并发连接测试"""

    def __init__(self, url: str, num_clients: int, duration: int):
        self.url = url
        self.num_clients = num_clients
        self.duration = duration
        self.clients: List[WebSocketLoadClient] = []
        self.monitor = PerformanceMonitor()
        self.stats = ConnectionStats()
        self.metrics = PerformanceMetrics()

    async def run(self) -> PerformanceMetrics:
        """运行测试"""
        logger.info("Starting concurrent WebSocket test")
        logger.info("URL: %s", self.url)
        logger.info("Clients: %d", self.num_clients)
        logger.info("Duration: %ds", self.duration)

        start_time = time.time()

        await self.monitor.start_monitoring()

        try:
            await self._connect_all()
            await self._run_test()
        finally:
            await self.monitor.stop_monitoring()
            await self._disconnect_all()

        end_time = time.time()

        self.metrics.duration = end_time - start_time
        self.metrics.peak_memory_mb = self.monitor.peak_memory_mb
        self.metrics.avg_cpu_percent = self.monitor.get_avg_cpu()
        self.metrics.peak_cpu_percent = self.monitor.get_peak_cpu()
        self.metrics.connection_stats = self.stats

        return self.metrics

    async def _connect_all(self) -> None:
        """建立所有连接"""
        logger.info("Connecting %d clients...", self.num_clients)

        tasks = []
        for i in range(self.num_clients):
            client = WebSocketLoadClient(i, self.url)
            self.clients.append(client)
            tasks.append(client.connect())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            self.stats.total += 1
            if result is True:
                self.stats.success += 1
            else:
                self.stats.failed += 1
                if isinstance(result, Exception):
                    self.stats.errors.append(str(result))

        logger.info("Connected: %d/%d", self.stats.success, self.num_clients)

    async def _run_test(self) -> None:
        """运行测试"""
        logger.info("Running test for %d seconds...", self.duration)

        end_time = time.time() + self.duration

        while time.time() < end_time:
            await self._send_heartbeats()
            await asyncio.sleep(1)

    async def _send_heartbeats(self) -> None:
        """发送心跳"""
        tasks = []
        for client in self.clients:
            if client.connected:
                tasks.append(client.heartbeat())

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if result is True:
                    self.stats.messages_sent += 1

    async def _disconnect_all(self) -> None:
        """断开所有连接"""
        logger.info("Disconnecting all clients...")

        tasks = []
        for client in self.clients:
            if client.connected:
                tasks.append(client.disconnect())

        await asyncio.gather(*tasks, return_exceptions=True)

        self.stats.disconnected = len([c for c in self.clients if not c.connected])


def print_report(metrics: PerformanceMetrics) -> None:
    """打印测试报告"""
    print("\n" + "=" * 60)
    print("WebSocket Concurrent Connection Test Report")
    print("=" * 60)

    print(f"\nTest Duration: {metrics.duration:.2f}s")

    print("\nConnection Statistics:")
    print(f"  Total:     {metrics.connection_stats.total}")
    print(f"  Success:   {metrics.connection_stats.success}")
    print(f"  Failed:    {metrics.connection_stats.failed}")
    print(f"  Disconnected: {metrics.connection_stats.disconnected}")

    success_rate = 0.0
    if metrics.connection_stats.total > 0:
        success_rate = metrics.connection_stats.success / metrics.connection_stats.total * 100
    print(f"  Success Rate: {success_rate:.1f}%")

    print("\nPerformance Metrics:")
    print(f"  Peak Memory: {metrics.peak_memory_mb:.2f} MB")
    print(f"  Avg CPU:     {metrics.avg_cpu_percent:.2f}%")
    print(f"  Peak CPU:    {metrics.peak_cpu_percent:.2f}%")

    print("\nTest Results:")
    passed = True

    if metrics.connection_stats.success < 1000:
        print("  ❌ FAILED: Less than 1000 successful connections")
        passed = False
    else:
        print("  ✅ PASSED: 1000+ concurrent connections")

    if metrics.peak_memory_mb > 2048:
        print("  ❌ FAILED: Memory usage exceeded 2GB")
        passed = False
    else:
        print("  ✅ PASSED: Memory usage within limit")

    if metrics.peak_cpu_percent > 80:
        print("  ❌ FAILED: CPU usage exceeded 80%")
        passed = False
    else:
        print("  ✅ PASSED: CPU usage within limit")

    if metrics.connection_stats.disconnected > 0:
        print(f"  ⚠️  WARNING: {metrics.connection_stats.disconnected} connections lost")
    else:
        print("  ✅ PASSED: No connections lost")

    print("\n" + "=" * 60)
    if passed:
        print("OVERALL: ✅ PASSED")
    else:
        print("OVERALL: ❌ FAILED")
    print("=" * 60 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="WebSocket Concurrent Connection Test",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=1000,
        help="Number of concurrent clients (default: 1000)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Test duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="ws://localhost:18008/ws/memories/live",
        help="WebSocket URL",
    )

    args = parser.parse_args()

    test = WebSocketConcurrentTest(args.url, args.clients, args.duration)

    try:
        metrics = asyncio.run(test.run())
        print_report(metrics)

        sys.exit(0 if metrics.connection_stats.success >= 1000 else 1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("Test failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
