"""WebSocket 消息延迟性能测试

验证 WebSocket 消息延迟 p99 < 100ms。

运行方式：
    uv run python tests/performance/test_websocket_latency.py --clients 10 --messages 100

测试指标：
- p99 延迟 < 100ms
- p95 延迟 < 50ms
- p50 延迟 < 20ms
- 吞吐量 ≥ 1000 msg/s
"""

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

from websockets.client import connect as ws_connect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LatencySample:
    """延迟样本"""

    client_id: int
    message_id: int
    latency_ms: float
    timestamp: float


@dataclass
class LatencyMetrics:
    """延迟指标"""

    total_messages: int = 0
    successful_messages: int = 0
    failed_messages: int = 0
    samples: List[LatencySample] = field(default_factory=list)
    duration: float = 0.0

    @property
    def p50(self) -> float:
        """p50 延迟（中位数）"""
        if not self.samples:
            return 0.0
        return statistics.median(s.latency_ms for s in self.samples)

    @property
    def p95(self) -> float:
        """p95 延迟"""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(s.latency_ms for s in self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def p99(self) -> float:
        """p99 延迟"""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(s.latency_ms for s in self.samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def min_latency(self) -> float:
        """最小延迟"""
        if not self.samples:
            return 0.0
        return min(s.latency_ms for s in self.samples)

    @property
    def max_latency(self) -> float:
        """最大延迟"""
        if not self.samples:
            return 0.0
        return max(s.latency_ms for s in self.samples)

    @property
    def avg_latency(self) -> float:
        """平均延迟"""
        if not self.samples:
            return 0.0
        return statistics.mean(s.latency_ms for s in self.samples)

    @property
    def throughput(self) -> float:
        """吞吐量（msg/s）"""
        if self.duration <= 0:
            return 0.0
        return self.successful_messages / self.duration

    @property
    def std_dev(self) -> float:
        """标准差"""
        if len(self.samples) < 2:
            return 0.0
        return statistics.stdev(s.latency_ms for s in self.samples)


class LatencyTestClient:
    """延迟测试客户端"""

    def __init__(self, client_id: int, url: str):
        self.client_id = client_id
        self.url = url
        self.samples: List[LatencySample] = []
        self.success_count = 0
        self.fail_count = 0

    async def measure_latency(self, message_id: int, timeout: float = 5.0) -> Optional[float]:
        """测量单条消息延迟

        Returns:
            延迟（毫秒），如果失败返回 None
        """
        try:
            start_time = time.perf_counter()

            async with ws_connect(self.url) as websocket:
                message = {
                    "type": "ping",
                    "client_id": self.client_id,
                    "message_id": message_id,
                    "timestamp": start_time,
                }

                await asyncio.wait_for(
                    websocket.send(json.dumps(message)),
                    timeout=timeout,
                )

                response = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=timeout,
                )

                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000

                sample = LatencySample(
                    client_id=self.client_id,
                    message_id=message_id,
                    latency_ms=latency_ms,
                    timestamp=start_time,
                )
                self.samples.append(sample)
                self.success_count += 1

                return latency_ms

        except Exception as e:
            logger.error("Client %d message %d failed: %s", self.client_id, message_id, e)
            self.fail_count += 1
            return None

    async def run_test(self, num_messages: int, delay: float = 0.01) -> None:
        """运行测试

        Args:
            num_messages: 发送消息数量
            delay: 消息间隔（秒）
        """
        for i in range(num_messages):
            await self.measure_latency(i)
            if delay > 0:
                await asyncio.sleep(delay)


class WebSocketLatencyTest:
    """WebSocket 延迟测试"""

    def __init__(self, url: str, num_clients: int, messages_per_client: int):
        self.url = url
        self.num_clients = num_clients
        self.messages_per_client = messages_per_client
        self.clients: List[LatencyTestClient] = []
        self.metrics = LatencyMetrics()

    async def run(self) -> LatencyMetrics:
        """运行测试"""
        logger.info("Starting WebSocket latency test")
        logger.info("URL: %s", self.url)
        logger.info("Clients: %d", self.num_clients)
        logger.info("Messages per client: %d", self.messages_per_client)
        logger.info("Total messages: %d", self.num_clients * self.messages_per_client)

        start_time = time.time()

        try:
            await self._run_all_clients()
        finally:
            end_time = time.time()
            self.metrics.duration = end_time - start_time

        self._aggregate_metrics()

        return self.metrics

    async def _run_all_clients(self) -> None:
        """运行所有客户端"""
        logger.info("Starting %d clients...", self.num_clients)

        self.clients = [LatencyTestClient(i, self.url) for i in range(self.num_clients)]

        tasks = [client.run_test(self.messages_per_client) for client in self.clients]

        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("All clients completed")

    def _aggregate_metrics(self) -> None:
        """聚合指标"""
        for client in self.clients:
            self.metrics.samples.extend(client.samples)
            self.metrics.successful_messages += client.success_count
            self.metrics.failed_messages += client.fail_count
            self.metrics.total_messages += client.success_count + client.fail_count


def print_report(metrics: LatencyMetrics) -> None:
    """打印测试报告"""
    print("\n" + "=" * 60)
    print("WebSocket Latency Test Report")
    print("=" * 60)

    print(f"\nTest Duration: {metrics.duration:.2f}s")
    print(f"Total Messages: {metrics.total_messages}")
    print(f"Successful: {metrics.successful_messages}")
    print(f"Failed: {metrics.failed_messages}")

    success_rate = 0.0
    if metrics.total_messages > 0:
        success_rate = metrics.successful_messages / metrics.total_messages * 100
    print(f"Success Rate: {success_rate:.1f}%")

    print("\nLatency Statistics:")
    print(f"  Min:    {metrics.min_latency:.2f} ms")
    print(f"  Avg:    {metrics.avg_latency:.2f} ms")
    print(f"  Max:    {metrics.max_latency:.2f} ms")
    print(f"  StdDev: {metrics.std_dev:.2f} ms")

    print("\nPercentiles:")
    print(f"  p50: {metrics.p50:.2f} ms")
    print(f"  p95: {metrics.p95:.2f} ms")
    print(f"  p99: {metrics.p99:.2f} ms")

    print(f"\nThroughput: {metrics.throughput:.2f} msg/s")

    print("\nTest Results:")
    passed = True

    if metrics.p50 > 20:
        print("  ❌ FAILED: p50 latency > 20ms")
        passed = False
    else:
        print("  ✅ PASSED: p50 latency within limit")

    if metrics.p95 > 50:
        print("  ❌ FAILED: p95 latency > 50ms")
        passed = False
    else:
        print("  ✅ PASSED: p95 latency within limit")

    if metrics.p99 > 100:
        print("  ❌ FAILED: p99 latency > 100ms")
        passed = False
    else:
        print("  ✅ PASSED: p99 latency within limit")

    if metrics.throughput < 1000:
        print("  ❌ FAILED: Throughput < 1000 msg/s")
        passed = False
    else:
        print("  ✅ PASSED: Throughput within limit")

    print("\n" + "=" * 60)
    if passed:
        print("OVERALL: ✅ PASSED")
    else:
        print("OVERALL: ❌ FAILED")
    print("=" * 60 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="WebSocket Latency Test",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=10,
        help="Number of concurrent clients (default: 10)",
    )
    parser.add_argument(
        "--messages",
        type=int,
        default=100,
        help="Messages per client (default: 100)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.01,
        help="Delay between messages in seconds (default: 0.01)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="ws://localhost:17999/ws/memories/live",
        help="WebSocket URL",
    )

    args = parser.parse_args()

    test = WebSocketLatencyTest(args.url, args.clients, args.messages)

    try:
        metrics = asyncio.run(test.run())
        print_report(metrics)

        passed = metrics.p50 <= 20 and metrics.p95 <= 50 and metrics.p99 <= 100 and metrics.throughput >= 1000
        sys.exit(0 if passed else 1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("Test failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
