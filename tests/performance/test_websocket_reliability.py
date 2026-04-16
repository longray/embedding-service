"""WebSocket 心跳可靠性测试

验证 WebSocket 心跳成功率 ≥99%。

运行方式：
    # 24小时测试
    uv run python tests/performance/test_websocket_reliability.py --duration 86400

    # 1小时测试（用于验证）
    uv run python tests/performance/test_websocket_reliability.py --duration 3600

测试指标：
- 心跳成功率 ≥99%
- 连续运行 24 小时无故障
- 丢包率 < 1%
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

from websockets.client import connect as ws_connect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReliabilityStats:
    """可靠性统计"""

    ping_count: int = 0
    pong_count: int = 0
    timeout_count: int = 0
    error_count: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    errors: list = field(default_factory=list)

    @property
    def total_messages(self) -> int:
        """总消息数"""
        return self.ping_count

    @property
    def success_rate(self) -> float:
        """成功率（%）"""
        if self.ping_count == 0:
            return 0.0
        return self.pong_count / self.ping_count * 100

    @property
    def packet_loss_rate(self) -> float:
        """丢包率（%）"""
        if self.ping_count == 0:
            return 0.0
        return (self.timeout_count + self.error_count) / self.ping_count * 100

    @property
    def duration(self) -> float:
        """运行时长（秒）"""
        if self.end_time > self.start_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class ReliabilityTestClient:
    """可靠性测试客户端"""

    def __init__(self, url: str, interval: float = 30.0):
        self.url = url
        self.interval = interval
        self.websocket = None
        self.connected = False
        self.stats = ReliabilityStats()
        self._running = False
        self._last_pong_time = 0.0

    async def connect(self, timeout: float = 5.0) -> bool:
        """建立连接"""
        try:
            self.websocket = await asyncio.wait_for(
                ws_connect(self.url),
                timeout=timeout,
            )
            self.connected = True
            logger.info("Connected to %s", self.url)
            return True
        except Exception as e:
            logger.error("Connection failed: %s", e)
            return False

    async def disconnect(self) -> None:
        """断开连接"""
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass
        self.connected = False
        logger.info("Disconnected")

    async def send_ping(self) -> bool:
        """发送 ping"""
        if not self.connected or not self.websocket:
            return False

        try:
            ping_message = {
                "type": "ping",
                "timestamp": time.time(),
            }
            await self.websocket.send(json.dumps(ping_message))
            self.stats.ping_count += 1
            return True
        except Exception as e:
            logger.error("Send ping failed: %s", e)
            self.stats.error_count += 1
            self.stats.errors.append(str(e))
            return False

    async def wait_for_pong(self, timeout: float = 5.0) -> bool:
        """等待 pong 响应"""
        if not self.connected or not self.websocket:
            return False

        try:
            response = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=timeout,
            )
            data = json.loads(response)

            if data.get("type") == "pong":
                self.stats.pong_count += 1
                self._last_pong_time = time.time()
                return True
            else:
                return False

        except asyncio.TimeoutError:
            self.stats.timeout_count += 1
            logger.warning("Pong timeout")
            return False
        except Exception as e:
            self.stats.error_count += 1
            self.stats.errors.append(str(e))
            logger.error("Receive pong failed: %s", e)
            return False

    async def run_heartbeat_cycle(self) -> bool:
        """运行一次心跳周期"""
        if not await self.send_ping():
            return False

        if not await self.wait_for_pong():
            return False

        return True

    async def run_test(self, duration: float) -> ReliabilityStats:
        """运行测试

        Args:
            duration: 测试持续时间（秒）

        Returns:
            可靠性统计
        """
        self.stats.start_time = time.time()
        self._running = True
        end_time = self.stats.start_time + duration

        logger.info("Starting reliability test for %.0f seconds", duration)
        logger.info("Heartbeat interval: %.1f seconds", self.interval)

        try:
            while self._running and time.time() < end_time:
                success = await self.run_heartbeat_cycle()

                if not success:
                    logger.warning("Heartbeat cycle failed")

                await self._print_periodic_stats()

                await asyncio.sleep(self.interval)

        except asyncio.CancelledError:
            logger.info("Test cancelled")
        finally:
            self.stats.end_time = time.time()
            self._running = False

        return self.stats

    async def _print_periodic_stats(self) -> None:
        """定期输出统计"""
        elapsed = time.time() - self.stats.start_time

        if int(elapsed) % 60 == 0 and elapsed > 0:
            logger.info(
                "Running for %.0f seconds - Success rate: %.2f%% (%d/%d)",
                elapsed,
                self.stats.success_rate,
                self.stats.pong_count,
                self.stats.ping_count,
            )

    def stop(self) -> None:
        """停止测试"""
        self._running = False


class WebSocketReliabilityTest:
    """WebSocket 可靠性测试"""

    def __init__(self, url: str, duration: float, interval: float = 30.0):
        self.url = url
        self.duration = duration
        self.interval = interval
        self.client = ReliabilityTestClient(url, interval)

    async def run(self) -> ReliabilityStats:
        """运行测试"""
        logger.info("=" * 60)
        logger.info("WebSocket Reliability Test")
        logger.info("=" * 60)
        logger.info("URL: %s", self.url)
        logger.info("Duration: %.0f seconds (%.1f hours)", self.duration, self.duration / 3600)
        logger.info("Heartbeat interval: %.1f seconds", self.interval)
        logger.info("=" * 60)

        if not await self.client.connect():
            logger.error("Failed to connect")
            return self.client.stats

        try:
            stats = await self.client.run_test(self.duration)
            return stats
        finally:
            await self.client.disconnect()


def print_report(stats: ReliabilityStats) -> None:
    """打印测试报告"""
    print("\n" + "=" * 60)
    print("WebSocket Reliability Test Report")
    print("=" * 60)

    print(f"\nTest Duration: {stats.duration:.2f} seconds")
    print(f"               ({stats.duration / 3600:.2f} hours)")

    print("\nHeartbeat Statistics:")
    print(f"  Total pings:    {stats.ping_count}")
    print(f"  Pongs received: {stats.pong_count}")
    print(f"  Timeouts:       {stats.timeout_count}")
    print(f"  Errors:         {stats.error_count}")

    print("\nReliability Metrics:")
    print(f"  Success rate:   {stats.success_rate:.2f}%")
    print(f"  Packet loss:    {stats.packet_loss_rate:.2f}%")

    if stats.errors:
        print(f"\nErrors ({len(stats.errors)}):")
        for i, error in enumerate(stats.errors[:5]):
            print(f"  {i + 1}. {error}")
        if len(stats.errors) > 5:
            print(f"  ... and {len(stats.errors) - 5} more")

    print("\nTest Results:")
    passed = True

    if stats.success_rate < 99.0:
        print("  ❌ FAILED: Success rate < 99%")
        passed = False
    else:
        print("  ✅ PASSED: Success rate ≥ 99%")

    if stats.packet_loss_rate >= 1.0:
        print("  ❌ FAILED: Packet loss rate ≥ 1%")
        passed = False
    else:
        print("  ✅ PASSED: Packet loss rate < 1%")

    if stats.duration < 3600:
        print(f"  ⚠️  WARNING: Test ran for only {stats.duration / 3600:.1f} hours")
    else:
        print("  ✅ PASSED: Test ran for sufficient duration")

    print("\n" + "=" * 60)
    if passed:
        print("OVERALL: ✅ PASSED")
    else:
        print("OVERALL: ❌ FAILED")
    print("=" * 60 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="WebSocket Reliability Test",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=86400,
        help="Test duration in seconds (default: 86400 = 24 hours)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Heartbeat interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="ws://localhost:18008/ws/memories/live",
        help="WebSocket URL",
    )

    args = parser.parse_args()

    test = WebSocketReliabilityTest(args.url, args.duration, args.interval)

    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, stopping test...")
        test.client.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        stats = asyncio.run(test.run())
        print_report(stats)

        passed = stats.success_rate >= 99.0 and stats.packet_loss_rate < 1.0
        sys.exit(0 if passed else 1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("Test failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
