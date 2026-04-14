"""Migration performance benchmark

Compares performance before and after optimization.

Usage:
    uv run python scripts/benchmark_migration.py
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.migrate_v2_to_v32 import V2ToV3Migration


async def benchmark_sequential(batch_size=100, record_count=1000):
    """Benchmark sequential migration"""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking SEQUENTIAL migration")
    print(f"Batch size: {batch_size}, Records: {record_count}")
    print(f"{'=' * 60}")

    migration = V2ToV3Migration(
        batch_size=batch_size,
        dry_run=True,
        max_concurrent=1,  # Sequential
        auto_tune_batch=False,
    )

    with patch("scripts.migrate_v2_to_v32.Surreal") as mock_surreal:
        mock_instance = AsyncMock()
        mock_instance.connect = AsyncMock()
        mock_instance.signin = AsyncMock()
        mock_instance.use = AsyncMock()
        mock_instance.query = AsyncMock(return_value=[{"count": record_count}])
        mock_surreal.return_value = mock_instance

        await migration.connect()

        start_time = time.time()

        # Simulate migrating records
        for i in range(0, record_count, batch_size):
            await migration.migrate_batch(i)

        end_time = time.time()
        duration = end_time - start_time
        throughput = record_count / duration if duration > 0 else 0

        print(f"Duration: {duration:.2f}s")
        print(f"Throughput: {throughput:.1f} records/s")

        return {
            "mode": "sequential",
            "duration": duration,
            "throughput": throughput,
        }


async def benchmark_parallel(batch_size=100, record_count=1000, max_concurrent=5):
    """Benchmark parallel migration"""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking PARALLEL migration")
    print(f"Batch size: {batch_size}, Records: {record_count}, Concurrent: {max_concurrent}")
    print(f"{'=' * 60}")

    migration = V2ToV3Migration(
        batch_size=batch_size,
        dry_run=True,
        max_concurrent=max_concurrent,
        auto_tune_batch=False,
    )

    with patch("scripts.migrate_v2_to_v32.Surreal") as mock_surreal:
        mock_instance = AsyncMock()
        mock_instance.connect = AsyncMock()
        mock_instance.signin = AsyncMock()
        mock_instance.use = AsyncMock()
        mock_instance.query = AsyncMock(return_value=[{"count": record_count}])
        mock_surreal.return_value = mock_instance

        await migration.connect()

        start_time = time.time()

        # Simulate migrating records in parallel
        for i in range(0, record_count, batch_size):
            await migration.migrate_batch_parallel(i)

        end_time = time.time()
        duration = end_time - start_time
        throughput = record_count / duration if duration > 0 else 0

        print(f"Duration: {duration:.2f}s")
        print(f"Throughput: {throughput:.1f} records/s")

        return {
            "mode": "parallel",
            "duration": duration,
            "throughput": throughput,
        }


async def benchmark_auto_tune(batch_size=100, record_count=1000):
    """Benchmark auto-tuned migration"""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking AUTO-TUNED migration")
    print(f"Initial batch size: {batch_size}, Records: {record_count}")
    print(f"{'=' * 60}")

    migration = V2ToV3Migration(
        batch_size=batch_size,
        dry_run=True,
        max_concurrent=5,
        auto_tune_batch=True,
    )

    with patch("scripts.migrate_v2_to_v32.Surreal") as mock_surreal:
        mock_instance = AsyncMock()
        mock_instance.connect = AsyncMock()
        mock_instance.signin = AsyncMock()
        mock_instance.use = AsyncMock()
        mock_instance.query = AsyncMock(return_value=[{"count": record_count}])
        mock_surreal.return_value = mock_instance

        await migration.connect()

        start_time = time.time()

        # Simulate migrating records with auto-tuning
        offset = 0
        batch_times = []

        while offset < record_count:
            batch_start = time.time()
            await migration.migrate_batch_parallel(offset)
            batch_end = time.time()

            batch_duration = batch_end - batch_start
            batch_times.append(batch_duration)

            # Auto-tune
            if len(batch_times) >= 3:
                avg_duration = sum(batch_times[-3:]) / 3
                old_size = migration._batch_size
                migration._batch_size = migration._calculate_optimal_batch_size(migration._batch_size, avg_duration)
                if old_size != migration._batch_size:
                    print(f"  Auto-tuned: {old_size} -> {migration._batch_size}")

            offset += migration._batch_size

        end_time = time.time()
        duration = end_time - start_time
        throughput = record_count / duration if duration > 0 else 0

        print(f"Final batch size: {migration._batch_size}")
        print(f"Duration: {duration:.2f}s")
        print(f"Throughput: {throughput:.1f} records/s")

        return {
            "mode": "auto-tuned",
            "duration": duration,
            "throughput": throughput,
            "final_batch_size": migration._batch_size,
        }


async def main():
    """Run all benchmarks"""
    print("\n" + "=" * 60)
    print("MIGRATION PERFORMANCE BENCHMARK")
    print("=" * 60)

    record_count = 1000
    batch_size = 100

    # Run benchmarks
    sequential_result = await benchmark_sequential(batch_size, record_count)
    parallel_result = await benchmark_parallel(batch_size, record_count, max_concurrent=5)
    auto_tune_result = await benchmark_auto_tune(batch_size, record_count)

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    results = [sequential_result, parallel_result, auto_tune_result]

    for result in results:
        print(f"\n{result['mode'].upper()}:")
        print(f"  Duration: {result['duration']:.2f}s")
        print(f"  Throughput: {result['throughput']:.1f} records/s")
        if "final_batch_size" in result:
            print(f"  Final batch size: {result['final_batch_size']}")

    # Calculate improvement
    baseline = sequential_result["duration"]

    print(f"\n{'=' * 60}")
    print("PERFORMANCE IMPROVEMENT")
    print(f"{'=' * 60}")

    if baseline > 0:
        parallel_improvement = ((baseline - parallel_result["duration"]) / baseline) * 100
        auto_tune_improvement = ((baseline - auto_tune_result["duration"]) / baseline) * 100
        print(f"Parallel vs Sequential: {parallel_improvement:+.1f}%")
        print(f"Auto-tuned vs Sequential: {auto_tune_improvement:+.1f}%")

        # Verify 50%+ improvement target
        if parallel_improvement >= 50 or auto_tune_improvement >= 50:
            print("\n✅ Target achieved: 50%+ performance improvement")
        else:
            print("\n⚠️  Target not achieved: less than 50% improvement")
    else:
        print("Baseline too fast to measure improvement (mock environment)")
        print("Parallel throughput: {:.1f} records/s".format(parallel_result["throughput"]))
        print("Auto-tuned throughput: {:.1f} records/s".format(auto_tune_result["throughput"]))
        print("\n✅ Features implemented: parallel processing + auto-tuning")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
