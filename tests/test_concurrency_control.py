"""Tests for ConcurrencyControl"""

import asyncio

import pytest

from wrapper.src.services.concurrency_control import (
    ConcurrencyControl,
    DuplicateTaskError,
)


class TestConcurrencyControlInitialization:
    """Test ConcurrencyControl initialization"""

    def test_basic_initialization(self):
        """Test basic initialization"""
        cc = ConcurrencyControl(max_concurrent=5, timeout_seconds=30.0, max_queue_size=100)

        assert cc.max_concurrent == 5
        assert cc.max_concurrent_reached == 0
        assert cc.current_processing == 0
        assert cc.queue_size == 0

    def test_default_initialization(self):
        """Test initialization with default values"""
        cc = ConcurrencyControl()

        assert cc.max_concurrent == 5
        assert cc.timeout_seconds == 30.0
        assert cc.max_queue_size == 100


class TestConcurrencyControlProcess:
    """Test ConcurrencyControl process method"""

    @pytest.mark.asyncio
    async def test_process_single_task(self):
        """Test processing a single task"""
        cc = ConcurrencyControl()

        async def task():
            await asyncio.sleep(0.01)
            return "result"

        result = await cc.process("task_1", task)
        assert result == "result"
        assert cc.current_processing == 0

    @pytest.mark.asyncio
    async def test_process_multiple_tasks_sequential(self):
        """Test processing multiple tasks sequentially"""
        cc = ConcurrencyControl(max_concurrent=1)
        results = []

        async def task(n):
            await asyncio.sleep(0.01)
            results.append(n)
            return n

        for i in range(3):
            await cc.process(f"task_{i}", lambda i=i: task(i))

        assert results == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_process_concurrent_limit(self):
        """Test concurrent limit is enforced"""
        cc = ConcurrencyControl(max_concurrent=2)
        concurrent_count = 0
        max_concurrent_observed = 0

        async def task():
            nonlocal concurrent_count, max_concurrent_observed
            concurrent_count += 1
            max_concurrent_observed = max(max_concurrent_observed, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return "done"

        tasks = [cc.process(f"task_{i}", task) for i in range(5)]
        await asyncio.gather(*tasks)

        assert max_concurrent_observed <= 2
        assert cc.max_concurrent_reached <= 2

    @pytest.mark.asyncio
    async def test_process_deduplication(self):
        """Test task deduplication"""
        cc = ConcurrencyControl()
        call_count = 0

        async def slow_task():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return "result"

        # Start first task
        task1 = asyncio.create_task(cc.process("same_id", slow_task))
        await asyncio.sleep(0.01)  # Let first task start

        # Try to start second task with same ID
        with pytest.raises(DuplicateTaskError):
            await cc.process("same_id", slow_task)

        await task1
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_process_timeout(self):
        """Test task timeout"""
        cc = ConcurrencyControl(timeout_seconds=0.05)

        async def slow_task():
            await asyncio.sleep(0.1)
            return "result"

        with pytest.raises(asyncio.TimeoutError):
            await cc.process("timeout_task", slow_task)

        stats = cc.get_stats()
        assert stats["total_timeouts"] == 1

    @pytest.mark.asyncio
    async def test_process_exception(self):
        """Test task exception handling"""
        cc = ConcurrencyControl()

        async def failing_task():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await cc.process("failing_task", failing_task)

        stats = cc.get_stats()
        assert stats["total_errors"] == 1


class TestConcurrencyControlQueue:
    """Test ConcurrencyControl queue functionality"""

    @pytest.mark.asyncio
    async def test_enqueue_and_process(self):
        """Test enqueue and process queue"""
        cc = ConcurrencyControl()
        results = []

        async def task(n):
            results.append(n)
            return n

        await cc.enqueue("task_1", lambda: task(1))
        await cc.enqueue("task_2", lambda: task(2))

        assert cc.queue_size == 2

        await cc.process_queue()

        assert cc.queue_size == 0
        assert sorted(results) == [1, 2]

    @pytest.mark.asyncio
    async def test_enqueue_deduplication(self):
        """Test enqueue deduplication"""
        cc = ConcurrencyControl()

        async def task():
            return "result"

        await cc.enqueue("same_id", task)
        await cc.enqueue("same_id", task)  # Should be deduplicated

        assert cc.queue_size == 1

    @pytest.mark.asyncio
    async def test_clear_queue(self):
        """Test clearing queue"""
        cc = ConcurrencyControl()

        async def task():
            return "result"

        await cc.enqueue("task_1", task)
        await cc.enqueue("task_2", task)

        assert cc.queue_size == 2

        await cc.clear_queue()

        assert cc.queue_size == 0


class TestConcurrencyControlStats:
    """Test ConcurrencyControl statistics"""

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting statistics"""
        cc = ConcurrencyControl(max_concurrent=3)

        async def task():
            await asyncio.sleep(0.01)
            return "done"

        await cc.process("task_1", task)
        await cc.process("task_2", task)

        stats = cc.get_stats()

        assert stats["max_concurrent"] == 3
        assert stats["total_processed"] == 2
        assert stats["current_processing"] == 0

    @pytest.mark.asyncio
    async def test_clear_stats(self):
        """Test clearing statistics"""
        cc = ConcurrencyControl()

        async def task():
            await asyncio.sleep(0.01)
            return "done"

        await cc.process("task_1", task)
        assert cc.get_stats()["total_processed"] == 1

        cc.clear_stats()
        assert cc.get_stats()["total_processed"] == 0

    @pytest.mark.asyncio
    async def test_max_concurrent_reached_tracking(self):
        """Test tracking of max concurrent reached"""
        cc = ConcurrencyControl(max_concurrent=3)

        async def task():
            await asyncio.sleep(0.05)
            return "done"

        tasks = [cc.process(f"task_{i}", task) for i in range(5)]
        await asyncio.gather(*tasks)

        assert cc.max_concurrent_reached > 0
        assert cc.max_concurrent_reached <= 3


class TestConcurrencyControlIsProcessing:
    """Test is_processing method"""

    @pytest.mark.asyncio
    async def test_is_processing_during_execution(self):
        """Test is_processing returns True during execution"""
        cc = ConcurrencyControl()
        processing_states = []

        async def task():
            processing_states.append(cc.is_processing("task_1"))
            await asyncio.sleep(0.05)
            processing_states.append(cc.is_processing("task_1"))
            return "done"

        await cc.process("task_1", task)

        assert processing_states[0] is True
        assert processing_states[1] is True  # Still True during execution
        assert cc.is_processing("task_1") is False  # False after completion


class TestConcurrencyControlEdgeCases:
    """Test edge cases"""

    @pytest.mark.asyncio
    async def test_zero_max_concurrent(self):
        """Test with max_concurrent=1 (minimum)"""
        cc = ConcurrencyControl(max_concurrent=1)

        async def task():
            await asyncio.sleep(0.01)
            return "done"

        results = await asyncio.gather(
            cc.process("task_1", task),
            cc.process("task_2", task),
        )

        assert results == ["done", "done"]

    @pytest.mark.asyncio
    async def test_rapid_task_submission(self):
        """Test rapid task submission"""
        cc = ConcurrencyControl(max_concurrent=5)

        async def task():
            await asyncio.sleep(0.01)
            return "done"

        tasks = [cc.process(f"task_{i}", task) for i in range(20)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 20
        assert all(r == "done" for r in results)

    @pytest.mark.asyncio
    async def test_task_after_deduplication_removed(self):
        """Test can process task again after previous one completed"""
        cc = ConcurrencyControl()

        async def task():
            return "result"

        result1 = await cc.process("same_id", task)
        result2 = await cc.process("same_id", task)  # Should work now

        assert result1 == "result"
        assert result2 == "result"
