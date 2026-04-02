"""去重决策逻辑"""

from typing import Any


class DedupMixin:
    """去重决策方法"""

    def _decide_duplicate_action(
        self,
        new_memory: dict[str, Any],
        old_record: dict[str, Any],
        similarity: float,
        mem_type: str,
    ) -> str:
        if similarity >= 0.95 and "source_id" not in new_memory:
            return "UPDATE"
        return "KEEP_BOTH"
