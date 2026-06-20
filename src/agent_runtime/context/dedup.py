# src/agent_runtime/context/dedup.py
from typing import List

from agent_runtime.context.models import ContextItem


class ContextDeduplicator:
    @staticmethod
    def deduplicate(items: List[ContextItem]) -> List[ContextItem]:
        seen_content = set()
        unique_items = []
        for item in items:
            normalized_content = item.content.strip()
            if normalized_content not in seen_content:
                seen_content.add(normalized_content)
                unique_items.append(item)
        return unique_items
