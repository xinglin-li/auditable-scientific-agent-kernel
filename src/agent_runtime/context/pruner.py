# src/agent_runtime/context/pruner.py
"""V2 Context pruning: remove stale or low-value items before budget assembly.

ContextPruner operates on the sorted item list *before* token budget truncation.
It removes items that are:
  - Stale (age_in_steps > max_age, and priority < stale_threshold)
  - Very low priority items when the item count is high
  - Retrieval evidence with insufficient citation confidence

This is distinct from budget truncation: pruning is a semantic decision,
not a token-count decision.
"""
from typing import List

from agent_runtime.context.models import ContextItem


class ContextPruner:
    """Remove stale or low-value context items before assembly."""

    @staticmethod
    def prune(
        items: List[ContextItem],
        *,
        max_age: int = 10,
        stale_priority_threshold: int = 60,
        max_items_before_low_priority_cut: int = 20,
        low_priority_cutoff: int = 20,
    ) -> tuple[List[ContextItem], int]:
        """Return (retained_items, pruned_count)."""
        pruned_count = 0
        retained: List[ContextItem] = []

        for item in items:
            # Rule 1: drop stale low-priority items
            if item.age_in_steps > max_age and item.priority < stale_priority_threshold:
                pruned_count += 1
                continue

            retained.append(item)

        # Rule 2: when retained list is still large, drop lowest-priority items
        if len(retained) > max_items_before_low_priority_cut:
            retained.sort(key=lambda x: x.priority, reverse=True)
            cutoff = max_items_before_low_priority_cut
            # Keep system instructions (priority >= 100) even beyond cutoff
            must_keep = [i for i in retained if i.priority >= 100]
            rest = [i for i in retained if i.priority < 100]
            extra_cut = cutoff - len(must_keep)
            if extra_cut > 0:
                rest = rest[:extra_cut]
            pruned_count += len(retained) - len(must_keep) - len(rest)
            retained = must_keep + rest
            retained.sort(key=lambda x: x.priority, reverse=True)

        return retained, pruned_count