# src/agent_runtime/context/assembler.py
"""V2 Context Assembler with pruning, budget reporting, and lost-in-the-middle mitigation.

Pipeline order:
  1. Deduplicate
  2. Prune (stale items, low-priority overflow)
  3. Sort by priority (descending)
  4. Truncate by token budget
  5. Apply lost-in-the-middle reordering
  6. Produce ContextBudgetReport
"""
from typing import List, Tuple

from agent_runtime.context.budget import TokenBudgetTracker
from agent_runtime.context.dedup import ContextDeduplicator
from agent_runtime.context.models import ContextBundle, ContextBudgetReport, ContextItem
from agent_runtime.context.pruner import ContextPruner


class ContextAssembler:
    def __init__(self, max_tokens: int = 1000):
        self.max_tokens = max_tokens

    def assemble(self, raw_items: List[ContextItem]) -> Tuple[ContextBundle, ContextBudgetReport]:
        """Assemble context and produce a structured budget report.

        Returns (bundle, report) so callers can log or record the report.
        """
        total_submitted = len(raw_items)

        # Step 1: deduplicate
        deduped = ContextDeduplicator.deduplicate(raw_items)
        deduped_count = len(deduped)

        # Step 2: prune stale / low-value items
        pruned, pruned_count = ContextPruner.prune(deduped)

        # Step 3: sort by priority
        sorted_items = sorted(pruned, key=lambda x: x.priority, reverse=True)

        # Step 4: token budget truncation
        retained_items: List[ContextItem] = []
        dropped_items: List[ContextItem] = []
        current_tokens = 0

        for item in sorted_items:
            item_tokens = TokenBudgetTracker.estimate_tokens(item.content)

            if current_tokens + item_tokens > self.max_tokens:
                if item.priority >= 100:
                    # System instructions are retained even if they exceed the nominal budget.
                    retained_items.append(item)
                    current_tokens += item_tokens
                else:
                    dropped_items.append(item)
            else:
                retained_items.append(item)
                current_tokens += item_tokens

        dropped_count = len(dropped_items)

        # Step 5: lost-in-the-middle mitigation
        # Reorder so that important items appear at BOTH the beginning and end.
        # Model attention is strongest at positions 0 and N-1.
        lost_in_middle = len(retained_items) > 8
        if lost_in_middle:
            retained_items = self._reorder_for_positional_bias(retained_items)

        # Step 6: compute distributions for the report
        trust_dist: dict[str, int] = {}
        kind_dist: dict[str, int] = {}
        for item in retained_items:
            trust_dist[item.trust_level] = trust_dist.get(item.trust_level, 0) + 1
            kind_dist[item.kind] = kind_dist.get(item.kind, 0) + 1

        utilization = current_tokens / max(self.max_tokens, 1)
        # Warn if budget is heavily taxed (>90%) with many items still
        if utilization > 0.9 and lost_in_middle:
            lost_in_middle_warning = True
        else:
            lost_in_middle_warning = False

        report = ContextBudgetReport(
            total_items_submitted=total_submitted,
            items_after_dedup=deduped_count,
            items_pruned=pruned_count,
            items_dropped_by_budget=dropped_count,
            items_retained=len(retained_items),
            estimated_tokens_used=current_tokens,
            max_tokens=self.max_tokens,
            utilization_ratio=round(utilization, 3),
            lost_in_middle_warning=lost_in_middle_warning,
            trust_distribution=trust_dist,
            kind_distribution=kind_dist,
        )

        bundle = ContextBundle(
            items=retained_items,
            total_estimated_tokens=current_tokens,
            max_tokens=self.max_tokens,
            dropped_items=dropped_items,
        )

        return bundle, report

    @staticmethod
    def _reorder_for_positional_bias(items: List[ContextItem]) -> List[ContextItem]:
        """Reorder so high-priority items appear at both front and back.

        Strategy: interleave high-priority items at the tail of the list,
        keeping system instructions at position 0, user messages at position 1,
        then distributing remaining items with priorities alternating front/back.
        """
        if len(items) <= 2:
            return items

        system = [i for i in items if i.priority >= 100]
        user = [i for i in items if i.kind == "user_message"]
        rest = [i for i in items if i not in system and i not in user]

        result: List[ContextItem] = system[:] + user[:]

        for idx, item in enumerate(rest):
            # Alternate: front-half goes next, back-half appended
            if idx % 2 == 0:
                result.insert(len(system) + len(user) + idx // 2, item)
            else:
                result.append(item)

        return result