# src/agent_runtime/context/models.py
from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class ContextItem(BaseModel):
    item_id: str = Field(..., description="Unique ID for this context slice")
    kind: str = Field(..., description="system_instruction, user_message, tool_result, retrieved_evidence, etc.")
    content: str = Field(..., description="The literal text payload.")
    priority: int = Field(..., description="Higher numbers mean higher retention priority (0-100).")
    provenance: Dict[str, str] = Field(default_factory=dict, description="Source trace, e.g., {'file': 'backtest.md', 'line': '12'}")
    trust_level: str = Field(..., description="system_trusted, user_supplied, retrieved_untrusted")
    # V2: staleness tracking for pruning
    age_in_steps: int = 0


class ContextBundle(BaseModel):
    items: List[ContextItem]
    total_estimated_tokens: int
    max_tokens: int
    dropped_items: List[ContextItem] = Field(default_factory=list)


class ContextBudgetReport(BaseModel):
    """V2: structured report of what the context assembler did.

    Produced alongside ContextBundle.Answers:
      - How much budget was available?
      - What was retained, dropped, or pruned?
      - Is the context at risk of lost-in-the-middle?
      - How many items by trust_level and kind?
    """
    total_items_submitted: int
    items_after_dedup: int
    items_pruned: int
    items_dropped_by_budget: int
    items_retained: int
    estimated_tokens_used: int
    max_tokens: int
    utilization_ratio: float
    lost_in_middle_warning: bool = False
    trust_distribution: Dict[str, int] = Field(default_factory=dict)
    kind_distribution: Dict[str, int] = Field(default_factory=dict)
