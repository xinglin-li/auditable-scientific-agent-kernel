# src/agent_runtime/graph/reducers.py
from typing import Any, List, Optional

def reduce_append(left: Optional[List[Any]], right: Optional[List[Any]]) -> List[Any]:
    """Standard append-only merge rule for trace_events, errors, and messages."""
    if left is None:
        left = []
    if right is None:
        right = []
    return left + list(right)

def reduce_unique_str_list(left: Optional[List[str]], right: Optional[List[str]]) -> List[str]:
    """Append unique values to string lists such as active_job_ids."""
    if left is None:
        left = []
    if right is None:
        right = []
    # Preserve insertion order while removing duplicates.
    seen = set(left)
    result = list(left)
    for item in right:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def reduce_artifact_refs(left: Optional[List[dict]], right: Optional[List[dict]]) -> List[dict]:
    """Append or update structured mappings such as artifact_refs and completed_job_results.
    Resolve conflicts using a unique identifier such as ``artifact_id`` or ``job_id``.
    """
    if left is None:
        left = []
    if right is None:
        right = []
    
    result = {m.get("artifact_id", m.get("job_id", idx)): m for idx, m in enumerate(left)}
    for item in right:
        # Upsert entries with a unique business key so the latest metadata wins.
        key = item.get("artifact_id", item.get("job_id", f"gen_{len(result)}"))
        result[key] = item
        
    return list(result.values())
