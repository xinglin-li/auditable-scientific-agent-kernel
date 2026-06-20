# src/agent_runtime/mcp_clients/models.py
from typing import Any, Literal, Optional, List
from pydantic import BaseModel

class RegisteredCapability(BaseModel):
    """Deterministic capability model governed by the main process."""
    server_id: str
    name: str
    kind: Literal["tool", "resource", "prompt"]
    description: str
    input_schema: Optional[dict] = None
    risk_level: Literal["low", "medium", "high"]
    requires_approval: bool
    allowed_roles: List[str]
    transport: Literal["stdio", "http"]
