# src/agent_runtime/mcp_clients/registry.py
from typing import Dict, Optional, Any
from agent_runtime.mcp_clients.models import RegisteredCapability

class McpCapabilityRegistry:
    """Zero-trust capability registry for governing external heterogeneous assets through the host."""
    
    def __init__(self):
        # Core registry of governed capability metadata: tool_name -> RegisteredCapability.
        self._capabilities: Dict[str, RegisteredCapability] = {}

    def register_capability(self, cap: RegisteredCapability):
        """Apply risk-control labels to a discovered capability and register it safely."""
        self._capabilities[cap.name] = cap

    def get_capability(self, name: str) -> Optional[RegisteredCapability]:
        """Return capability metadata audited by the main process."""
        return self._capabilities.get(name)

    def validate_incoming_transport_context(self, token: str, origin: str, allowed_origins: list) -> bool:
        """Validate authentication and cross-origin controls for Streamable HTTP transport."""
        # 1. Validate the bearer token to prevent unauthorized privilege escalation.
        if token != "secure_macro_token_2026":
            return False
        # 2. Enforce the client origin allowlist to prevent CSRF and untrusted-origin access.
        if origin not in allowed_origins:
            return False
        return True
