# src/agent_runtime/mcp_clients/policy.py
from typing import Any, Dict, Literal, Tuple
from agent_runtime.mcp_clients.registry import McpCapabilityRegistry

class McpApprovalPolicyGate:
    """Policy gateway that converts model requests into safe execution decisions."""
    
    def __init__(self, registry: McpCapabilityRegistry):
        self.registry = registry
    
    def evaluate_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        current_user_role: str
    ) -> Tuple[Literal["allow", "deny", "route_to_hitl"], str]:
        """Evaluate a request according to least-privilege rules."""
        
        # 1. Block unknown capabilities proposed by the model or injected as unregistered tools.
        cap = self.registry.get_capability(tool_name)
        if not cap:
            return "deny", f"Security Violation: Capability '{tool_name}' is not registered in the host."
        
        # 2. Enforce RBAC for the user role associated with the active graph thread.
        if current_user_role not in cap.allowed_roles:
            return "deny", f"Access Denied: Role '{current_user_role}' has insufficient privileges for '{tool_name}'."
        
        # 3. Strictly validate model-supplied arguments against the input schema.
        if cap.input_schema:
            properties = cap.input_schema.get("properties", {})
            for key in arguments.keys():
                if key not in properties:
                    return "deny", f"Schema Error: Malicious or invalid argument key '{key}' detected."

        # 4. Route risky operations through the human-in-the-loop approval branch.
        if cap.risk_level == "high" or cap.requires_approval:
            # Block high-risk or explicitly gated operations and route them to approval.
            return "route_to_hitl", f"HITL Guard Engaged: '{tool_name}' exhibits high-risk status. Control routed to human approval lock."

        # The request passed all checks and may proceed through the MCP client.
        return "allow", "Verification passed."
