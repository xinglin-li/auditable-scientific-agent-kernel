# tests/test_mcp_capability_registry.py
import pytest
from agent_runtime.mcp_clients.models import RegisteredCapability
from agent_runtime.mcp_clients.registry import McpCapabilityRegistry
from agent_runtime.mcp_clients.policy import McpApprovalPolicyGate

@pytest.fixture
def populated_registry():
    """Initialize a governed capability registry."""
    reg = McpCapabilityRegistry()
    # Register a low-risk tool that requires no approval.
    reg.register_capability(RegisteredCapability(
        server_id="macro_srv_01", name="fetch_series", kind="tool",
        description="Fetch reader series", risk_level="low",
        requires_approval=False, allowed_roles=["researcher", "admin"], transport="stdio"
    ))
    # Register a high-risk tool requiring human approval, such as deleting a data asset.
    reg.register_capability(RegisteredCapability(
        server_id="macro_srv_01", name="delete_artifact", kind="tool",
        description="Delete file storage", risk_level="high",
        requires_approval=True, allowed_roles=["admin"], transport="stdio"
    ))
    return reg

def test_http_transport_origin_and_token_boundary(populated_registry):
    """Verify that remote transport rejects invalid tokens and forged origins."""
    allowed = ["https://xinglin-systematic.sas.com"]
    
    # Scenario A: supply an invalid token.
    assert populated_registry.validate_incoming_transport_context(
        token="BAD_HACKER_TOKEN", origin=allowed[0], allowed_origins=allowed
    ) is False
    
    # Scenario B: supply a valid token from a malicious cross-site origin.
    assert populated_registry.validate_incoming_transport_context(
        token="secure_macro_token_2026", origin="https://evil-malicious-site.com", allowed_origins=allowed
    ) is False

def test_policy_gate_rbac_and_unregistered_tool_blocking(populated_registry):
    """Verify that policy blocks unregistered tools and calls from insufficiently privileged roles."""
    gate = McpApprovalPolicyGate(populated_registry)
    
    # 1. Block a black-box injected tool that was never registered.
    decision, msg = gate.evaluate_tool_call("drop_all_tables", {}, "admin")
    assert decision == "deny"
    assert "not registered" in msg
    
    # 2. Block a low-privilege researcher from invoking the high-risk delete_artifact tool.
    decision_rbac, msg_rbac = gate.evaluate_tool_call("delete_artifact", {}, "researcher")
    assert decision_rbac == "deny"
    assert "Access Denied" in msg_rbac
