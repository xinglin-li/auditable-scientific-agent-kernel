# tests/test_mcp_approval_policy.py
import pytest
from tests.test_mcp_capability_registry import populated_registry
from agent_runtime.mcp_clients.policy import McpApprovalPolicyGate

def test_policy_gate_high_risk_routing_to_hitl(populated_registry):
    """Verify direct access for low-risk tools and approval routing for high-risk tools."""
    gate = McpApprovalPolicyGate(populated_registry)
    
    # Scenario one: allow a low-risk read tool with an authorized role and valid arguments.
    decision_low, msg_low = gate.evaluate_tool_call("fetch_series", {}, "researcher")
    assert decision_low == "allow"
    
    # Scenario two: require approval for a high-risk privileged tool, even for an admin.
    decision_high, msg_high = gate.evaluate_tool_call("delete_artifact", {}, "admin")
    assert decision_high == "route_to_hitl"
    assert "HITL Guard Engaged" in msg_high

def test_policy_gate_argument_schema_injection_blocking(populated_registry):
    """Verify that the policy gateway rejects schema-injection arguments from the model."""
    # Configure strict argument constraints for fetch_series.
    cap = populated_registry.get_capability("fetch_series")
    cap.input_schema = {"type": "object", "properties": {"series_id": {"type": "string"}}}
    
    gate = McpApprovalPolicyGate(populated_registry)
    
    # Have the model supply an extra noncompliant argument resembling SQL or path injection.
    malicious_args = {"series_id": "CPI", "malicious_injected_sql_field": "DROP TABLE users;"}
    
    decision, msg = gate.evaluate_tool_call("fetch_series", malicious_args, "researcher")
    assert decision == "deny"
    assert "Schema Error" in msg
