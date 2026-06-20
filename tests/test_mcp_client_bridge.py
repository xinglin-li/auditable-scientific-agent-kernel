# tests/test_mcp_client_bridge.py
import sys
import os
import pytest
import asyncio
import json
from agent_runtime.mcp.client import AsyncMcpStdioClient
from agent_runtime.mcp.bridge import McpToolBridge

# 1. Build an independent external Python script that simulates a standard MCP server.
MOCK_SERVER_CODE = """
import sys
import json

def listen_loop():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")
            
            if method == "tools/list":
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [
                            {
                                "name": "mcp_arima_forecast",
                                "description": "Execute cross-process ARIMA calculation",
                                "inputSchema": {"type": "object", "properties": {"series_id": {"type": "string"}}}
                            }
                        ]
                    }
                }
            elif method == "tools/call":
                tool_name = req.get("params", {}).get("name")
                args = req.get("params", {}).get("arguments", {})
                sid = args.get("series_id", "UNKNOWN")
                
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Mcp server output successfully for {sid}"}]
                    }
                }
            else:
                res = {"jsonrpc": "2.0", "id": req_id, "error": {"message": "Method not found"}}
                
            sys.stdout.write(json.dumps(res) + "\\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(str(e) + "\\n")
            sys.stderr.flush()

if __name__ == "__main__":
    listen_loop()
"""

@pytest.fixture
def mock_server_file(tmp_path):
    """Create the external server script in an isolated temporary directory."""
    server_path = tmp_path / "mock_mcp_server.py"
    server_path.write_text(MOCK_SERVER_CODE, encoding="utf-8")
    return str(server_path)

@pytest.mark.asyncio
async def test_mcp_stdio_client_and_bridge_full_lifecycle(mock_server_file):
    """Integration-test the framework-free cross-process communication channel."""
    # 1. Start the external script with the current Python interpreter.
    client = AsyncMcpStdioClient(command=sys.executable, args=[mock_server_file])
    
    await client.start()
    
    try:
        # 2. Verify tools/list communication across the process pipes.
        tools_list = await client.list_tools()
        assert len(tools_list) == 1
        assert tools_list[0]["name"] == "mcp_arima_forecast"
        assert "ARIMA calculation" in tools_list[0]["description"]
        
        # 3. Verify that the bridge dynamically wraps a closure.
        bridge = McpToolBridge(mcp_client=client)
        native_tool = bridge.convert_to_native_tool(tools_list[0])
        
        # Execute the asynchronous wrapper exposed as a local callable.
        execution_result = await native_tool({"series_id": "USA_CPI_2026"})
        
        # 4. Verify the complete JSON-RPC round trip and response payload.
        assert execution_result["tool_name"] == "mcp_arima_forecast"
        assert execution_result["is_success"] is True
        assert "Mcp server output successfully for USA_CPI_2026" in execution_result["raw_output"]
        
    finally:
        # 5. Close the client and ensure no zombie subprocess remains.
        await client.stop()
