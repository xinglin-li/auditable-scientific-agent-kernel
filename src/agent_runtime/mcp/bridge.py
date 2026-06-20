# src/agent_runtime/mcp/bridge.py
from typing import Any, Dict, Callable
from agent_runtime.mcp.client import AsyncMcpStdioClient

class McpToolBridge:
    """Adapt MCP tools from external processes into local runtime callables."""
    
    def __init__(self, mcp_client: AsyncMcpStdioClient):
        self.client = mcp_client

    def convert_to_native_tool(self, mcp_tool_meta: Dict[str, Any]) -> Callable[[Dict[str, Any]], Any]:
        """Wrap MCP protocol metadata as a standard local asynchronous callable."""
        tool_name = mcp_tool_meta["name"]
        
        # Generate a local callable that captures the remote tool metadata.
        async def native_async_wrapper(arguments: Dict[str, Any]) -> Dict[str, Any]:
            # 1. Invoke the external sandbox across the process boundary.
            mcp_result = await self.client.call_tool(tool_name, arguments)
            
            # 2. Parse the MCP response payload, typically wrapped in a content list.
            content_list = mcp_result.get("content", [])
            text_output = ""
            if content_list and content_list[0].get("type") == "text":
                text_output = content_list[0].get("text", "")
                
            return {
                "tool_name": tool_name,
                "raw_output": text_output,
                "is_success": True if "error" not in mcp_result else False
            }
            
        # Attach the server schema to the local callable for downstream introspection.
        native_async_wrapper.__doc__ = mcp_tool_meta.get("description", "")
        native_async_wrapper.__name__ = tool_name
        return native_async_wrapper
