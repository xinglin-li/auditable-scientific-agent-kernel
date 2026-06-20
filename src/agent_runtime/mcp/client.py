# src/agent_runtime/mcp/client.py
import asyncio
import json
from typing import Any, Dict, List, Optional

class AsyncMcpStdioClient:
    """Framework-free MCP stdio client for cross-process communication over standard pipes."""
    
    def __init__(self, command: str, args: Optional[List[str]] = None):
        self.command = command
        self.args = args or []
        self.process: Optional[asyncio.subprocess.Process] = None
        self.read_task: Optional[asyncio.Task] = None
        self.request_id = 0
        # Map pending request IDs to their response futures: id -> asyncio.Future.
        self._pending_requests: Dict[int, asyncio.Future] = {}

    async def start(self):
        """Start the external subprocess and connect to its stdio pipes."""
        self.process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        # Start a non-blocking background task that continuously parses server JSON frames.
        self.read_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self):
        """Read and parse JSON-RPC 2.0 response frames."""
        try:
            while self.process and self.process.stdout:
                line = await self.process.stdout.readline()
                if not line:
                    break
                
                # Each line must contain one valid JSON document.
                data = json.loads(line.decode("utf-8").strip())
                resp_id = data.get("id")
                
                # Resolve the matching pending future when a response arrives.
                if resp_id is not None and resp_id in self._pending_requests:
                    future = self._pending_requests.pop(resp_id)
                    if not future.done():
                        future.set_result(data)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            # Fail every pending future so a reader error cannot deadlock callers.
            for future in self._pending_requests.values():
                if not future.done():
                    future.set_exception(e)

    async def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Write a JSON-RPC 2.0 request to the subprocess."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP Server 进程未启动或管道已破裂。")
            
        self.request_id += 1
        current_id = self.request_id
        
        payload = {
            "jsonrpc": "2.0",
            "id": current_id,
            "method": method,
            "params": params or {}
        }
        
        # Create a future for the pending response.
        future = asyncio.get_running_loop().create_future()
        self._pending_requests[current_id] = future
        
        # Serialize to stdin with a trailing newline as the frame delimiter.
        json_bytes = (json.dumps(payload) + "\n").encode("utf-8")
        self.process.stdin.write(json_bytes)
        await self.process.stdin.drain()
        
        # Wait asynchronously for _read_loop to resolve the future.
        return await future

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Call the MCP list-tools primitive to discover server capabilities."""
        response = await self._send_request("tools/list")
        if "error" in response:
            raise RuntimeError(f"MCP list_tools 失败: {response['error']}")
        return response.get("result", {}).get("tools", [])

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call the MCP tool-execution primitive across the process boundary."""
        params = {"name": tool_name, "arguments": arguments}
        response = await self._send_request("tools/call", params=params)
        if "error" in response:
            raise RuntimeError(f"MCP call_tool 执行致命失败: {response['error']}")
        return response.get("result", {})

    async def stop(self):
        """Shut down the process tree cleanly without leaving zombie processes."""
        if self.read_task:
            self.read_task.cancel()
            try:
                await self.read_task
            except asyncio.CancelledError:
                pass
        if self.process:
            try:
                self.process.terminate()
                await self.process.wait()
            except Exception:
                pass
