"""Interactive stdio client for the installed official Colab MCP server.

The subprocess and its local connection are owned by this client. Only tool
names and results are printed, never the server's pairing token or browser URL.
Close with {"op":"quit"}. This is a fallback when the desktop's live tool
catalog has not reloaded the configured server yet.
"""
from __future__ import annotations

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = StdioServerParameters(
    command=r"C:\Users\Samsung\.local\bin\uvx.exe",
    args=["--offline", "--python",
          r"C:\Users\Samsung\AppData\Roaming\uv\python\cpython-3.14.5-windows-x86_64-none\python.exe",
          "--from", "git+https://github.com/googlecolab/colab-mcp@b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1",
          "colab-mcp"],
)


def send(value):
    print(json.dumps(value, ensure_ascii=False, default=str), flush=True)


async def run():
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            send({"ready": True, "server": info.serverInfo.name})
            while True:
                line = await asyncio.to_thread(sys.stdin.readline)
                if not line:
                    break
                try:
                    request = json.loads(line)
                    operation = request.get("op")
                    if operation == "quit":
                        send({"closed": True})
                        break
                    if operation == "tools":
                        catalog = await session.list_tools()
                        send({"tools": [tool.model_dump(mode="json", exclude_none=True)
                                         for tool in catalog.tools]})
                    elif operation == "call":
                        result = await session.call_tool(request["name"], request.get("arguments", {}),
                                                         read_timeout_seconds=__import__("datetime").timedelta(seconds=180))
                        send({"result": result.model_dump(mode="json", exclude_none=True)})
                    else:
                        raise ValueError("Unsupported operation")
                except Exception as exc:
                    send({"error": type(exc).__name__, "message": str(exc)})


if __name__ == "__main__":
    asyncio.run(run())
