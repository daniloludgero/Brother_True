"""Verify the official MCP stdio handshake without opening a browser or running code.

Run with the installed colab-mcp environment's Python. The client SDK owns and
closes its subprocess. No token or user configuration is printed.
"""
from __future__ import annotations

import argparse
import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def probe(args):
    params = StdioServerParameters(command=args.command, args=args.server_args)
    async with asyncio.timeout(45):
        async with stdio_client(params) as (receive, send):
            async with ClientSession(receive, send) as client:
                result = await client.initialize()
                catalog = await client.list_tools()
                print(json.dumps({
                    "server": result.serverInfo.model_dump(),
                    "protocol": result.protocolVersion,
                    "tool_list_changed": result.capabilities.tools.listChanged
                        if result.capabilities.tools else None,
                    "tools": [tool.name for tool in catalog.tools],
                    "colab_browser_paired": False,
                    "notebook_code_executed": False,
                }, ensure_ascii=False), flush=True)
                if "open_colab_browser_connection" not in {tool.name for tool in catalog.tools}:
                    raise RuntimeError("Official Colab connection bootstrap tool was not advertised")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True)
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.server_args[:1] == ["--"]:
        args.server_args = args.server_args[1:]
    asyncio.run(probe(args))
