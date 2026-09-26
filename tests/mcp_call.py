"""Call one inference-ops MCP tool from a terminal (operator calibration / debugging).
  MCP_URL=http://<host>:8765/mcp MCP_AUTH_TOKEN=... python -m tests.mcp_call <tool> '<json args>'
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import httpx2
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def main() -> None:
    tool, args = sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    headers = {"Authorization": f"Bearer {os.environ['MCP_AUTH_TOKEN']}"}
    async with httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(1800)) as http:
        async with streamable_http_client(os.environ.get("MCP_URL", "http://127.0.0.1:8765/mcp"), http_client=http) as (r, w, *_):
            async with ClientSession(r, w) as s:
                await s.initialize()
                res = await s.call_tool(tool, args)
                if res.is_error:
                    sys.exit("TOOL_ERROR: " + " ".join(getattr(c, "text", "") for c in res.content))
                print(json.dumps(res.structured_content or json.loads(res.content[0].text), indent=1, default=str))


if __name__ == "__main__":
    asyncio.run(main())
