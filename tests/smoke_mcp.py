"""End-to-end smoke test of the MCP server over real streamable HTTP (as TrueForge will call it).

Works against the fake local stack or the real AWS stack:
  MCP_URL=http://127.0.0.1:8765/mcp MCP_AUTH_TOKEN=... python -m tests.smoke_mcp [--quick]
It deploys to SHADOW only; it never calls a gated production tool except to prove refusal.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import httpx2
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = os.environ.get("MCP_URL", "http://127.0.0.1:8765/mcp")
TOKEN = os.environ.get("MCP_AUTH_TOKEN", "")
QUICK = "--quick" in sys.argv


def show(title: str, obj: object, limit: int = 1500) -> None:
    text = json.dumps(obj, indent=1, default=str)
    print(f"\n=== {title}\n{text[:limit]}{' ...' if len(text) > limit else ''}")


async def call(s: ClientSession, name: str, **args: object) -> dict:
    res = await s.call_tool(name, args)
    if res.is_error:
        return {"TOOL_ERROR": " ".join(getattr(c, "text", "") for c in res.content)}
    return res.structured_content or json.loads(res.content[0].text)


async def main() -> None:
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    async with httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(900)) as http:
        async with streamable_http_client(URL, http_client=http) as (read, write, *_):
            async with ClientSession(read, write) as s:
                await s.initialize()
                tools = (await s.list_tools()).tools
                show("tools", [{"name": t.name, "readOnly": t.annotations and t.annotations.read_only_hint,
                                "destructive": t.annotations and t.annotations.destructive_hint} for t in tools], 4000)
                show("get_slo_status", await call(s, "get_slo_status", window_minutes=1))
                show("compare_windows", await call(s, "compare_windows", baseline_start="-100s",
                                                   baseline_end="-55s", incident_start="-40s"), 2500)
                show("get_change_history", await call(s, "get_change_history"))
                wl = await call(s, "capture_workload", start="-40s")
                show("capture_workload", wl)
                dur = 20 if QUICK else 45
                d0 = await call(s, "deploy_shadow", changes={}, reason="reproduce production config")
                show("deploy_shadow (repro)", d0)
                r0 = await call(s, "run_load_test", workload_id=wl["id"], duration_s=dur,
                                hypothesis="Current prod config reproduces the incident on shadow")
                show("run_load_test (repro)", r0, 2500)
                d1 = await call(s, "deploy_shadow", changes={"enable_prefix_caching": True, "max_num_seqs": 16},
                                reason="bound admission to KV capacity + reuse shared system prompt")
                show("deploy_shadow (candidate)", d1)
                r1 = await call(s, "run_load_test", workload_id=wl["id"], duration_s=dur,
                                hypothesis="Fewer concurrent seqs stops preemption; prefix cache cuts prefill")
                show("run_load_test (candidate)", r1, 2500)
                show("list_experiments", await call(s, "list_experiments"), 2500)
                changes = {"enable_prefix_caching": True, "max_num_seqs": 16}
                show("plan_production_change", await call(s, "plan_production_change", changes=changes,
                                                          evidence_run_ids=[r1.get("id", "?")]), 2500)
                show("apply_production_config WITHOUT evidence (must refuse)",
                     await call(s, "apply_production_config", changes=changes, evidence_run_ids=[],
                                justification="smoke", blast_radius="smoke", expected_result="smoke"))
                show("deploy_shadow out-of-range (must refuse)",
                     await call(s, "deploy_shadow", changes={"gpu_memory_utilization": 0.99}, reason="smoke"))
                show("deploy_shadow change model (must refuse)",
                     await call(s, "deploy_shadow", changes={"model": "gpt2"}, reason="smoke"))


if __name__ == "__main__":
    asyncio.run(main())
