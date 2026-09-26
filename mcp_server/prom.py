"""Prometheus as the engine-metrics source (used when PROMETHEUS_URL is set).

Env:
  PROMETHEUS_URL           e.g. http://10.0.1.20:9090
  PROMETHEUS_JOB_PROD      scrape job for the prod vLLM (default vllm-prod)
  PROMETHEUS_BEARER_TOKEN  or PROMETHEUS_USER + PROMETHEUS_PASSWORD, if your Prometheus needs auth
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from common.promtext import ALIASES


class PromError(RuntimeError):
    pass


def enabled() -> bool:
    return bool(os.environ.get("PROMETHEUS_URL"))


def _client() -> httpx.AsyncClient:
    headers, auth = {}, None
    if os.environ.get("PROMETHEUS_BEARER_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['PROMETHEUS_BEARER_TOKEN']}"
    elif os.environ.get("PROMETHEUS_USER"):
        auth = (os.environ["PROMETHEUS_USER"], os.environ.get("PROMETHEUS_PASSWORD", ""))
    return httpx.AsyncClient(base_url=os.environ["PROMETHEUS_URL"].rstrip("/"), headers=headers, auth=auth,
                             timeout=30)


async def query_range(promql: str, start: float, end: float, step: float | None = None) -> list[dict[str, Any]]:
    step = step or max(5.0, (end - start) / 1000)
    async with _client() as c:
        try:
            r = await c.get("/api/v1/query_range", params={"query": promql, "start": start, "end": end, "step": step})
        except httpx.HTTPError as e:
            raise PromError(f"Prometheus unreachable: {type(e).__name__}: {e}") from None
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if r.status_code != 200 or body.get("status") != "success":
        raise PromError(f"Prometheus query failed (HTTP {r.status_code}): {body.get('error') or r.text[:300]}")
    return body["data"]["result"]


async def engine_samples(start: float, end: float) -> list[dict[str, Any]]:
    """Same shape as the controller's sampler: [{"t", "kv_cache_usage", "running", ...}], so the
    rest of the server doesn't care which source is used."""
    job = os.environ.get("PROMETHEUS_JOB_PROD", "vllm-prod")
    by_t: dict[float, dict[str, Any]] = {}
    for key, names in ALIASES.items():
        promql = " or ".join(f'sum({n}{{job="{job}"}})' for n in names)
        for series in await query_range(promql, start, end):
            for t, v in series["values"]:
                by_t.setdefault(float(t), {"t": float(t)})[key] = float(v)
    capacity = f'sum(vllm:num_requests_waiting_by_reason{{job="{job}",reason="capacity"}})'
    for series in await query_range(capacity, start, end):
        for t, v in series["values"]:
            by_t.setdefault(float(t), {"t": float(t)})["waiting_for_kv_capacity"] = float(v)
    return [by_t[t] for t in sorted(by_t)]
