"""Open-loop replay harness: re-sends a captured production workload against a server while
sampling the engine's own /metrics once a second.

Open-loop (arrivals don't wait for responses) matters: a closed-loop test would hide queueing,
which is exactly the failure we're trying to reproduce.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from common import promtext
from workload.client import send


async def _sample_engine(metrics_url: str, stop: asyncio.Event, samples: list[dict[str, Any]]) -> None:
    async with httpx.AsyncClient(timeout=3.0) as c:
        while not stop.is_set():
            try:
                r = await c.get(metrics_url)
                samples.append({"t": time.time(), **promtext.extract(r.text)})
            except httpx.HTTPError:
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass


def engine_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse 1 s engine samples into what matters for KV-cache/queueing diagnosis."""
    if not samples:
        return {"samples": 0}

    def vals(k: str) -> list[float]:
        return [s[k] for s in samples if s.get(k) is not None]

    def delta(k: str) -> float | None:
        v = vals(k)
        return round(v[-1] - v[0], 1) if len(v) >= 2 else None

    kv, waiting, running = vals("kv_cache_usage"), vals("waiting"), vals("running")
    span = max(1.0, samples[-1]["t"] - samples[0]["t"])
    hits, queries = delta("prefix_hits_total"), delta("prefix_queries_total")
    hit_rate = round(hits / queries, 3) if hits is not None and queries else None
    if hit_rate is None and vals("prefix_hit_rate"):
        hit_rate = round(sum(vals("prefix_hit_rate")) / len(vals("prefix_hit_rate")), 3)
    gen = delta("generation_tokens_total")
    return {
        "samples": len(samples),
        "kv_cache_usage_max": round(max(kv), 3) if kv else None,
        "kv_cache_usage_mean": round(sum(kv) / len(kv), 3) if kv else None,
        "running_mean": round(sum(running) / len(running), 1) if running else None,
        "waiting_mean": round(sum(waiting) / len(waiting), 1) if waiting else None,
        "waiting_max": max(waiting) if waiting else None,
        "preemptions": delta("preemptions_total"),
        "prefix_cache_hit_rate": hit_rate,
        "engine_generation_tokens_per_s": round(gen / span, 1) if gen is not None else None,
    }


async def run(base_url: str, model: str, items: list[dict[str, Any]], prompts: dict[str, dict[str, Any]],
              rate_multiplier: float = 1.0, duration_s: float = 60.0, max_inflight: int = 256,
              request_timeout_s: float = 120.0, drain_timeout_s: float = 60.0) -> dict[str, Any]:
    """items: [{"prompt_id", "max_tokens", "offset_s"}] with offsets relative to the capture start."""
    if not items:
        raise ValueError("empty workload")
    items = sorted(items, key=lambda i: i["offset_s"])
    span = max(1.0, items[-1]["offset_s"] + 1.0)
    records: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    stop = asyncio.Event()
    sem = asyncio.Semaphore(max_inflight)
    dropped = 0

    limits = httpx.Limits(max_connections=max_inflight, max_keepalive_connections=max_inflight)
    async with httpx.AsyncClient(limits=limits) as client:
        sampler = asyncio.create_task(_sample_engine(f"{base_url}/metrics", stop, samples))

        async def fire(item: dict[str, Any]) -> None:
            async with sem:
                records.append(await send(client, base_url, model, item["prompt_id"], prompts[item["prompt_id"]],
                                          max_tokens=item.get("max_tokens"), timeout_s=request_timeout_s))

        tasks: list[asyncio.Task[None]] = []
        t0 = time.monotonic()
        loop_i = 0
        while True:
            base = loop_i * span
            for it in items:
                at = (base + it["offset_s"]) / rate_multiplier
                if at >= duration_s:
                    break
                wait = at - (time.monotonic() - t0)
                if wait > 0:
                    await asyncio.sleep(wait)
                if it["prompt_id"] not in prompts:
                    dropped += 1
                    continue
                tasks.append(asyncio.create_task(fire(it)))
            else:
                loop_i += 1
                continue
            break

        issued_s = time.monotonic() - t0
        done, pending = await asyncio.wait(tasks, timeout=drain_timeout_s) if tasks else (set(), set())
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        cancelled = len(tasks) - len(records)
        now = time.time()
        for _ in range(cancelled):
            records.append({"request_id": None, "prompt_id": None, "t_start": now, "t_end": now,
                            "status": "cancelled", "error": f"still in flight after {drain_timeout_s}s drain",
                            "e2e_ms": None, "ttft_ms": None, "prompt_tokens": None, "output_tokens": None})
        stop.set()
        await sampler

    return {
        "records": records,
        "issued_duration_s": round(issued_s, 1),
        "dropped_unknown_prompts": dropped,
        "engine": engine_summary(samples),
    }
