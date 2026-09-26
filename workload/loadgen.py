"""Production traffic: the "users". Runs continuously on the prod node against the prod vLLM.

Poisson arrivals; the traffic mix comes from a scenario file that the chaos tooling swaps at
runtime. Every completed request is written to the request log (SQLite), which is the
client-side truth the agent later reads.

The scenario name is deliberately NOT written to the request log: in real life nobody labels
traffic "long_context_shift". The agent must infer the shift from prompt token counts.

  python -m workload.loadgen --base-url http://127.0.0.1:8100 --model qwen2.5-7b \
      --dataset data/dataset.json --scenario-file state/scenario.json --db state/requests.db
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx

from workload import dataset as ds
from workload.client import send

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  request_id TEXT, prompt_id TEXT, t_start REAL, t_end REAL, max_tokens INTEGER,
  prompt_tokens INTEGER, output_tokens INTEGER, ttft_ms REAL, e2e_ms REAL, status TEXT, error TEXT);
CREATE INDEX IF NOT EXISTS idx_requests_t ON requests(t_start);
"""
FIELDS = ["request_id", "prompt_id", "t_start", "t_end", "max_tokens", "prompt_tokens", "output_tokens",
          "ttft_ms", "e2e_ms", "status", "error"]
DEFAULT_SCENARIO = {"rps": 2.0, "long_share": 0.05}


def open_db(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def read_scenario(path: str) -> dict[str, Any]:
    try:
        with open(path) as f:
            sc = json.load(f)
        sc.setdefault("started_at", os.path.getmtime(path))
        return sc
    except (OSError, ValueError):
        return dict(DEFAULT_SCENARIO, started_at=time.time())


def current_rate(sc: dict[str, Any], now: float) -> tuple[float, float]:
    """(rps, long_share) right now; supports an optional initial burst."""
    burst = sc.get("burst")
    if burst and now - sc["started_at"] < burst["duration_s"]:
        return burst["rps"], burst.get("long_share", sc["long_share"])
    return sc["rps"], sc["long_share"]


async def main_async(args: argparse.Namespace) -> None:
    prompts = ds.load(args.dataset)["prompts"]
    short_ids = [k for k, p in prompts.items() if p["kind"] == "short"]
    long_ids = [k for k, p in prompts.items() if p["kind"] == "long"]
    conn = open_db(args.db)
    rng = random.Random()
    inflight: set[asyncio.Task[None]] = set()
    limits = httpx.Limits(max_connections=args.max_inflight, max_keepalive_connections=args.max_inflight)

    async with httpx.AsyncClient(limits=limits) as client:
        async def one(pid: str) -> None:
            rec = await send(client, args.base_url, args.model, pid, prompts[pid], timeout_s=args.timeout)
            conn.execute(f"INSERT INTO requests ({','.join(FIELDS)}) VALUES ({','.join('?' * len(FIELDS))})",
                         [rec[k] for k in FIELDS])
            conn.commit()

        sc, sc_checked = read_scenario(args.scenario_file), 0.0
        while True:
            now = time.time()
            if now - sc_checked > 1.0:
                sc, sc_checked = read_scenario(args.scenario_file), now
            rps, long_share = current_rate(sc, now)
            if rps <= 0:
                await asyncio.sleep(1.0)
                continue
            await asyncio.sleep(rng.expovariate(rps))
            if len(inflight) >= args.max_inflight:
                continue  # client-side cap; shows up as lower offered load, never silently queued
            pid = rng.choice(long_ids if rng.random() < long_share else short_ids)
            task = asyncio.create_task(one(pid))
            inflight.add(task)
            task.add_done_callback(inflight.discard)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "qwen2.5-7b"))
    ap.add_argument("--dataset", default="data/dataset.json")
    ap.add_argument("--scenario-file", default="state/scenario.json")
    ap.add_argument("--db", default="state/requests.db")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-inflight", type=int, default=400)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
