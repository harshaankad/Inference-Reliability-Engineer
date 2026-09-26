"""Operator calibration tool: what production looks like over the last N minutes (the same client-side
and engine summaries the agent sees), plus SLO verdicts against a policy file.

  PROD_CONTROLLER_URL=... CONTROLLER_TOKEN=... python -m chaos.measure [minutes] [--policy mcp_server/policy.yaml]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import httpx
import yaml

from common import stats
from workload.loadtest import engine_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("minutes", nargs="?", type=float, default=5)
    ap.add_argument("--policy", default="mcp_server/policy.yaml")
    args = ap.parse_args()
    c = httpx.Client(base_url=os.environ.get("PROD_CONTROLLER_URL", "http://127.0.0.1:9000"), timeout=60,
                     headers={"Authorization": f"Bearer {os.environ.get('CONTROLLER_TOKEN', '')}"})
    end = time.time()
    start = end - args.minutes * 60
    records = c.get("/traffic/requests", params={"start": start, "end": end}).json()["records"]
    samples = c.get("/slots/prod/engine/series", params={"start": start, "end": end}).json()["samples"]
    slo = yaml.safe_load(open(args.policy))["slos"]
    summary = stats.summarize(records, duration_s=end - start, slo=slo)
    verdict = stats.evaluate_slos(summary, slo)
    print(json.dumps({"window_min": args.minutes, "client_side": summary, "engine": engine_summary(samples),
                      "slo_verdict": {k: (v["value"], v["pass"]) for k, v in verdict["slos"].items()},
                      "all_pass": verdict["all_pass"]}, indent=1))


if __name__ == "__main__":
    main()
