"""Operator tooling for the demo: inject traffic scenarios and reset. NOT exposed to the agent.

  export PROD_CONTROLLER_URL=http://<prod-ip>:9000 SHADOW_CONTROLLER_URL=http://<shadow-ip>:9000 CONTROLLER_TOKEN=...
  python -m chaos.chaos status
  python -m chaos.chaos traffic long_context_shift     # the flagship incident (no config change!)
  python -m chaos.chaos traffic healthy|surge|burst
  python -m chaos.chaos reset [--clear-evidence state/mcp]   # healthy traffic + initial config on prod and shadow
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _client(url_env: str, default: str) -> tuple[httpx.Client, str]:
    token = os.environ.get("CONTROLLER_TOKEN", "")
    return httpx.Client(base_url=os.environ.get(url_env, default), timeout=30,
                        headers={"Authorization": f"Bearer {token}"}), url_env


def prod() -> httpx.Client:
    return _client("PROD_CONTROLLER_URL", "http://127.0.0.1:9000")[0]


def shadow() -> httpx.Client:
    return _client("SHADOW_CONTROLLER_URL", "http://127.0.0.1:9001")[0]


def traffic(name: str) -> None:
    path = ROOT / "workload" / "scenarios" / f"{name}.json"
    if not path.exists():
        sys.exit(f"no scenario {name}; have: {[p.stem for p in path.parent.glob('*.json')]}")
    r = prod().post("/traffic/scenario", json=json.loads(path.read_text()))
    r.raise_for_status()
    print(f"traffic -> {name}: {r.json()}")


def deploy(c: httpx.Client, slot: str, config: dict, message: str, wait_s: float = 1200) -> None:
    r = c.post(f"/slots/{slot}/deploy", json={"config": config, "author": "platform-team", "message": message})
    r.raise_for_status()
    dep = r.json()
    deadline = time.time() + wait_s
    while dep["status"] not in ("healthy", "failed", "rolled_back", "rollback_failed", "error") and time.time() < deadline:
        time.sleep(5)
        dep = c.get(f"/deploys/{dep['id']}").json()
        print(f"  {slot}: {dep['status']}", flush=True)
    print(f"{slot}: {dep['status']} (v{dep.get('version')}, {dep.get('downtime_s')}s)")
    if dep["status"] != "healthy":
        print(dep.get("failure_log_tail") or dep.get("error"))


def initial_config() -> dict:
    return yaml.safe_load((ROOT / "infra" / "configs" / "prod_initial.yaml").read_text())


def status() -> None:
    p = prod()
    print("prod:", json.dumps(p.get("/slots/prod").json().get("active_version"), indent=1)[:800])
    print("scenario:", p.get("/traffic/scenario").json())
    try:
        print("shadow:", (shadow().get("/slots/shadow").json().get("active_version") or {}).get("config"))
    except httpx.HTTPError as e:
        print("shadow unreachable:", e)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    t = sub.add_parser("traffic")
    t.add_argument("scenario")
    r = sub.add_parser("reset")
    r.add_argument("--clear-evidence", metavar="MCP_STATE_DIR")
    r.add_argument("--skip-shadow", action="store_true")
    sub.add_parser("deploy-initial")
    args = ap.parse_args()

    if args.cmd == "status":
        status()
    elif args.cmd == "traffic":
        traffic(args.scenario)
    elif args.cmd in ("reset", "deploy-initial"):
        if args.cmd == "reset":
            traffic("healthy")
        cfg = initial_config()
        msg = "Initial prod config for qwen2.5-7b (reviewed)"
        cur = prod().get("/slots/prod").json().get("active_version") or {}
        if cur.get("config") != cfg:
            deploy(prod(), "prod", cfg, msg)
        else:
            print("prod already on the initial config")
        if not getattr(args, "skip_shadow", False):
            deploy(shadow(), "shadow", cfg, "shadow baseline")
        if getattr(args, "clear_evidence", None):
            shutil.rmtree(args.clear_evidence, ignore_errors=True)
            print(f"cleared {args.clear_evidence}")


if __name__ == "__main__":
    main()
