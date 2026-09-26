"""Evidence registry: every captured workload and every shadow experiment, stored as JSON files.

apply_production_config reads this to decide whether a candidate config has been proven, so
the agent cannot talk its way into an untested change even after a human clicks Allow.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

STATE = Path(os.environ.get("MCP_STATE_DIR", "state/mcp"))


def _dir(kind: str) -> Path:
    p = STATE / kind
    p.mkdir(parents=True, exist_ok=True)
    return p


def new_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%H%M%S')}-{uuid.uuid4().hex[:4]}"


def save(kind: str, obj: dict[str, Any]) -> None:
    (_dir(kind) / f"{obj['id']}.json").write_text(json.dumps(obj))


def load(kind: str, obj_id: str) -> dict[str, Any]:
    p = _dir(kind) / f"{obj_id}.json"
    if not p.exists():
        raise KeyError(f"no {kind[:-1]} with id {obj_id}")
    return json.loads(p.read_text())


def all_(kind: str) -> list[dict[str, Any]]:
    out = [json.loads(p.read_text()) for p in _dir(kind).glob("*.json")]
    return sorted(out, key=lambda o: o["created_at"])
