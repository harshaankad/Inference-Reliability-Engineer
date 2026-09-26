"""Node controller: runs on each GPU host and owns the vLLM container(s) on it.

Only this service touches Docker and the GPU. The MCP server talks to it over the VPC with a
bearer token; vLLM itself binds to 127.0.0.1 and is never reachable from outside the host.

Env:
  CONTROLLER_TOKEN   bearer token (required unless ALLOW_NO_AUTH=1, dev only)
  SLOTS              "name:gpu:port[,name:gpu:port]"  e.g. "prod:0:8100" or "shadow:0:8100"
  ENGINE_DRIVER      docker (real, vLLM container) | process (real, `vllm serve` child process;
                     for hosts without Docker, e.g. RunPod) | fake (local plumbing tests only)
  MODEL              HF model id (default Qwen/Qwen2.5-7B-Instruct)
  SERVED_MODEL_NAME  name clients use (default qwen2.5-7b)
  VLLM_IMAGE         docker image (default vllm/vllm-openai:latest; pin after calibration)
  HF_CACHE           host dir for weights (default /opt/hf)
  STATE_DIR          default ./state
  DATASET_PATH       default ./data/dataset.json
  REQUEST_DB         prod traffic request log (default $STATE_DIR/requests.db)
  SCENARIO_FILE      loadgen scenario file (default $STATE_DIR/scenario.json)
  HEALTH_TIMEOUT_S   max wait for vLLM to become healthy (default 900)

Run:  uvicorn controller.app:app --host 0.0.0.0 --port 9000
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from common import promtext
from common import vllm_config as vc
from workload import dataset as ds
from workload import loadtest
from workload.client import send

TOKEN = os.environ.get("CONTROLLER_TOKEN", "")
ALLOW_NO_AUTH = os.environ.get("ALLOW_NO_AUTH") == "1"
DRIVER = os.environ.get("ENGINE_DRIVER", "docker")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
SERVED = os.environ.get("SERVED_MODEL_NAME", "qwen2.5-7b")
IMAGE = os.environ.get("VLLM_IMAGE", "vllm/vllm-openai:latest")
HF_CACHE = os.environ.get("HF_CACHE", "/opt/hf")
STATE = Path(os.environ.get("STATE_DIR", "state"))
DATASET_PATH = os.environ.get("DATASET_PATH", "data/dataset.json")
REQUEST_DB = os.environ.get("REQUEST_DB", str(STATE / "requests.db"))
SCENARIO_FILE = os.environ.get("SCENARIO_FILE", str(STATE / "scenario.json"))
HEALTH_TIMEOUT_S = float(os.environ.get("HEALTH_TIMEOUT_S", "900"))
SAMPLE_EVERY_S = 5.0


def _parse_slots(spec: str) -> dict[str, dict[str, Any]]:
    slots = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        name, gpu, port = part.split(":")
        slots[name] = {"gpu": gpu, "port": int(port)}
    return slots


SLOTS = _parse_slots(os.environ.get("SLOTS", "prod:0:8100"))


# --------------------------------------------------------------------------- storage
class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS engine (slot TEXT, t REAL, data TEXT);
          CREATE INDEX IF NOT EXISTS idx_engine ON engine(slot, t);
          CREATE TABLE IF NOT EXISTS gpu (t REAL, data TEXT);
          CREATE INDEX IF NOT EXISTS idx_gpu ON gpu(t);
        """)

    def add_engine(self, slot: str, sample: dict[str, Any]) -> None:
        self.db.execute("INSERT INTO engine VALUES (?,?,?)", (slot, sample["t"], json.dumps(sample)))
        self.db.commit()

    def add_gpu(self, sample: dict[str, Any]) -> None:
        self.db.execute("INSERT INTO gpu VALUES (?,?)", (sample["t"], json.dumps(sample)))
        self.db.commit()

    def engine(self, slot: str, start: float, end: float) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT data FROM engine WHERE slot=? AND t BETWEEN ? AND ? ORDER BY t",
                               (slot, start, end)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def gpu(self, start: float, end: float) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT data FROM gpu WHERE t BETWEEN ? AND ? ORDER BY t", (start, end)).fetchall()
        return [json.loads(r[0]) for r in rows]


def _history_path(slot: str) -> Path:
    return STATE / slot / "history.json"


def load_history(slot: str) -> list[dict[str, Any]]:
    try:
        return json.loads(_history_path(slot).read_text())
    except (OSError, ValueError):
        return []


def save_history(slot: str, history: list[dict[str, Any]]) -> None:
    p = _history_path(slot)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(history, indent=1))


def active_version(slot: str) -> dict[str, Any] | None:
    return next((h for h in reversed(load_history(slot)) if h["status"] == "active"), None)


# --------------------------------------------------------------------------- engine drivers
async def _run(*cmd: str, timeout: float = 120) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return 124, "timeout"
    return proc.returncode or 0, out.decode(errors="replace")


class DockerDriver:
    def container(self, slot: str) -> str:
        return f"vllm-{slot}"

    def command(self, slot: str, config: dict[str, Any]) -> list[str]:
        s = SLOTS[slot]
        return ["docker", "run", "-d", "--name", self.container(slot), "--gpus", f"device={s['gpu']}",
                "--ipc=host", "-p", f"127.0.0.1:{s['port']}:8000", "-v", f"{HF_CACHE}:/root/.cache/huggingface",
                "-e", "HF_TOKEN", "--entrypoint", "vllm", IMAGE, "serve", MODEL,
                "--served-model-name", SERVED, "--port", "8000", *vc.render_flags(config)]

    async def start(self, slot: str, config: dict[str, Any]) -> str:
        code, out = await _run(*self.command(slot, config), timeout=600)
        if code != 0:
            raise RuntimeError(f"docker run failed: {out[-500:]}")
        return out.strip()

    async def stop(self, slot: str) -> None:
        await _run("docker", "rm", "-f", self.container(slot), timeout=120)

    async def logs(self, slot: str, tail: int) -> str:
        _, out = await _run("docker", "logs", "--tail", str(tail), self.container(slot), timeout=30)
        return out

    async def status(self, slot: str) -> str:
        code, out = await _run("docker", "inspect", "-f", "{{.State.Status}}", self.container(slot), timeout=15)
        return out.strip() if code == 0 else "absent"


class ProcessDriver:
    """Runs the engine as a child process (own process group) instead of a Docker container.
    Subclasses define the command. Used on hosts without Docker, e.g. GPU containers on RunPod."""

    log_name = "engine.log"

    def __init__(self) -> None:
        self.procs: dict[str, subprocess.Popen[bytes]] = {}

    def _log(self, slot: str) -> Path:
        return STATE / slot / self.log_name

    def env(self, slot: str) -> dict[str, str]:
        return dict(os.environ)

    def command(self, slot: str, config: dict[str, Any]) -> list[str]:
        raise NotImplementedError

    def orphan_pattern(self, slot: str) -> str | None:
        return None  # regex for pkill -f, to clean up engines left over from a controller restart

    async def start(self, slot: str, config: dict[str, Any]) -> str:
        self._log(slot).parent.mkdir(parents=True, exist_ok=True)
        log = open(self._log(slot), "ab")
        self.procs[slot] = subprocess.Popen(self.command(slot, config), stdout=log, stderr=subprocess.STDOUT,
                                            env=self.env(slot), start_new_session=True)
        return str(self.procs[slot].pid)

    async def stop(self, slot: str) -> None:
        p = self.procs.pop(slot, None)
        if p and p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
                await asyncio.get_running_loop().run_in_executor(None, lambda: p.wait(timeout=60))
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        pattern = self.orphan_pattern(slot)
        if pattern:
            await _run("pkill", "-f", pattern, timeout=15)
            await asyncio.sleep(2)

    async def logs(self, slot: str, tail: int) -> str:
        try:
            return "\n".join(self._log(slot).read_text(errors="replace").splitlines()[-tail:])
        except OSError:
            return ""

    async def status(self, slot: str) -> str:
        p = self.procs.get(slot)
        if p is None:
            return "absent"
        return "running" if p.poll() is None else "exited"


class VllmProcessDriver(ProcessDriver):
    """Real vLLM, launched as `vllm serve` (the binary from the vllm/vllm-openai image or a pip install)."""

    log_name = "vllm.log"

    def env(self, slot: str) -> dict[str, str]:
        return {**os.environ, "CUDA_VISIBLE_DEVICES": str(SLOTS[slot]["gpu"])}

    def command(self, slot: str, config: dict[str, Any]) -> list[str]:
        return [os.environ.get("VLLM_BIN", "vllm"), "serve", MODEL, "--served-model-name", SERVED,
                "--host", "127.0.0.1", "--port", str(SLOTS[slot]["port"]), *vc.render_flags(config)]

    def orphan_pattern(self, slot: str) -> str | None:
        return f"vllm serve .*--port {SLOTS[slot]['port']}( |$)"


class FakeDriver(ProcessDriver):
    """LOCAL PLUMBING TESTS ONLY. Spawns dev/fake_vllm.py instead of a GPU engine.
    Never use for calibration or the demo: its numbers are not real."""

    log_name = "fake_vllm.log"

    def command(self, slot: str, config: dict[str, Any]) -> list[str]:
        return [sys.executable, "-m", "dev.fake_vllm", "--port", str(SLOTS[slot]["port"]),
                "--model", SERVED, "--config", json.dumps(config)]


driver: DockerDriver | ProcessDriver = {"fake": FakeDriver, "process": VllmProcessDriver}.get(DRIVER, DockerDriver)()
store = Store(STATE / "controller.db")
deploys: dict[str, dict[str, Any]] = {}
slot_locks = {s: asyncio.Lock() for s in SLOTS}
busy: dict[str, str | None] = {s: None for s in SLOTS}  # "deploying" | "load_test" | None
_dataset: dict[str, Any] | None = None


def dataset() -> dict[str, Any]:
    global _dataset
    if _dataset is None:
        _dataset = ds.load(DATASET_PATH)
    return _dataset


def base_url(slot: str) -> str:
    return f"http://127.0.0.1:{SLOTS[slot]['port']}"


async def healthy(slot: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            return (await c.get(f"{base_url(slot)}/health")).status_code == 200
    except httpx.HTTPError:
        return False


async def wait_healthy(slot: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if await healthy(slot):
            return True
        if await driver.status(slot) in ("exited", "dead", "absent"):
            await asyncio.sleep(2)
            if await driver.status(slot) in ("exited", "dead", "absent"):
                return False
        await asyncio.sleep(3)
    return False


async def _do_deploy(dep: dict[str, Any], auto_rollback: bool) -> None:
    slot, config = dep["slot"], dep["config"]
    prev = active_version(slot)
    busy[slot] = "deploying"
    try:
        dep["status"] = "stopping"
        t0 = time.time()
        await driver.stop(slot)
        dep["status"] = "starting"
        await driver.start(slot, config)
        ok = await wait_healthy(slot, HEALTH_TIMEOUT_S)
        history = load_history(slot)
        entry = {"version": (history[-1]["version"] + 1) if history else 1, "ts": time.time(),
                 "config": config, "config_hash": vc.config_hash(config, MODEL), "author": dep["author"],
                 "message": dep["message"], "command": shlex.join(driver.command(slot, config)),
                 "deploy_id": dep["id"]}
        if ok:
            for h in history:
                if h["status"] == "active":
                    h["status"] = "superseded"
            entry.update(status="active", downtime_s=round(time.time() - t0, 1))
            history.append(entry)
            save_history(slot, history)
            dep.update(status="healthy", version=entry["version"], downtime_s=entry["downtime_s"])
            return
        tail = await driver.logs(slot, 60)
        entry.update(status="failed", failure_log_tail=tail[-3000:])
        history.append(entry)
        save_history(slot, history)
        dep.update(status="failed", failure_log_tail=tail[-3000:])
        if auto_rollback and prev:
            dep["status"] = "rolling_back"
            await driver.stop(slot)
            await driver.start(slot, prev["config"])
            ok = await wait_healthy(slot, HEALTH_TIMEOUT_S)
            dep.update(status="rolled_back" if ok else "rollback_failed", rolled_back_to=prev["version"],
                       downtime_s=round(time.time() - t0, 1))
    except Exception as e:  # surface everything to the caller; never leave a deploy "in progress"
        dep.update(status="error", error=f"{type(e).__name__}: {e}")
    finally:
        dep["finished_at"] = time.time()
        busy[slot] = None


# --------------------------------------------------------------------------- samplers
async def _nvidia_smi() -> list[dict[str, Any]] | dict[str, str]:
    if DRIVER == "fake":
        return {"note": "fake driver: no GPU"}
    code, out = await _run("nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,"
                           "temperature.gpu,power.draw", "--format=csv,noheader,nounits", timeout=15)
    if code != 0:
        return {"error": out[-300:]}
    gpus = []
    for line in out.strip().splitlines():
        idx, name, used, total, util, temp, power = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(idx), "name": name, "memory_used_mib": float(used),
                     "memory_total_mib": float(total), "utilization_pct": float(util),
                     "temperature_c": float(temp), "power_w": float(power) if power not in ("[N/A]", "") else None})
    return gpus


async def sampler_loop() -> None:
    async with httpx.AsyncClient(timeout=3.0) as c:
        while True:
            now = time.time()
            for slot in SLOTS:
                if busy[slot] == "deploying":
                    continue
                try:
                    r = await c.get(f"{base_url(slot)}/metrics")
                    if r.status_code == 200:
                        store.add_engine(slot, {"t": now, **promtext.extract(r.text)})
                except httpx.HTTPError:
                    pass
            gpu = await _nvidia_smi()
            if isinstance(gpu, list):
                store.add_gpu({"t": now, "gpus": gpu})
            await asyncio.sleep(SAMPLE_EVERY_S)


@asynccontextmanager
async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
    if not TOKEN and not ALLOW_NO_AUTH:
        raise RuntimeError("CONTROLLER_TOKEN is required (set ALLOW_NO_AUTH=1 only for local dev)")
    task = asyncio.create_task(sampler_loop())
    yield
    task.cancel()


app = FastAPI(title="inference node controller", lifespan=lifespan)


def auth(authorization: str = Header(default="")) -> None:
    if ALLOW_NO_AUTH and not TOKEN:
        return
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(401, "bad token")


def _slot(slot: str) -> str:
    if slot not in SLOTS:
        raise HTTPException(404, f"unknown slot {slot}; this node has {sorted(SLOTS)}")
    return slot


# --------------------------------------------------------------------------- API
@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "slots": sorted(SLOTS), "driver": DRIVER}


@app.get("/slots/{slot}", dependencies=[Depends(auth)])
async def slot_info(slot: str) -> dict[str, Any]:
    _slot(slot)
    act = active_version(slot)
    running_requests = None
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            running_requests = promtext.extract((await c.get(f"{base_url(slot)}/metrics")).text).get("running")
    except httpx.HTTPError:
        pass
    return {
        "slot": slot, "model": MODEL, "served_model_name": SERVED, "gpu": SLOTS[slot]["gpu"],
        "container_status": await driver.status(slot), "healthy": await healthy(slot), "busy": busy[slot],
        "active_version": act and {k: act[k] for k in ("version", "ts", "config", "config_hash", "author",
                                                         "message", "command", "downtime_s")},
        "requests_running_now": running_requests,
        "dataset_fingerprint": ds.fingerprint(dataset()) if Path(DATASET_PATH).exists() else None,
    }


@app.get("/slots/{slot}/history", dependencies=[Depends(auth)])
async def slot_history(slot: str) -> list[dict[str, Any]]:
    return [{k: v for k, v in h.items() if k != "failure_log_tail"} for h in load_history(_slot(slot))]


class DeployReq(BaseModel):
    config: dict[str, Any]
    author: str = "unknown"
    message: str = ""
    auto_rollback: bool = True


@app.post("/slots/{slot}/deploy", dependencies=[Depends(auth)])
async def deploy(slot: str, req: DeployReq) -> dict[str, Any]:
    _slot(slot)
    try:
        config = vc.validate(req.config)
    except vc.ConfigError as e:
        raise HTTPException(400, str(e))
    if busy[slot]:
        raise HTTPException(409, f"slot {slot} is busy: {busy[slot]}")
    dep = {"id": uuid.uuid4().hex[:10], "slot": slot, "config": config, "author": req.author,
           "message": req.message, "status": "queued", "started_at": time.time(),
           "config_hash": vc.config_hash(config, MODEL)}
    deploys[dep["id"]] = dep
    busy[slot] = "deploying"
    asyncio.create_task(_do_deploy(dep, req.auto_rollback))
    return dep


@app.get("/deploys/{deploy_id}", dependencies=[Depends(auth)])
async def deploy_status(deploy_id: str) -> dict[str, Any]:
    if deploy_id not in deploys:
        raise HTTPException(404, "unknown deploy")
    return deploys[deploy_id]


@app.get("/slots/{slot}/logs", dependencies=[Depends(auth)])
async def logs(slot: str, tail: int = Query(200, le=5000), grep: str | None = None) -> dict[str, Any]:
    text = await driver.logs(_slot(slot), tail)
    lines = text.splitlines()
    if grep:
        rx = re.compile(grep, re.IGNORECASE)
        lines = [ln for ln in lines if rx.search(ln)]
    return {"slot": slot, "lines": lines[-tail:]}


@app.get("/slots/{slot}/metrics", dependencies=[Depends(auth)])
async def slot_metrics(slot: str) -> PlainTextResponse:
    """Prometheus scrape target: vLLM's own /metrics, proxied (vLLM binds to 127.0.0.1 only)."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{base_url(_slot(slot))}/metrics")
        return PlainTextResponse(r.text, status_code=r.status_code)
    except httpx.HTTPError:
        return PlainTextResponse("# engine down\n", status_code=503)


@app.get("/gpu/metrics", dependencies=[Depends(auth)])
async def gpu_metrics() -> PlainTextResponse:
    """Prometheus scrape target: nvidia-smi as gauges (no DCGM exporter needed)."""
    gpus = await _nvidia_smi()
    lines = []
    if isinstance(gpus, list):
        for name, key in [("ff_gpu_memory_used_mib", "memory_used_mib"), ("ff_gpu_memory_total_mib", "memory_total_mib"),
                          ("ff_gpu_utilization_pct", "utilization_pct"), ("ff_gpu_temperature_c", "temperature_c"),
                          ("ff_gpu_power_w", "power_w")]:
            lines.append(f"# TYPE {name} gauge")
            lines += [f'{name}{{gpu="{g["index"]}",name="{g["name"]}"}} {g[key]}' for g in gpus if g[key] is not None]
    return PlainTextResponse("\n".join(lines) + "\n")


@app.get("/slots/{slot}/engine/series", dependencies=[Depends(auth)])
async def engine_series(slot: str, start: float, end: float) -> dict[str, Any]:
    return {"slot": _slot(slot), "samples": store.engine(slot, start, end), "sample_every_s": SAMPLE_EVERY_S}


@app.get("/gpu", dependencies=[Depends(auth)])
async def gpu(start: float | None = None, end: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"now": await _nvidia_smi()}
    if start is not None and end is not None:
        out["samples"] = store.gpu(start, end)
    return out


class LoadTestReq(BaseModel):
    items: list[dict[str, Any]]
    rate_multiplier: float = Field(1.0, gt=0, le=10)
    duration_s: float = Field(60, ge=10, le=240)
    max_inflight: int = Field(256, ge=1, le=1024)
    request_timeout_s: float = Field(120, ge=5, le=600)
    drain_timeout_s: float = Field(60, ge=0, le=300)


jobs: dict[str, dict[str, Any]] = {}


def _start_job(kind: str, slot: str, coro: Any) -> dict[str, Any]:
    """Run a long task in the background; clients poll GET /jobs/{id}. Keeps every HTTP request short,
    which matters behind proxies with ~100 s timeouts (e.g. RunPod)."""
    job = {"id": uuid.uuid4().hex[:10], "kind": kind, "slot": slot, "status": "running", "started_at": time.time()}
    jobs[job["id"]] = job

    async def runner() -> None:
        try:
            job["result"] = await coro
            job["status"] = "done"
        except Exception as e:  # surfaced to the poller
            job.update(status="error", error=f"{type(e).__name__}: {e}")
        job["finished_at"] = time.time()

    asyncio.create_task(runner())
    return {k: v for k, v in job.items() if k != "result"}


@app.get("/jobs/{job_id}", dependencies=[Depends(auth)])
async def job_status(job_id: str) -> dict[str, Any]:
    if job_id not in jobs:
        raise HTTPException(404, "unknown job")
    return jobs[job_id]


async def _precheck(slot: str) -> None:
    _slot(slot)
    if busy[slot]:
        raise HTTPException(409, f"slot {slot} is busy: {busy[slot]}")
    if not await healthy(slot):
        raise HTTPException(409, f"slot {slot} is not healthy; deploy a config first")


async def _loadtest(slot: str, req: LoadTestReq) -> dict[str, Any]:
    act = active_version(slot)
    async with slot_locks[slot]:
        busy[slot] = "load_test"
        try:
            result = await loadtest.run(base_url(slot), SERVED, req.items, dataset()["prompts"],
                                        rate_multiplier=req.rate_multiplier, duration_s=req.duration_s,
                                        max_inflight=req.max_inflight, request_timeout_s=req.request_timeout_s,
                                        drain_timeout_s=req.drain_timeout_s)
        finally:
            busy[slot] = None
    result["config"] = act and act["config"]
    result["config_hash"] = act and act["config_hash"]
    return result


@app.post("/slots/{slot}/loadtest", dependencies=[Depends(auth)])
async def run_loadtest(slot: str, req: LoadTestReq, background: bool = False) -> dict[str, Any]:
    await _precheck(slot)
    if background:
        busy[slot] = "load_test"  # claim the slot now so a second request gets 409 immediately
        async def go() -> dict[str, Any]:
            busy[slot] = None
            return await _loadtest(slot, req)
        return _start_job("load_test", slot, go())
    return await _loadtest(slot, req)


class QualityReq(BaseModel):
    n: int = Field(40, ge=1, le=200)
    concurrency: int = Field(8, ge=1, le=64)


async def _quality(slot: str, req: QualityReq) -> dict[str, Any]:
    prompts = dataset()["prompts"]
    golden = sorted(k for k, p in prompts.items() if p["kind"] == "golden")[: req.n]
    sem = asyncio.Semaphore(req.concurrency)
    async with slot_locks[slot]:
        busy[slot] = "load_test"
        try:
            async with httpx.AsyncClient() as c:
                async def one(pid: str) -> dict[str, Any]:
                    async with sem:
                        rec = await send(c, base_url(slot), SERVED, pid, prompts[pid], temperature=0.0,
                                         timeout_s=300, keep_text=True)
                    expected = prompts[pid]["answer"]
                    return {"prompt_id": pid, "expected": expected, "output": (rec.get("text") or "")[:200],
                            "correct": expected in (rec.get("text") or ""), "status": rec["status"],
                            "prompt_tokens": rec["prompt_tokens"]}
                rows = await asyncio.gather(*(one(p) for p in golden))
        finally:
            busy[slot] = None
    act = active_version(slot)
    return {"rows": rows, "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
            "config_hash": act and act["config_hash"], "config": act and act["config"]}


@app.post("/slots/{slot}/quality", dependencies=[Depends(auth)])
async def quality(slot: str, req: QualityReq, background: bool = False) -> dict[str, Any]:
    """Golden long-context prompts at temperature 0; each has one exact answer (a needle code)."""
    await _precheck(slot)
    if background:
        busy[slot] = "load_test"
        async def go() -> dict[str, Any]:
            busy[slot] = None
            return await _quality(slot, req)
        return _start_job("quality", slot, go())
    return await _quality(slot, req)


# ---- production traffic (prod node only) -------------------------------------------------
_REQ_FIELDS = ["request_id", "prompt_id", "t_start", "t_end", "max_tokens", "prompt_tokens", "output_tokens",
               "ttft_ms", "e2e_ms", "status", "error"]


@app.get("/traffic/requests", dependencies=[Depends(auth)])
async def traffic_requests(start: float, end: float, limit: int = Query(20000, le=100000)) -> dict[str, Any]:
    if not Path(REQUEST_DB).exists():
        return {"records": [], "note": "no request log on this node"}
    conn = sqlite3.connect(REQUEST_DB)
    try:
        rows = conn.execute(f"SELECT {','.join(_REQ_FIELDS)} FROM requests WHERE t_start BETWEEN ? AND ? "
                            f"ORDER BY t_start LIMIT ?", (start, end, limit)).fetchall()
    finally:
        conn.close()
    return {"records": [dict(zip(_REQ_FIELDS, r)) for r in rows]}


@app.get("/traffic/scenario", dependencies=[Depends(auth)])
async def get_scenario() -> dict[str, Any]:
    try:
        return json.loads(Path(SCENARIO_FILE).read_text())
    except (OSError, ValueError):
        return {}


@app.post("/traffic/scenario", dependencies=[Depends(auth)])
async def set_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    """Chaos tooling only. Not exposed through the MCP server."""
    if "rps" not in scenario or "long_share" not in scenario:
        raise HTTPException(400, "scenario needs rps and long_share")
    scenario = dict(scenario, started_at=time.time())
    Path(SCENARIO_FILE).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(SCENARIO_FILE).with_suffix(".tmp")
    tmp.write_text(json.dumps(scenario))
    tmp.replace(SCENARIO_FILE)
    return scenario
