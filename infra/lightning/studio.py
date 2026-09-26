"""Manage the GPU nodes on Lightning AI: two Studios on GCP with 1 x NVIDIA L4 each,
ff-prod (production vLLM + the users) and ff-shadow (experiments). Needs LIGHTNING_API_KEY in the
environment (never commit it). ROLE is prod or shadow.

  python -m infra.lightning.studio up ROLE        # create/start the Studio, expose :9000, save the URL
  python -m infra.lightning.studio push ROLE      # upload this repo + write ~/.ff/node.env (tokens)
  python -m infra.lightning.studio setup ROLE     # install vLLM, weights, dataset, start (detached)
  python -m infra.lightning.studio logs ROLE [setup|controller|loadgen|engine]
  python -m infra.lightning.studio start ROLE     # restart controller/engine/loadgen after a Studio restart
  python -m infra.lightning.studio run ROLE "<cmd>"
  python -m infra.lightning.studio stop ROLE      # stop the Studio (stops GPU billing)
  python -m infra.lightning.studio status

Env: LIGHTNING_TEAMSPACE (default vision-model), LIGHTNING_USER (default holamigotesthello),
LIGHTNING_CLOUD (default GCP), LIGHTNING_MACHINE (default L4),
CONTROLLER_TOKEN / HF_TOKEN (default: read from AWS SSM /firefighter/*).
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from lightning_sdk import Machine, Studio

ROOT = Path(__file__).resolve().parents[2]
FF = "/teamspace/studios/this_studio/ff"
STATE_FILE = ROOT / "infra" / "lightning" / "studio.env"   # git-ignored: studio URL
EXCLUDE = {".venv", "state", "data", ".git", "__pycache__", ".pytest_cache"}
ROLES = ("prod", "shadow")


def logs(role: str) -> dict[str, str]:
    return {"setup": f"{FF}/logs/setup.log", "controller": f"{FF}/logs/controller.log",
            "loadgen": f"{FF}/logs/loadgen.log", "start": f"{FF}/logs/start.log",
            "engine": f"{FF}/state/{role}/vllm.log"}


def studio(role: str, create: bool = False) -> Studio:
    return Studio(name=f"ff-{role}",
                  teamspace=os.environ.get("LIGHTNING_TEAMSPACE", "vision-model"),
                  user=os.environ.get("LIGHTNING_USER", "holamigotesthello"),
                  cloud=os.environ.get("LIGHTNING_CLOUD", "GCP"), create_ok=create)


def secret(name: str, ssm_name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    out = subprocess.run(["aws", "ssm", "get-parameter", "--region", os.environ.get("AWS_REGION", "ap-south-1"),
                          "--name", f"/firefighter/{ssm_name}", "--with-decryption", "--query", "Parameter.Value",
                          "--output", "text"], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"set {name} or make /firefighter/{ssm_name} readable in AWS SSM: {out.stderr.strip()}")
    return out.stdout.strip()


def read_state() -> dict[str, str]:
    if not STATE_FILE.exists():
        return {}
    return dict(line.split("=", 1) for line in STATE_FILE.read_text().split() if "=" in line)


def up(role: str) -> None:
    s = studio(role, create=True)
    machine = getattr(Machine, os.environ.get("LIGHTNING_MACHINE", "L4"))
    if str(s.status) != "Status.Running" and "Running" not in str(s.status):
        print(f"starting {s.name} on {machine} ...", flush=True)
        s.start(machine)
    s.auto_sleep = False
    print(s.run("nvidia-smi --query-gpu=index,name,memory.total --format=csv"))
    existing = [e for e in s.list_ports() if "9000" in [str(p) for p in (e.ports or [])]]
    urls = [u for e in (existing or s.add_ports({"controller": 9000})) for u in e.urls]
    state = read_state()
    state[f"{role.upper()}_CONTROLLER_URL"] = urls[0]
    STATE_FILE.write_text("".join(f"{k}={v}\n" for k, v in sorted(state.items())))
    print(f"{role} controller URL: {urls[0]}  (saved to {STATE_FILE.relative_to(ROOT)})")


def push(role: str) -> None:
    s = studio(role)
    with tempfile.TemporaryDirectory() as tmp:
        tgz = Path(tmp) / "app.tgz"
        with tarfile.open(tgz, "w:gz") as tar:
            for p in ROOT.iterdir():
                if p.name in EXCLUDE or p.name.endswith((".env", ".sqlite")):
                    continue
                tar.add(p, arcname=p.name, filter=lambda ti: None if any(
                    part in EXCLUDE for part in Path(ti.name).parts) or ti.name.endswith("instances.env") else ti)
        s.upload_file(str(tgz), "ff/app.tgz", progress_bar=False)  # remote path is relative to the Studio home
    env = {"CONTROLLER_TOKEN": secret("CONTROLLER_TOKEN", "controller_token"),
           "HF_TOKEN": secret("HF_TOKEN", "hf_token"),
           "MODEL": os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct"), "SERVED_MODEL_NAME": "qwen2.5-7b"}
    env_lines = "\n".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    print(s.run(f"mkdir -p {FF}/app {FF}/logs ~/.ff && tar xzf {FF}/app.tgz -C {FF}/app && "
                f"umask 077 && printf '%s\\n' {shlex.quote(env_lines)} > ~/.ff/node.env && echo pushed"))


def detached(role: str, cmd: str, log: str) -> None:
    studio(role).run_and_detach(f"cd {FF}/app && nohup bash -lc {shlex.quote(cmd)} > {log} 2>&1 &", timeout=5)
    print(f"started in background; follow with: python -m infra.lightning.studio logs {role}")


def main() -> None:
    args = sys.argv[1:]
    cmd = args[0] if args else "status"
    if cmd == "status":
        for role in ROLES:
            try:
                s = studio(role)
                print(f"ff-{role}: {s.status} {s.machine}")
            except Exception as e:
                print(f"ff-{role}: {type(e).__name__}: {e}")
        print(read_state())
        return
    if len(args) < 2 or args[1] not in ROLES:
        sys.exit(__doc__)
    role = args[1]
    if cmd == "up":
        up(role)
    elif cmd == "push":
        push(role)
    elif cmd == "setup":
        detached(role, f"bash infra/lightning/setup_studio.sh {role}", logs(role)["setup"])
    elif cmd == "start":
        detached(role, f"bash infra/lightning/start.sh {role}", logs(role)["start"])
    elif cmd == "logs":
        which = args[2] if len(args) > 2 else "setup"
        print(studio(role).run(f"tail -n {os.environ.get('N', '40')} {logs(role).get(which, which)}"))
    elif cmd == "run":
        print(studio(role).run(args[2]))
    elif cmd == "stop":
        studio(role).stop()
        print(f"ff-{role} stopped (GPU billing stopped); `up {role}` + `start {role}` brings it back")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
