# Inference Firefighter — Build Roadmap (v2: AWS + OpenAI + traffic scenarios)

**Hackathon:** Agents That Act — TrueFoundry × Polaris (one day, Bengaluru)
**Harness:** TrueForge · **Cloud:** AWS (organizer-provided) · **Agent LLM:** OpenAI (organizer-provided)

**One-liner:** An AI inference engineer that works out *why* production inference got slow. It could be a bad config change, a surge of users, a change in what users send, or all of these at once. It reproduces the problem on a real GPU in a shadow environment, proves the right fix (tune config, scale out, or shed load), and only then asks permission to touch production.

**What changed from v1**
- Everything runs in AWS: the TrueForge harness, our tool server, monitoring, and the GPU inference fleet.
- The agent's model is OpenAI, configured in TrueForge (Settings → Models).
- The agent now handles **traffic-driven incidents**, not only config regressions. It covers user surges, prompt-length shifts, output-length shifts, transient bursts, overload thrashing, and mixed cases. Each class has its own correct fix, and some have "do nothing" as the right answer.

---

## 0. How this project meets every hackathon rule

| Rule | How we satisfy it |
|---|---|
| Runs on TrueForge | The agent is a saved TrueForge agent (OpenAI model + our MCP server + runbook skill + sandbox + approvals). It runs from the TrueForge chat UI hosted on our AWS control-plane instance. |
| Reaches a **real system** | Real **vLLM** on real **AWS GPU instances** serving real traffic, real **Prometheus/DCGM** metrics, real container logs, a real **nginx gateway**, a real **GitHub** config repo, and real **AWS EC2** capacity actions |
| Generated code runs **safely in a sandbox** | All analysis code the agent writes runs in a **Daytona sandbox** through TrueForge **Code Mode**. Examples: p95 calculations, before/after traffic comparisons, capacity-curve fitting, replica math, quality scoring, charts. The sandbox has no credentials and no network path to production. |
| **Stops before irreversible actions** | Every production-changing tool is gated by `require_approval_for_tools`: config changes, scaling (which costs money), admission/rate-limit policy, and rollback. The MCP server also refuses actions that lack validated evidence. |
| Built during the hackathon | Pre-event work is accounts, docs and planning only. The repo is created at kickoff. |
| AI tools disclosed | README "AI assistance" section |
| Can explain architecture/code | One owner per component + a teach-back before judging |
| Credentials | Organizer AWS/OpenAI credentials are **never** in the repo or video. On AWS we use an **IAM instance role** (no static keys at all). The OpenAI key lives only in TrueForge's model settings. `gitleaks` runs as a pre-commit hook. |

---

## 1. What we're building

### 1.1 The incident classes the agent must handle

The core idea: **latency and throughput problems have different causes, and each cause has a different correct fix.** Applying the wrong fix wastes money, as in scaling out for a config bug, or doesn't work, as in tuning config for a 3× user surge. The agent's job is to **classify first, then fix**.

| # | Scenario (how we trigger it) | What the metrics show | Correct remediation |
|---|---|---|---|
| **S1** | **Config regression.** Bad commit lowers `gpu_memory_utilization` and raises `max_num_seqs`. | Latency jumps exactly at a deploy event. Offered load and traffic mix unchanged. KV cache ≈100%, preemptions ↑. | Config fix **inside constraints** (e.g. memory budget) → `apply_production_config` |
| **S2** | **User surge.** Many more users; RPS ramps 2–4×. | Offered RPS ↑, same token mix. Queue (`num_requests_waiting`) grows. Served throughput **plateaus** at capacity, so **goodput and per-user tokens/s fall**. GPU util ≈ max. | Capacity problem: `scale_replicas` (costs $/hr → approval). If the GPU was under-used, a config efficiency gain first. Rate limiting as a stopgap if the budget is capped. |
| **S3** | **Prompt-length shift.** Same RPS, far more long-context (RAG) prompts. | TTFT ↑ a lot, prefill tokens/s ↑, KV pressure ↑. Decode speed mostly unchanged. | Prefill-side tuning: chunked-prefill token budget (`max_num_batched_tokens`), prefix caching, FP8 KV cache. Or a dedicated long-context replica (stretch). |
| **S4** | **Output-length shift.** Users ask for much longer answers (`max_tokens` ↑). | End-to-end latency ↑, concurrency ↑, time per output token roughly stable at first. Requests stay in flight longer. | Decode-side: balance `max_num_seqs`, scale out, or cap `max_tokens` per tier. That last one is a **product decision**, so the agent **asks the human** (`ask_user_question`). |
| **S5** | **Transient burst.** 30–60 s spike, then back to normal. | Short p95 spike, queue drains by itself, no deploy event. | **No production change.** The agent explains why and suggests alert tuning. Knowing when *not* to act is part of the grade. |
| **S6** | **Overload thrashing.** Surge + too-high `max_num_seqs`. | Offered load ↑ **but served throughput ↓** (collapse), preemptions spike, KV thrash. | Counter-intuitive fix: **lower** `max_num_seqs` (fewer concurrent sequences means more throughput), then scale if still needed |
| **S7** | **Mixed: config change + traffic change at the same time** (the flagship demo) | Both a deploy event and a traffic shift in the same window | **Counterfactual replay** (§1.3) to split the blame, then a fix that addresses both causes |
| S8 *(stretch)* | **Prefix caching disabled** in a "refactor" commit | TTFT ↑ for requests sharing a long system prompt; prefix-cache hit rate → 0 | Re-enable prefix caching |
| S9 *(stretch)* | **Traffic drop / over-provisioned** | Low GPU util, very low queue, 2 replicas idle | Scale **in** to save $/hr (approval) |

### 1.2 Metrics: what "throughput decreased" actually means
"Throughput" means several different things, and the agent has to be precise about which one moved. We expose all of these:

| Metric | Meaning | Source |
|---|---|---|
| **Offered load** | Requests arriving per second (the demand) | gateway / loadgen |
| **Served throughput** | Completed req/s and **output tokens/s** (what the fleet delivers) | `vllm:generation_tokens_total`, gateway |
| **Per-user speed** | Tokens/s each user sees; time per output token (TPOT / inter-token latency) | vLLM histograms, loadgen |
| **Goodput** | Requests/s completed **within SLO**; this is what users feel | loadgen histogram vs SLO |
| **Latency** | p50/p95/p99 end-to-end, TTFT | vLLM + loadgen histograms |
| **Saturation** | Waiting queue, running seqs, KV-cache usage %, preemptions/s, GPU utilization, GPU memory | vLLM `/metrics`, **DCGM exporter** |
| **Traffic shape** | Prompt-token and output-token distributions, long/short share, arrival burstiness | `vllm:request_prompt_tokens`, `vllm:request_generation_tokens`, loadgen request log |

When "a lot of users started using it", the typical signature is: offered load ↑, served throughput flat (at capacity), **per-user speed ↓ and goodput ↓**, queue ↑. When the server is thrashing (S6), even served throughput falls.

### 1.3 How the agent reasons (the method we teach it in the runbook skill)
1. **Before vs now.** Compare the incident window against a healthy baseline window on every metric group in §1.2. This is done in code in the sandbox.
2. **Change events.** List everything that changed in the window: config commits, deploys, scaling events, traffic-shape changes.
3. **Classify** using the table in §1.1 (config / load / shape / transient / thrash / mixed). Give a confidence level and the evidence for and against.
4. **Counterfactual replay (for mixed or unclear cases).** Capture a traffic sample from *before* (W₀) and from *now* (W₁). Replay all four combinations in shadow:

   |  | Old config C₀ | New config C₁ |
   |---|---|---|
   | Old traffic W₀ | healthy baseline | effect of config alone |
   | New traffic W₁ | effect of traffic alone | reproduces incident |

   This attributes the regression to config, to traffic, or to the interaction between them, using real measurements rather than guesswork.
5. **Capacity curve (for load problems).** Replay the current traffic shape at 0.5×, 1×, 1.5×, 2×… the rate against one replica in shadow. Find the **knee**, i.e. the max RPS that still meets SLO, for each candidate config. Then:
   `replicas_needed = ceil(peak_RPS × headroom / capacity_per_replica)`.
6. **Pick the cheapest fix that passes all SLOs and constraints**, in this order: config tuning (free) → scale out ($) → admission control (hurts some users). Anything involving a business trade-off gets asked as a question, e.g. "cap free-tier max_tokens vs +$X/hr".
7. **Prove it** with a longer replay (and a quality eval if the fix could affect outputs, e.g. FP8 KV cache). Present an evidence table, then call the gated tool.
8. **Verify after approval.** Watch production recover. If it doesn't, propose a rollback.

### 1.4 AWS architecture

```
  Team laptop ──(AWS SSM port-forward, no open ports)──┐
                                                       ▼
┌──────────────────────── AWS VPC (organizer account) ───────────────────────────────────┐
│                                                                                        │
│  CONTROL PLANE  (CPU EC2, e.g. m6i.xlarge; IAM instance role, least privilege)          │
│  ┌──────────────────────────────────────────────────────────────────────────────┐      │
│  │ TrueForge (Docker Compose hosted mode: server+UI, Postgres, Redis) :8791       │      │
│  │   Agent "inference-firefighter" — OpenAI model, runbook skill, sandbox,        │      │
│  │   subagents, Generative UI, APPROVAL GATE on prod tools                         │      │
│  │ inference-ops MCP server :8000/mcp   (our code; the ONLY path to prod)          │      │
│  │ Prometheus + Grafana · nginx gateway (router + rate limits) · loadgen (users)   │      │
│  └────────────┬──────────────────────────────────────┬───────────────────────────┘      │
│               │ private VPC traffic only             │                                  │
│  ┌────────────▼───────────────────┐   ┌──────────────▼─────────────────────────┐        │
│  │ PROD GPU fleet                  │   │ SHADOW GPU (experiments only)           │        │
│  │ replica slots r1 (active),      │   │ slots s1, s2 (parallel experiments)     │        │
│  │ r2 (spare, for scale-out)       │   │ replay runner · load sweeps · quality   │        │
│  │ vLLM + DCGM exporter + deploy-  │   │ eval · deploy-controller               │        │
│  │ controller                      │   │                                         │        │
│  └─────────────────────────────────┘   └─────────────────────────────────────────┘        │
└────────────────────────────────────────────────────────────────────────────────────────┘
        ▲                                                   ▲
        │ GitHub: inference-prod-config (source of truth)   │ Daytona sandbox (agent code;
        │ deployment.yaml — every prod change is a commit   │ MCP calls bridged via harness)
```

**GPU layout — pick based on what the organizers give us:**

| Option | Instances | Pros | Cons |
|---|---|---|---|
| **A (recommended)** | 1 × `g6.12xlarge` (4 × NVIDIA L4, 24 GB each): GPU0 = prod r1, GPU1 = prod r2 (spare), GPU2–3 = shadow s1/s2 | One instance, fast scale-out (start a container on a free GPU in about 1 min), simple networking | Needs 48 vCPU of G-instance quota; scale-out is "within a host" |
| **B (more cloud-native)** | Prod = Auto Scaling Group of `g6.xlarge` with a **warm pool** (pre-baked AMI with image + weights), shadow = 1–2 × `g6.xlarge` | Scale-out is a real `SetDesiredCapacity` call | Slower (2–5 min per new replica even from a warm pool); more moving parts |
| **C (minimum)** | 2 × `g6.xlarge` (prod, shadow) | Smallest quota | No spare GPU, so S2 falls back to config tuning + admission control (no scale-out) |

The deploy-controller abstracts over "slots" (host, GPU index, port), so the MCP server and the agent don't care which option we use. **Start with A or C. Upgrade to B only if time allows.**

GPU note: **L4 (Ada, `g6`)** supports FP8 KV cache. If only `g5` (A10G, Ampere) is available, don't rely on FP8 KV cache. Use the non-FP8 fixes in §Phase 4.

**Why TrueForge runs in hosted mode on AWS:** the docs say local `npx` mode is for your own machine and shouldn't be exposed. On EC2 we run the **Docker Compose hosted mode** (server + Postgres + Redis) bound to localhost. The team reaches it through **SSM port-forwarding**, so no public URL is needed and neither is OIDC.
```bash
aws ssm start-session --target <control-instance-id> \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["8791"],"localPortNumber":["8791"]}'
# then open http://localhost:8791 (Grafana the same way on :3000)
```

---

## 2. Tech decisions (locked)

| Decision | Choice |
|---|---|
| Harness | TrueForge, Docker Compose hosted mode on the control-plane EC2 (`PUBLIC_BASE_URL=http://localhost:8791`) |
| Agent LLM | **OpenAI** (organizer key) under Settings → Models. Use the most capable tool-calling GPT model offered (`openai/<model-id>`), `reasoning_effort: medium`, `temperature: 0.2` where supported. |
| Sandbox | **Daytona**, the only TrueForge sandbox provider. **Not provided by organizers**, so we need our own account/key with *Sandboxes* + *Snapshots write* permission. |
| Tools | Our Python MCP server (`mcp` SDK / FastMCP, streamable HTTP) on the control plane |
| Serving | vLLM (`vllm/vllm-openai` Docker image), **Qwen2.5-3B-Instruct** (fallback 1.5B) |
| Gateway | **nginx**: `least_conn` upstream across active prod replicas, `proxy_buffering off` for streaming, `limit_req` zones for admission control. Config is rendered by the controller from `deployment.yaml`. |
| Metrics | Prometheus (5 s scrape) ← vLLM `/metrics`, **NVIDIA DCGM exporter** (GPU util/mem), loadgen, nginx exporter. Grafana dashboards. |
| AMI | AWS **Deep Learning Base OSS Nvidia Driver GPU AMI** (Ubuntu), which ships with drivers, Docker and the NVIDIA container toolkit |
| Access | **SSM Session Manager** (no SSH ports open); security groups allow only intra-VPC traffic |
| AWS auth for our code | **IAM instance role** on the control plane (no access keys). Least privilege: `ec2:Describe*`, plus for option B `autoscaling:SetDesiredCapacity`/`Describe*` on **our ASG only**, plus `ssm:*Session*` for ops |
| Prod source of truth | GitHub repo `inference-prod-config` → `deployment.yaml` (vLLM flags + replica count + admission policy). Every production action = a commit + a controller reconcile. |
| Analysis | TrueForge **Code Mode** in Daytona (`from mcp_client import call_tool`) |

---

## 3. Team and ownership

| Role | Owns |
|---|---|
| **P1 — AWS & Infra** | VPC/SG/IAM, EC2 instances, vLLM, DCGM, nginx gateway, Prometheus/Grafana, deploy-controller, scale-out path, teardown |
| **P2 — Tools** | `inference-ops` MCP server: all tools, guardrails, evidence registry, GitHub + AWS integration |
| **P3 — Agent** | TrueForge (hosted) setup, OpenAI + Daytona config, agent spec, instructions, runbook skill, Generative UI, behavior tuning, scenario scorecard |
| **P4 — Workload & Story** | Datasets, loadgen with **scenario engine**, request log, replay/sweep runner, quality eval, SLO policy, demo video, README |

With 3 people, P4's work is split between P1 and P3. With 2, it's Infra + Workload and Tools + Agent + Story.

---

## Phase −1 — Before the event (accounts, research, reading only; no project code)

**Ask the organizers (as early as possible):**
- [ ] Which **AWS region**, and are **GPU instances** allowed? Which types (`g6.*` preferred, `g5.*` acceptable)?
- [ ] What's the **vCPU quota for "Running On-Demand G and VT instances"**? Option A needs 48 vCPU, option C needs 8.
- [ ] Do we get console access, IAM user/role creation, and SSM?
- [ ] Which **OpenAI models** and what rate limits (tokens/min) are on the provided key?
- [ ] Is there a budget cap on the credits?

**Our own accounts:**
- [ ] **Daytona** account + API key (Sandboxes + Snapshots write)
- [ ] **GitHub** (the config repo is created on the day; fine-grained PAT scoped to that one repo)
- [ ] **Hugging Face** token
- [ ] Local tools: AWS CLI v2 + Session Manager plugin, Node ≥ 22.14, Python 3.11+, Docker, `gh`, `gitleaks`

**Reading:**
- [ ] TrueForge: Quickstart (hosted Docker Compose tab), Create an Agent (full spec), Sandbox, MCP Servers, Code Mode, Subagents, Use an agent (approvals, `ask_user_question`), Sessions
- [ ] vLLM: engine args (`--gpu-memory-utilization`, `--max-num-seqs`, `--max-num-batched-tokens`, `--max-model-len`, `--kv-cache-dtype`, `--enable-prefix-caching`, chunked prefill), metrics docs
- [ ] AWS: SSM port forwarding, DLAMI, (option B) ASG warm pools
- [ ] Agree on roles; everyone reads this roadmap

---

## Phase 1 — Kickoff and de-risking spikes (H0:00 → H1:00)

**1.1 Repo scaffolding (P2)**
```
inference-firefighter/
├── README.md                      # incl. AI-assistance disclosure
├── .env.example                   # names only, never values
├── infra/
│   ├── aws/                       # launch scripts / CLI commands, IAM policy JSON, teardown.sh
│   ├── control/docker-compose.yml # prometheus, grafana, nginx, loadgen, mcp server
│   ├── gpu/docker-compose.yml     # vllm slots + dcgm-exporter + deploy-controller
│   ├── prometheus/  grafana/  nginx/
├── controller/                    # FastAPI deploy-controller (slot abstraction)
├── workload/
│   ├── build_dataset.py
│   ├── loadgen.py                 # scenario engine + request log + metrics
│   ├── scenarios/*.yaml           # baseline, surge, long_prompt, long_output, burst, thrash, mixed
│   ├── replay.py  sweep.py
│   └── golden_set.jsonl
├── mcp_server/  server.py  guardrails.py  evidence.py  policy.yaml
├── agent/  spec.json  create_agent.py  instructions.md
├── skills/incident-runbook/SKILL.md
└── chaos/  run_scenario.sh  reset.sh
```
Also create the private repo **`inference-prod-config`** with the healthy `deployment.yaml` (§Phase 2.5).

**1.2 Spike A — AWS GPU + vLLM (P1)** ⚠️ the biggest risk, start at minute 0
- [ ] Launch the GPU instance(s) (DLAMI, 200 GB gp3) and the control-plane instance, with SSM role attached. Check `nvidia-smi`.
- [ ] Run vLLM on one GPU:
  ```bash
  docker run -d --gpus '"device=0"' -p 8100:8000 --ipc=host \
    -v /opt/hf:/root/.cache/huggingface vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-3B-Instruct --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 --max-model-len 16384
  ```
- [ ] From the control plane: `curl /v1/chat/completions` and `curl /metrics` work. **Write down the exact vLLM metric names** for this version (e.g. `vllm:kv_cache_usage_perc` vs `vllm:gpu_cache_usage_perc`) and share them with P2.
- [ ] Time a vLLM cold restart (sets the experiment budget).
- **Fallback:** quota denied → escalate to organizers right away and meanwhile use option C, or whatever single GPU is available.

**1.3 Spike B — TrueForge hosted on AWS + OpenAI + Daytona (P3)**
- [ ] On the control plane: `git clone https://github.com/truefoundry/trueforge && cd trueforge && cp packages/trueforge/.env.example packages/trueforge/.env && docker compose up -d --build`. Make sure ports are bound to localhost and not exposed in the security group.
- [ ] SSM port-forward 8791 → open the UI. Settings → Models → **OpenAI** (organizer key). Settings → Sandbox providers → **Daytona**.
- [ ] Test: an agent with the sandbox enabled runs `python -c "import numpy; print(numpy.__version__)"`.

**1.4 Spike C — MCP + approval + Code Mode bridge (P2 + P3) — the critical integration**
- [ ] 20-line FastMCP server on the control plane with `ping()` (read-only) and `dangerous_action()` (destructive).
- [ ] Settings → Connectors → **Add MCP Server** → `http://<mcp-host>:8000/mcp`. Because TrueForge runs in Docker, use the compose network hostname or the instance's private IP, **not** `localhost`.
- [ ] Gate `dangerous_action` → the chat shows **Allow/Deny**. ✅
- [ ] Code Mode: "write a Python script that calls `ping` 5× through `call_tool` and prints the results". ✅ when it works from the Daytona sandbox through the harness bridge.
- **Fallback:** if the bridge fails, the agent calls MCP tools directly. Large results get offloaded to sandbox files automatically, and the agent's analysis code reads those files. We still meet "generated code runs in the sandbox".

**1.5 Spike D — Datasets (P4)**
- [ ] ~2k short chat prompts; ~500 long RAG prompts (4k–10k tokens, public-domain text + question); a "long answer" prompt set (asks for essays/code, `max_tokens` 1024–2048). **No real user data.**

**Checkpoint H1:00:** GPU + vLLM live, TrueForge + OpenAI + Daytona live, approval + Code Mode proven. Freeze the architecture.

---

## Phase 2 — Build the production world on AWS (H1:00 → H3:30) — P1 + P4

### 2.1 GPU host (`infra/gpu/docker-compose.yml`)
- [ ] vLLM **slot containers** (prod r1, prod r2 spare, shadow s1, s2), each pinned to one GPU with its own port. The controller renders their flags.
- [ ] **DCGM exporter** (GPU utilization, memory, SM activity per GPU).
- [ ] Model weights cached on local disk (`/opt/hf`), images pre-pulled.

### 2.2 Deploy-controller (`controller/`, on the GPU host; bearer token; VPC-only)
- `GET /state`: slots, their configs, health, uptime, GPU assignment
- `POST /reconcile {deployment}`: brings **prod** to the desired `deployment.yaml`:
  - vLLM flags changed → rolling restart. With the spare slot we can do *start new → health-check → swap in nginx → stop old*, i.e. near-zero downtime.
  - `replicas` changed → start/stop prod slots → re-render the nginx upstream → `nginx -s reload`
  - `admission` changed → re-render the nginx `limit_req` → reload
  - Auto-rollback if a new replica isn't healthy within N s
  - Posts a **Grafana annotation** for every action ("deploy sha abc123", "scaled 1→2")
- `POST /shadow/{slot}/deploy {config}`: shadow only
- `GET /logs?slot=&since=&grep=`
- **Option B:** replica changes call `autoscaling:SetDesiredCapacity` on the prod ASG, and new instances register with nginx on boot.

### 2.3 Gateway + loadgen (control plane) — P4
- [ ] **nginx** gateway in front of prod replicas (`least_conn`, streaming-safe), nginx-prometheus-exporter.
- [ ] **loadgen.py = the "users"**. It hits the gateway, not vLLM directly.
  - **Scenario engine:** a YAML timeline of phases, e.g.:
    ```yaml
    # scenarios/surge.yaml — "users flood in after a launch"
    phases:
      - {duration: 300, rps: 4,  mix: {short: 0.9, long: 0.1}, output_tokens: [64, 256]}
      - {duration: 120, rps_ramp_to: 12, mix: {short: 0.9, long: 0.1}}
      - {duration: 900, rps: 12, mix: {short: 0.9, long: 0.1}}
    ```
    Supports `rps`, `rps_ramp_to`, `mix` (short/long prompts), `output_tokens` range, `burst` (spike for N s), and Poisson arrivals.
  - Measures per request on the client side: TTFT, end-to-end latency, output tokens, tokens/s, status. Exposes Prometheus histograms labeled by `bucket=short|long`, plus a **goodput** counter (met SLO yes/no).
  - Appends to the **request log** (SQLite): `ts, request_id, prompt_ref, prompt_tokens, max_tokens, output_tokens, bucket, ttft_ms, e2e_ms, status`. This is our "gateway log", which the agent samples for replay and before/after comparison.
  - Control API: `POST /scenario {name}` for chaos scripts, `GET /state`.

### 2.4 Replay, sweep and quality eval runners (GPU host or control plane) — P4
- [ ] `replay.py`: replays a captured workload against a shadow slot, **preserving relative arrival times** with an optional `rate_multiplier` (1.0 = as captured, 2.0 = twice the users). Duration ≤ 60–90 s. Returns per-request records + summary.
- [ ] `sweep.py`: runs replay at multipliers `[0.5, 1, 1.5, 2, 2.5, 3]` for 30–45 s each and returns per-level summaries. The agent finds the knee in code.
- [ ] Quality eval: ~100 golden prompts (incl. long-context) with **baseline outputs** recorded from the healthy config at temperature 0. It returns candidate + baseline outputs, and the agent scores them in the sandbox.

### 2.5 Config repo `inference-prod-config` (healthy baseline — calibrate in Phase 4)
```yaml
# deployment.yaml
model: Qwen/Qwen2.5-3B-Instruct
replicas: 1
vllm:
  gpu_memory_utilization: 0.90
  max_num_seqs: 64
  max_num_batched_tokens: 8192
  max_model_len: 16384
  enable_prefix_caching: true
  kv_cache_dtype: auto
admission:
  enabled: false          # rate_limit_rps / burst when enabled
```
```yaml
# constraints.yaml  (read by the agent via get_policy)
max_gpu_memory_utilization: 0.60   # reranker sidecar co-located (added with change #42)
max_replicas: 2
max_hourly_gpu_spend_usd: 3.0      # budget guard for scale-out
```

### 2.6 Grafana dashboard "Inference — prod"
Row 1, users: offered RPS · goodput · p95 e2e · p95 TTFT · error rate
Row 2, throughput: served req/s · output tokens/s · per-user tokens/s (p50)
Row 3, saturation: waiting vs running · KV usage % · preemptions/s · GPU util/mem (DCGM)
Row 4, traffic shape: long/short share · prompt-token p95 · output-token p95
Annotations: deploys, scaling, admission changes, scenario phase changes

**Checkpoint H3:30:** live traffic through nginx → vLLM, Grafana populated, a reconcile (config change + scale 1→2) works, and a shadow replay + sweep return data.

---

## Phase 3 — The `inference-ops` MCP server (H1:00 → H4:30) — P2

Build against mocks first, then switch to real endpoints around H3:00.

### 3.1 Conventions
```python
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP("inference-ops", host="0.0.0.0", port=8000)
RO   = ToolAnnotations(readOnlyHint=True)
EXP  = ToolAnnotations(readOnlyHint=False, destructiveHint=False)  # shadow only
PROD = ToolAnnotations(readOnlyHint=False, destructiveHint=True)   # gated
...
mcp.run(transport="streamable-http")   # /mcp
```
- Descriptions explain *when* to use the tool and *what it returns*.
- Direct results are compact summaries. Bulk per-request data goes in large results that TrueForge offloads to the sandbox for analysis in code.
- Every result carries an `evidence_id`, and every experiment is stored in the **evidence registry** (`evidence.py`, SQLite).

### 3.2 Observe tools (read-only → no approval)
| Tool | Returns |
|---|---|
| `get_slo_snapshot(window="10m")` | All §1.2 metric groups for prod + which SLOs are breached + since when |
| `compare_windows(baseline_window, incident_window)` | Per-metric deltas (%) across load, throughput, goodput, latency, saturation and traffic shape. This is the agent's first classification input. |
| `query_metrics(promql, start, end, step)` | Raw Prometheus ranges for custom analysis in code |
| `get_traffic_profile(window)` | RPS over time, prompt/output token percentiles, long/short share, burstiness (from the request log) |
| `get_change_events(since)` | A merged timeline of config commits (GitHub diffs), deploys, scaling events, admission changes |
| `get_logs(slot, since, grep, limit)` | vLLM logs (preemptions, OOM, restarts) |
| `get_deployment_state()` | Current `deployment.yaml`, live replicas, slot health, spare capacity |
| `get_policy()` | SLO thresholds + constraints + cost rates ($/GPU-hour) |
| `capture_workload(window, n)` | Samples request records from a window → `workload_id` (e.g. W₀ = before, W₁ = now) |

### 3.3 Experiment tools (shadow only → no approval, guarded)
| Tool | Behavior |
|---|---|
| `deploy_shadow(slot, vllm_config)` | Allowlisted knobs within safe ranges, **shadow slots only**. Returns `config_hash`. |
| `run_replay(slot, workload_id, rate_multiplier=1.0, duration_s=60)` | Summary + per-request records. Stores `(config_hash, workload_id, multiplier, results, slo_pass)`. |
| `run_capacity_sweep(slot, workload_id, multipliers)` | Per-level summaries → the agent computes the knee / capacity per replica in code |
| `run_quality_eval(slot, n)` | Candidate vs baseline outputs for scoring in the sandbox |

Allowlisted vLLM knobs: `gpu_memory_utilization` 0.30–0.95 · `max_num_seqs` 8–512 · `max_num_batched_tokens` 512–32768 · `max_model_len` 2048–32768 · `enable_prefix_caching` · `kv_cache_dtype` auto|fp8. The model is not changeable.

### 3.4 Production tools (GATED → human approval)
| Tool | Server-side preconditions (checked even after approval) | Effect |
|---|---|---|
| `apply_production_config(vllm_config, justification, evidence_ids)` | Config hash validated in shadow on the **current** traffic workload, all SLOs passing; respects constraints | Commit `deployment.yaml` → reconcile (rolling) → health-check → auto-rollback |
| `scale_replicas(target, justification, evidence_ids)` | A capacity sweep in the evidence registry supports `target`; `target ≤ max_replicas`; projected $/hr ≤ budget | Commit → reconcile (start/stop replica, update nginx) → returns the new $/hr |
| `set_admission_policy(rate_limit_rps, burst, justification)` | Must be ≥ the measured capacity floor (can't set it absurdly low) | Commit → nginx `limit_req` reload |
| `rollback_production(to_sha)` | `to_sha` exists in repo history | Redeploy that commit |
| `publish_incident_report(markdown)` *(stretch)* | none | GitHub issue in the config repo |

The approval card the human sees shows the tool arguments: diff, justification, evidence IDs, and for scaling the **$/hr impact**.

### 3.5 Tests (P2)
- [ ] Script that calls every tool directly (no LLM) against real endpoints.
- [ ] Negative tests: out-of-range knob → refused. Prod config without evidence → refused. Scale beyond `max_replicas` or budget → refused. Shadow tool targeting prod → refused.

**Checkpoint H4:30:** MCP registered in TrueForge; all tools work from chat.

---

## Phase 4 — Scenarios: inject, calibrate, verify the "right answer" by hand (H3:30 → H5:30) — P1 + P4

For every scenario, we (humans, no agent) confirm it: (1) produces a clear, real degradation, (2) has the expected metric signature from §1.1, (3) has a remediation that actually works.

### 4.1 `chaos/run_scenario.sh <name>`
| Scenario | Injection |
|---|---|
| `s1_config` | Commit bad config (`gpu_memory_utilization 0.90→0.45`, `max_num_seqs 64→256`, msg: *"Reserve GPU memory for reranker; raise max_num_seqs for throughput"*) + add memory constraint → reconcile |
| `s2_surge` | loadgen `surge.yaml` (RPS 3× over 2 min, same mix) |
| `s3_long_prompt` | loadgen mix short/long 0.9/0.1 → 0.6/0.4 at the same RPS |
| `s4_long_output` | loadgen `output_tokens` 64–256 → 1024–2048 |
| `s5_burst` | loadgen 45 s burst at 4× then back to normal |
| `s6_thrash` | `max_num_seqs → 256` + surge + moderate memory |
| `s7_mixed` (flagship) | `s1_config` + `s3_long_prompt` within the same 2 minutes |
| `s8_prefix` (stretch) | Commit `enable_prefix_caching: false` + traffic sharing a long system prompt |

`chaos/reset.sh` restores the healthy commit, replicas=1, admission off, baseline scenario, and clears the evidence registry. **It must bring the system back to healthy in < 3 min.**

### 4.2 Calibration targets (measure on the real GPUs)
- [ ] Healthy baseline: p95 e2e, TTFT, goodput ≈ offered load, queue ≈ 0. **Set the baseline RPS at ~50–60% of one replica's capacity** (find it with a sweep).
- [ ] Each scenario: ≥ **3× p95 degradation** (S5 just a short spike), visible in Grafana within 2–3 min.
- [ ] **S2 surge sizing:** pick the surge so that **one replica can't meet SLO under any config, but two can.** That makes scale-out the provably correct answer. Verify with sweeps.
- [ ] **S1/S7 fix:** verify that a config inside the 0.60 memory budget passes all SLOs. Preferred: `kv_cache_dtype: fp8` + `gpu_memory_utilization 0.55` + `max_num_seqs 64` + tuned `max_num_batched_tokens`. Verify FP8 works on our GPU/vLLM version. If it doesn't, calibrate a non-FP8 winner.
- [ ] Verify the **wrong fixes fail**: e.g. revert memory (violates constraint), `max_model_len 4096` (long requests error), scale-out for S1 (costs money, and `compare_windows` shows no load change, so it's unjustified).
- [ ] **S6:** verify that lowering `max_num_seqs` *raises* served throughput.
- [ ] Write the real SLO thresholds into `policy.yaml` **from these measurements**. Record all numbers in `docs/calibration.md` for the README and Q&A.

Example `policy.yaml` (replace with calibrated values):
```yaml
slos:
  p95_e2e_latency_ms:   {max: 1500}
  p95_ttft_ms:          {max: 700}
  error_rate:           {max: 0.005}
  goodput_ratio:        {min: 0.95}     # goodput / offered load
  per_user_tokens_per_s_p50: {min: 25}
  quality_agreement:    {min: 0.97}
cost:
  gpu_hour_usd: 0.80                    # verify against the instance price in our region
```

**Checkpoint H5:30:** S1, S2, S5 and S7 are calibrated and manually proven (the others are nice-to-have).

---

## Phase 5 — The agent in TrueForge (H4:00 → H7:30) — P3 (+P2)

### 5.1 Runbook skill (`skills/incident-runbook/SKILL.md`, public GitHub repo → Settings → Skills → Import)
It encodes §1.3 as a procedure:
1. **Triage:** `get_slo_snapshot`, `get_policy`. Which SLOs are breached, since when, and how badly?
2. **Before vs now** (Code Mode): `compare_windows` + `get_traffic_profile` + `get_change_events`. Compute deltas in code and plot the incident timeline with change events overlaid.
3. **Classify** using the signature table (config / surge / prompt-shift / output-shift / transient / thrash / mixed). Give confidence + evidence for and against. **If transient** (queue drained, no change event, spike < 2 min): recommend no production change, explain why, stop.
4. **Reproduce:** `capture_workload` (current window) → replay against the current prod config in shadow. It must reproduce. If it doesn't, go back to step 3.
5. **Attribute** (mixed/unclear): 2×2 counterfactual replay (W₀/W₁ × C₀/C₁) → percentage of the regression due to config vs traffic.
6. **Remediate by class:**
   - Config → experiment with configs **within constraints** (one variable, then combine) → validate
   - Load → capacity sweep per candidate config → can config alone reach the required capacity? If not → `replicas_needed` → cost → `scale_replicas`
   - Shape shift → prefill-side (S3) or decode-side (S4) knobs; product trade-offs (e.g. capping `max_tokens`) → **ask the human** with `ask_user_question`
   - Thrash → try **lower** `max_num_seqs` first
   - Budget blocks scaling → propose `set_admission_policy` as a stopgap and explain who gets throttled
7. **Budget:** ≤ 6 shadow experiments, 60 s replays; use **parallel subagents** (one per shadow slot) for independent experiments. Subagents return only summaries.
8. **Evidence card** (Generative UI): candidates × SLOs pass/fail table, p95 + goodput bar chart, capacity curve (for load cases), exact diff / scaling plan with $/hr.
9. **Gated action** with justification + evidence IDs. On **Deny**, read the reason, adapt, re-validate, and ask again.
10. **Verify** prod recovery for 2–3 min with `get_slo_snapshot`; propose rollback if it hasn't recovered.
11. **Report:** timeline, classification, root cause, attribution, fix, evidence, cost impact, follow-ups.

### 5.2 Instructions (`agent/instructions.md`) — short; the procedure lives in the skill
```
You are Inference Firefighter, the on-call inference reliability engineer for a vLLM
fleet on AWS. You restore latency/throughput SLOs.

- First decide WHY: config change, more users, different traffic shape, a transient
  burst, overload thrashing, or a mix. Different causes need different fixes; never
  scale out to hide a config bug, and never tune config to absorb a load the hardware
  cannot serve.
- Production is read-only to you except through the gated tools, which a human approves.
- Experiment only in shadow. Reproduce before fixing. Compute every number in code in
  the sandbox; never estimate from raw JSON.
- Respect get_policy constraints and budget. Prefer the cheapest fix that passes all SLOs.
- If no change is needed, say so and stop.
- Ask the human when a fix is a business trade-off.
- Follow the incident-runbook skill. Present evidence with Generative UI.
```

### 5.3 Agent spec (`agent/spec.json`, created with `trueforge-sdk` in `create_agent.py`)
```json
{
  "model": { "name": "openai/<model-id>", "params": { "reasoning_effort": "medium" } },
  "instructions": "<instructions.md>",
  "mcp_servers": [{
    "name": "inference-ops",
    "enable_tools": ["@all"],
    "require_approval_for_tools": [
      "apply_production_config", "scale_replicas", "set_admission_policy",
      "rollback_production", "publish_incident_report"
    ],
    "preload": true
  }],
  "skills": [{ "name": "incident-runbook" }],
  "config": {
    "sandbox": { "enabled": true },
    "generative_ui": { "enabled": true },
    "ask_user_questions": { "enabled": true },
    "dynamic_sub_agents": { "enabled": true },
    "iteration_limit": 200
  }
}
```
- Gated tools are **named explicitly** as well as annotated `destructiveHint`, because tools without annotations don't match `@destructive`.
- `create_agent.py`: `TrueForge(base_url="http://localhost:8791", timeout=600)` → `client.agents.create(name="inference-firefighter", manifest=spec)` (or `agents.update`). The agent is versioned in git.
- OpenAI rate limits: parallel subagents multiply token usage. If you hit TPM limits, cap parallelism in the runbook ("at most 2 subagents").

### 5.4 Scenario scorecard: iterate until it generalizes
Run each scenario, `reset.sh` in between, and review every run in **Sessions** (timeline, tool calls, tokens). Fix failures in the **runbook, instructions or tool descriptions**, never by hard-coding scenario answers.

| Scenario | Correct class? | Correct fix? | Approval shown? | Recovered? | Time | Tokens |
|---|---|---|---|---|---|---|
| S1 config | | | | | | |
| S2 surge | | | | | | |
| S5 burst (expect *no action*) | | | n/a | n/a | | |
| S7 mixed | | | | | | |
| S3 / S4 / S6 | | | | | | |

This table goes into the README as the **agent scorecard**. It's strong evidence that the agent isn't scripted.

Common problems → fixes:
| Symptom | Fix |
|---|---|
| Scales out for a config bug | Runbook: "classify before remediating"; `scale_replicas` requires sweep evidence showing 1 replica can't meet SLO |
| Tunes config endlessly during a surge | Runbook: capacity sweep early for load-class incidents; experiment budget |
| Acts on a transient burst | Transient rule in step 3; `compare_windows` exposes "queue drained" + "no change events" |
| Eyeballs numbers | Strengthen "compute in code"; bulk data only in offloaded results |
| Reverts a constraint-violating change | `get_policy` surfaces constraints prominently; the server rejects them anyway |
| Too slow | Parallel shadow slots, 45–60 s replays, fewer sweep levels |

**Checkpoint H7:30:** S1, S2, S5 and S7 are each solved correctly in ≥ 2 of 3 runs; the approval pause always appears; the fixes work.

---

## Phase 6 — Polish that impresses (H7:30 → H8:45) — pick in order, stop at the freeze

1. **Deny → adapt → approve.** Deny the first proposal with a reason ("keep max_num_seqs ≥ 96 for evening peak" or "budget: no second replica, find another way"). The agent re-validates and comes back with a different proposal.
2. **Live recovery shot:** Grafana next to the TrueForge chat. The deploy/scale annotation appears, p95 drops and goodput recovers.
3. **Cost-aware approval card:** "+1 replica = +$0.80/hr; restores goodput from 61% → 99%".
4. **Incident report** posted as a GitHub issue (gated).
5. **Alert-triggered start:** a Prometheus alert → webhook → script opens a TrueForge session via SDK with the alert payload. Verify the session shows in the UI before relying on it.
6. **Stretch scenarios** S3/S4/S6/S8/S9 added to the scorecard.

---

## Phase 7 — Freeze, harden, rehearse (H8:45 → H9:20)

- [ ] **Feature freeze at H8:45.**
- [ ] 2 clean end-to-end rehearsals of the demo sequence (§8.1).
- [ ] **Record a backup video** of the best clean run as soon as you have one.
- [ ] Keep all EC2 instances running; images pulled; weights cached.
- [ ] Phone hotspot ready. The SSM session needs internet; everything else runs in AWS.
- [ ] Screen hygiene: no AWS console pages showing keys, no TrueForge Settings with keys, no `.env`, no terminal history with tokens.

---

## Phase 8 — Demo video, README, submission (H9:20 → H10:00)

### 8.1 Demo video (~3 min): flagship mixed incident + a quick surge
| Time | Screen | Voice-over |
|---|---|---|
| 0:00–0:15 | Grafana: p95 4×, goodput collapsing, alert | "Production inference on AWS just degraded. Was it a config change? More users? Different traffic? Usually it's a mix, and the fix depends on which." |
| 0:15–0:45 | TrueForge: `compare_windows`, change events, Code Mode script in Daytona, classification | "The agent compares before and now across load, throughput and saturation, using code it writes and runs in a sandbox. It finds two things: a config commit, and a shift to long-context traffic." |
| 0:45–1:20 | 2×2 counterfactual replay in shadow GPUs (parallel subagents), attribution chart | "To split the blame, it replays old and new traffic against old and new config on a shadow GPU. Result: 70% config, 30% traffic." |
| 1:20–1:50 | Candidates table: revert rejected (memory budget), fast config rejected (errors), winner passes | "It won't just revert, because that would starve the reranker. It finds a config that fixes both causes, inside the budget." |
| 1:50–2:15 | Approval card → Deny with reason → re-validate → Allow; Grafana recovers | "It only touches production with my approval, and when I push back, it adapts." |
| 2:15–2:45 | Montage (⏩): user surge scenario → capacity sweep → "config can't fix this" → `scale_replicas` card with $/hr → Allow → goodput recovers. Burst scenario → "no action needed". | "Same agent, different problem: a user surge. It proves one GPU can't serve this load under any config and asks to add a replica, with the cost. And for a harmless spike, it knows to do nothing." |
| 2:45–3:00 | Scorecard table + title | "Agents that act: diagnose, experiment safely, prove it, then ask." |

Mark sped-up segments on screen (⏩ 4×). Show no secrets.

### 8.2 README checklist
- [ ] Problem → solution → architecture diagram (AWS)
- [ ] Hackathon requirement mapping (§0)
- [ ] Incident classes + metric signatures (§1.1–1.2) + method (§1.3)
- [ ] Tool list labelled read / shadow / **gated**
- [ ] **Agent scorecard** + calibration numbers
- [ ] Setup: AWS launch + IAM, TrueForge compose, OpenAI/Daytona config, `create_agent.py`, `run_scenario.sh`, `reset.sh`, `teardown.sh`
- [ ] **AI assistance disclosure** (required, and specific)
- [ ] Limitations & future work (multi-node ASG autoscaling, speculative decoding, disaggregated prefill/decode, canary rollouts, predictive scaling)
- [ ] `gitleaks detect` clean

### 8.3 Teach-back + Q&A prep (15 min before judging)
- *"Is it fake?"* Real vLLM on real AWS GPUs. The degradation comes from real KV-cache preemption and real queueing. Every number is from Prometheus/DCGM or our replays.
- *"Is the agent scripted?"* Scorecard across 4+ scenario types, with different correct fixes, including "do nothing". Sessions shows every decision.
- *"What stops it breaking prod or burning money?"* Approval gate; server-side evidence, constraint and budget checks; auto-rollback; `max_replicas`; an IAM role scoped to our resources only.
- *"Why the sandbox?"* The agent writes and runs analysis code in isolated Daytona. It has no credentials and no route to prod, and its MCP calls are bridged through the harness.
- *"How do you tell more users from a bad config?"* `compare_windows` (offered load vs shape vs change events) + counterfactual replay + capacity sweep.

### 8.4 After the event
- [ ] `infra/aws/teardown.sh`: terminate GPU instances. Credits are finite.

---

## 9. Master timeline (assumes a ~10-hour build window — rescale if different)

| Time | P1 AWS & Infra | P2 Tools | P3 Agent | P4 Workload & Story |
|---|---|---|---|---|
| H0:00–1:00 | Spike A: EC2 GPU + vLLM | Repo + Spike C (MCP/approval/Code Mode) | Spike B: TrueForge on EC2, OpenAI, Daytona | Spike D: datasets |
| H1:00–2:30 | GPU compose, DCGM, controller (reconcile) | Observe tools (mocked) | Runbook skill + instructions draft | Loadgen scenario engine + request log |
| H2:30–3:30 | nginx gateway, Prometheus, Grafana, scale path | Experiment tools + guardrails + evidence registry | Agent spec + `create_agent.py` | Replay + sweep + golden set |
| **H3:30** | **Checkpoint: the world is live** | switch to real endpoints | | |
| H3:30–5:30 | Scenario injection + calibration | Prod tools + preconditions + tests | First end-to-end runs (S1) | Calibration + `policy.yaml` |
| **H5:30** | **Checkpoint: S1/S2/S5/S7 manually proven** | | | |
| H5:30–7:30 | Stability, annotations, reset speed | Tool description tuning | Scorecard iteration (S1, S2, S5, S7) | Demo script, README draft |
| **H7:30** | **Checkpoint: scorecard ≥ 2/3 on core scenarios** | | | |
| H7:30–8:45 | Stretch scenarios | Incident report tool | Deny→adapt beat, evidence card polish | Backup video |
| **H8:45** | **FEATURE FREEZE** | | | |
| H8:45–9:20 | Rehearsals | Rehearsals | Rehearsals | Final recording |
| H9:20–10:00 | Teach-back, teardown plan | README tools | Teach-back | Submit |

---

## 10. Risk register

| Risk | Impact | Mitigation / fallback |
|---|---|---|
| No / low GPU quota in the organizer AWS account | Fatal | Ask before the event; start Spike A at minute 0; fall back to option C (2 × g6.xlarge), or a single GPU hosting prod + shadow at ~0.45 memory each |
| Only `g5` (Ampere) available | Med | Skip the FP8 KV fix; calibrate a non-FP8 winner |
| Code Mode bridge issue with the MCP server | High | Spike C; fallback: direct tool calls + automatic offload to sandbox files + agent analysis code |
| TrueForge-in-Docker can't reach the MCP server | Med | Use the compose service name / private IP, not `localhost`; or run the MCP server inside the same compose network |
| Scale-out too slow for the demo | Med | Option A (spare GPU slot, ~1 min) instead of new EC2 boot; label sped-up footage |
| Scenario not dramatic / wrong fix wins | High | Phase 4 calibration: every scenario manually proven before agent work |
| Agent picks the wrong class | High | Signature table + `compare_windows` + counterfactual replay; scorecard iteration |
| OpenAI rate limits (TPM) | Med | Cap subagent parallelism; compact tool outputs; offloading |
| Credits run out / runaway cost | Med | `max_replicas` + budget guard; AWS Budgets alert; teardown script |
| Secrets exposure (organizer keys) | High (rules) | Instance role instead of keys; keys only in TrueForge settings; gitleaks; screen hygiene |
| Venue Wi-Fi | High | Hotspot; everything else runs in AWS; backup video |

---

## 11. Definition of done

- [ ] Real vLLM on AWS GPUs serving scenario-driven traffic through an nginx gateway, with Prometheus/DCGM/Grafana.
- [ ] One command injects each core scenario (S1 config, S2 surge, S5 burst, S7 mixed); one command resets.
- [ ] The agent on TrueForge (OpenAI model) **classifies** each correctly and applies the **matching** fix: config tune, scale-out, or no action.
- [ ] The agent's analysis code runs in the **Daytona sandbox**.
- [ ] Reproduce-in-shadow before fixing; counterfactual attribution for the mixed case; capacity sweep for the surge.
- [ ] **TrueForge approval pause** before every production change; Deny → adapt works.
- [ ] After approval: GitHub commit + reconcile + measured recovery in Grafana.
- [ ] README with scorecard, AI disclosure, setup; no secrets anywhere.
- [ ] ≤ 3-minute video (sped-up parts labeled) + backup recording.
- [ ] Every team member can explain the architecture and their code.

## 12. MVP cut line (protect in this order)

1. **Must:** AWS GPU + vLLM + Prometheus; S1 config regression end to end; observe tools; `deploy_shadow` + `run_replay`; sandboxed analysis code; gated `apply_production_config` with commit + reconcile; approval in the UI.
2. **Should:** S2 surge with `run_capacity_sweep` + gated `scale_replicas`; S5 "no action"; `compare_windows`-based classification; evidence card; Grafana recovery shot.
3. **Nice:** S7 counterfactual 2×2, parallel subagents, Deny→adapt, quality eval, admission control, S3/S4/S6/S8/S9, incident report, alert-triggered start, ASG option B.

Cut from the bottom. Never cut the three hackathon requirements: **real system, sandboxed code, approval gate**.
