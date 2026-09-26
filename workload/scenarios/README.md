# Traffic scenarios

Loaded by `workload/loadgen.py` via the prod controller (`POST /traffic/scenario`, used only by `chaos/chaos.py`).
The agent never sees scenario names; it has to infer what changed from the request log and engine metrics.

| File | What it simulates |
|---|---|
| `healthy.json` | Normal traffic: mostly short chat prompts |
| `long_context_shift.json` | **Flagship.** Same request rate, but product starts sending long RAG contexts (~50% long) |
| `surge.json` | Many more users (3x rps), same mix |
| `burst.json` | 45 s spike at 5x, then back to normal (correct answer: no production change) |

The numbers are starting points. **Calibrate on the real g6.xlarge (L4)** (see README, "Calibration").
