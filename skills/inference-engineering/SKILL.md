---
name: inference-engineering
description: Inference-engineering reference distilled from "Inference Engineering" (Philip Kiely, Baseten). Mental models, decision rules and anti-patterns for LLM serving performance - prefill vs decode bottlenecks, KV-cache budgeting, batching, prefix caching, quantization, speculative decoding, parallelism, benchmarking and safe rollout. Load when choosing or justifying a serving-config change, reading latency/throughput/memory metrics, or designing an experiment.
---

# Inference Engineering (distilled)

Source: Philip Kiely, *Inference Engineering* (Baseten, 2025), free to read at
https://www.baseten.co/inference-engineering/book/. This skill paraphrases and organises the book's
ideas for an on-call agent; section numbers (§) point to the original. It is not a substitute for the
book. `references/chapters.md` has per-chapter notes, `references/glossary.md` has terms.

## How to use this skill
1. Name the bottleneck with the mental models below (compute vs memory bandwidth vs KV capacity vs load).
2. Pick levers from the decision rules; prefer the lossless ones first.
3. Design the experiment with the benchmarking rules; change one thing at a time.
4. Ship with the rollout rules; watch the metrics together, not one at a time.

## Mental models

**M1. Two phases, two bottlenecks (§1.4, §2.2, §2.4).** Prefill builds the KV cache for the whole prompt
and is *compute-bound*; it sets **TTFT**. Decode generates one token per forward pass and is
*memory-bandwidth-bound* at low/medium batch sizes; it sets per-user **TPS / inter-token latency**.
Diagnose which phase is hurting before touching knobs: long prompts inflate TTFT, long answers inflate
end-to-end time.

**M2. Arithmetic intensity (§2.4).** A GPU is balanced when the work done per byte moved matches its
ops:byte ratio. Batching raises decode's arithmetic intensity (more compute per weight read), which is
why throughput grows with batch size until compute or memory runs out.

**M3. The VRAM budget (§3.1.2, §5.3.2, §5.4).** Memory = weights + activations/buffers + KV cache. Plan
for weights plus *at least ~50% headroom* for KV cache (more for long context or big batches). Whatever
the engine reserves after weights is the KV pool; when it fills, requests wait, get preempted/recomputed,
or old cache entries are evicted. Roughly 1 GB per billion parameters in FP8, 2 GB in BF16.

**M4. Latency vs throughput is a dial, not a win (§1.2.2, §7.2.1).** Bigger batches/concurrency =
more total tokens/s and worse per-user latency. Online products (chat, agents, voice) optimise latency;
offline jobs optimise throughput. Per-replica batch size and the autoscaler's concurrency target should
match.

**M5. Constraints buy performance (§1, §5 intro).** The more you know and fix about the workload
(sequence lengths, prompt structure, traffic shape, SLOs), the more you can specialise and the faster
it gets. Real traffic breaks assumptions, so tuning is continuous, not one-off.

**M6. Percentiles and scope (§1.4.1-1.4.2).** Latency is right-skewed; judge P50 *and* P90/P95/P99, never
the mean. Separate on-GPU inference time from end-to-end time (queueing, network): fast inference with
slow end-to-end means an infrastructure/queueing problem, not a model-performance one.

**M7. Techniques interact (§5 intro).** Optimisations can help or fight each other (e.g. KV-cache
quantisation helps disaggregation and prefix caching; large batches starve speculative decoding).
Aim for a balanced set, and expect non-obvious winners that only experiments reveal.

## Decision rules

| Situation (evidence) | Levers, in preferred order | Why / caveat |
|---|---|---|
| Many requests share a long prefix (system prompt, RAG scaffold, multi-turn history, shared code) | **Prefix caching** (`enable_prefix_caching`); keep novel tokens *late* in the prompt | Lossless; skips prefill for the shared prefix, lowering TTFT and freeing compute. Matching stops at the first differing token (§5.3.1) |
| KV cache near 100%, requests waiting for KV capacity or being preempted, long prompts | Bound concurrency to what the cache holds (`max_num_seqs`); prefix caching; **FP8 KV cache** (`kv_cache_dtype=fp8`); a little more `gpu_memory_utilization`; KV offload to host memory at larger scale | Over-admission makes sequences fight for cache. FP8 KV roughly doubles capacity but is *moderately* quality-sensitive: gate it on an eval (§5.1.2, §5.3.2) |
| Long prompts stall short ones (short-prompt TTFT spikes) | Chunked prefill with a sensible per-step token budget (`max_num_batched_tokens`); prefix caching | Splits big prefills so decode keeps flowing (§5.3.4). A bigger budget is not automatically better when the KV cache is the limit |
| Load exceeds what one replica can serve under *any* config | **Scale out** replicas (traffic-based autoscaling, utilisation as a lagging check); queue with priorities if capacity lags | A capacity problem is not a config problem (§7.2). Concurrency target = batch size |
| Need more speed and quality risk is acceptable | FP8 (floating-point, ideally microscaled) on weights/activations, then KV cache; leave attention/softmax in high precision | One precision step ≈ 30-50% faster, not 2×. Sensitivity: weights < activations < KV cache < attention. Avoid integer formats for quality-sensitive work (§5.1) |
| Low batch, decode-bound, want higher per-user TPS | Speculative decoding (EAGLE, n-gram for code/edit tasks) | Helps TPS only, never TTFT; must be disabled at high batch sizes (§5.2) |
| Model or KV cache doesn't fit one GPU | Tensor parallelism within an NVLink node; expert parallelism for MoE throughput | Lovelace GPUs (L4/L40) have no NVLink, so parallelism across them is inefficient: prefer replicas (§3.2.2, §5.4) |
| Huge prefill-heavy traffic on 100B+ models (100M-1B+ tokens/day) | Disaggregated prefill/decode (xPyD), conditional disaggregation | Otherwise it wastes GPUs; use them for replicas instead (§5.5.2) |

## Experiment and benchmark rules (§4.5, §7.4)
- **Baseline first**, then one change at a time; test changes individually *and* combined.
- The best benchmark is **shadowed real production traffic**. If simulating, match input/output lengths,
  arrival pattern and jitter, real request contents (cache hits depend on them) and sampling parameters.
- Send enough traffic for stable numbers; repeat runs when results are close.
- Track quality alongside speed for any lossy change: perplexity, a public benchmark, and above all a
  product-specific eval; the bar is "no difference beyond noise".
- Profile only when benchmarks can't explain a result.

## Shipping rules (§7.2-7.4)
- Prefer **canary** rollouts (small traffic share, watch, ramp) over blue-green at GPU scale; keep a
  rollback path and enough warm replicas so the canary doesn't queue.
- Watch together: request volume, input/output lengths, status codes, TTFT/TPS/end-to-end at P50/P90/P99,
  replicas (active and starting), GPU/CPU/memory utilisation, queue depth. A latency spike can be volume
  *or* longer prompts; only the combination tells you which.
- Expect hardware failure (order of one per ~50k GPU-hours) and cold starts (GPU, image, weights, engine
  start/compile); cache weights and compiled engines close to the GPU.

## Anti-patterns
- Optimising the **mean** latency, or a single metric in isolation.
- Benchmarking with toy prompts, fixed synthetic lengths or unrealistic concurrency.
- Changing several knobs at once, then guessing which one helped.
- Treating "bigger batch / bigger token budget" as always faster when the KV cache is the constraint.
- Quantising attention/softmax, or using integer formats for quality-sensitive production traffic.
- Turning on speculative decoding for high-batch, throughput-oriented serving.
- Reaching for disaggregation or multi-node parallelism at small scale.
- Scaling out to hide a configuration bug, or re-tuning config to absorb load the hardware can't serve.
- Scaling on GPU utilisation alone (it lags; a few huge prompts look like heavy traffic).

## Cheatsheet (§3.2, §5.1, §7.3)
| GPU | FP8 dense | Memory | Bandwidth | Notes |
|---|---|---|---|---|
| L4 (Ada) | ~242 TFLOPS | 24 GB | ~300 GB/s | Cheap, small models; FP8 capable; no NVLink |
| L40 (Ada) | ~362 TFLOPS | 48 GB | ~864 GB/s | Usually beaten by MIG slices of H100 |
| H100 / H200 (Hopper) | ~1,979 TFLOPS | 80 / 141 GB | 3.35 / 4.8 TB/s | Workhorses; FP8, FlashAttention 3 |
| B200 / B300 (Blackwell) | ~5 PFLOPS | 192 / 288 GB | up to 8 TB/s | FP4 / microscaling formats |

- Prefill wants FLOPS; decode wants memory bandwidth (pick hardware accordingly).
- Inference engines: vLLM (broadest model/hardware support, good on smaller/older GPUs), SGLang (strong on
  large MoE throughput), TensorRT-LLM (highest performance, more engineering) (§4.3).
- Network rule of thumb: ~5 ms per time zone of distance (§7.3.2).

## Mapping to vLLM knobs (for this fleet)
| Knob | Book concept |
|---|---|
| `max_num_seqs` | concurrency / batch size cap (M4, KV over-admission) |
| `max_num_batched_tokens` | per-step token budget for chunked prefill (§5.3.4) |
| `enable_prefix_caching` | prefix KV reuse (§5.3.1) |
| `kv_cache_dtype` | KV-cache quantisation (§5.1.2); quality-gate it |
| `gpu_memory_utilization` | size of the KV pool after weights (M3, §5.3.2) |
| `max_model_len` | longest accepted sequence; lowering it only rejects long requests |
