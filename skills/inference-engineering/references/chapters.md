# Chapter notes: *Inference Engineering* (Philip Kiely, Baseten)

Paraphrased study notes, one block per chapter. Read the original for detail, figures and formulas:
https://www.baseten.co/inference-engineering/book/

## Ch. 0: Inference
Inference (serving trained generative models) is where most AI value and spend now sits. Inference
engineering spans the stack from CUDA kernels to Kubernetes, with three goals: faster, cheaper, more
reliable. Three layers of work: runtime (making one model instance fast), infrastructure (scaling and
availability across GPUs, clusters and clouds), and tooling (developer experience around both).

## Ch. 1: Prerequisites
- Know the requirements before optimising: model, interface, end-to-end latency budget, unit economics,
  usage pattern. Early products should use shared per-token APIs; move to dedicated deployments for
  scale, specialisation (custom model, SLOs) or multi-model orchestration (§1.1).
- Use case drives everything: online (latency-optimised) vs offline (throughput-optimised); consumer
  (cost, spiky virality) vs B2B (latency, uptime); compliance constrains where GPUs can be (§1.2).
- The biggest performance decision is the model: find the smallest model that passes your evals;
  fine-tuning and distillation can shrink it (§1.3).
- Metrics: TTFT (prefill), TPS/ITL (decode); distinguish perceived per-user TPS from total service TPS;
  report percentiles; separate inference-only from end-to-end latency (§1.4).

## Ch. 2: Models
- Neural nets are mostly matmuls in linear layers plus activations; transformers add attention (§2.1).
- LLM inference: tokenize + chat template, prefill (build KV cache, first token), decode (one token per
  forward pass, sampling via temperature/top-k/top-p) until a stop token or limit (§2.2).
- config.json describes the architecture; stick to popular architectures for the best engine support.
- Attention relates each token to prior tokens; with a KV cache it costs linear time per new token, but
  the cache itself grows with sequence length and lives in GPU memory (§2.2.3).
- Mixture of Experts adds sparsity: few active parameters per token, but under batching nearly all
  experts end up active; enables expert parallelism (§2.2.4).
- Bottlenecks: prefill and image/video generation are compute-bound, decode is memory-bound; compare an
  algorithm's arithmetic intensity to the GPU's ops:byte ratio (roofline) (§2.4).
- Attention optimisations (FlashAttention, PagedAttention, sliding/sparse attention) make long context
  tractable (§2.5).

## Ch. 3: Hardware
- GPUs = many SMs with Tensor Cores; judge compute by dense Tensor Core FLOPS at the precision you use;
  FLOPS roughly double per precision halving (§3.1.1).
- Memory hierarchy: HBM/VRAM (capacity and bandwidth) feeding L2/L1 SRAM caches. VRAM must hold weights
  plus generous KV headroom; bandwidth governs decode speed (§3.1.2).
- Generations: Hopper (H100/H200, FP8) and Blackwell (B200/B300, FP4 and microscaling) are the inference
  mainstream; Ada Lovelace (L4/L40) suits small, cost-sensitive models and lacks NVLink; Rubin adds HBM4
  and a prefill-oriented CPX chip (§3.2).
- Instances bundle GPU, CPU, RAM, disk, network and interconnect; any of them can bottleneck. NVLink/
  NVSwitch inside a node, InfiniBand between nodes; MIG slices big GPUs for small models (§3.3).
- Alternatives (AMD, TPU, Trainium/Inferentia, wafer-scale and SRAM-heavy startups) compete on bandwidth,
  power or platform integration but must rebuild the software stack (§3.4). Local inference trades cost
  and privacy for weak, fragmented hardware (§3.5).

## Ch. 4: Software
- CUDA kernels, graphs, driver and runtime; most engineers select rather than write kernels (cuBLAS,
  CUTLASS/CuTe, FlashInfer, DeepGEMM); kernel fusion removes memory round-trips (§4.1).
- PyTorch is the base; torch.compile fuses/selects kernels but not custom plugin kernels; safetensors
  for weights, ONNX for graph + weights; TensorRT / ONNX Runtime as optimised runtimes (§4.2).
- Engines: vLLM (broadest model and hardware support, easy, good on smaller GPUs), SGLang (strong for
  large MoE throughput, customisable, diffusion support), TensorRT-LLM (best performance, most effort,
  NVIDIA only). All do continuous batching, quantisation, speculation, prefix caching, parallelism,
  disaggregation (§4.3).
- NVIDIA Dynamo orchestrates engines at scale: KV-aware routing, disaggregation, multi-node, SLA-based
  planning; unnecessary for small deployments (§4.4).
- Benchmarking: shadow real traffic or simulate its lengths, arrival pattern, contents and parameters;
  baseline first; one change at a time; repeat runs; profile (PyTorch Profiler, Nsight) only to explain
  results (§4.5).

## Ch. 5: Techniques
- Quantisation: FP8-class floating formats are the production sweet spot (~30-50% faster per precision
  step); sensitivity rises from weights to activations to KV cache to attention; microscaling formats
  (MXFP8/MXFP4/NVFP4) trade a little overhead for accuracy; verify with perplexity, benchmarks and custom
  evals; aim for no perceptible loss (§5.1).
- Speculative decoding (draft-target, Medusa, EAGLE, n-gram/lookahead): more tokens per forward pass,
  improves TPS not TTFT, depends on acceptance rate and draft cost, only pays at low batch (§5.2).
- Caching: prefix caching reuses KV for shared prefixes (put unique tokens last); KV storage tiers
  (VRAM, host RAM, local SSD, network SSD) and offloading; cache-aware routing; long-context tools:
  FlashAttention, PagedAttention, chunked prefill (§5.3).
- Parallelism: tensor parallelism is the in-node default for latency; expert parallelism for MoE
  throughput; pipeline parallelism mainly across nodes (TP within, PP between); extra nodes are often
  better spent on replicas (§5.4).
- Disaggregation: separate prefill and decode engines (xPyD), conditional disaggregation for mixed
  traffic; worth it only for very high volume, very large models and prefill-heavy traffic (§5.5).

## Ch. 6: Modalities
The same autoregressive toolkit covers VLMs (vision encoder cost, image tokens), embeddings (throughput,
batching), ASR and TTS (streaming, time-to-first-word metrics); image and video generation are iterative
denoising, compute-bound, with their own caching, few-step and parallelism tricks; each modality needs
its own latency and quality metrics.

## Ch. 7: Production
- Containers with pinned dependencies; start from engine base images; NIMs as references (§7.1).
- Autoscaling on traffic (proactive) plus utilisation (lagging); configure min/max replicas, window,
  scale-down delay, concurrency target = batch size; continuous batching; cold starts (GPU, image,
  weights, engine compile) drive how aggressively you can scale down; routing, load balancing and
  queues (including priority queues); scale-to-zero only for bursty or offline use (§7.2).
- Multi-cloud capacity as one pool for capacity, redundancy, latency and compliance; reserved + on-demand
  + spot; geo-aware balancing; plan for GPU failures; active-active or active-passive (§7.3).
- Testing (manual, load, shadow) and canary rollouts rather than blue-green at GPU scale; cost is a
  function of batch size, traffic and sequence lengths; observe volume, sizes, codes, latency percentiles,
  replicas, utilisation and queue depth together (§7.4).
- Client code matters: session reuse, async jobs for throughput work, streaming protocols (WebSockets,
  gRPC) for real-time modalities (§7.5).
