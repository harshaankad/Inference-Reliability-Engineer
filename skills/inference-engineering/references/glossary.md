# Glossary (serving-focused, paraphrased)

Terms follow the usage in *Inference Engineering* (Philip Kiely, Baseten); definitions are our own
summaries. The book's Appendix A has the full glossary.

| Term | Meaning |
|---|---|
| Arithmetic intensity | Work done per byte of memory moved by an algorithm; compared to the GPU's ops:byte ratio it tells you whether you're compute-bound or memory-bound |
| Autoscaling | Automatically changing the number of replicas to keep SLOs without paying for idle GPUs; driven by traffic (leading) and utilisation (lagging) |
| Batch size / concurrency | How many sequences are processed together; the main latency-vs-throughput dial |
| Canary deployment | Sending a small share of live traffic to a new version, watching it, then ramping up |
| Chunked prefill | Splitting a long prompt's prefill into pieces that interleave with other requests' decode, so one long prompt doesn't stall everyone |
| Cold start | Time to bring a new replica to its first response: GPU, image, weights, engine start/compile |
| Compute-bound | Limited by FLOPS (LLM prefill, image/video generation) |
| Continuous (in-flight) batching | Token-level batching that swaps requests in and out as they finish, instead of waiting for fixed batches |
| Decode | The autoregressive phase generating one token per forward pass; memory-bandwidth-bound at low/medium batch |
| Disaggregation | Running prefill and decode on separate engines/GPUs (xPyD); pays off only at large scale |
| End-to-end latency | What the user experiences, including queueing and network, not just GPU time |
| FP8 / microscaling (MXFP8, NVFP4) | Low-precision floating formats; microscaling uses per-block scale factors to keep accuracy |
| Goodput | Requests (or tokens) served within the SLO; throughput that actually counts |
| Inter-token latency (ITL) | Time between consecutive output tokens (10 ms ITL = 100 tokens/s per user) |
| KV cache | Stored keys/values for every token in a sequence so attention doesn't recompute them; grows with sequence length and competes for GPU memory |
| KV-cache offloading | Moving colder KV blocks from VRAM to host RAM or SSD to extend capacity |
| KV-cache quantisation | Storing the KV cache at lower precision (e.g. FP8) to fit more tokens; moderately quality-sensitive |
| Memory-bound | Limited by memory bandwidth (LLM decode at low/medium batch) |
| Mixture of Experts (MoE) | Sparse layers of many expert matrices with a router; few active parameters per token |
| Ops:byte ratio | A GPU's FLOPS divided by its memory bandwidth; the balance point for arithmetic intensity |
| PagedAttention | Managing the KV cache in fixed-size pages/blocks to avoid fragmentation and allow sharing |
| Perplexity | How "surprised" a model is by reference text; a quick check for quantisation damage |
| Prefill | Processing the whole prompt to build the KV cache and produce the first token; compute-bound; sets TTFT |
| Prefix caching | Reusing the KV cache of an identical prompt prefix across requests, skipping that prefill |
| Preemption | An engine evicting a running sequence's KV cache under memory pressure and recomputing it later |
| Shadow traffic | Copying real production requests to a test system to benchmark it without affecting users |
| Speculative decoding | Drafting several tokens cheaply and verifying them in one forward pass; raises TPS, not TTFT; needs spare compute (low batch) |
| Tensor parallelism (TP) | Splitting each layer's tensors across GPUs; needs fast interconnect (NVLink) |
| Tokens per second (TPS) | Per-user decode speed (perceived TPS) or total service output (total TPS); say which |
| Time to first token (TTFT) | Delay until the first output token; driven by prefill and queueing |
| VRAM | GPU high-bandwidth memory holding weights, activations and the KV cache |
