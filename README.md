# LLM_Sizing_Guide
A calculator to estimate the memory footprint, capacity, and latency based on your planned LLM application's requirements on different GPU architectures.
Blog: https://blogs.vmware.com/cloud-foundation/2024/09/25/llm-inference-sizing-and-performance-guidance/

## Calculation model

The calculator estimates **one stream** across the selected devices. It counts active weight traffic by component precision (for example MXFP4 expert matrices plus BF16 attention, router and output matrices), context-dependent attention/cache/state work, and memory in bytes, and it checks workload admission against the memory budget. Theoretical GPU peaks are the default. Tensor-parallel communication, matrix-kernel precision, cache sharding and a runtime memory reserve can be specified explicitly. Optional `--efficiency calibrated` applies GPU decode factors fitted to public batch-1 measurements; prefill factors remain provisional and CPU rates are fixed effective values. Results are idealized single-stream estimates, not aggregate serving capacity.

# Usage
Prerequisite: `pip install -r requirements.txt`

Here are the Flags and their abbreviations for the script.
- num_gpu ('-g'): Specify the number of GPUs you plan to use for your deployment.
- prompt_sz ('-p'): Define the average size of the input prompts you expect to process.
- response_sz ('-r'): Set the average size of the responses you expect to generate.
- n_concurrent_req ('-c'): Indicate the number of concurrent requests for memory sizing only; this does not scale the output or decode rates.
- profile ('--profile'): `via` sizes the VMware Intelligent Assist workload (defaults to 16000-token prompts, 512-token answers, 4 concurrent users) using the VIA model and GPU lists.
- device ('--device'): `gpu` (default) or `cpu` to size CPU-only inference VMs (llama.cpp engine in VCF Private AI Services).
- models / devices ('--models', '--devices'): comma-separated name filters, e.g. `--models "gpt-oss,Qwen3.8" --devices "H100,B200"`.
- efficiency ('--efficiency'): `theoretical` (an idealized hardware estimate, the default for both GPU profiles) or `calibrated` (GPU decode factors fitted to public batch-1 measurements: 0.70 bandwidth efficiency plus 0.5 ms/token for dense and MoE models; prefill compute factors of 0.38 dense / 0.15 MoE remain provisional). CPU devices always use and report calibrated effective rates, even when `--efficiency theoretical` is supplied.
- memory_utilization ('--memory-utilization'): fraction of device memory available for weights, cache and state. Default 0.9 for `--profile via` GPU runs (a 10% runtime reserve) and 1.0 for the default profile and CPU devices; an explicit value always overrides the default.

Baseline GPU TFLOPS in `configs/gpu_specs.py` are dense FP16/BF16 Tensor Core figures; optional FP8/FP4 peaks are separately labeled. NVIDIA datasheets often show the 2:4 sparsity figure, which is 2x higher (e.g. H100 PCIe 1,513 sparse = 756.5 dense); inference does not use sparsity.

By modifying these variables, you can easily estimate the performance characteristics of your LLM deployment and make informed decisions about your infrastructure requirements.

## Interpreting performance estimates

Theoretical mode uses peak hardware rates and explicit estimates of work. Memory is calculated in bytes and displayed in decimal GB; only the per-token cache column uses GiB. It includes growing KV, bounded sliding-window caches and recurrent state. The memory reserve (`--memory-utilization`; 0.9 for `--profile via` GPU runs, otherwise 1.0) changes capacity, `Max Concurrent Req` and OOM decisions, not latency for workloads that fit. The memory-footprint table shows demand and excludes the reserve; a value of 1.0 is a raw hardware budget, not a runtime recommendation.

- **Single-request output rate** = response tokens / end-to-end seconds, including prompt prefill.
- **Single-stream decode rate** = `1000 / TPOT_ms`, with TPOT evaluated at the midpoint of the generated context. It excludes prefill.
- Both rates describe one stream across the selected devices. Concurrency affects memory admission only; it does not model batching or multiply throughput. **Max Concurrent Req** is a memory limit, not a latency guarantee.
- **Status** explains unavailable results. Both rates are `OOM` if weights plus the requested concurrency's cache/state exceed the memory budget, and `N/A` for unsupported configurations or contexts exceeding the model limit. Warnings, tables and CSVs use the same admission calculation.

Terminal and CSV reports use **Single-request output rate** in place of **Output Tokens Throughput**, and include **Single-stream decode rate** and **Status**. Update consumers of the old CSV header. Zero/negative workloads and empty model/device selections are rejected.

### Work and deployment assumptions

The per-token projection estimate is `max(matrix FLOPs / compute peak, active weight bytes / bandwidth)`. Attention is a separate compute/bandwidth bound, added to projections. Prefill counts causal attention pairs; decode counts the cached context at the response midpoint. GQA includes sliding windows. MLA assumes absorbed latent attention. GLM DSA attends to at most 2,048 cached tokens while its indexer scans the full context. Recurrent state has read/write traffic and an approximate six scalar operations per FP32 state element; prefill assumes fused state reuse. Softmax, top-k selection, activation traffic, kernel launch costs and imperfect overlap/utilization are not fully modeled.

GPT-OSS active weight bytes count the selected MXFP4 matrices and BF16 attention/router/output matrices separately (about 5.00 GB for 120B, not the 2.75 GB implied by applying the checkpoint-average bytes per parameter to every active parameter). Kimi and GLM use config-derived precision splits; their rounded active-parameter counts and cache layouts remain approximations. Checkpoint memory and active decode traffic are different quantities. GLM NVFP4 leaves attention/shared experts in BF16, so a smaller checkpoint need not imply fewer active bytes than its FP8 version. See [GPT-OSS configuration](https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json), [precision format](https://github.com/openai/gpt-oss#precision-format), [Kimi configuration](https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json), [GLM FP8 configuration](https://huggingface.co/zai-org/GLM-5.2-FP8/raw/main/config.json), and [GLM NVFP4 configuration](https://huggingface.co/nvidia/GLM-5.2-NVFP4/raw/main/config.json).

- `--parallelism ideal` (default) explicitly assumes perfectly pooled/sharded hardware and zero communication. It is not a deployment topology.
- `--parallelism tensor --interconnect-gbps N` adds two BF16 activation ring all-reduces per layer. `N` is **one-way GB/s per GPU**, not Gbit/s or aggregate bidirectional bandwidth. Transfer bytes per collective are `2 × (GPUs−1)/GPUs × tokens × hidden size × 2`. `--collective-latency-us` is required together with `--interconnect-gbps` for multi-GPU tensor mode and adds that latency per collective. An explicit 0 is only a lower bound and prints a warning; 10 us is an illustrative assumption, not a measured hardware latency. Communication is not overlapped. This models a balanced tensor-parallel scenario; expert parallelism, all-to-all communication, pipeline parallelism and replicas are not modeled.
- `--kv-cache-sharding auto` pools cache in ideal mode. Tensor mode replicates MLA/DSA cache and partitions GQA by KV heads (replicating heads when necessary). `sharded` assumes an engine with cache/context parallelism; `replicated` explicitly duplicates the cache. Recurrent state is assumed evenly partitioned. Per-rank imbalance and communication for cache parallelism are not modeled.
- `--compute-precision bf16` (default) is an explicit matrix-kernel assumption, independent of checkpoint storage. `fp8`/`fp4` use cataloged dense peaks for eligible matrix components, retaining BF16 for other components and attention. Missing hardware peaks/checkpoint support produce `N/A`; selecting a precision does not establish engine support. Current optional peaks cover H100/H200 FP8 and B200/B300 FP8/FP4.

TTFT retains the calculator's prefill-plus-first-decode convention. E2E is prefill plus response tokens × midpoint TPOT. These are idealized estimates, not a guaranteed ceiling. Prefix caching, speculative decoding, scheduling, batching and aggregate serving throughput remain outside this calculator. CPU effective rates are fixed constants. GPU decode calibration factors were fitted after these corrections; prefill factors remain provisional.

Example of explicit tensor assumptions (900 GB/s and 5 us are scenario inputs, not measured fabric values):

```bash
python LLM_size_pef_calculator.py --profile via --models Kimi --devices B300 -g 8 --parallelism tensor --interconnect-gbps 900 --collective-latency-us 5 --memory-utilization 0.9
```

# Example output
```bash
✗ python LLM_size_pef_calculator.py -g 4 -p 4096 -r 256 -c 10
 num_gpu = 4, prompt_size = 4096 tokens, response_size = 256 tokens
 n_concurrent_request = 10, profile = default, device = gpu, efficiency = theoretical
Estimate mode: theoretical (idealized hardware estimate).
Single-request output rate includes prefill; single-stream decode rate excludes prefill. Neither rate represents aggregate serving capacity.
Concurrency affects memory sizing only; Max Concurrent Req is a memory limit, not a latency target.
TPOT uses midpoint decode context. Attention/cache/state work is estimated; batching, prefix reuse, speculative decoding and queueing are not modeled.
Assumptions: parallelism=ideal, compute=bf16, memory utilization=1, KV sharding=auto.
Memory reserve: 0% of device memory (profile default). The memory-footprint table excludes the reserve; Max Concurrent Req and OOM decisions apply it. Latency is unchanged for workloads that fit.
Ideal pooling assumes perfect partitioning and zero communication; it is not a deployment topology.

******************** Estimate LLM Memory Footprint ********************
| Model           |   Input Size (tokens) |   Output Size (tokens) |   Concurrent Requests | KV Cache Size per Token   | Weights   | Memory Footprint   |
|-----------------+-----------------------+------------------------+-----------------------+---------------------------+-----------+--------------------|
| Llama-3.1-8B    |                  4096 |                    256 |                    10 | 0.000122 GiB/token        | 16.0 GB   | 21.70 GB           |
| Llama-3.1-70B   |                  4096 |                    256 |                    10 | 0.000305 GiB/token        | 140.0 GB  | 154.26 GB          |
| Mistral-7B-v0.3 |                  4096 |                    256 |                    10 | 0.000122 GiB/token        | 14.0 GB   | 19.70 GB           |
| Qwen2.5-14B     |                  4096 |                    256 |                    10 | 0.000183 GiB/token        | 29.4 GB   | 37.96 GB           |

******************** Estimate LLM Capacity and Latency ********************
| Model           | GPU      |   Input Size (tokens) |   Output Size (tokens) |   Concurrent Requests |   Max # KV Cache Tokens | Prefill Time   | TPOT (ms)   | TTFT    | E2E Latency   | Single-request output rate   | Single-stream decode rate   |   Max Concurrent Req | Status   |
|-----------------+----------+-----------------------+------------------------+-----------------------+-------------------------+----------------+-------------+---------+---------------+------------------------------+-----------------------------+----------------------+----------|
| Llama-3.1-8B    | L40s     |                  4096 |                    256 |                    10 |                 1342773 | 0.012 ms       | 4.790 ms    | 0.053 s | 1.3 s         | 200.86 tokens/sec            | 208.78 tokens/sec           |                  308 | OK       |
| Llama-3.1-8B    | H100 NVL |                  4096 |                    256 |                    10 |                 2746582 | 0.005 ms       | 1.061 ms    | 0.022 s | 0.3 s         | 874.99 tokens/sec            | 942.39 tokens/sec           |                  631 | OK       |
| Llama-3.1-8B    | H200 NVL |                  4096 |                    256 |                    10 |                 4180908 | 0.005 ms       | 0.862 ms    | 0.022 s | 0.2 s         | 1059.42 tokens/sec           | 1159.87 tokens/sec          |                  960 | OK       |
| Llama-3.1-8B    | MI300X   |                  4096 |                    256 |                    10 |                 5737304 | 0.003 ms       | 0.781 ms    | 0.014 s | 0.2 s         | 1200.36 tokens/sec           | 1280.69 tokens/sec          |                 1318 | OK       |
| Llama-3.1-70B   | L40s     |                  4096 |                    256 |                    10 |                  158691 | 0.100 ms       | 40.910 ms   | 0.452 s | 10.9 s        | 23.52 tokens/sec             | 24.44 tokens/sec            |                   36 | OK       |
| Llama-3.1-70B   | H100 NVL |                  4096 |                    256 |                    10 |                  720214 | 0.043 ms       | 9.063 ms    | 0.187 s | 2.5 s         | 102.47 tokens/sec            | 110.34 tokens/sec           |                  165 | OK       |
| Llama-3.1-70B   | H200 NVL |                  4096 |                    256 |                    10 |                 1293945 | 0.043 ms       | 7.364 ms    | 0.186 s | 2.1 s         | 124.07 tokens/sec            | 135.80 tokens/sec           |                  297 | OK       |
| Llama-3.1-70B   | MI300X   |                  4096 |                    256 |                    10 |                 1916503 | 0.028 ms       | 6.669 ms    | 0.121 s | 1.8 s         | 140.57 tokens/sec            | 149.95 tokens/sec           |                  440 | OK       |
| Mistral-7B-v0.3 | L40s     |                  4096 |                    256 |                    10 |                 1358032 | 0.010 ms       | 4.211 ms    | 0.047 s | 1.1 s         | 228.43 tokens/sec            | 237.47 tokens/sec           |                  312 | OK       |
| Mistral-7B-v0.3 | H100 NVL |                  4096 |                    256 |                    10 |                 2761840 | 0.005 ms       | 0.933 ms    | 0.019 s | 0.3 s         | 994.94 tokens/sec            | 1071.90 tokens/sec          |                  634 | OK       |
| Mistral-7B-v0.3 | H200 NVL |                  4096 |                    256 |                    10 |                 4196166 | 0.005 ms       | 0.758 ms    | 0.019 s | 0.2 s         | 1204.58 tokens/sec           | 1319.26 tokens/sec          |                  964 | OK       |
| Mistral-7B-v0.3 | MI300X   |                  4096 |                    256 |                    10 |                 5752563 | 0.003 ms       | 0.686 ms    | 0.012 s | 0.2 s         | 1364.96 tokens/sec           | 1456.69 tokens/sec          |                 1321 | OK       |
| Qwen2.5-14B     | L40s     |                  4096 |                    256 |                    10 |                  827026 | 0.022 ms       | 8.747 ms    | 0.098 s | 2.3 s         | 109.96 tokens/sec            | 114.32 tokens/sec           |                  190 | OK       |
| Qwen2.5-14B     | H100 NVL |                  4096 |                    256 |                    10 |                 1762898 | 0.009 ms       | 1.938 ms    | 0.040 s | 0.5 s         | 478.87 tokens/sec            | 516.04 tokens/sec           |                  405 | OK       |
| Qwen2.5-14B     | H200 NVL |                  4096 |                    256 |                    10 |                 2719116 | 0.009 ms       | 1.574 ms    | 0.040 s | 0.4 s         | 579.75 tokens/sec            | 635.12 tokens/sec           |                  624 | OK       |
| Qwen2.5-14B     | MI300X   |                  4096 |                    256 |                    10 |                 3756713 | 0.006 ms       | 1.426 ms    | 0.026 s | 0.4 s         | 656.99 tokens/sec            | 701.28 tokens/sec           |                  863 | OK       |
```

Note: TTFT now includes the full prompt prefill (`prompt_size x prefill time per token + TPOT`). Earlier versions reported only one token's prefill time, which understated TTFT.

# Sizing VMware Intelligent Assist (VIA)
Model specs support MoE (active vs. total parameters), quantized checkpoints, hybrid/linear attention (only full-attention layers keep KV) and MLA-style caches; see `configs/model_specs.py`. Architecture values were checked against each model's `config.json`. CPU VM assumptions (Intel Xeon with and without AMX, AMD EPYC) and their calibration sources are in `configs/cpu_specs.py`. llama.cpp only uses AMX for Q4_0/Q4_1/Q8_0/Q4_K/Q5_K/Q6_K/IQ4_XS GGUFs, so the CPU table's `Accel` column shows which kernel path each estimate assumes. GLM-5.2 and Kimi-K3 cache layouts are estimates derived from their configs.

`--profile via` defaults to theoretical GPU estimates and reserves 10% of GPU memory (`--memory-utilization 0.9`; pass `1` for the raw budget). To opt into calibration, add `--efficiency calibrated`, for example:

```bash
python LLM_size_pef_calculator.py --profile via --efficiency calibrated -g 8 --models Kimi --devices B300
```

Calibration applies class-wide GPU efficiencies and a fixed decode overhead. The decode factors (0.70 bandwidth efficiency plus 0.5 ms/token, the same for dense and MoE models) were fitted to public batch-1 measurements after active-weight bytes were corrected to component precision. The H100 gpt-oss-20b vLLM result is a visible outlier excluded from fitting. Prefill compute factors (0.38 dense, 0.15 MoE) remain provisional and CPU rates are unchanged. The same factors are extrapolated to other configurations; there are no Kimi-K3, GLM-5.2, or B300 anchors in the validation set. `validation/sanity_check.py` reports a legacy-input comparison and a separate source-aware comparison (documented context lengths, KV precision and hardware specifications) and checks only documented configurations, not aggregate serving performance.

## GPU: gpt-oss and Qwen on a single GPU
```bash
✗ python LLM_size_pef_calculator.py --profile via -g 1 --models "gpt-oss,Qwen"
 num_gpu = 1, prompt_size = 16000 tokens, response_size = 512 tokens
 n_concurrent_request = 4, profile = via, device = gpu, efficiency = theoretical
Estimate mode: theoretical (idealized hardware estimate).
Single-request output rate includes prefill; single-stream decode rate excludes prefill. Neither rate represents aggregate serving capacity.
Concurrency affects memory sizing only; Max Concurrent Req is a memory limit, not a latency target.
TPOT uses midpoint decode context. Attention/cache/state work is estimated; batching, prefix reuse, speculative decoding and queueing are not modeled.
Assumptions: parallelism=ideal, compute=bf16, memory utilization=0.9, KV sharding=auto.
Memory reserve: 10% of device memory (profile default). The memory-footprint table excludes the reserve; Max Concurrent Req and OOM decisions apply it. Latency is unchanged for workloads that fit.

******************** Estimate LLM Memory Footprint ********************
| Model                |   Input Size (tokens) |   Output Size (tokens) |   Concurrent Requests | KV Cache Size per Token   | Weights   | Memory Footprint   |
|----------------------+-----------------------+------------------------+-----------------------+---------------------------+-----------+--------------------|
| gpt-oss-20b (MXFP4)  |                 16000 |                    512 |                     4 | 0.000023 GiB/token        | 13.0 GB   | 14.64 GB           |
| gpt-oss-120b (MXFP4) |                 16000 |                    512 |                     4 | 0.000034 GiB/token        | 63.0 GB   | 65.45 GB           |
| Qwen3.5-9B (Q4_K_M)  |                 16000 |                    512 |                     4 | 0.000031 GiB/token        | 5.8 GB    | 8.16 GB            |
| Qwen3.5-9B (BF16)    |                 16000 |                    512 |                     4 | 0.000031 GiB/token        | 18.0 GB   | 20.36 GB           |
| Qwen3.8-27B (Q4_K_M) |                 16000 |                    512 |                     4 | 0.000061 GiB/token        | 18.0 GB   | 22.93 GB           |
| Qwen3.8-27B (FP8)    |                 16000 |                    512 |                     4 | 0.000061 GiB/token        | 29.0 GB   | 33.93 GB           |
| Qwen3.8-27B (BF16)   |                 16000 |                    512 |                     4 | 0.000061 GiB/token        | 56.0 GB   | 60.93 GB           |

Warning gpt-oss-120b (MXFP4) on 1x L4: OOM: requested weights/cache/state exceed memory budget.
Memory limit for this context: 0 concurrent requests.
Even one request does not fit; use more memory or a shorter supported context.

Warning gpt-oss-120b (MXFP4) on 1x RTX PRO 4500 BW: OOM: requested weights/cache/state exceed memory budget.
Memory limit for this context: 0 concurrent requests.
Even one request does not fit; use more memory or a shorter supported context.

Warning gpt-oss-120b (MXFP4) on 1x L40s: OOM: requested weights/cache/state exceed memory budget.
Memory limit for this context: 0 concurrent requests.
Even one request does not fit; use more memory or a shorter supported context.

Warning Qwen3.8-27B (Q4_K_M) on 1x L4: OOM: requested weights/cache/state exceed memory budget.
Memory limit for this context: 2 concurrent requests.

Warning Qwen3.8-27B (FP8) on 1x L4: OOM: requested weights/cache/state exceed memory budget.
Memory limit for this context: 0 concurrent requests.
Even one request does not fit; use more memory or a shorter supported context.

Warning Qwen3.8-27B (FP8) on 1x RTX PRO 4500 BW: OOM: requested weights/cache/state exceed memory budget.
Memory limit for this context: 0 concurrent requests.
Even one request does not fit; use more memory or a shorter supported context.

Warning Qwen3.8-27B (BF16) on 1x L4: OOM: requested weights/cache/state exceed memory budget.
Memory limit for this context: 0 concurrent requests.
Even one request does not fit; use more memory or a shorter supported context.

Warning Qwen3.8-27B (BF16) on 1x RTX PRO 4500 BW: OOM: requested weights/cache/state exceed memory budget.
Memory limit for this context: 0 concurrent requests.
Even one request does not fit; use more memory or a shorter supported context.

Warning Qwen3.8-27B (BF16) on 1x L40s: OOM: requested weights/cache/state exceed memory budget.
Memory limit for this context: 0 concurrent requests.
Even one request does not fit; use more memory or a shorter supported context.

******************** Estimate LLM Capacity and Latency ********************
| Model                | GPU             |   Input Size (tokens) |   Output Size (tokens) |   Concurrent Requests |   Max # KV Cache Tokens | Prefill Time   | TPOT (ms)   | TTFT    | E2E Latency   | Single-request output rate   | Single-stream decode rate   |   Max Concurrent Req | Status                                                  |
|----------------------+-----------------+-----------------------+------------------------+-----------------------+-------------------------+----------------+-------------+---------+---------------+------------------------------+-----------------------------+----------------------+---------------------------------------------------------|
| gpt-oss-20b (MXFP4)  | L4              |                 16000 |                    512 |                     4 |                  349806 | 0.073 ms       | 13.695 ms   | 1.177 s | 8.2 s         | 62.63 tokens/sec             | 73.02 tokens/sec            |                   21 | OK                                                      |
| gpt-oss-20b (MXFP4)  | RTX PRO 4500 BW |                 16000 |                    512 |                     4 |                  642775 | 0.043 ms       | 5.136 ms    | 0.699 s | 3.3 s         | 154.09 tokens/sec            | 194.72 tokens/sec           |                   38 | OK                                                      |
| gpt-oss-20b (MXFP4)  | L40s            |                 16000 |                    512 |                     4 |                 1228713 | 0.024 ms       | 4.755 ms    | 0.394 s | 2.8 s         | 181.34 tokens/sec            | 210.30 tokens/sec           |                   73 | OK                                                      |
| gpt-oss-20b (MXFP4)  | RTX PRO 6000 BW |                 16000 |                    512 |                     4 |                 2986525 | 0.018 ms       | 2.573 ms    | 0.284 s | 1.6 s         | 320.26 tokens/sec            | 388.72 tokens/sec           |                  179 | OK                                                      |
| gpt-oss-20b (MXFP4)  | A100 80GB SXM   |                 16000 |                    512 |                     4 |                 2400588 | 0.028 ms       | 2.015 ms    | 0.453 s | 1.5 s         | 345.29 tokens/sec            | 496.30 tokens/sec           |                  144 | OK                                                      |
| gpt-oss-20b (MXFP4)  | H100 SXM        |                 16000 |                    512 |                     4 |                 2400588 | 0.009 ms       | 1.226 ms    | 0.143 s | 0.8 s         | 664.79 tokens/sec            | 815.40 tokens/sec           |                  144 | OK                                                      |
| gpt-oss-20b (MXFP4)  | H100 NVL        |                 16000 |                    512 |                     4 |                 2913283 | 0.011 ms       | 1.053 ms    | 0.170 s | 0.7 s         | 723.32 tokens/sec            | 949.27 tokens/sec           |                  175 | OK                                                      |
| gpt-oss-20b (MXFP4)  | H200 SXM        |                 16000 |                    512 |                     4 |                 4634474 | 0.009 ms       | 0.856 ms    | 0.143 s | 0.6 s         | 882.01 tokens/sec            | 1168.34 tokens/sec          |                  278 | OK                                                      |
| gpt-oss-20b (MXFP4)  | B200            |                 16000 |                    512 |                     4 |                 6062697 | 0.004 ms       | 0.514 ms    | 0.063 s | 0.3 s         | 1572.96 tokens/sec           | 1947.23 tokens/sec          |                  364 | OK                                                      |
| gpt-oss-20b (MXFP4)  | B300            |                 16000 |                    512 |                     4 |                10017775 | 0.004 ms       | 0.514 ms    | 0.063 s | 0.3 s         | 1572.96 tokens/sec           | 1947.23 tokens/sec          |                  602 | OK                                                      |
| gpt-oss-120b (MXFP4) | L4              |                 16000 |                    512 |                     4 |                       0 | OOM            | OOM         | OOM     | OOM           | OOM                          | OOM                         |                    0 | OOM: requested weights/cache/state exceed memory budget |
| gpt-oss-120b (MXFP4) | RTX PRO 4500 BW |                 16000 |                    512 |                     4 |                       0 | OOM            | OOM         | OOM     | OOM           | OOM                          | OOM                         |                    0 | OOM: requested weights/cache/state exceed memory budget |
| gpt-oss-120b (MXFP4) | L40s            |                 16000 |                    512 |                     4 |                       0 | OOM            | OOM         | OOM     | OOM           | OOM                          | OOM                         |                    0 | OOM: requested weights/cache/state exceed memory budget |
| Qwen3.8-27B (Q4_K_M) | L4              |                 16000 |                    512 |                     4 |                   52642 | OOM            | OOM         | OOM     | OOM           | OOM                          | OOM                         |                    2 | OOM: requested weights/cache/state exceed memory budget |
| Qwen3.8-27B (FP8)    | L4              |                 16000 |                    512 |                     4 |                       0 | OOM            | OOM         | OOM     | OOM           | OOM                          | OOM                         |                    0 | OOM: requested weights/cache/state exceed memory budget |
| Qwen3.8-27B (FP8)    | RTX PRO 4500 BW |                 16000 |                    512 |                     4 |                       0 | OOM            | OOM         | OOM     | OOM           | OOM                          | OOM                         |                    0 | OOM: requested weights/cache/state exceed memory budget |
| Qwen3.8-27B (BF16)   | L4              |                 16000 |                    512 |                     4 |                       0 | OOM            | OOM         | OOM     | OOM           | OOM                          | OOM                         |                    0 | OOM: requested weights/cache/state exceed memory budget |
| Qwen3.8-27B (BF16)   | RTX PRO 4500 BW |                 16000 |                    512 |                     4 |                       0 | OOM            | OOM         | OOM     | OOM           | OOM                          | OOM                         |                    0 | OOM: requested weights/cache/state exceed memory budget |
| Qwen3.8-27B (BF16)   | L40s            |                 16000 |                    512 |                     4 |                       0 | OOM            | OOM         | OOM     | OOM           | OOM                          | OOM                         |                    0 | OOM: requested weights/cache/state exceed memory budget |
[... 51 fitting rows omitted; run the command for the full table ...]
```

## GPU: frontier MoE models on an 8-GPU node
```bash
✗ python LLM_size_pef_calculator.py --profile via -g 8 --models "GLM,Kimi" --devices "RTX PRO 6000,H200,B200,B300"
 num_gpu = 8, prompt_size = 16000 tokens, response_size = 512 tokens
 n_concurrent_request = 4, profile = via, device = gpu, efficiency = theoretical
Estimate mode: theoretical (idealized hardware estimate).
Single-request output rate includes prefill; single-stream decode rate excludes prefill. Neither rate represents aggregate serving capacity.
Concurrency affects memory sizing only; Max Concurrent Req is a memory limit, not a latency target.
TPOT uses midpoint decode context. Attention/cache/state work is estimated; batching, prefix reuse, speculative decoding and queueing are not modeled.
Assumptions: parallelism=ideal, compute=bf16, memory utilization=0.9, KV sharding=auto.
Memory reserve: 10% of device memory (profile default). The memory-footprint table excludes the reserve; Max Concurrent Req and OOM decisions apply it. Latency is unchanged for workloads that fit.
Ideal pooling assumes perfect partitioning and zero communication; it is not a deployment topology.

******************** Estimate LLM Memory Footprint ********************
| Model           |   Input Size (tokens) |   Output Size (tokens) |   Concurrent Requests | KV Cache Size per Token   | Weights   | Memory Footprint   |
|-----------------+-----------------------+------------------------+-----------------------+---------------------------+-----------+--------------------|
| GLM-5.2 (FP8)   |                 16000 |                    512 |                     4 | 0.000044 GiB/token        | 756.0 GB  | 759.15 GB          |
| GLM-5.2 (NVFP4) |                 16000 |                    512 |                     4 | 0.000044 GiB/token        | 440.0 GB  | 443.15 GB          |
| Kimi-K3 (MXFP4) |                 16000 |                    512 |                     4 | 0.000026 GiB/token        | 1561.0 GB | 1564.55 GB         |

Warning GLM-5.2 (FP8) on 8x RTX PRO 6000 BW: OOM: requested weights/cache/state exceed memory budget.
Memory limit for this context: 0 concurrent requests.
Even one request does not fit; use more memory or a shorter supported context.

Warning Kimi-K3 (MXFP4) on 8x RTX PRO 6000 BW: OOM: requested weights/cache/state exceed memory budget.
Memory limit for this context: 0 concurrent requests.
Even one request does not fit; use more memory or a shorter supported context.

Warning Kimi-K3 (MXFP4) on 8x H200 SXM: OOM: requested weights/cache/state exceed memory budget.
Memory limit for this context: 0 concurrent requests.
Even one request does not fit; use more memory or a shorter supported context.

Warning Kimi-K3 (MXFP4) on 8x B200: OOM: requested weights/cache/state exceed memory budget.
Memory limit for this context: 0 concurrent requests.
Even one request does not fit; use more memory or a shorter supported context.

******************** Estimate LLM Capacity and Latency ********************
| Model           | GPU             |   Input Size (tokens) |   Output Size (tokens) |   Concurrent Requests |   Max # KV Cache Tokens | Prefill Time   | TPOT (ms)   | TTFT    | E2E Latency   | Single-request output rate   | Single-stream decode rate   |   Max Concurrent Req | Status                                                  |
|-----------------+-----------------+-----------------------+------------------------+-----------------------+-------------------------+----------------+-------------+---------+---------------+------------------------------+-----------------------------+----------------------+---------------------------------------------------------|
| GLM-5.2 (FP8)   | RTX PRO 6000 BW |                 16000 |                    512 |                     4 |                       0 | OOM            | OOM         | OOM     | OOM           | OOM                          | OOM                         |                    0 | OOM: requested weights/cache/state exceed memory budget |
| GLM-5.2 (FP8)   | H200 SXM        |                 16000 |                    512 |                     4 |                 5433962 | 0.013 ms       | 1.073 ms    | 0.208 s | 0.8 s         | 677.14 tokens/sec            | 931.67 tokens/sec           |                  329 | OK                                                      |
| GLM-5.2 (FP8)   | B200            |                 16000 |                    512 |                     4 |                11320754 | 0.006 ms       | 0.644 ms    | 0.091 s | 0.4 s         | 1217.39 tokens/sec           | 1552.79 tokens/sec          |                  685 | OK                                                      |
| GLM-5.2 (FP8)   | B300            |                 16000 |                    512 |                     4 |                27622641 | 0.006 ms       | 0.644 ms    | 0.091 s | 0.4 s         | 1217.39 tokens/sec           | 1552.79 tokens/sec          |                 1672 | OK                                                      |
| GLM-5.2 (NVFP4) | RTX PRO 6000 BW |                 16000 |                    512 |                     4 |                 5266247 | 0.026 ms       | 3.724 ms    | 0.413 s | 2.3 s         | 221.12 tokens/sec            | 268.52 tokens/sec           |                  318 | OK                                                      |
| GLM-5.2 (NVFP4) | H200 SXM        |                 16000 |                    512 |                     4 |                12058700 | 0.013 ms       | 1.239 ms    | 0.208 s | 0.8 s         | 608.83 tokens/sec            | 807.08 tokens/sec           |                  730 | OK                                                      |
| GLM-5.2 (NVFP4) | B200            |                 16000 |                    512 |                     4 |                17945492 | 0.006 ms       | 0.743 ms    | 0.092 s | 0.5 s         | 1085.95 tokens/sec           | 1345.14 tokens/sec          |                 1086 | OK                                                      |
| GLM-5.2 (NVFP4) | B300            |                 16000 |                    512 |                     4 |                34247379 | 0.006 ms       | 0.743 ms    | 0.092 s | 0.5 s         | 1085.95 tokens/sec           | 1345.14 tokens/sec          |                 2074 | OK                                                      |
| Kimi-K3 (MXFP4) | RTX PRO 6000 BW |                 16000 |                    512 |                     4 |                       0 | OOM            | OOM         | OOM     | OOM           | OOM                          | OOM                         |                    0 | OOM: requested weights/cache/state exceed memory budget |
| Kimi-K3 (MXFP4) | H200 SXM        |                 16000 |                    512 |                     4 |                       0 | OOM            | OOM         | OOM     | OOM           | OOM                          | OOM                         |                    0 | OOM: requested weights/cache/state exceed memory budget |
| Kimi-K3 (MXFP4) | B200            |                 16000 |                    512 |                     4 |                       0 | OOM            | OOM         | OOM     | OOM           | OOM                          | OOM                         |                    0 | OOM: requested weights/cache/state exceed memory budget |
| Kimi-K3 (MXFP4) | B300            |                 16000 |                    512 |                     4 |                18524667 | 0.014 ms       | 2.155 ms    | 0.223 s | 1.3 s         | 386.62 tokens/sec            | 464.11 tokens/sec           |                  578 | OK                                                      |
```
Note: Kimi-K3's weights (1,561 GB) do not fit on 8x RTX PRO 6000 (768 GB), 8x H200 (1,128 GB) or 8x B200 (1,440 GB); 8x B300 (2,304 GB) is the smallest single-node fit. GLM-5.2 FP8 (756 GB) fits the raw 768 GB of 8x RTX PRO 6000 (about 12 GB left for cache) but not the 691 GB available after the default 10% VIA runtime reserve; pass `--memory-utilization 1` to see the raw-budget result.

## CPU-only inference VMs
```bash
✗ python LLM_size_pef_calculator.py --profile via --device cpu -c 1 --models "gpt-oss,Q4_K_M"
 num_gpu = 1, prompt_size = 16000 tokens, response_size = 512 tokens
 n_concurrent_request = 1, profile = via, device = cpu, efficiency = calibrated
Estimate mode: calibrated CPU rates (always used for CPU devices).
CPU efficiency factors are unchanged and have not been refitted to corrected work estimates.
Single-request output rate includes prefill; single-stream decode rate excludes prefill. Neither rate represents aggregate serving capacity.
Concurrency affects memory sizing only; Max Concurrent Req is a memory limit, not a latency target.
TPOT uses midpoint decode context. Attention/cache/state work is estimated; batching, prefix reuse, speculative decoding and queueing are not modeled.
Assumptions: parallelism=ideal, compute=bf16, memory utilization=1, KV sharding=auto.
Memory reserve: 0% of device memory (profile default). The memory-footprint table excludes the reserve; Max Concurrent Req and OOM decisions apply it. Latency is unchanged for workloads that fit.

******************** Estimate LLM Memory Footprint ********************
| Model                |   Input Size (tokens) |   Output Size (tokens) |   Concurrent Requests | KV Cache Size per Token   | Weights   | Memory Footprint   |
|----------------------+-----------------------+------------------------+-----------------------+---------------------------+-----------+--------------------|
| gpt-oss-20b (MXFP4)  |                 16000 |                    512 |                     1 | 0.000023 GiB/token        | 13.0 GB   | 13.41 GB           |
| gpt-oss-120b (MXFP4) |                 16000 |                    512 |                     1 | 0.000034 GiB/token        | 63.0 GB   | 63.61 GB           |
| Qwen3.5-9B (Q4_K_M)  |                 16000 |                    512 |                     1 | 0.000031 GiB/token        | 5.8 GB    | 6.39 GB            |
| Qwen3.8-27B (Q4_K_M) |                 16000 |                    512 |                     1 | 0.000061 GiB/token        | 18.0 GB   | 19.23 GB           |

******************** Estimate LLM Capacity and Latency ********************
| Model                | GPU                              |   Input Size (tokens) |   Output Size (tokens) |   Concurrent Requests |   Max # KV Cache Tokens | Prefill Time   | TPOT (ms)   | TTFT      | E2E Latency   | Single-request output rate   | Single-stream decode rate   |   Max Concurrent Req | Status   | Accel   |
|----------------------+----------------------------------+-----------------------+------------------------+-----------------------+-------------------------+----------------+-------------+-----------+---------------+------------------------------+-----------------------------+----------------------+----------+---------|
| gpt-oss-20b (MXFP4)  | Xeon 3rd Gen 32c / 8ch DDR4-3200 |                 16000 |                    512 |                     1 |                 4679233 | 4.166 ms       | 111.447 ms  | 66.763 s  | 123.7 s       | 4.14 tokens/sec              | 8.97 tokens/sec             |                  281 | OK       | AVX-512 |
| gpt-oss-20b (MXFP4)  | Xeon 5th Gen 32c / 8ch DDR5-5600 |                 16000 |                    512 |                     1 |                 4679233 | 4.166 ms       | 63.684 ms   | 66.715 s  | 99.3 s        | 5.16 tokens/sec              | 15.70 tokens/sec            |                  281 | OK       | AVX-512 |
| gpt-oss-20b (MXFP4)  | Xeon 6700P 64c / 8ch DDR5-6400   |                 16000 |                    512 |                     1 |                 9887567 | 2.083 ms       | 55.724 ms   | 33.382 s  | 61.9 s        | 8.28 tokens/sec              | 17.95 tokens/sec            |                  594 | OK       | AVX-512 |
| gpt-oss-20b (MXFP4)  | Xeon 6900P 64c / 12ch DDR5-6400  |                 16000 |                    512 |                     1 |                 9887567 | 2.083 ms       | 37.149 ms   | 33.363 s  | 52.3 s        | 9.78 tokens/sec              | 26.92 tokens/sec            |                  594 | OK       | AVX-512 |
| gpt-oss-20b (MXFP4)  | EPYC 9004 32c / 12ch DDR5-4800   |                 16000 |                    512 |                     1 |                 4679233 | 4.166 ms       | 49.532 ms   | 66.701 s  | 92.0 s        | 5.56 tokens/sec              | 20.19 tokens/sec            |                  281 | OK       | AVX-512 |
| gpt-oss-20b (MXFP4)  | EPYC 9005 64c / 12ch DDR5-6400   |                 16000 |                    512 |                     1 |                 9887567 | 2.083 ms       | 37.149 ms   | 33.363 s  | 52.3 s        | 9.78 tokens/sec              | 26.92 tokens/sec            |                  594 | OK       | AVX-512 |
| gpt-oss-120b (MXFP4) | Xeon 3rd Gen 32c / 8ch DDR4-3200 |                 16000 |                    512 |                     1 |                 1763109 | 5.965 ms       | 152.001 ms  | 95.584 s  | 173.3 s       | 2.96 tokens/sec              | 6.58 tokens/sec             |                  105 | OK       | AVX-512 |
| gpt-oss-120b (MXFP4) | Xeon 5th Gen 32c / 8ch DDR5-5600 |                 16000 |                    512 |                     1 |                 1763109 | 5.965 ms       | 86.858 ms   | 95.519 s  | 139.9 s       | 3.66 tokens/sec              | 11.51 tokens/sec            |                  105 | OK       | AVX-512 |
| gpt-oss-120b (MXFP4) | Xeon 6700P 64c / 8ch DDR5-6400   |                 16000 |                    512 |                     1 |                 5235332 | 2.982 ms       | 76.001 ms   | 47.792 s  | 86.6 s        | 5.91 tokens/sec              | 13.16 tokens/sec            |                  314 | OK       | AVX-512 |
| gpt-oss-120b (MXFP4) | Xeon 6900P 64c / 12ch DDR5-6400  |                 16000 |                    512 |                     1 |                 5235332 | 2.982 ms       | 50.667 ms   | 47.767 s  | 73.7 s        | 6.95 tokens/sec              | 19.74 tokens/sec            |                  314 | OK       | AVX-512 |
| gpt-oss-120b (MXFP4) | EPYC 9004 32c / 12ch DDR5-4800   |                 16000 |                    512 |                     1 |                 1763109 | 5.965 ms       | 67.556 ms   | 95.500 s  | 130.0 s       | 3.94 tokens/sec              | 14.80 tokens/sec            |                  105 | OK       | AVX-512 |
| gpt-oss-120b (MXFP4) | EPYC 9005 64c / 12ch DDR5-6400   |                 16000 |                    512 |                     1 |                 5235332 | 2.982 ms       | 50.667 ms   | 47.767 s  | 73.7 s        | 6.95 tokens/sec              | 19.74 tokens/sec            |                  314 | OK       | AVX-512 |
| Qwen3.5-9B (Q4_K_M)  | Xeon 3rd Gen 32c / 8ch DDR4-3200 |                 16000 |                    512 |                     1 |                 3727722 | 9.055 ms       | 57.108 ms   | 144.933 s | 174.1 s       | 2.94 tokens/sec              | 17.51 tokens/sec            |                  206 | OK       | AVX-512 |
| Qwen3.5-9B (Q4_K_M)  | Xeon 5th Gen 32c / 8ch DDR5-5600 |                 16000 |                    512 |                     1 |                 3727722 | 4.527 ms       | 32.633 ms   | 72.471 s  | 89.1 s        | 5.74 tokens/sec              | 30.64 tokens/sec            |                  206 | OK       | AMX     |
| Qwen3.5-9B (Q4_K_M)  | Xeon 6700P 64c / 8ch DDR5-6400   |                 16000 |                    512 |                     1 |                 7633972 | 2.264 ms       | 28.554 ms   | 36.248 s  | 50.8 s        | 10.07 tokens/sec             | 35.02 tokens/sec            |                  423 | OK       | AMX     |
| Qwen3.5-9B (Q4_K_M)  | Xeon 6900P 64c / 12ch DDR5-6400  |                 16000 |                    512 |                     1 |                 7633972 | 2.264 ms       | 19.036 ms   | 36.238 s  | 46.0 s        | 11.14 tokens/sec             | 52.53 tokens/sec            |                  423 | OK       | AMX     |
| Qwen3.5-9B (Q4_K_M)  | EPYC 9004 32c / 12ch DDR5-4800   |                 16000 |                    512 |                     1 |                 3727722 | 9.055 ms       | 25.381 ms   | 144.901 s | 157.9 s       | 3.24 tokens/sec              | 39.40 tokens/sec            |                  206 | OK       | AVX-512 |
| Qwen3.5-9B (Q4_K_M)  | EPYC 9005 64c / 12ch DDR5-6400   |                 16000 |                    512 |                     1 |                 7633972 | 4.527 ms       | 19.036 ms   | 72.457 s  | 82.2 s        | 6.23 tokens/sec              | 52.53 tokens/sec            |                  423 | OK       | AVX-512 |
| Qwen3.8-27B (Q4_K_M) | Xeon 3rd Gen 32c / 8ch DDR4-3200 |                 16000 |                    512 |                     1 |                 1676177 | 27.164 ms      | 171.922 ms  | 434.800 s | 522.7 s       | 0.98 tokens/sec              | 5.82 tokens/sec             |                   89 | OK       | AVX-512 |
| Qwen3.8-27B (Q4_K_M) | Xeon 5th Gen 32c / 8ch DDR5-5600 |                 16000 |                    512 |                     1 |                 1676177 | 13.582 ms      | 98.241 ms   | 217.412 s | 267.6 s       | 1.91 tokens/sec              | 10.18 tokens/sec            |                   89 | OK       | AMX     |
| Qwen3.8-27B (Q4_K_M) | Xeon 6700P 64c / 8ch DDR5-6400   |                 16000 |                    512 |                     1 |                 3629302 | 6.791 ms       | 85.961 ms   | 108.743 s | 152.7 s       | 3.35 tokens/sec              | 11.63 tokens/sec            |                  193 | OK       | AMX     |
| Qwen3.8-27B (Q4_K_M) | Xeon 6900P 64c / 12ch DDR5-6400  |                 16000 |                    512 |                     1 |                 3629302 | 6.791 ms       | 57.307 ms   | 108.714 s | 138.0 s       | 3.71 tokens/sec              | 17.45 tokens/sec            |                  193 | OK       | AMX     |
| Qwen3.8-27B (Q4_K_M) | EPYC 9004 32c / 12ch DDR5-4800   |                 16000 |                    512 |                     1 |                 1676177 | 27.164 ms      | 76.410 ms   | 434.705 s | 473.8 s       | 1.08 tokens/sec              | 13.09 tokens/sec            |                   89 | OK       | AVX-512 |
| Qwen3.8-27B (Q4_K_M) | EPYC 9005 64c / 12ch DDR5-6400   |                 16000 |                    512 |                     1 |                 3629302 | 13.582 ms      | 57.307 ms   | 217.371 s | 246.7 s       | 2.08 tokens/sec              | 17.45 tokens/sec            |                  193 | OK       | AVX-512 |
```

## Sanity check against published measurements
```bash
✗ python validation/sanity_check.py
GPU MoE decode uses the dense decode factors (0.70 bandwidth efficiency + 0.5 ms/token);
prefill compute factors are unchanged and provisional. Kimi, GLM and B300 have no anchors.

== 1. Legacy-input comparison (original inputs, measurements and hold-out membership) ==
Decode anchors are context-free and prefill anchors use the projection-only cost; they do not
validate long-context attention, parallelism or serving throughput.
| Type        | Model                        | Device                          | Engine           | Metric        |   Measured |   Estimate |   Estimate / Measured | Source                               |
|-------------+------------------------------+---------------------------------+------------------+---------------+------------+------------+-----------------------+--------------------------------------|
| calibration | Llama-3.1-8B (BF16)          | H100 SXM                        | vLLM             | decode tok/s  |     146    |     136.56 |                  0.94 | FlashFormer, arXiv:2505.22758        |
| calibration | Qwen3.5-27B (Q4_K_M)         | RTX PRO 6000 BW                 | llama.cpp        | decode tok/s  |      60.9  |      61.9  |                  1.02 | hardware-corner.net RTX PRO 6000     |
| calibration | Qwen3-14B (Q4_K_M)           | RTX PRO 6000 BW                 | llama.cpp        | decode tok/s  |     114.4  |     116.95 |                  1.02 | hardware-corner.net RTX PRO 6000     |
| hold-out    | Qwen3-8B (Q4_K_M)            | RTX PRO 6000 BW                 | llama.cpp        | decode tok/s  |     173.7  |     201.1  |                  1.16 | hardware-corner.net RTX PRO 6000     |
| hold-out    | Qwen3-32B (Q4_K_M)           | RTX PRO 6000 BW                 | llama.cpp        | decode tok/s  |      54.9  |      54.91 |                  1    | hardware-corner.net RTX PRO 6000     |
| calibration | Qwen3.5-27B (Q4_K_M)         | RTX PRO 6000 BW                 | llama.cpp        | prefill tok/s |    3338.5  |    3518.52 |                  1.05 | hardware-corner.net RTX PRO 6000     |
| calibration | Qwen3-8B (Q4_K_M)            | RTX PRO 6000 BW                 | llama.cpp        | prefill tok/s |   10964.1  |   11585.4  |                  1.06 | hardware-corner.net RTX PRO 6000     |
| calibration | Qwen3-14B (Q4_K_M)           | RTX PRO 6000 BW                 | llama.cpp        | prefill tok/s |    6865.6  |    6418.92 |                  0.93 | hardware-corner.net RTX PRO 6000     |
| calibration | Qwen3-32B (Q4_K_M)           | RTX PRO 6000 BW                 | llama.cpp        | prefill tok/s |    3137.2  |    2896.34 |                  0.92 | hardware-corner.net RTX PRO 6000     |
| calibration | gpt-oss-20b (MXFP4)          | L4                              | vLLM 0.15        | decode tok/s  |      60    |      55.11 |                  0.92 | devforth.io L4/L40S/H100 gpt-oss-20b |
| outlier     | gpt-oss-20b (MXFP4)          | H100 SXM                        | vLLM 0.15        | decode tok/s  |     220    |     480.7  |                  2.19 | devforth.io L4/L40S/H100 gpt-oss-20b |
| calibration | gpt-oss-120b (MXFP4)         | RTX PRO 6000 BW                 | llama.cpp        | decode tok/s  |     210.1  |     201.12 |                  0.96 | hardware-corner.net RTX PRO 6000     |
| hold-out    | gpt-oss-20b (MXFP4)          | L40s                            | vLLM 0.15        | decode tok/s  |     160    |     150.89 |                  0.94 | devforth.io L4/L40S/H100 gpt-oss-20b |
| calibration | gpt-oss-20b (MXFP4)          | L4                              | vLLM 0.15        | TTFT s        |       3.5  |       6.11 |                  1.75 | devforth.io, ~13k-token prompt       |
| calibration | gpt-oss-20b (MXFP4)          | L40s                            | vLLM 0.15        | TTFT s        |       1.17 |       2.04 |                  1.75 | devforth.io, ~13k-token prompt       |
| calibration | gpt-oss-20b (MXFP4)          | H100 SXM                        | vLLM 0.15        | TTFT s        |       0.8  |       0.75 |                  0.93 | devforth.io, ~13k-token prompt       |
| calibration | gpt-oss-120b (MXFP4)         | RTX PRO 6000 BW                 | llama.cpp        | prefill tok/s |    4578.6  |    7352.94 |                  1.61 | hardware-corner.net RTX PRO 6000     |
| calibration | gpt-oss-20b (MXFP4)          | EPYC 9755 128c / 12ch DDR5-6400 | llama.cpp native | prefill tok/s |    1187.3  |    1173.33 |                  0.99 | AMD ZenDNN llama.cpp article, 2026   |
| calibration | Llama-3.1-8B (Q4_K_M)        | EPYC 9554 64c / 12ch DDR5-4800  | llama.cpp        | decode tok/s  |      49.8  |      51.51 |                  1.03 | ahelpme.com EPYC 9554                |
| calibration | Qwen3-Coder-30B-A3B (Q4_K_M) | EPYC 9554 64c / 12ch DDR5-4800  | llama.cpp        | decode tok/s  |      42.2  |      41.33 |                  0.98 | ahelpme.com EPYC 9554                |
| hold-out    | QwQ-32B (Q4_K_M)             | EPYC 9554 64c / 12ch DDR5-4800  | llama.cpp        | decode tok/s  |      14    |      12.77 |                  0.91 | ahelpme.com EPYC 9554                |
| hold-out    | Llama-3.3-70B (Q4_K_M)       | EPYC 9554 64c / 12ch DDR5-4800  | llama.cpp        | decode tok/s  |       7.14 |       5.96 |                  0.84 | ahelpme.com EPYC 9554                |
calibration     throughput estimate/measured: min 0.92, max 1.61 (n=13)
calibration     TTFT       estimate/measured: min 0.93, max 1.75 (n=3)
hold-out        throughput estimate/measured: min 0.84, max 1.16 (n=5)
outlier         throughput estimate/measured: min 2.19, max 2.19 (n=1)
The H100 gpt-oss-20b decode result is an outlier excluded from fitting; no hardware-specific
overhead was added to explain it.

== 2. Source-aware comparison (documented conditions; evaluation only, nothing tuned) ==
| Type            | Model                | Device                   | Engine               | Metric        |   Context/prompt |   Measured |   Estimate |   Estimate / Measured | Source                                               |
|-----------------+----------------------+--------------------------+----------------------+---------------+------------------+------------+------------+-----------------------+------------------------------------------------------|
| evaluation      | gpt-oss-20b (MXFP4)  | L4                       | vLLM 0.15.1 (FP8 KV) | TTFT s        |           13,000 |       3.5  |       6.11 |                  1.75 | devforth.io single-user ~10k-word prompt             |
| evaluation      | gpt-oss-20b (MXFP4)  | L40s                     | vLLM 0.15.1 (FP8 KV) | TTFT s        |           13,000 |       1.17 |       2.04 |                  1.75 | devforth.io single-user ~10k-word prompt             |
| evaluation      | gpt-oss-20b (MXFP4)  | H100 SXM                 | vLLM 0.15.1 (FP8 KV) | TTFT s        |           13,000 |       0.8  |       0.75 |                  0.93 | devforth.io single-user ~10k-word prompt             |
| evaluation      | gpt-oss-20b (MXFP4)  | L4                       | vLLM 0.15.1 (FP8 KV) | decode tok/s  |            7,694 |      62.39 |      53.75 |                  0.86 | devforth.io sequence-length table                    |
| evaluation      | gpt-oss-20b (MXFP4)  | L4                       | vLLM 0.15.1 (FP8 KV) | decode tok/s  |           14,338 |      60.73 |      52.65 |                  0.87 | devforth.io sequence-length table                    |
| evaluation      | gpt-oss-20b (MXFP4)  | L40s                     | vLLM 0.15.1 (FP8 KV) | decode tok/s  |            7,658 |     160.18 |     147.37 |                  0.92 | devforth.io sequence-length table                    |
| evaluation      | gpt-oss-20b (MXFP4)  | L40s                     | vLLM 0.15.1 (FP8 KV) | decode tok/s  |           14,289 |     156.02 |     144.51 |                  0.93 | devforth.io sequence-length table                    |
| outlier         | gpt-oss-20b (MXFP4)  | H100 SXM                 | vLLM 0.15.1 (FP8 KV) | decode tok/s  |            7,681 |     228.18 |     471.43 |                  2.07 | devforth.io sequence-length table                    |
| outlier         | gpt-oss-20b (MXFP4)  | H100 SXM                 | vLLM 0.15.1 (FP8 KV) | decode tok/s  |           14,310 |     223.6  |     463.84 |                  2.07 | devforth.io sequence-length table                    |
| validation-only | gpt-oss-20b (MXFP4)  | L4                       | vLLM 0.15.1 (FP8 KV) | decode tok/s  |           61,434 |      54.08 |      45.98 |                  0.85 | devforth.io sequence-length table (L4, long context) |
| evaluation      | Qwen3-8B (Q4_K_M)    | RTX PRO 6000 Workstation | llama.cpp            | decode tok/s  |            4,096 |     173.7  |     201.11 |                  1.16 | hardware-corner.net 4K tier                          |
| evaluation      | Qwen3-14B (Q4_K_M)   | RTX PRO 6000 Workstation | llama.cpp            | decode tok/s  |            4,096 |     114.4  |     121.68 |                  1.06 | hardware-corner.net 4K tier                          |
| evaluation      | Qwen3.5-27B (Q4_K_M) | RTX PRO 6000 Workstation | llama.cpp            | decode tok/s  |            4,096 |      60.9  |      67.02 |                  1.1  | hardware-corner.net 4K tier                          |
| evaluation      | Qwen3-32B (Q4_K_M)   | RTX PRO 6000 Workstation | llama.cpp            | decode tok/s  |            4,096 |      54.9  |      58.28 |                  1.06 | hardware-corner.net 4K tier                          |
| evaluation      | gpt-oss-120b (MXFP4) | RTX PRO 6000 Workstation | llama.cpp            | decode tok/s  |            4,096 |     210.1  |     216.72 |                  1.03 | hardware-corner.net 4K tier                          |
| evaluation      | Qwen3-8B (Q4_K_M)    | RTX PRO 6000 Workstation | llama.cpp            | prefill tok/s |            4,096 |   10964.1  |   10790.4  |                  0.98 | hardware-corner.net 4K tier                          |
| evaluation      | Qwen3-14B (Q4_K_M)   | RTX PRO 6000 Workstation | llama.cpp            | prefill tok/s |            4,096 |    6865.6  |    6074.53 |                  0.88 | hardware-corner.net 4K tier                          |
| evaluation      | Qwen3.5-27B (Q4_K_M) | RTX PRO 6000 Workstation | llama.cpp            | prefill tok/s |            4,096 |    3338.5  |    3452.63 |                  1.03 | hardware-corner.net 4K tier                          |
| evaluation      | Qwen3-32B (Q4_K_M)   | RTX PRO 6000 Workstation | llama.cpp            | prefill tok/s |            4,096 |    3137.2  |    2718.32 |                  0.87 | hardware-corner.net 4K tier                          |
| evaluation      | gpt-oss-120b (MXFP4) | RTX PRO 6000 Workstation | llama.cpp            | prefill tok/s |            4,096 |    4578.6  |    6918    |                  1.51 | hardware-corner.net 4K tier                          |
| evaluation      | Llama-3.1-8B         | H100 SXM                 | vLLM                 | decode tok/s  |              192 |     146    |     136.36 |                  0.93 | FlashFormer Table 6 (vLLM column)                    |
| evaluation      | Llama-3.1-8B         | H100 SXM                 | vLLM                 | decode tok/s  |            1,088 |     144    |     135.43 |                  0.94 | FlashFormer Table 6 (vLLM column)                    |
| evaluation      | Llama-3.1-8B         | H100 SXM                 | vLLM                 | decode tok/s  |            3,136 |     141    |     133.36 |                  0.95 | FlashFormer Table 6 (vLLM column)                    |
| evaluation      | Llama-3.1-8B         | H100 SXM                 | vLLM                 | decode tok/s  |            6,208 |     138    |     130.38 |                  0.94 | FlashFormer Table 6 (vLLM column)                    |
evaluation      throughput estimate/measured: min 0.86, max 1.51 (n=18)
evaluation      TTFT       estimate/measured: min 0.93, max 1.75 (n=3)
outlier         throughput estimate/measured: min 2.07, max 2.07 (n=2)
validation-only throughput estimate/measured: min 0.85, max 0.85 (n=1)
Counts are per comparison; observations shared with the legacy set are not pooled.

Sources and conditions (source-aware set):
- devforth.io single-user ~10k-word prompt: https://devforth.io/insights/self-hosted-gpt-real-response-time-token-throughput-and-cost-on-l4-l40s-and-h100-for-gpt-oss-20b/
    Narrative single-user run (~10k words, assumed ~13K tokens); response length not documented (128 assumed for the midpoint context).
- devforth.io sequence-length table: https://devforth.io/insights/self-hosted-gpt-real-response-time-token-throughput-and-cost-on-l4-l40s-and-h100-for-gpt-oss-20b/
    Rows used (input/thinking/output/effective length/TTFT) are recorded in each anchor's note in this file.
- devforth.io sequence-length table (L4, long context): https://devforth.io/insights/self-hosted-gpt-real-response-time-token-throughput-and-cost-on-l4-l40s-and-h100-for-gpt-oss-20b/
    Reported fields: input 61357, thinking 62, output 93, effective sequence length 61266, TTFT 29.654 s, decoding 54.082 tok/s, thinking time 1.103 s. The effective length is smaller than input_tokens, and the interval behind decoding_tokens/s is not stated, so the token-counting convention is ambiguous. TTFT is not scored here.
- hardware-corner.net 4K tier: https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-pro-6000-blackwell/
    4K-context tier; llama.cpp build 8288 is listed on the gpt-oss Q8_0 row. KV precision (assumed FP16) and generation length (assumed: decode evaluated at 4,096-token context) are not documented; 1,790 GB/s is the page's workstation bandwidth.
- FlashFormer Table 6 (vLLM column): https://arxiv.org/html/2505.22758v2
    Batch one, one H100 SXM, 128 generated tokens, CUDA graphs enabled; decoding speed only.
For throughput rows < 1 means conservative; for TTFT rows > 1 means conservative.
```
All outputs above were produced by running the exact commands shown against this branch's code (`tabulate` column padding may differ slightly on your machine). The single-GPU VIA performance table is excerpted, and timestamped CSV-save messages are omitted.
