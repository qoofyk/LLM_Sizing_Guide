# LLM_Sizing_Guide
A calculator to estimate the memory footprint, capacity, and latency based on your planned LLM application's requirements on different GPU architectures.
Blog: https://blogs.vmware.com/cloud-foundation/2024/09/25/llm-inference-sizing-and-performance-guidance/

# Usage
Prerequisite: `pip install -r requirements.txt`

Here are the Flags and their abbreviations for the script.
- num_gpu ('-g'): Specify the number of GPUs you plan to use for your deployment.
- prompt_sz ('-p'): Define the average size of the input prompts you expect to process.
- response_sz ('-r'): Set the average size of the responses you expect to generate.
- n_concurrent_req ('-c'): Indicate the number of concurrent requests you anticipate handling.
- profile ('--profile'): `via` sizes the VMware Intelligent Assist workload (defaults to 16000-token prompts, 512-token answers, 4 concurrent users) using the VIA model and GPU lists.
- device ('--device'): `gpu` (default) or `cpu` to size CPU-only inference VMs (llama.cpp engine in VCF Private AI Services).
- models / devices ('--models', '--devices'): comma-separated name filters, e.g. `--models "gpt-oss,Qwen3.8" --devices "H100,B200"`.
- efficiency ('--efficiency'): `theoretical` (peak hardware, the default) or `calibrated` (fitted to published batch-1 measurements; the default with `--profile via`). CPU devices always use calibrated values.

All GPU TFLOPS in `configs/gpu_specs.py` are dense FP16/BF16 Tensor Core figures. NVIDIA datasheets often show the 2:4 sparsity figure, which is 2x higher (e.g. H100 PCIe 1,513 sparse = 756.5 dense); inference does not use sparsity.

By modifying these variables, you can easily estimate the performance characteristics of your LLM deployment and make informed decisions about your infrastructure requirements.

# Example output
```bash
✗ python LLM_size_pef_calculator.py -g 4 -p 4096 -r 256 -c 10
 num_gpu = 4, prompt_size = 4096 tokens, response_size = 256 tokens
 n_concurrent_request = 10, profile = default, device = gpu, efficiency = theoretical
******************** Estimate LLM Memory Footprint ********************
| Model           | Input Size (tokens) | Output Size (tokens) | Concurrent Requests | KV Cache Size per Token | Weights  | Memory Footprint |
|-----------------+---------------------+----------------------+---------------------+-------------------------+----------+------------------|
| Llama-3.1-8B    | 4096                | 256                  | 10                  | 0.000122 GiB/token      | 16.0 GB  | 21.31 GB         |
| Llama-3.1-70B   | 4096                | 256                  | 10                  | 0.000305 GiB/token      | 140.0 GB | 153.28 GB        |
| Mistral-7B-v0.3 | 4096                | 256                  | 10                  | 0.000122 GiB/token      | 14.0 GB  | 19.31 GB         |
| Qwen2.5-14B     | 4096                | 256                  | 10                  | 0.000183 GiB/token      | 29.4 GB  | 37.37 GB         |
******************** Estimate LLM Capacity and Latency ********************
| Model           | GPU      | Input Size (tokens) | Output Size (tokens) | Concurrent Requests | Max # KV Cache Tokens | Prefill Time | TPOT (ms) | TTFT    | E2E Latency | Output Tokens Throughput | Max Concurrent Req |
|-----------------+----------+---------------------+----------------------+---------------------+-----------------------+--------------+-----------+---------+-------------+--------------------------+--------------------|
| Llama-3.1-8B    | L40s     | 4096                | 256                  | 10                  | 1441792               | 0.011 ms     | 4.630 ms  | 0.050 s | 1.2 s       | 208.05 tokens/sec        | 331                |
| Llama-3.1-8B    | H100 NVL | 4096                | 256                  | 10                  | 2949120               | 0.005 ms     | 1.026 ms  | 0.021 s | 0.3 s       | 907.24 tokens/sec        | 677                |
| Llama-3.1-8B    | H200 NVL | 4096                | 256                  | 10                  | 4489216               | 0.005 ms     | 0.833 ms  | 0.020 s | 0.2 s       | 1098.98 tokens/sec       | 1031               |
| Llama-3.1-8B    | MI300X   | 4096                | 256                  | 10                  | 6160384               | 0.003 ms     | 0.755 ms  | 0.013 s | 0.2 s       | 1244.27 tokens/sec       | 1415               |
| Llama-3.1-70B   | L40s     | 4096                | 256                  | 10                  | 170393                | 0.097 ms     | 40.509 ms | 0.437 s | 10.8 s      | 23.78 tokens/sec         | 39                 |
| Llama-3.1-70B   | H100 NVL | 4096                | 256                  | 10                  | 773324                | 0.042 ms     | 8.974 ms  | 0.181 s | 2.5 s       | 103.68 tokens/sec        | 177                |
| Llama-3.1-70B   | H200 NVL | 4096                | 256                  | 10                  | 1389363               | 0.042 ms     | 7.292 ms  | 0.179 s | 2.0 s       | 125.60 tokens/sec        | 319                |
| Llama-3.1-70B   | MI300X   | 4096                | 256                  | 10                  | 2057830               | 0.027 ms     | 6.604 ms  | 0.116 s | 1.8 s       | 142.20 tokens/sec        | 472                |
| Mistral-7B-v0.3 | L40s     | 4096                | 256                  | 10                  | 1458176               | 0.010 ms     | 4.051 ms  | 0.044 s | 1.1 s       | 237.78 tokens/sec        | 335                |
| Mistral-7B-v0.3 | H100 NVL | 4096                | 256                  | 10                  | 2965504               | 0.004 ms     | 0.897 ms  | 0.018 s | 0.2 s       | 1036.85 tokens/sec       | 681                |
| Mistral-7B-v0.3 | H200 NVL | 4096                | 256                  | 10                  | 4505600               | 0.004 ms     | 0.729 ms  | 0.018 s | 0.2 s       | 1255.98 tokens/sec       | 1035               |
| Mistral-7B-v0.3 | MI300X   | 4096                | 256                  | 10                  | 6176768               | 0.003 ms     | 0.660 ms  | 0.012 s | 0.2 s       | 1422.02 tokens/sec       | 1419               |
| Qwen2.5-14B     | L40s     | 4096                | 256                  | 10                  | 888012                | 0.020 ms     | 8.507 ms  | 0.092 s | 2.3 s       | 113.23 tokens/sec        | 204                |
| Qwen2.5-14B     | H100 NVL | 4096                | 256                  | 10                  | 1892898               | 0.009 ms     | 1.885 ms  | 0.038 s | 0.5 s       | 493.74 tokens/sec        | 434                |
| Qwen2.5-14B     | H200 NVL | 4096                | 256                  | 10                  | 2919628               | 0.009 ms     | 1.531 ms  | 0.038 s | 0.4 s       | 598.08 tokens/sec        | 670                |
| Qwen2.5-14B     | MI300X   | 4096                | 256                  | 10                  | 4033740               | 0.006 ms     | 1.387 ms  | 0.024 s | 0.4 s       | 677.15 tokens/sec        | 926                |
```

Note: TTFT now includes the full prompt prefill (`prompt_size x prefill time per token + TPOT`). Earlier versions reported only one token's prefill time, which understated TTFT.

# Sizing VMware Intelligent Assist (VIA)
Model specs support MoE (active vs. total parameters), quantized checkpoints, hybrid/linear attention (only full-attention layers keep KV) and MLA-style caches; see `configs/model_specs.py`. Architecture values were checked against each model's `config.json`. CPU VM assumptions (Intel Xeon with and without AMX, AMD EPYC) and their calibration sources are in `configs/cpu_specs.py`. llama.cpp only uses AMX for Q4_0/Q4_1/Q8_0/Q4_K/Q5_K/Q6_K/IQ4_XS GGUFs, so the CPU table's `Accel` column shows which kernel path each estimate assumes. GLM-5.2 and Kimi-K3 cache layouts are estimates derived from their configs.

`--profile via` uses the calibrated efficiency model. `validation/sanity_check.py` compares it against published measurements (see the last section).

## GPU: gpt-oss and Qwen on a single GPU
```bash
✗ python LLM_size_pef_calculator.py --profile via -g 1 --models "gpt-oss,Qwen"
 num_gpu = 1, prompt_size = 16000 tokens, response_size = 512 tokens
 n_concurrent_request = 4, profile = via, device = gpu, efficiency = calibrated

******************** Estimate LLM Memory Footprint ********************
| Model                | Input Size (tokens) | Output Size (tokens) | Concurrent Requests | KV Cache Size per Token | Weights | Memory Footprint |
|----------------------+---------------------+----------------------+---------------------+-------------------------+---------+------------------|
| gpt-oss-20b (MXFP4)  | 16000               | 512                  | 4                   | 0.000023 GiB/token      | 13.0 GB | 14.51 GB         |
| gpt-oss-120b (MXFP4) | 16000               | 512                  | 4                   | 0.000034 GiB/token      | 63.0 GB | 65.27 GB         |
| Qwen3.5-9B (Q4_K_M)  | 16000               | 512                  | 4                   | 0.000031 GiB/token      | 5.8 GB  | 8.02 GB          |
| Qwen3.5-9B (BF16)    | 16000               | 512                  | 4                   | 0.000031 GiB/token      | 18.0 GB | 20.22 GB         |
| Qwen3.8-27B (Q4_K_M) | 16000               | 512                  | 4                   | 0.000061 GiB/token      | 18.0 GB | 22.63 GB         |
| Qwen3.8-27B (FP8)    | 16000               | 512                  | 4                   | 0.000061 GiB/token      | 29.0 GB | 33.63 GB         |
| Qwen3.8-27B (BF16)   | 16000               | 512                  | 4                   | 0.000061 GiB/token      | 56.0 GB | 60.63 GB         |

!!!! Warning gpt-oss-120b (MXFP4): weights (63 GB) do not fit in 1x L4 (24 GB)

!!!! Warning gpt-oss-120b (MXFP4): weights (63 GB) do not fit in 1x RTX PRO 4500 BW (32 GB)

!!!! Warning gpt-oss-120b (MXFP4): weights (63 GB) do not fit in 1x L40s (48 GB)

!!!! Warning Qwen3.8-27B (FP8): weights (29 GB) do not fit in 1x L4 (24 GB)

!!!! Warning Qwen3.8-27B (FP8): n_concurrent_request=4 is TOO Large!!!
Causing OOM with ISL=16000 and OSL=512 using 1x RTX PRO 4500 BW
Max number of concurrent requests that can be set for this use case: 2
Ignore the rows in the following table which contains RTX PRO 4500 BW and rerun the calculator with this number

!!!! Warning Qwen3.8-27B (BF16): weights (56 GB) do not fit in 1x L4 (24 GB)

!!!! Warning Qwen3.8-27B (BF16): weights (56 GB) do not fit in 1x RTX PRO 4500 BW (32 GB)

!!!! Warning Qwen3.8-27B (BF16): weights (56 GB) do not fit in 1x L40s (48 GB)

******************** Estimate LLM Capacity and Latency ********************
| Model                | GPU             | Input Size (tokens) | Output Size (tokens) | Concurrent Requests | Max # KV Cache Tokens | Prefill Time | TPOT (ms) | TTFT     | E2E Latency | Output Tokens Throughput | Max Concurrent Req |
|----------------------+-----------------+---------------------+----------------------+---------------------+-----------------------+--------------+-----------+----------+-------------+--------------------------+--------------------|
| gpt-oss-20b (MXFP4)  | L4              | 16000               | 512                  | 4                   | 480597                | 0.397 ms     | 16.928 ms | 6.364 s  | 15.0 s      | 34.10 tokens/sec         | 29                 |
| gpt-oss-20b (MXFP4)  | RTX PRO 4500 BW | 16000               | 512                  | 4                   | 830122                | 0.236 ms     | 7.598 ms  | 3.791 s  | 7.7 s       | 66.72 tokens/sec         | 50                 |
| gpt-oss-20b (MXFP4)  | H100 SXM        | 16000               | 512                  | 4                   | 2927274               | 0.049 ms     | 3.337 ms  | 0.779 s  | 2.5 s       | 206.07 tokens/sec        | 177                |
| gpt-oss-120b (MXFP4) | L4              | 16000               | 512                  | 4                   | 0                     | OOM          | OOM       | OOM      | OOM         | OOM                      | OOM                |
| gpt-oss-120b (MXFP4) | RTX PRO 6000 BW | 16000               | 512                  | 4                   | 961194                | 0.136 ms     | 5.445 ms  | 2.181 s  | 5.0 s       | 103.15 tokens/sec        | 58                 |
| gpt-oss-120b (MXFP4) | H100 SXM        | 16000               | 512                  | 4                   | 495160                | 0.069 ms     | 3.642 ms  | 1.103 s  | 3.0 s       | 172.72 tokens/sec        | 29                 |
| Qwen3.8-27B (FP8)    | RTX PRO 4500 BW | 16000               | 512                  | 4                   | 49152                 | 0.700 ms     | 52.286 ms | 11.253 s | 38.0 s      | 13.48 tokens/sec         | 2                  |
| Qwen3.8-27B (FP8)    | RTX PRO 6000 BW | 16000               | 512                  | 4                   | 1097728               | 0.284 ms     | 26.441 ms | 4.574 s  | 18.1 s      | 28.31 tokens/sec         | 57                 |
| Qwen3.8-27B (FP8)    | H100 SXM        | 16000               | 512                  | 4                   | 835584                | 0.144 ms     | 12.867 ms | 2.311 s  | 8.9 s       | 57.62 tokens/sec         | 44                 |
| Qwen3.8-27B (BF16)   | RTX PRO 6000 BW | 16000               | 512                  | 4                   | 655360                | 0.284 ms     | 50.594 ms | 4.598 s  | 30.5 s      | 16.81 tokens/sec         | 34                 |
[... 60 more rows: 7 models x 10 GPUs = 70 rows in total; run the command for the full table ...]
```

## GPU: frontier MoE models on an 8-GPU node
```bash
✗ python LLM_size_pef_calculator.py --profile via -g 8 --models "GLM,Kimi" --devices "RTX PRO 6000,H200,B200,B300"
 num_gpu = 8, prompt_size = 16000 tokens, response_size = 512 tokens
 n_concurrent_request = 4, profile = via, device = gpu, efficiency = calibrated

******************** Estimate LLM Memory Footprint ********************
| Model           | Input Size (tokens) | Output Size (tokens) | Concurrent Requests | KV Cache Size per Token | Weights   | Memory Footprint |
|-----------------+---------------------+----------------------+---------------------+-------------------------+-----------+------------------|
| GLM-5.2 (FP8)   | 16000               | 512                  | 4                   | 0.000044 GiB/token      | 756.0 GB  | 758.93 GB        |
| GLM-5.2 (NVFP4) | 16000               | 512                  | 4                   | 0.000044 GiB/token      | 440.0 GB  | 442.93 GB        |
| Kimi-K3 (MXFP4) | 16000               | 512                  | 4                   | 0.000026 GiB/token      | 1561.0 GB | 1564.42 GB       |

!!!! Warning Kimi-K3 (MXFP4): weights (1561 GB) do not fit in 8x RTX PRO 6000 BW (768 GB)

!!!! Warning Kimi-K3 (MXFP4): weights (1561 GB) do not fit in 8x H200 SXM (1128 GB)

!!!! Warning Kimi-K3 (MXFP4): weights (1561 GB) do not fit in 8x B200 (1440 GB)

******************** Estimate LLM Capacity and Latency ********************
| Model           | GPU             | Input Size (tokens) | Output Size (tokens) | Concurrent Requests | Max # KV Cache Tokens | Prefill Time | TPOT (ms) | TTFT    | E2E Latency | Output Tokens Throughput | Max Concurrent Req |
|-----------------+-----------------+---------------------+----------------------+---------------------+-----------------------+--------------+-----------+---------+-------------+--------------------------+--------------------|
| GLM-5.2 (FP8)   | RTX PRO 6000 BW | 16000               | 512                  | 4                   | 270691                | 0.133 ms     | 8.363 ms  | 2.142 s | 6.4 s       | 79.81 tokens/sec         | 16                 |
| GLM-5.2 (FP8)   | H200 SXM        | 16000               | 512                  | 4                   | 8391427               | 0.067 ms     | 4.117 ms  | 1.082 s | 3.2 s       | 160.71 tokens/sec        | 508                |
| GLM-5.2 (FP8)   | B200            | 16000               | 512                  | 4                   | 15429399              | 0.030 ms     | 3.270 ms  | 0.477 s | 2.1 s       | 238.32 tokens/sec        | 934                |
| GLM-5.2 (FP8)   | B300            | 16000               | 512                  | 4                   | 34919166              | 0.030 ms     | 3.270 ms  | 0.477 s | 2.1 s       | 238.32 tokens/sec        | 2114               |
| GLM-5.2 (NVFP4) | RTX PRO 6000 BW | 16000               | 512                  | 4                   | 7398893               | 0.133 ms     | 5.703 ms  | 2.139 s | 5.1 s       | 101.32 tokens/sec        | 448                |
| GLM-5.2 (NVFP4) | H200 SXM        | 16000               | 512                  | 4                   | 15519629              | 0.067 ms     | 3.232 ms  | 1.081 s | 2.7 s       | 187.35 tokens/sec        | 939                |
| GLM-5.2 (NVFP4) | B200            | 16000               | 512                  | 4                   | 22557601              | 0.030 ms     | 2.739 ms  | 0.477 s | 1.9 s       | 272.84 tokens/sec        | 1366               |
| GLM-5.2 (NVFP4) | B300            | 16000               | 512                  | 4                   | 42047368              | 0.030 ms     | 2.739 ms  | 0.477 s | 1.9 s       | 272.84 tokens/sec        | 2546               |
| Kimi-K3 (MXFP4) | RTX PRO 6000 BW | 16000               | 512                  | 4                   | 0                     | OOM          | OOM       | OOM     | OOM         | OOM                      | OOM                |
| Kimi-K3 (MXFP4) | H200 SXM        | 16000               | 512                  | 4                   | 0                     | OOM          | OOM       | OOM     | OOM         | OOM                      | OOM                |
| Kimi-K3 (MXFP4) | B200            | 16000               | 512                  | 4                   | 0                     | OOM          | OOM       | OOM     | OOM         | OOM                      | OOM                |
| Kimi-K3 (MXFP4) | B300            | 16000               | 512                  | 4                   | 28855258              | 0.077 ms     | 3.812 ms  | 1.236 s | 3.2 s       | 160.79 tokens/sec        | 868                |
```
Note: Kimi-K3's weights (1,561 GB) do not fit on 8x RTX PRO 6000 (768 GB), 8x H200 (1,128 GB) or 8x B200 (1,440 GB); 8x B300 (2,304 GB) is the smallest single-node fit. GLM-5.2 FP8 (756 GB) technically fits 8x RTX PRO 6000 but leaves only ~12 GB for KV cache.

## CPU-only inference VMs
```bash
✗ python LLM_size_pef_calculator.py --profile via --device cpu -c 1 --models "gpt-oss,Q4_K_M"
 num_gpu = 1, prompt_size = 16000 tokens, response_size = 512 tokens
 n_concurrent_request = 1, profile = via, device = cpu, efficiency = calibrated

******************** Estimate LLM Memory Footprint ********************
| Model                | Input Size (tokens) | Output Size (tokens) | Concurrent Requests | KV Cache Size per Token | Weights | Memory Footprint |
|----------------------+---------------------+----------------------+---------------------+-------------------------+---------+------------------|
| gpt-oss-20b (MXFP4)  | 16000               | 512                  | 1                   | 0.000023 GiB/token      | 13.0 GB | 13.38 GB         |
| gpt-oss-120b (MXFP4) | 16000               | 512                  | 1                   | 0.000034 GiB/token      | 63.0 GB | 63.57 GB         |
| Qwen3.5-9B (Q4_K_M)  | 16000               | 512                  | 1                   | 0.000031 GiB/token      | 5.8 GB  | 6.35 GB          |
| Qwen3.8-27B (Q4_K_M) | 16000               | 512                  | 1                   | 0.000061 GiB/token      | 18.0 GB | 19.16 GB         |

******************** Estimate LLM Capacity and Latency ********************
| Model                | GPU                              | Input Size (tokens) | Output Size (tokens) | Concurrent Requests | Max # KV Cache Tokens | Prefill Time | TPOT (ms)  | TTFT      | E2E Latency | Output Tokens Throughput | Max Concurrent Req | Accel   |
|----------------------+----------------------------------+---------------------+----------------------+---------------------+-----------------------+--------------+------------+-----------+-------------+--------------------------+--------------------+---------|
| gpt-oss-20b (MXFP4)  | Xeon 3rd Gen 32c / 8ch DDR4-3200 | 16000               | 512                  | 1                   | 5024426               | 3.409 ms     | 60.743 ms  | 54.606 s  | 85.6 s      | 5.98 tokens/sec          | 304                | AVX-512 |
| gpt-oss-20b (MXFP4)  | Xeon 5th Gen 32c / 8ch DDR5-5600 | 16000               | 512                  | 1                   | 5024426               | 3.409 ms     | 34.710 ms  | 54.580 s  | 72.3 s      | 7.08 tokens/sec          | 304                | AVX-512 |
| gpt-oss-20b (MXFP4)  | Xeon 6700P 64c / 8ch DDR5-6400   | 16000               | 512                  | 1                   | 10616832              | 1.705 ms     | 30.372 ms  | 27.303 s  | 42.8 s      | 11.96 tokens/sec         | 642                | AVX-512 |
| gpt-oss-20b (MXFP4)  | Xeon 6900P 64c / 12ch DDR5-6400  | 16000               | 512                  | 1                   | 10616832              | 1.705 ms     | 20.248 ms  | 27.293 s  | 37.6 s      | 13.60 tokens/sec         | 642                | AVX-512 |
| gpt-oss-20b (MXFP4)  | EPYC 9004 32c / 12ch DDR5-4800   | 16000               | 512                  | 1                   | 5024426               | 3.409 ms     | 26.997 ms  | 54.572 s  | 68.4 s      | 7.49 tokens/sec          | 304                | AVX-512 |
| gpt-oss-20b (MXFP4)  | EPYC 9005 64c / 12ch DDR5-6400   | 16000               | 512                  | 1                   | 10616832              | 1.705 ms     | 20.248 ms  | 27.293 s  | 37.6 s      | 13.60 tokens/sec         | 642                | AVX-512 |
| gpt-oss-120b (MXFP4) | Xeon 3rd Gen 32c / 8ch DDR4-3200 | 16000               | 512                  | 1                   | 1893262               | 4.830 ms     | 74.622 ms  | 77.347 s  | 115.5 s     | 4.43 tokens/sec          | 114                | AVX-512 |
| gpt-oss-120b (MXFP4) | Xeon 5th Gen 32c / 8ch DDR5-5600 | 16000               | 512                  | 1                   | 1893262               | 4.830 ms     | 42.641 ms  | 77.315 s  | 99.1 s      | 5.17 tokens/sec          | 114                | AVX-512 |
| gpt-oss-120b (MXFP4) | Xeon 6700P 64c / 8ch DDR5-6400   | 16000               | 512                  | 1                   | 5621532               | 2.415 ms     | 37.311 ms  | 38.674 s  | 57.7 s      | 8.87 tokens/sec          | 340                | AVX-512 |
| gpt-oss-120b (MXFP4) | Xeon 6900P 64c / 12ch DDR5-6400  | 16000               | 512                  | 1                   | 5621532               | 2.415 ms     | 24.874 ms  | 38.661 s  | 51.4 s      | 9.97 tokens/sec          | 340                | AVX-512 |
| gpt-oss-120b (MXFP4) | EPYC 9004 32c / 12ch DDR5-4800   | 16000               | 512                  | 1                   | 1893262               | 4.830 ms     | 33.165 ms  | 77.306 s  | 94.3 s      | 5.43 tokens/sec          | 114                | AVX-512 |
| gpt-oss-120b (MXFP4) | EPYC 9005 64c / 12ch DDR5-6400   | 16000               | 512                  | 1                   | 5621532               | 2.415 ms     | 24.874 ms  | 38.661 s  | 51.4 s      | 9.97 tokens/sec          | 340                | AVX-512 |
| Qwen3.5-9B (Q4_K_M)  | Xeon 3rd Gen 32c / 8ch DDR4-3200 | 16000               | 512                  | 1                   | 4004249               | 8.523 ms     | 51.491 ms  | 136.415 s | 162.7 s     | 3.15 tokens/sec          | 220                | AVX-512 |
| Qwen3.5-9B (Q4_K_M)  | Xeon 5th Gen 32c / 8ch DDR5-5600 | 16000               | 512                  | 1                   | 4004249               | 4.261 ms     | 29.424 ms  | 68.211 s  | 83.2 s      | 6.15 tokens/sec          | 220                | AMX     |
| Qwen3.5-9B (Q4_K_M)  | Xeon 6700P 64c / 8ch DDR5-6400   | 16000               | 512                  | 1                   | 8198553               | 2.131 ms     | 25.746 ms  | 34.117 s  | 47.3 s      | 10.83 tokens/sec         | 451                | AMX     |
| Qwen3.5-9B (Q4_K_M)  | Xeon 6900P 64c / 12ch DDR5-6400  | 16000               | 512                  | 1                   | 8198553               | 2.131 ms     | 17.164 ms  | 34.108 s  | 42.9 s      | 11.94 tokens/sec         | 451                | AMX     |
| Qwen3.5-9B (Q4_K_M)  | EPYC 9004 32c / 12ch DDR5-4800   | 16000               | 512                  | 1                   | 4004249               | 8.523 ms     | 22.885 ms  | 136.387 s | 148.1 s     | 3.46 tokens/sec          | 220                | AVX-512 |
| Qwen3.5-9B (Q4_K_M)  | EPYC 9005 64c / 12ch DDR5-6400   | 16000               | 512                  | 1                   | 8198553               | 4.261 ms     | 17.164 ms  | 68.199 s  | 77.0 s      | 6.65 tokens/sec          | 451                | AVX-512 |
| Qwen3.8-27B (Q4_K_M) | Xeon 3rd Gen 32c / 8ch DDR4-3200 | 16000               | 512                  | 1                   | 1802240               | 25.568 ms    | 159.801 ms | 409.251 s | 490.9 s     | 1.04 tokens/sec          | 95                 | AVX-512 |
| Qwen3.8-27B (Q4_K_M) | Xeon 5th Gen 32c / 8ch DDR5-5600 | 16000               | 512                  | 1                   | 1802240               | 12.784 ms    | 91.315 ms  | 204.637 s | 251.3 s     | 2.04 tokens/sec          | 95                 | AMX     |
| Qwen3.8-27B (Q4_K_M) | Xeon 6700P 64c / 8ch DDR5-6400   | 16000               | 512                  | 1                   | 3899392               | 6.392 ms     | 79.901 ms  | 102.353 s | 143.2 s     | 3.58 tokens/sec          | 205                | AMX     |
| Qwen3.8-27B (Q4_K_M) | Xeon 6900P 64c / 12ch DDR5-6400  | 16000               | 512                  | 1                   | 3899392               | 6.392 ms     | 53.267 ms  | 102.326 s | 129.5 s     | 3.95 tokens/sec          | 205                | AMX     |
| Qwen3.8-27B (Q4_K_M) | EPYC 9004 32c / 12ch DDR5-4800   | 16000               | 512                  | 1                   | 1802240               | 25.568 ms    | 71.023 ms  | 409.162 s | 445.5 s     | 1.15 tokens/sec          | 95                 | AVX-512 |
| Qwen3.8-27B (Q4_K_M) | EPYC 9005 64c / 12ch DDR5-6400   | 16000               | 512                  | 1                   | 3899392               | 12.784 ms    | 53.267 ms  | 204.599 s | 231.8 s     | 2.21 tokens/sec          | 205                | AVX-512 |
```

## Sanity check against published measurements
```bash
✗ python validation/sanity_check.py
| Type        | Model                        | Device                          | Engine           | Metric        | Measured  | Estimate  | Estimate / Measured | Source                               |
|-------------+------------------------------+---------------------------------+------------------+---------------+-----------+-----------+---------------------+--------------------------------------|
| calibration | Llama-3.1-8B (BF16)          | H100 SXM                        | vLLM             | decode tok/s  | 146.00    | 136.56    | 0.94                | FlashFormer, arXiv:2505.22758        |
| calibration | Qwen3.5-27B (Q4_K_M)         | RTX PRO 6000 BW                 | llama.cpp        | decode tok/s  | 60.90     | 61.90     | 1.02                | hardware-corner.net RTX PRO 6000     |
| calibration | Qwen3-14B (Q4_K_M)           | RTX PRO 6000 BW                 | llama.cpp        | decode tok/s  | 114.40    | 116.95    | 1.02                | hardware-corner.net RTX PRO 6000     |
| hold-out    | Qwen3-8B (Q4_K_M)            | RTX PRO 6000 BW                 | llama.cpp        | decode tok/s  | 173.70    | 201.10    | 1.16                | hardware-corner.net RTX PRO 6000     |
| hold-out    | Qwen3-32B (Q4_K_M)           | RTX PRO 6000 BW                 | llama.cpp        | decode tok/s  | 54.90     | 54.91     | 1.00                | hardware-corner.net RTX PRO 6000     |
| calibration | Qwen3.5-27B (Q4_K_M)         | RTX PRO 6000 BW                 | llama.cpp        | prefill tok/s | 3,338.50  | 3,518.52  | 1.05                | hardware-corner.net RTX PRO 6000     |
| calibration | Qwen3-8B (Q4_K_M)            | RTX PRO 6000 BW                 | llama.cpp        | prefill tok/s | 10,964.10 | 11,585.37 | 1.06                | hardware-corner.net RTX PRO 6000     |
| calibration | Qwen3-14B (Q4_K_M)           | RTX PRO 6000 BW                 | llama.cpp        | prefill tok/s | 6,865.60  | 6,418.92  | 0.93                | hardware-corner.net RTX PRO 6000     |
| calibration | Qwen3-32B (Q4_K_M)           | RTX PRO 6000 BW                 | llama.cpp        | prefill tok/s | 3,137.20  | 2,896.34  | 0.92                | hardware-corner.net RTX PRO 6000     |
| calibration | gpt-oss-20b (MXFP4)          | L4                              | vLLM 0.15        | decode tok/s  | 60.00     | 59.07     | 0.98                | devforth.io L4/L40S/H100 gpt-oss-20b |
| calibration | gpt-oss-20b (MXFP4)          | H100 SXM                        | vLLM 0.15        | decode tok/s  | 220.00    | 299.68    | 1.36                | devforth.io L4/L40S/H100 gpt-oss-20b |
| calibration | gpt-oss-120b (MXFP4)         | RTX PRO 6000 BW                 | llama.cpp        | decode tok/s  | 210.10    | 183.65    | 0.87                | hardware-corner.net RTX PRO 6000     |
| hold-out    | gpt-oss-20b (MXFP4)          | L40s                            | vLLM 0.15        | decode tok/s  | 160.00    | 139.21    | 0.87                | devforth.io L4/L40S/H100 gpt-oss-20b |
| calibration | gpt-oss-20b (MXFP4)          | L4                              | vLLM 0.15        | TTFT s        | 3.50      | 5.17      | 1.48                | devforth.io, ~13k-token prompt       |
| calibration | gpt-oss-20b (MXFP4)          | L40s                            | vLLM 0.15        | TTFT s        | 1.17      | 1.73      | 1.48                | devforth.io, ~13k-token prompt       |
| calibration | gpt-oss-20b (MXFP4)          | H100 SXM                        | vLLM 0.15        | TTFT s        | 0.80      | 0.63      | 0.79                | devforth.io, ~13k-token prompt       |
| calibration | gpt-oss-120b (MXFP4)         | RTX PRO 6000 BW                 | llama.cpp        | prefill tok/s | 4,578.60  | 7,352.94  | 1.61                | hardware-corner.net RTX PRO 6000     |
| calibration | gpt-oss-20b (MXFP4)          | EPYC 9755 128c / 12ch DDR5-6400 | llama.cpp native | prefill tok/s | 1,187.30  | 1,173.33  | 0.99                | AMD ZenDNN llama.cpp article, 2026   |
| calibration | Llama-3.1-8B (Q4_K_M)        | EPYC 9554 64c / 12ch DDR5-4800  | llama.cpp        | decode tok/s  | 49.80     | 51.51     | 1.03                | ahelpme.com EPYC 9554                |
| calibration | Qwen3-Coder-30B-A3B (Q4_K_M) | EPYC 9554 64c / 12ch DDR5-4800  | llama.cpp        | decode tok/s  | 42.20     | 41.33     | 0.98                | ahelpme.com EPYC 9554                |
| hold-out    | QwQ-32B (Q4_K_M)             | EPYC 9554 64c / 12ch DDR5-4800  | llama.cpp        | decode tok/s  | 14.00     | 12.77     | 0.91                | ahelpme.com EPYC 9554                |
| hold-out    | Llama-3.3-70B (Q4_K_M)       | EPYC 9554 64c / 12ch DDR5-4800  | llama.cpp        | decode tok/s  | 7.14      | 5.96      | 0.84                | ahelpme.com EPYC 9554                |
calibration estimate/measured: min 0.79, max 1.61 (n=17)
hold-out    estimate/measured: min 0.84, max 1.16 (n=5)
For throughput rows < 1 means conservative; for TTFT rows > 1 means conservative.
```
All outputs above were produced by running the exact commands shown against this branch's code (`tabulate` column padding may differ slightly on your machine).
