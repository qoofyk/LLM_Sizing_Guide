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

By modifying these variables, you can easily estimate the performance characteristics of your LLM deployment and make informed decisions about your infrastructure requirements.

# Example output
```bash
✗ python LLM_size_pef_calculator.py -g 4 -p 4096 -r 256 -c 10
 num_gpu = 4, prompt_size = 4096 tokens, response_size = 256 tokens
 n_concurrent_request = 10, profile = default, device = gpu
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
Model specs support MoE (active vs. total parameters), quantized checkpoints, hybrid/linear attention (only full-attention layers keep KV) and MLA-style caches; see `configs/model_specs.py`. CPU VM assumptions and calibration sources are in `configs/cpu_specs.py`. Values marked as estimates in the configs (e.g. GLM-5.2 and Kimi-K3 KV cache) are not published by the model vendors.

## GPU: gpt-oss and Qwen on a single GPU
```bash
✗ python LLM_size_pef_calculator.py --profile via -g 1 --models "gpt-oss,Qwen"
 num_gpu = 1, prompt_size = 16000 tokens, response_size = 512 tokens
 n_concurrent_request = 4, profile = via, device = gpu

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
!!!! Warning Qwen3.8-27B (BF16): weights (56 GB) do not fit in 1x L40s (48 GB)
[... same warning for every model/GPU pair whose weights alone exceed that GPU's memory ...]

******************** Estimate LLM Capacity and Latency ********************
| Model                | GPU           | ... | Prefill Time | TPOT (ms) | TTFT    | E2E Latency | Output Tokens Throughput | Max Concurrent Req |
|----------------------+---------------+ ... +--------------+-----------+---------+-------------+--------------------------+--------------------|
| gpt-oss-20b (MXFP4)  | L4            | ... | 0.060 ms     | 7.464 ms  | 0.960 s | 4.8 s       | 107.25 tokens/sec        | 29                 |
| gpt-oss-20b (MXFP4)  | L40s          | ... | 0.020 ms     | 2.592 ms  | 0.321 s | 1.6 s       | 311.21 tokens/sec        | 92                 |
| gpt-oss-20b (MXFP4)  | H100 SXM      | ... | 0.007 ms     | 0.668 ms  | 0.117 s | 0.5 s       | 1116.30 tokens/sec       | 177                |
| gpt-oss-20b (MXFP4)  | B200          | ... | 0.003 ms     | 0.280 ms  | 0.051 s | 0.2 s       | 2632.24 tokens/sec       | 441                |
| gpt-oss-120b (MXFP4) | L4            | ... | OOM          | OOM       | OOM     | OOM         | OOM                      | OOM                |
| gpt-oss-120b (MXFP4) | A100 80GB SXM | ... | 0.033 ms     | 1.349 ms  | 0.524 s | 1.2 s       | 421.81 tokens/sec        | 29                 |
| gpt-oss-120b (MXFP4) | H100 SXM      | ... | 0.010 ms     | 0.821 ms  | 0.166 s | 0.6 s       | 874.67 tokens/sec        | 29                 |
| Qwen3.8-27B (BF16)   | H100 SXM      | ... | 0.055 ms     | 16.716 ms | 0.890 s | 9.4 s       | 54.28 tokens/sec         | 20                 |
| Qwen3.8-27B (FP8)    | H100 SXM      | ... | 0.055 ms     | 8.657 ms  | 0.882 s | 5.3 s       | 96.51 tokens/sec         | 44                 |
[... 7 models x 8 GPUs = 51 rows total; run the command to see the full cross product ...]
```

## GPU: frontier MoE models on an 8-GPU node
```bash
✗ python LLM_size_pef_calculator.py --profile via -g 8 --models "GLM,Kimi" --devices "H200,B200,B300"
 num_gpu = 8, prompt_size = 16000 tokens, response_size = 512 tokens
 n_concurrent_request = 4, profile = via, device = gpu

******************** Estimate LLM Memory Footprint ********************
| Model           | Input Size (tokens) | Output Size (tokens) | Concurrent Requests | KV Cache Size per Token | Weights   | Memory Footprint |
|-----------------+---------------------+----------------------+---------------------+-------------------------+-----------+------------------|
| GLM-5.2 (FP8)   | 16000               | 512                  | 4                   | 0.000051 GiB/token      | 756.0 GB  | 759.38 GB        |
| GLM-5.2 (NVFP4) | 16000               | 512                  | 4                   | 0.000051 GiB/token      | 440.0 GB  | 443.38 GB        |
| Kimi-K3 (MXFP4) | 16000               | 512                  | 4                   | 0.000065 GiB/token      | 1561.0 GB | 1565.31 GB       |

!!!! Warning Kimi-K3 (MXFP4): weights (1561 GB) do not fit in 8x H200 SXM (1128 GB)
!!!! Warning Kimi-K3 (MXFP4): weights (1561 GB) do not fit in 8x B200 (1440 GB)

******************** Estimate LLM Capacity and Latency ********************
| Model           | GPU      | Input Size (tokens) | Output Size (tokens) | Concurrent Requests | Max # KV Cache Tokens | Prefill Time | TPOT (ms) | TTFT    | E2E Latency | Output Tokens Throughput | Max Concurrent Req |
|-----------------+----------+---------------------+----------------------+---------------------+-----------------------+--------------+-----------+---------+-------------+--------------------------+--------------------|
| GLM-5.2 (FP8)   | H200 SXM | 16000               | 512                  | 4                   | 7262399               | 0.010 ms     | 1.058 ms  | 0.163 s | 0.7 s       | 727.65 tokens/sec        | 439                |
| GLM-5.2 (FP8)   | B200     | 16000               | 512                  | 4                   | 13353443              | 0.004 ms     | 0.635 ms  | 0.072 s | 0.4 s       | 1292.04 tokens/sec       | 808                |
| GLM-5.2 (FP8)   | B300     | 16000               | 512                  | 4                   | 30220951              | 0.004 ms     | 0.635 ms  | 0.072 s | 0.4 s       | 1292.04 tokens/sec       | 1830               |
| GLM-5.2 (NVFP4) | H200 SXM | 16000               | 512                  | 4                   | 13431534              | 0.010 ms     | 0.616 ms  | 0.162 s | 0.5 s       | 1073.13 tokens/sec       | 813                |
| GLM-5.2 (NVFP4) | B200     | 16000               | 512                  | 4                   | 19522578              | 0.004 ms     | 0.370 ms  | 0.071 s | 0.3 s       | 1966.52 tokens/sec       | 1182               |
| GLM-5.2 (NVFP4) | B300     | 16000               | 512                  | 4                   | 36390086              | 0.004 ms     | 0.370 ms  | 0.071 s | 0.3 s       | 1966.52 tokens/sec       | 2203               |
| Kimi-K3 (MXFP4) | H200 SXM | 16000               | 512                  | 4                   | 0                     | OOM          | OOM       | OOM     | OOM         | OOM                      | OOM                |
| Kimi-K3 (MXFP4) | B200     | 16000               | 512                  | 4                   | 0                     | OOM          | OOM       | OOM     | OOM         | OOM                      | OOM                |
| Kimi-K3 (MXFP4) | B300     | 16000               | 512                  | 4                   | 11397002              | 0.012 ms     | 0.906 ms  | 0.186 s | 0.6 s       | 789.24 tokens/sec        | 690                |
```
Note: Kimi-K3's weights (1,561 GB) do not fit on 8x H200 (1,128 GB) or 8x B200 (1,440 GB); 8x B300 (2,304 GB) is the smallest single-node fit.

## CPU-only inference VMs
```bash
✗ python LLM_size_pef_calculator.py --profile via --device cpu -c 1 --models "gpt-oss,Q4_K_M"
 num_gpu = 1, prompt_size = 16000 tokens, response_size = 512 tokens
 n_concurrent_request = 1, profile = via, device = cpu

******************** Estimate LLM Memory Footprint ********************
| Model                | Input Size (tokens) | Output Size (tokens) | Concurrent Requests | KV Cache Size per Token | Weights | Memory Footprint |
|----------------------+---------------------+----------------------+---------------------+-------------------------+---------+------------------|
| gpt-oss-20b (MXFP4)  | 16000               | 512                  | 1                   | 0.000023 GiB/token      | 13.0 GB | 13.38 GB         |
| gpt-oss-120b (MXFP4) | 16000               | 512                  | 1                   | 0.000034 GiB/token      | 63.0 GB | 63.57 GB         |
| Qwen3.5-9B (Q4_K_M)  | 16000               | 512                  | 1                   | 0.000031 GiB/token      | 5.8 GB  | 6.35 GB          |
| Qwen3.8-27B (Q4_K_M) | 16000               | 512                  | 1                   | 0.000061 GiB/token      | 18.0 GB | 19.16 GB         |

******************** Estimate LLM Capacity and Latency ********************
| Model                | GPU                      | Input Size (tokens) | Output Size (tokens) | Concurrent Requests | Max # KV Cache Tokens | Prefill Time | TPOT (ms)  | TTFT      | E2E Latency | Output Tokens Throughput | Max Concurrent Req |
|----------------------+--------------------------+---------------------+----------------------+---------------------+-----------------------+--------------+------------+-----------+-------------+--------------------------+--------------------|
| gpt-oss-20b (MXFP4)  | CPU 16c / 8ch DDR5-4800  | 16000               | 512                  | 1                   | 2228224               | 6.818 ms     | 18.235 ms  | 109.109 s | 118.4 s     | 4.32 tokens/sec          | 134                |
| gpt-oss-20b (MXFP4)  | CPU 32c / 8ch DDR5-4800  | 16000               | 512                  | 1                   | 5024426               | 3.409 ms     | 18.235 ms  | 54.564 s  | 63.9 s      | 8.01 tokens/sec          | 304                |
| gpt-oss-20b (MXFP4)  | CPU 32c / 12ch DDR5-4800 | 16000               | 512                  | 1                   | 5024426               | 3.409 ms     | 12.170 ms  | 54.558 s  | 60.8 s      | 8.42 tokens/sec          | 304                |
| gpt-oss-20b (MXFP4)  | CPU 64c / 12ch DDR5-6400 | 16000               | 512                  | 1                   | 10616832              | 1.705 ms     | 9.117 ms   | 27.282 s  | 31.9 s      | 16.03 tokens/sec         | 642                |
| gpt-oss-120b (MXFP4) | CPU 16c / 8ch DDR5-4800  | 16000               | 512                  | 1                   | 29127                 | 9.659 ms     | 22.401 ms  | 154.568 s | 166.0 s     | 3.08 tokens/sec          | 1                  |
| gpt-oss-120b (MXFP4) | CPU 32c / 8ch DDR5-4800  | 16000               | 512                  | 1                   | 1893262               | 4.830 ms     | 22.401 ms  | 77.295 s  | 88.7 s      | 5.77 tokens/sec          | 114                |
| gpt-oss-120b (MXFP4) | CPU 32c / 12ch DDR5-4800 | 16000               | 512                  | 1                   | 1893262               | 4.830 ms     | 14.950 ms  | 77.288 s  | 84.9 s      | 6.03 tokens/sec          | 114                |
| gpt-oss-120b (MXFP4) | CPU 64c / 12ch DDR5-6400 | 16000               | 512                  | 1                   | 5621532               | 2.415 ms     | 11.201 ms  | 38.648 s  | 44.4 s      | 11.54 tokens/sec         | 340                |
| Qwen3.5-9B (Q4_K_M)  | CPU 16c / 8ch DDR5-4800  | 16000               | 512                  | 1                   | 1907097               | 17.045 ms    | 47.231 ms  | 272.775 s | 296.9 s     | 1.72 tokens/sec          | 105                |
| Qwen3.5-9B (Q4_K_M)  | CPU 32c / 8ch DDR5-4800  | 16000               | 512                  | 1                   | 4004249               | 8.523 ms     | 47.231 ms  | 136.411 s | 160.5 s     | 3.19 tokens/sec          | 220                |
| Qwen3.5-9B (Q4_K_M)  | CPU 32c / 12ch DDR5-4800 | 16000               | 512                  | 1                   | 4004249               | 8.523 ms     | 31.522 ms  | 136.395 s | 152.5 s     | 3.36 tokens/sec          | 220                |
| Qwen3.5-9B (Q4_K_M)  | CPU 64c / 12ch DDR5-6400 | 16000               | 512                  | 1                   | 8198553               | 4.261 ms     | 23.616 ms  | 68.205 s  | 80.3 s      | 6.38 tokens/sec          | 451                |
| Qwen3.8-27B (Q4_K_M) | CPU 16c / 8ch DDR5-4800  | 16000               | 512                  | 1                   | 753664                | 51.136 ms    | 146.580 ms | 818.328 s | 893.2 s     | 0.57 tokens/sec          | 39                 |
| Qwen3.8-27B (Q4_K_M) | CPU 32c / 8ch DDR5-4800  | 16000               | 512                  | 1                   | 1802240               | 25.568 ms    | 146.580 ms | 409.237 s | 484.1 s     | 1.06 tokens/sec          | 95                 |
| Qwen3.8-27B (Q4_K_M) | CPU 32c / 12ch DDR5-4800 | 16000               | 512                  | 1                   | 1802240               | 25.568 ms    | 97.826 ms  | 409.189 s | 459.2 s     | 1.12 tokens/sec          | 95                 |
| Qwen3.8-27B (Q4_K_M) | CPU 64c / 12ch DDR5-6400 | 16000               | 512                  | 1                   | 3899392               | 12.784 ms    | 73.290 ms  | 204.619 s | 242.1 s     | 2.12 tokens/sec          | 205                |
```
All three outputs above were produced by running the exact commands shown against this branch's code (`tabulate` formatting may render slightly differently on your machine).
