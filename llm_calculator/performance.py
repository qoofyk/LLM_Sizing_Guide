"""Performance calculation utilities for LLM inference."""

from dataclasses import dataclass
from typing import Union, Optional
from configs.gpu_specs import GPUSpec
from configs.model_specs import ModelSpec

BYTES_IN_GiB = 1_073_741_824

@dataclass
class PerformanceMetrics:
    kv_cache_tokens: int
    prefill_time_per_token: Union[float, str]
    tpot: Union[float, str]
    ttft: Union[float, str]
    e2e_latency: Union[float, str]
    throughput: Union[float, str]
    max_concurrent_requests: Union[int, str] = "OOM"

class PerformanceCalculator:
    """Calculator for LLM inference performance metrics."""

    def __init__(self, num_gpu: int):
        self.num_gpu = num_gpu

    def calc_kv_cache_size_per_token(self, model: ModelSpec) -> float:
        """Calculate KV cache size per token in GiB."""
        if model.kv_bytes_per_token_override is not None:
            return model.kv_bytes_per_token_override / BYTES_IN_GiB
        d_head = model.head_dim or model.d_model / model.n_heads
        n_kv_layers = model.n_kv_layers or model.n_layers
        bytes_per_value = model.kv_dtype_bytes
        return 2 * n_kv_layers * model.n_kv_heads * d_head * bytes_per_value / BYTES_IN_GiB

    def calc_memory_footprint(self,
                            model: ModelSpec,
                            n_concurrent_request: int,
                            context_window: int) -> float:
        """Calculate total memory footprint in GB."""
        kv_cache_size_per_token = self.calc_kv_cache_size_per_token(model)
        return (kv_cache_size_per_token * context_window * n_concurrent_request +
                model.state_gb_per_request * n_concurrent_request +
                model.weight_gb)

    def calc_kv_cache_tokens(self,
                           gpu: GPUSpec,
                           model: ModelSpec,
                           kv_cache_size: float) -> float:
        """Calculate maximum number of tokens that can fit in KV cache."""
        result = (self.num_gpu * gpu.memory_gb - model.weight_gb) / kv_cache_size
        return max(0, result)

    def calc_prefill_time_per_token(self,
                                  model: ModelSpec,
                                  gpu: GPUSpec) -> Union[float, str]:
        """Calculate prefill time per token in milliseconds."""
        result = (2 * model.active_params / self.num_gpu) / gpu.fp16_tflops
        return result if result >= 0 else "OOM"

    def calc_tpot(self,
                 model: ModelSpec,
                 gpu: GPUSpec) -> Union[float, str]:
        """Calculate token processing time (TPOT) in milliseconds."""
        effective_bw = gpu.memory_bandwidth_gbps * gpu.bw_efficiency
        result = (model.active_weight_gb / self.num_gpu) / effective_bw * 1000
        return result if result >= 0 else "OOM"

    def calc_e2e_latency(self,
                        prefill_time_per_token: float,
                        tpot: float,
                        prompt_size: int,
                        response_size: int) -> float:
        """Calculate end-to-end latency in seconds."""
        return (prompt_size * prefill_time_per_token + response_size * tpot) / 1000

    def calculate_metrics(self,
                        model: ModelSpec,
                        gpu: GPUSpec,
                        prompt_size: int,
                        response_size: int) -> PerformanceMetrics:
        """Calculate all performance metrics for given model and GPU configuration."""
        kv_cache_size_per_token = self.calc_kv_cache_size_per_token(model)
        kv_cache_tokens = self.calc_kv_cache_tokens(gpu, model, kv_cache_size_per_token)
        prefill_time_per_token = self.calc_prefill_time_per_token(model, gpu)
        tpot = self.calc_tpot(model, gpu)
        weights_fit = model.weight_gb < self.num_gpu * gpu.memory_gb

        if not weights_fit or isinstance(prefill_time_per_token, str) or isinstance(tpot, str):
            return PerformanceMetrics(
                kv_cache_tokens=int(kv_cache_tokens),
                prefill_time_per_token="OOM",
                tpot="OOM",
                ttft="OOM",
                e2e_latency="OOM",
                throughput="OOM"
            )

        # TTFT: prefill the whole prompt, then decode the first token (seconds)
        ttft = (prompt_size * prefill_time_per_token + tpot) / 1000
        e2e_latency = self.calc_e2e_latency(
            prefill_time_per_token, tpot, prompt_size, response_size
        )
        throughput = response_size / e2e_latency if e2e_latency > 0 else "OOM"

        # Concurrency bounded by KV cache plus fixed per-request state
        context_window = prompt_size + response_size
        per_request_gib = (kv_cache_size_per_token * context_window +
                           model.state_gb_per_request)
        free_gb = self.num_gpu * gpu.memory_gb - model.weight_gb
        max_concurrent = int(free_gb // per_request_gib) if per_request_gib > 0 else 0

        return PerformanceMetrics(
            kv_cache_tokens=int(kv_cache_tokens),
            prefill_time_per_token=prefill_time_per_token,
            tpot=tpot,
            ttft=ttft,
            e2e_latency=e2e_latency,
            throughput=throughput,
            max_concurrent_requests=max_concurrent
        )
