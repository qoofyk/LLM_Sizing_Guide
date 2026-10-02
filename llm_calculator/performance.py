"""Performance calculation utilities for LLM inference."""

from dataclasses import dataclass
from typing import Tuple, Union, Optional
from configs.cpu_specs import CPUSpec, CPU_BW_EFFICIENCY
from configs.gpu_specs import GPUSpec
from configs.model_specs import ModelSpec

BYTES_IN_GiB = 1_073_741_824

# Calibrated GPU model for batch-1 inference, fitted to public measurements
# (see validation/sanity_check.py for every data point and its source):
#   TPOT    = active weight bytes / (peak BW x bandwidth eff.) + fixed overhead
#   prefill = 2 x active params / (dense peak FLOPS x compute eff.)
# Small-active MoE models (gpt-oss) pay a fixed per-token cost (routing, many small
# kernels) that dominates on fast GPUs, so a bandwidth-only formula overstates them.
GPU_CALIBRATION = {
    #         compute eff., bandwidth eff., overhead per output token (ms)
    "dense": {"compute": 0.38, "bandwidth": 0.70, "overhead_ms": 0.5},
    "moe":   {"compute": 0.15, "bandwidth": 0.50, "overhead_ms": 2.0},
}

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

    def __init__(self, num_gpu: int, calibrated: bool = False):
        self.num_gpu = num_gpu
        self.calibrated = calibrated

    def uses_amx(self, model: ModelSpec, device: GPUSpec) -> bool:
        return isinstance(device, CPUSpec) and device.amx and model.amx_eligible

    def effective_rates(self, model: ModelSpec, device: GPUSpec) -> Tuple[float, float]:
        """Return (effective TFLOPS, effective GB/s) for this model on this device."""
        model_class = "moe" if model.is_moe else "dense"
        if isinstance(device, CPUSpec):
            # CPU values are always effective (calibrated) numbers.
            tflops = device.amx_tflops if self.uses_amx(model, device) else device.fp16_tflops
            return tflops, device.memory_bandwidth_gbps * CPU_BW_EFFICIENCY[model_class]
        tflops = device.fp16_tflops
        bw = device.memory_bandwidth_gbps * device.bw_efficiency
        if self.calibrated:
            cal = GPU_CALIBRATION[model_class]
            tflops *= cal["compute"]
            bw *= cal["bandwidth"]
        return tflops, bw

    def decode_overhead_ms(self, model: ModelSpec, device: GPUSpec) -> float:
        """Fixed per-output-token cost (calibrated GPU mode only)."""
        if self.calibrated and not isinstance(device, CPUSpec):
            return GPU_CALIBRATION["moe" if model.is_moe else "dense"]["overhead_ms"]
        return 0.0

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
        tflops, _ = self.effective_rates(model, gpu)
        result = (2 * model.active_params / self.num_gpu) / tflops
        return result if result >= 0 else "OOM"

    def calc_tpot(self,
                 model: ModelSpec,
                 gpu: GPUSpec) -> Union[float, str]:
        """Calculate token processing time (TPOT) in milliseconds."""
        _, effective_bw = self.effective_rates(model, gpu)
        result = ((model.active_weight_gb / self.num_gpu) / effective_bw * 1000 +
                  self.decode_overhead_ms(model, gpu))
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
