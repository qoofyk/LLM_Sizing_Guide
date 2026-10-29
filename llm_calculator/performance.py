"""Analytical single-stream inference estimates; all internal memory uses bytes."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union

from configs.cpu_specs import CPUSpec, CPU_BW_EFFICIENCY
from configs.gpu_specs import GPUSpec
from configs.model_specs import ModelSpec

BYTES_IN_GiB = 1_073_741_824
BYTES_IN_GB = 1_000_000_000

# Historical factors, unchanged. They have NOT been refitted after adding
# component weight traffic and attention costs; calibrated mode is experimental.
GPU_CALIBRATION = {
    "dense": {"compute": 0.38, "bandwidth": 0.70, "overhead_ms": 0.5},
    "moe": {"compute": 0.15, "bandwidth": 0.50, "overhead_ms": 2.0},
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
    decode_throughput: Union[float, str] = "OOM"
    status: str = "OK"


class PerformanceCalculator:
    """Estimate one stream; concurrency is used only for admission/memory sizing.

    ideal: perfectly balanced, pooled devices, no communication cost.
    tensor: balanced weights, two ring all-reduces per layer, explicit link
    bandwidth/latency, and architecture-dependent KV replication.
    """

    def __init__(self, num_gpu: int, calibrated: bool = False,
                 parallelism: str = "ideal", interconnect_gbps: Optional[float] = None,
                 collective_latency_us: float = 0.0, memory_utilization: float = 1.0,
                 compute_precision: str = "bf16", kv_cache_sharding: str = "auto"):
        if num_gpu < 1:
            raise ValueError("num_gpu must be positive")
        if parallelism not in ("ideal", "tensor"):
            raise ValueError("parallelism must be ideal or tensor")
        if compute_precision not in ("bf16", "fp8", "fp4"):
            raise ValueError("compute_precision must be bf16, fp8, or fp4")
        if kv_cache_sharding not in ("auto", "sharded", "replicated"):
            raise ValueError("kv_cache_sharding must be auto, sharded, or replicated")
        if not math.isfinite(memory_utilization) or not 0 < memory_utilization <= 1:
            raise ValueError("memory_utilization must be in (0, 1]")
        if not math.isfinite(collective_latency_us) or collective_latency_us < 0:
            raise ValueError("collective_latency_us must be finite and nonnegative")
        if interconnect_gbps is not None and (not math.isfinite(interconnect_gbps) or interconnect_gbps <= 0):
            raise ValueError("interconnect_gbps must be finite and positive")
        if parallelism == "tensor" and num_gpu > 1 and interconnect_gbps is None:
            raise ValueError("tensor parallelism requires --interconnect-gbps (one-way GB/s per GPU)")
        self.num_gpu = num_gpu
        self.calibrated = calibrated
        self.parallelism = parallelism
        self.interconnect_gbps = interconnect_gbps
        self.collective_latency_us = collective_latency_us
        self.memory_utilization = memory_utilization
        self.compute_precision = compute_precision
        self.kv_cache_sharding = kv_cache_sharding

    def uses_amx(self, model: ModelSpec, device: GPUSpec) -> bool:
        return isinstance(device, CPUSpec) and device.amx and model.amx_eligible

    def effective_rates(self, model: ModelSpec, device: GPUSpec) -> Tuple[float, float]:
        """Return BF16/effective CPU TFLOPS and decimal GB/s per device."""
        model_class = "moe" if model.is_moe else "dense"
        if isinstance(device, CPUSpec):
            tflops = device.amx_tflops if self.uses_amx(model, device) else device.fp16_tflops
            return tflops, device.memory_bandwidth_gbps * CPU_BW_EFFICIENCY[model_class]
        tflops = device.fp16_tflops
        bw = device.memory_bandwidth_gbps * device.bw_efficiency
        if self.calibrated:
            tflops *= GPU_CALIBRATION[model_class]["compute"]
            bw *= GPU_CALIBRATION[model_class]["bandwidth"]
        return tflops, bw

    def decode_overhead_ms(self, model: ModelSpec, device: GPUSpec) -> float:
        """Historical per-output-token cost; GPU calibrated mode only."""
        if self.calibrated and not isinstance(device, CPUSpec):
            return GPU_CALIBRATION["moe" if model.is_moe else "dense"]["overhead_ms"]
        return 0.0

    def kv_bytes_per_token(self, model: ModelSpec) -> float:
        """Growing cache bytes, excluding fixed recurrent/sliding-window state."""
        if model.kv_bytes_per_token_override is not None:
            return model.kv_bytes_per_token_override
        head_dim = model.head_dim or model.d_model / model.n_heads
        layers = model.n_layers if model.n_kv_layers is None else model.n_kv_layers
        return 2 * layers * model.n_kv_heads * head_dim * model.kv_dtype_bytes

    def calc_kv_cache_size_per_token(self, model: ModelSpec) -> float:
        """Growing cache in GiB/token (public reporting API; not internal units)."""
        return self.kv_bytes_per_token(model) / BYTES_IN_GiB

    def sliding_bytes(self, model: ModelSpec, context: float) -> float:
        """Bounded sliding-window cache bytes for one logical request."""
        head_dim = model.head_dim or model.d_model / model.n_heads
        return (2 * model.sliding_layers * model.n_kv_heads * head_dim *
                model.kv_dtype_bytes * min(context, model.sliding_window))

    def kv_replication(self, model: ModelSpec) -> float:
        """Aggregate cache copies across ranks, relative to one logical cache."""
        if self.kv_cache_sharding == "replicated":
            return self.num_gpu
        if self.kv_cache_sharding == "sharded" or self.parallelism == "ideal":
            return 1.0
        if model.attention_type in ("mla", "dsa"):
            return self.num_gpu
        return max(1.0, self.num_gpu / model.n_kv_heads)

    def request_memory_bytes(self, model: ModelSpec, context: int) -> float:
        """One request's aggregate cache/state, including TP cache duplication."""
        cache = self.kv_bytes_per_token(model) * context + self.sliding_bytes(model, context)
        # Recurrent state is partitioned by heads in the balanced tensor model.
        return cache * self.kv_replication(model) + model.state_gb_per_request * BYTES_IN_GB

    def available_memory_bytes(self, gpu: GPUSpec) -> float:
        """Aggregate device capacity after the configured runtime reserve."""
        return self.num_gpu * gpu.memory_gb * BYTES_IN_GB * self.memory_utilization

    def calc_memory_footprint(self, model: ModelSpec, n_concurrent_request: int,
                              context_window: int) -> float:
        """Aggregate required memory in decimal GB; runtime reserve is separate."""
        return (model.weight_gb * BYTES_IN_GB +
                n_concurrent_request * self.request_memory_bytes(model, context_window)) / BYTES_IN_GB

    def max_concurrent_requests(self, model: ModelSpec, gpu: GPUSpec, context: int) -> int:
        """Memory-feasible request count, including fixed state and cache copies."""
        if context <= 0 or context > model.max_context_window:
            return 0
        free = max(0, self.available_memory_bytes(gpu) - model.weight_gb * BYTES_IN_GB)
        per_request = self.request_memory_bytes(model, context)
        return int(free // per_request) if per_request > 0 else 0

    def calc_kv_cache_tokens(self, gpu: GPUSpec, model: ModelSpec,
                             kv_cache_size: Optional[float] = None) -> float:
        """One-request memory token budget, reserving recurrent/sliding state.

        kv_cache_size is retained for call compatibility; units come from the
        model. This is a memory budget, not the model's maximum context length.
        """
        free = self.available_memory_bytes(gpu) - model.weight_gb * BYTES_IN_GB
        free -= model.state_gb_per_request * BYTES_IN_GB
        if free <= 0:
            return 0
        replication = self.kv_replication(model)
        growing = self.kv_bytes_per_token(model) * replication
        # Exact piecewise solution when the available cache is smaller than a window.
        sliding_per_token = self.sliding_bytes(model, 1) * replication
        window = model.sliding_window
        if window and free < (growing + sliding_per_token) * window:
            return free / (growing + sliding_per_token)
        return max(0, (free - sliding_per_token * window) / growing) if growing else 0

    def configuration_error(self, model: ModelSpec, gpu: GPUSpec) -> str:
        """Explain an unsupported kernel/partitioning assumption, or return empty."""
        if isinstance(gpu, CPUSpec):
            if self.num_gpu != 1 or self.parallelism != "ideal" or self.compute_precision != "bf16":
                return "Unsupported CPU configuration: use one VM, ideal, bf16/effective compute"
        if self.parallelism == "tensor" and model.n_heads % self.num_gpu:
            return "Unsupported tensor size: attention heads must divide evenly"
        if (self.parallelism == "tensor" and model.attention_type == "gqa" and
                model.n_kv_heads % self.num_gpu and self.num_gpu % model.n_kv_heads):
            return "Unsupported tensor size: KV heads cannot be partitioned evenly"
        if self.compute_precision != "bf16":
            if not getattr(gpu, self.compute_precision + "_tflops", None):
                return "Unsupported compute precision on this device"
            allowed = {"FP8", "NVFP4", "MXFP4"} if self.compute_precision == "fp8" else {"NVFP4", "MXFP4"}
            if model.precision not in allowed:
                return "Unsupported compute precision for this checkpoint"
        return ""

    def projection_compute_ms(self, model: ModelSpec, gpu: GPUSpec) -> float:
        """Per-token matrix FLOPs at an explicitly selected kernel precision."""
        bf16, _ = self.effective_rates(model, gpu)
        if self.compute_precision == "bf16":
            return 2 * model.active_params / (self.num_gpu * bf16)
        peak = getattr(gpu, self.compute_precision + "_tflops", None)
        if not peak:
            raise ValueError("No peak specified for the selected compute precision")
        low_params = model.low_precision_params_billion
        if low_params is None:
            low_params = model.active_params if model.precision == "FP8" else 0.0
        low_params = min(model.active_params, low_params)
        effective_low = peak * bf16 / gpu.fp16_tflops
        return 2 / self.num_gpu * ((model.active_params - low_params) / bf16 + low_params / effective_low)

    @staticmethod
    def causal_pairs(tokens: int, window: int = 0) -> float:
        """Number of causal query/key pairs, optionally window-limited."""
        if window and tokens > window:
            return window * (window + 1) / 2 + (tokens - window) * window
        return tokens * (tokens + 1) / 2

    def attention_flops(self, model: ModelSpec, pairs: float, sliding_pairs: float,
                        index_pairs: float) -> float:
        """Attention dot products; MLA assumes absorbed latent attention."""
        head_dim = model.head_dim or model.d_model / model.n_heads
        key_dim = model.attention_key_dim or head_dim
        value_dim = model.attention_value_dim or head_dim
        layers = model.n_layers if model.n_kv_layers is None else model.n_kv_layers
        return (2 * layers * model.n_heads * (key_dim + value_dim) * pairs +
                4 * model.sliding_layers * model.n_heads * head_dim * sliding_pairs +
                2 * model.indexer_layers * model.indexer_heads * model.indexer_dim * index_pairs)

    def communication_ms(self, model: ModelSpec, tokens: int = 1) -> float:
        """Two BF16 activation ring all-reduces per layer; no overlap assumed."""
        if self.parallelism == "ideal" or self.num_gpu == 1:
            return 0.0
        payload_bytes = tokens * model.d_model * 2
        ring_bytes = 2 * (self.num_gpu - 1) / self.num_gpu * payload_bytes
        return 2 * model.n_layers * (ring_bytes / (self.interconnect_gbps * 1e9) * 1000 +
                                     self.collective_latency_us / 1000)

    def calc_prefill_time_per_token(self, model: ModelSpec, gpu: GPUSpec,
                                    prompt_size: int = 0) -> float:
        """Average prefill ms/token; zero prompt gives the projection-only anchor."""
        projection = self.projection_compute_ms(model, gpu)
        if prompt_size <= 0:
            return projection
        tflops, bw = self.effective_rates(model, gpu)
        matrix_ms = max(projection * prompt_size,
                        model.active_weight_gb / (bw * self.num_gpu) * 1000)
        pairs = self.causal_pairs(prompt_size, model.sparse_top_k)
        flops = self.attention_flops(model, pairs,
                                    self.causal_pairs(prompt_size, model.sliding_window),
                                    self.causal_pairs(prompt_size))
        # Six scalar operations per FP32 recurrent-state element, an approximate
        # update/readout count. Prefill assumes fused state reuse across tokens.
        flops += 6 * model.state_gb_per_request * 1e9 / 4 * prompt_size
        cache_bytes = self.kv_bytes_per_token(model) * prompt_size + self.sliding_bytes(model, prompt_size)
        state_bytes = 2 * model.state_gb_per_request * 1e9
        attention_ms = max(flops / (tflops * self.num_gpu * 1e12) * 1000,
                           (cache_bytes * self.kv_replication(model) + state_bytes) /
                           (bw * self.num_gpu * 1e9) * 1000)
        return (matrix_ms + attention_ms + self.communication_ms(model, prompt_size)) / prompt_size

    def calc_tpot(self, model: ModelSpec, gpu: GPUSpec, context_size: float = 0) -> float:
        """Decode ms at the given cached context, with weight and attention work."""
        tflops, bw = self.effective_rates(model, gpu)
        matrix_ms = max(model.active_weight_gb / (self.num_gpu * bw) * 1000,
                        self.projection_compute_ms(model, gpu))
        selected = min(context_size, model.sparse_top_k) if model.sparse_top_k else context_size
        growing = self.kv_bytes_per_token(model)
        # DSA reads selected latent cache entries but scans the full index cache.
        cache_bytes = ((growing - model.indexer_bytes_per_token) * selected +
                       model.indexer_bytes_per_token * context_size +
                       self.sliding_bytes(model, context_size))
        flops = self.attention_flops(model, selected, min(context_size, model.sliding_window), context_size)
        state_bytes = 2 * model.state_gb_per_request * 1e9
        flops += 6 * model.state_gb_per_request * 1e9 / 4
        attention_ms = max(flops / (tflops * self.num_gpu * 1e12) * 1000,
                           (cache_bytes * self.kv_replication(model) + state_bytes) /
                           (bw * self.num_gpu * 1e9) * 1000)
        return matrix_ms + attention_ms + self.communication_ms(model) + self.decode_overhead_ms(model, gpu)

    def calc_e2e_latency(self, prefill_time_per_token: float, tpot: float,
                         prompt_size: int, response_size: int) -> float:
        """Prefill plus modeled output decode, converted from ms to seconds."""
        return (prompt_size * prefill_time_per_token + response_size * tpot) / 1000

    def calculate_metrics(self, model: ModelSpec, gpu: GPUSpec, prompt_size: int,
                          response_size: int, n_concurrent_request: int = 1) -> PerformanceMetrics:
        """Return unavailable rates for invalid or memory-infeasible workloads."""
        if prompt_size < 1 or response_size < 1 or n_concurrent_request < 1:
            raise ValueError("prompt, response and concurrency must be positive")
        context = prompt_size + response_size
        kv_tokens = int(self.calc_kv_cache_tokens(gpu, model))
        max_requests = self.max_concurrent_requests(model, gpu, context)
        status = self.configuration_error(model, gpu)
        if not status and context > model.max_context_window:
            status = "Context limit exceeded"
        if not status and n_concurrent_request > max_requests:
            status = "OOM: requested weights/cache/state exceed memory budget"
        if status:
            unavailable = "OOM" if status.startswith("OOM") else "N/A"
            return PerformanceMetrics(kv_tokens, unavailable, unavailable, unavailable,
                                      unavailable, unavailable, max_requests, unavailable, status)
        prefill = self.calc_prefill_time_per_token(model, gpu, prompt_size)
        # Midpoint-context approximation, not a serving-time or percentile model.
        tpot = self.calc_tpot(model, gpu, prompt_size + (response_size - 1) / 2)
        first_tpot = self.calc_tpot(model, gpu, prompt_size)
        ttft = (prompt_size * prefill + first_tpot) / 1000
        e2e = self.calc_e2e_latency(prefill, tpot, prompt_size, response_size)
        return PerformanceMetrics(kv_tokens, prefill, tpot, ttft, e2e,
                                  response_size / e2e, max_requests, 1000 / tpot)
