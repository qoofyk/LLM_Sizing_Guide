"""Model specifications for LLM performance calculations."""

from dataclasses import dataclass
from typing import List, Optional

# GGUF types with Intel AMX kernels in llama.cpp (ggml-cpu/amx/common.h,
# qtype_has_amx_kernels). MXFP4 (gpt-oss) and BF16/F16 are not on the list.
AMX_QUANT_TYPES = {"Q4_0", "Q4_1", "Q8_0", "Q4_K_S", "Q4_K_M",
                   "Q5_K_S", "Q5_K_M", "Q6_K", "IQ4_XS"}

@dataclass
class ModelSpec:
    name: str
    params_billion: float
    d_model: int
    n_heads: int
    n_kv_heads: int
    n_layers: int
    max_context_window: int
    # Optional fields for MoE, quantized and hybrid-attention models.
    # Defaults reproduce the original dense FP16 calculation.
    precision: str = "FP16"
    active_params_billion: Optional[float] = None   # MoE: params touched per token
    weight_bytes_per_param: float = 2.0             # FP16/BF16=2, FP8=1, 4-bit~0.55
    weight_gb_override: Optional[float] = None      # published checkpoint size
    n_kv_layers: Optional[int] = None               # layers that keep a growing KV cache
    head_dim: Optional[int] = None                  # when != d_model / n_heads
    kv_dtype_bytes: float = 2.0                     # FP16 KV=2, FP8 KV=1
    kv_bytes_per_token_override: Optional[float] = None  # MLA/DSA/KDA caches
    state_gb_per_request: float = 0.0               # fixed recurrent state (linear attention)

    @property
    def label(self) -> str:
        return self.name if self.precision == "FP16" else f"{self.name} ({self.precision})"

    @property
    def active_params(self) -> float:
        return self.active_params_billion or self.params_billion

    @property
    def is_moe(self) -> bool:
        return self.active_params < self.params_billion

    @property
    def amx_eligible(self) -> bool:
        return self.precision in AMX_QUANT_TYPES

    @property
    def weight_gb(self) -> float:
        if self.weight_gb_override is not None:
            return self.weight_gb_override
        return self.params_billion * self.weight_bytes_per_param

    @property
    def active_weight_gb(self) -> float:
        """GB of weights read per decoded token (scaled by the average bytes/param)."""
        return self.active_params * self.weight_gb / self.params_billion

MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        "Llama-3.1-8B", 8, 4096, 32, 8, 32, 131072
    ),
    ModelSpec(
        "Llama-3.1-70B", 70, 8192, 64, 8, 80, 131072
    ),
    ModelSpec(
        "Mistral-7B-v0.3", 7, 4096, 32, 8, 32, 32768
    ),
    ModelSpec(
        "Qwen2.5-14B", 14.7, 5120, 40, 8, 48, 131072
    ),
]

# Models relevant to VMware Intelligent Assist (VIA) on VCF Private AI Services.
# Architecture values checked against each model's config.json (ModelScope mirror,
# Oct 2026); weight sizes from vLLM recipes and llama.cpp/Unsloth GGUF listings.
VIA_MODEL_SPECS: List[ModelSpec] = [
    # gpt-oss: MoE with MXFP4 experts; every other layer is 128-token sliding
    # window, so only half the layers grow KV with context.
    ModelSpec("gpt-oss-20b", 20.9, 2880, 64, 8, 24, 131072,
              precision="MXFP4", active_params_billion=3.6,
              weight_gb_override=13.0, n_kv_layers=12, head_dim=64),
    ModelSpec("gpt-oss-120b", 116.8, 2880, 64, 8, 36, 131072,
              precision="MXFP4", active_params_billion=5.1,
              weight_gb_override=63.0, n_kv_layers=18, head_dim=64),
    # Qwen3.5-9B / Qwen3.8-27B: 3 Gated DeltaNet (linear) : 1 gated full-attention
    # layer; only the full-attention layers keep KV. Linear layers keep a fixed state.
    ModelSpec("Qwen3.5-9B", 9.0, 4096, 16, 4, 32, 262144,
              precision="Q4_K_M", weight_gb_override=5.8,
              n_kv_layers=8, head_dim=256, state_gb_per_request=0.05),
    ModelSpec("Qwen3.5-9B", 9.0, 4096, 16, 4, 32, 262144,
              precision="BF16", weight_gb_override=18.0,
              n_kv_layers=8, head_dim=256, state_gb_per_request=0.05),
    ModelSpec("Qwen3.8-27B", 27, 5120, 24, 4, 64, 262144,
              precision="Q4_K_M", weight_gb_override=18.0,
              n_kv_layers=16, head_dim=256, state_gb_per_request=0.15),
    ModelSpec("Qwen3.8-27B", 27, 5120, 24, 4, 64, 262144,
              precision="FP8", weight_gb_override=29.0,
              n_kv_layers=16, head_dim=256, state_gb_per_request=0.15),
    ModelSpec("Qwen3.8-27B", 27, 5120, 24, 4, 64, 262144,
              precision="BF16", weight_gb_override=56.0,
              n_kv_layers=16, head_dim=256, state_gb_per_request=0.15),
    # GLM-5.2: MoE 744B/40B active with DeepSeek Sparse Attention (MLA-based).
    # config.json: 78 layers, kv_lora_rank 512 + qk_rope 64 per layer; the DSA
    # indexer (128-dim key) is "full" on 20 layers and shared on the rest.
    # FP8 KV: 78 x 576 B + 20 x ~132 B ~= 47.6 KB/token (layout is vLLM's, estimate).
    ModelSpec("GLM-5.2", 744, 6144, 64, 1, 78, 1048576,
              precision="FP8", active_params_billion=40,
              weight_gb_override=756, kv_dtype_bytes=1,
              kv_bytes_per_token_override=47_600),
    ModelSpec("GLM-5.2", 744, 6144, 64, 1, 78, 1048576,
              precision="NVFP4", active_params_billion=40,
              weight_gb_override=440, kv_dtype_bytes=1,
              kv_bytes_per_token_override=47_600),
    # Kimi-K3: MoE 2.8T (16 of 896 experts), official 104B active, MXFP4 checkpoint
    # of 1,561 GB (attention/shared experts stay BF16). config.json: 93 layers =
    # 69 KDA (linear) + 24 MLA. Only MLA layers grow KV: 24 x 576 x 2 B = 27.6 KB/token.
    # KDA state: 69 layers x 96 heads x 128 x 128, assumed FP32 ~= 0.43 GB/request.
    ModelSpec("Kimi-K3", 2800, 7168, 96, 1, 93, 1048576,
              precision="MXFP4", active_params_billion=104,
              weight_gb_override=1561,
              kv_bytes_per_token_override=27_648, state_gb_per_request=0.43),
]

ALL_MODEL_SPECS: List[ModelSpec] = MODEL_SPECS + VIA_MODEL_SPECS

# Commented out models for reference
"""
LEGACY_MODEL_SPECS = [
    ModelSpec("Llama-3-8B", 8, 4096, 32, 8, 32, 8192),
    ModelSpec("Llama-3-70B", 70, 8192, 64, 8, 80, 8192),
    ModelSpec("Llama-2-7B", 7, 4096, 32, 32, 32, 8192),  # Uses MHA
    ModelSpec("Falcon-7B", 7, 4544, 71, 1, 32, 2048),    # Uses MQA
    ModelSpec("Falcon-40B", 40, 8192, 128, 1, 60, 2048), # Uses MQA
    ModelSpec("Falcon-180B", 180, 14848, 232, 1, 80, 2048), # Uses MQA
]
"""
