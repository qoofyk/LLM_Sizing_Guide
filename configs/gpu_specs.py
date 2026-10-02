"""GPU specifications for LLM performance calculations.

fp16_tflops is the DENSE FP16/BF16 Tensor Core peak. NVIDIA datasheets often
print the 2:4 structured-sparsity figure, which is exactly 2x the dense one
(e.g. H100 PCIe: 1,513 TFLOPS sparse = 756.5 dense; HGX B200/B300 footnote:
"Specification in Sparse. Dense is 1/2 sparse spec shown"). LLM inference
does not use 2:4 sparsity, so always enter the dense value here.
"""

from dataclasses import dataclass
from typing import List

@dataclass
class GPUSpec:
    name: str
    fp16_tflops: float
    memory_gb: int
    memory_bandwidth_gbps: int
    # Fraction of peak memory bandwidth achieved during decode. 1.0 keeps the
    # original (theoretical) calculation; CPU devices use a calibrated value.
    bw_efficiency: float = 1.0

# Use your specific GPUs here
GPU_SPECS: List[GPUSpec] = [
    # GPUSpec("L40", 181, 48, 864),
    GPUSpec("L40s", 362, 48, 864),
    # GPUSpec("H100 PCIe", 756.5, 80, 2000),
    # GPUSpec("H100 SXM", 989.5, 80, 3350),
    GPUSpec("H100 NVL", 835.5, 94, 3900),
    # GPUSpec("H200 SXM", 989.5, 141, 4800),
    GPUSpec("H200 NVL", 835.5, 141, 4800),
    GPUSpec("MI300X", 1307, 192, 5300)
]

# GPUs relevant to VIA sizing, from entry-level PCIe cards to 8-GPU HGX nodes.
# RTX PRO Server Edition datasheets list sparse TFLOPS without a footnote;
# dense = 1/2 (cross-checked with SM count x boost clock x 1,024 FLOP/clk/SM).
VIA_GPU_SPECS: List[GPUSpec] = [
    GPUSpec("L4", 121, 24, 300),
    GPUSpec("RTX PRO 4500 BW", 203, 32, 800),    # 406 sparse, 32 GB GDDR7, 165 W
    GPUSpec("L40s", 362, 48, 864),
    GPUSpec("RTX PRO 6000 BW", 500, 96, 1597),   # 1 PFLOP sparse, 96 GB GDDR7
    GPUSpec("A100 80GB SXM", 312, 80, 2039),
    GPUSpec("H100 SXM", 989.5, 80, 3350),
    GPUSpec("H100 NVL", 835.5, 94, 3900),
    GPUSpec("H200 SXM", 989.5, 141, 4800),
    GPUSpec("B200", 2250, 180, 8000),            # HGX: 36 PF sparse / 8 GPUs / 2
    GPUSpec("B300", 2250, 288, 8000),
]

# Commented out GPUs for reference
"""
LEGACY_GPU_SPECS = [
    GPUSpec("A10", 125, 24, 600),
    GPUSpec("A30", 165, 24, 933),   # 330 is the sparse figure
    GPUSpec("A100 40 GB", 312, 40, 1555),
    GPUSpec("A100 40 GB SXM", 312, 40, 1555),
    GPUSpec("A100 80 GB PCIe", 312, 80, 1935),
    GPUSpec("A100 80 GB SXM", 312, 80, 2039),
]
"""
