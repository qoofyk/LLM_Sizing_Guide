"""GPU specifications for LLM performance calculations."""

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
VIA_GPU_SPECS: List[GPUSpec] = [
    GPUSpec("L4", 121, 24, 300),
    GPUSpec("L40s", 362, 48, 864),
    GPUSpec("A100 80GB SXM", 312, 80, 2039),
    GPUSpec("H100 SXM", 989.5, 80, 3350),
    GPUSpec("H100 NVL", 835.5, 94, 3900),
    GPUSpec("H200 SXM", 989.5, 141, 4800),
    GPUSpec("B200", 2250, 180, 8000),
    GPUSpec("B300", 2250, 288, 8000),
]

# Commented out GPUs for reference
"""
LEGACY_GPU_SPECS = [
    GPUSpec("A10", 125, 24, 600),
    GPUSpec("A30", 330, 24, 933),
    GPUSpec("A100 40 GB", 312, 40, 1555),
    GPUSpec("A100 40 GB SXM", 312, 40, 1555),
    GPUSpec("A100 80 GB PCIe", 312, 80, 1935),
    GPUSpec("A100 80 GB SXM", 312, 80, 2039),
]
"""
