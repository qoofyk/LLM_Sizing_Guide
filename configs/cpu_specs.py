"""CPU inference VM specifications (llama.cpp engine in VCF Private AI Services).

Each entry is one inference VM pinned to a single socket / NUMA node:
- memory_gb: RAM given to the VM (fully reserved)
- memory_bandwidth_gbps: peak DDR bandwidth of that socket with ALL channels
  populated (channels x MT/s x 8 bytes)
- fp16_tflops: *effective* llama.cpp prompt-processing throughput on the AVX-512 /
  AVX2 path, ~0.066 TFLOPS per physical core. Calibrated from AMD's public
  gpt-oss-20b pp512 result (1,187 tok/s, one 128-core EPYC 9755 socket, native
  backend). This is an MoE figure, so it is conservative for dense models.

Decode efficiency (fraction of peak bandwidth), from public llama-bench tg128 on a
single-socket EPYC 9554 with all 12 DDR5 channels populated (ahelpme.com, 2025):
  dense: Llama-3.1-8B Q4_K_M 0.53, QwQ-32B Q4_K_M 0.60, Llama-3.3-70B Q4_K_M 0.66
  MoE:   Qwen3-Coder-30B-A3B Q4_K_M 0.18 (per-token overhead dominates, as on GPUs)

Intel Xeon 4th Gen and later add AMX. llama.cpp uses it only for some GGUF
types (Q4_0/Q4_1/Q8_0/Q4_K/Q5_K/Q6_K/IQ4_XS); MXFP4 (gpt-oss) and BF16/F16 stay
on AVX-512. With AMX we assume 2x effective prefill throughput (Intel reports ~2x
for Llama-3.2-3B with vs. without AMX; AMD's ZenDNN shows 2.18x over the native
path on EPYC, so 2x is a reasonable "optimized CPU kernel" figure). Decode is
bandwidth-bound either way, so AMX does not change the decode efficiency.
"""

from dataclasses import dataclass
from typing import Dict, List
from configs.gpu_specs import GPUSpec

TFLOPS_PER_CORE = 0.066        # AVX-512 / AVX2 path (native llama.cpp)
AMX_TFLOPS_PER_CORE = 0.132    # AMX path, AMX-eligible GGUF types only
CPU_BW_EFFICIENCY: Dict[str, float] = {"dense": 0.55, "moe": 0.18}

@dataclass
class CPUSpec(GPUSpec):
    cores: int = 0
    amx: bool = False

    @property
    def amx_tflops(self) -> float:
        return self.cores * AMX_TFLOPS_PER_CORE

def cpu(name: str, cores: int, ram_gb: int, bw_gbps: float, amx: bool) -> CPUSpec:
    return CPUSpec(name, cores * TFLOPS_PER_CORE, ram_gb, bw_gbps,
                   CPU_BW_EFFICIENCY["dense"], cores=cores, amx=amx)

CPU_SPECS: List[CPUSpec] = [
    # Intel without AMX: Xeon 3rd Gen (Ice Lake), 8ch DDR4-3200 = 204.8 GB/s
    cpu("Xeon 3rd Gen 32c / 8ch DDR4-3200", 32, 128, 204.8, amx=False),
    # Intel with AMX: Xeon 5th Gen (Emerald Rapids), 8ch DDR5-5600 = 358.4 GB/s
    cpu("Xeon 5th Gen 32c / 8ch DDR5-5600", 32, 128, 358.4, amx=True),
    # Intel with AMX: Xeon 6700P (Granite Rapids-SP), 8ch DDR5-6400 = 409.6 GB/s
    cpu("Xeon 6700P 64c / 8ch DDR5-6400", 64, 256, 409.6, amx=True),
    # Intel with AMX: Xeon 6900P (Granite Rapids-AP), 12ch DDR5-6400 = 614.4 GB/s
    cpu("Xeon 6900P 64c / 12ch DDR5-6400", 64, 256, 614.4, amx=True),
    # AMD (AVX-512, no AMX): EPYC 9004 (Genoa), 12ch DDR5-4800 = 460.8 GB/s
    cpu("EPYC 9004 32c / 12ch DDR5-4800", 32, 128, 460.8, amx=False),
    # AMD (AVX-512, no AMX): EPYC 9005 (Turin), 12ch DDR5-6400 = 614.4 GB/s
    cpu("EPYC 9005 64c / 12ch DDR5-6400", 64, 256, 614.4, amx=False),
]
