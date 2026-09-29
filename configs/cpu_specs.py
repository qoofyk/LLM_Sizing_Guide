"""CPU inference VM specifications (llama.cpp engine in VCF Private AI Services).

CPU VMs reuse the GPUSpec shape so the same calculator applies:
- memory_gb: RAM given to the inference VM (fully reserved)
- memory_bandwidth_gbps: peak DDR5 bandwidth of the socket/NUMA node the VM runs on
  (channels x MT/s x 8 bytes). It assumes ALL channels are populated.
- fp16_tflops: *effective* llama.cpp prompt-processing throughput, ~0.066 TFLOPS per
  physical core. Calibrated from AMD's gpt-oss-20b pp512 result (1,187 tok/s on one
  128-core EPYC 9755 socket, native backend). AMX/ZenDNN builds can be 2x+ faster.
- bw_efficiency: 0.4, calibrated from gpt-oss-20b decode (~26 tok/s) on a
  lightly populated EPYC 9334 (Leaseweb, 2026).
"""

from typing import List
from configs.gpu_specs import GPUSpec

CPU_BW_EFFICIENCY = 0.4
TFLOPS_PER_CORE = 0.066

CPU_SPECS: List[GPUSpec] = [
    # Xeon 4th/5th Gen (Sapphire/Emerald Rapids): 8ch DDR5-4800 = 307 GB/s
    GPUSpec("CPU 16c / 8ch DDR5-4800", 16 * TFLOPS_PER_CORE, 64, 307, CPU_BW_EFFICIENCY),
    GPUSpec("CPU 32c / 8ch DDR5-4800", 32 * TFLOPS_PER_CORE, 128, 307, CPU_BW_EFFICIENCY),
    # EPYC 9004/9005 (Genoa/Turin): 12ch DDR5-4800/6000 = 460-576 GB/s
    GPUSpec("CPU 32c / 12ch DDR5-4800", 32 * TFLOPS_PER_CORE, 128, 460, CPU_BW_EFFICIENCY),
    # Xeon 6 (Granite Rapids) 12ch DDR5-6400 = 614 GB/s; EPYC Turin 12ch DDR5-6000 = 576 GB/s
    GPUSpec("CPU 64c / 12ch DDR5-6400", 64 * TFLOPS_PER_CORE, 256, 614, CPU_BW_EFFICIENCY),
]
