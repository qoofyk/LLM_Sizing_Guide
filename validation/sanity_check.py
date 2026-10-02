#!/usr/bin/env python3
"""Compare calculator estimates with published batch-1 measurements.

Each anchor is either a "calibration" point (used to fit the efficiency factors in
llm_calculator/performance.py and configs/cpu_specs.py) or a "hold-out" point that
was not used for fitting. Hold-out rows are the real test of the model. All anchors
are public; NDA vendor data is deliberately not included here.

Run from the repo root:  python validation/sanity_check.py
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tabulate import tabulate  # noqa: E402
from configs.cpu_specs import cpu  # noqa: E402
from configs.gpu_specs import GPUSpec  # noqa: E402
from configs.model_specs import ModelSpec  # noqa: E402
from llm_calculator.performance import PerformanceCalculator  # noqa: E402

def m(name, params, precision, weight_gb, active=None):
    # Only params, active params and weight size matter for these estimates.
    return ModelSpec(name, params, 4096, 32, 8, 32, 131072, precision=precision,
                     weight_gb_override=weight_gb, active_params_billion=active)

H100_SXM = GPUSpec("H100 SXM", 989.5, 80, 3350)
L4 = GPUSpec("L4", 121, 24, 300)
PRO6000 = GPUSpec("RTX PRO 6000 BW", 500, 96, 1597)
EPYC_9554 = cpu("EPYC 9554 64c / 12ch DDR5-4800", 64, 192, 460.8, amx=False)
EPYC_9755 = cpu("EPYC 9755 128c / 12ch DDR5-6400", 128, 768, 614.4, amx=False)

@dataclass
class Anchor:
    kind: str          # "calibration" or "hold-out"
    model: ModelSpec
    device: GPUSpec
    engine: str
    metric: str        # "decode" or "prefill" (tok/s, batch 1)
    measured: float
    source: str

GPT_OSS_20B = m("gpt-oss-20b", 20.9, "MXFP4", 13.0, active=3.6)
GPT_OSS_120B = m("gpt-oss-120b", 116.8, "MXFP4", 63.0, active=5.1)

L40S = GPUSpec("L40s", 362, 48, 864)
VIA_LIKE_PROMPT = 13_000   # devforth.io used ~10k-word prompts (~13k tokens)

ANCHORS = [
    # --- GPU, dense (fit: bandwidth eff. 0.70 + 0.5 ms/token, prefill eff. 0.38) ---
    Anchor("calibration", m("Llama-3.1-8B", 8, "BF16", 16.0), H100_SXM, "vLLM",
           "decode", 146, "FlashFormer, arXiv:2505.22758"),
    Anchor("calibration", m("Qwen3.5-27B", 27, "Q4_K_M", 17.5), PRO6000, "llama.cpp",
           "decode", 60.9, "hardware-corner.net RTX PRO 6000"),
    Anchor("calibration", m("Qwen3-14B", 14.8, "Q4_K_M", 9.0), PRO6000, "llama.cpp",
           "decode", 114.4, "hardware-corner.net RTX PRO 6000"),
    Anchor("hold-out", m("Qwen3-8B", 8.2, "Q4_K_M", 5.0), PRO6000, "llama.cpp",
           "decode", 173.7, "hardware-corner.net RTX PRO 6000"),
    Anchor("hold-out", m("Qwen3-32B", 32.8, "Q4_K_M", 19.8), PRO6000, "llama.cpp",
           "decode", 54.9, "hardware-corner.net RTX PRO 6000"),
    Anchor("calibration", m("Qwen3.5-27B", 27, "Q4_K_M", 17.5), PRO6000, "llama.cpp",
           "prefill", 3338.5, "hardware-corner.net RTX PRO 6000"),
    Anchor("calibration", m("Qwen3-8B", 8.2, "Q4_K_M", 5.0), PRO6000, "llama.cpp",
           "prefill", 10964.1, "hardware-corner.net RTX PRO 6000"),
    Anchor("calibration", m("Qwen3-14B", 14.8, "Q4_K_M", 9.0), PRO6000, "llama.cpp",
           "prefill", 6865.6, "hardware-corner.net RTX PRO 6000"),
    Anchor("calibration", m("Qwen3-32B", 32.8, "Q4_K_M", 19.8), PRO6000, "llama.cpp",
           "prefill", 3137.2, "hardware-corner.net RTX PRO 6000"),
    # --- GPU, MoE (fit: bandwidth eff. 0.50 + 2.0 ms/token, prefill eff. 0.15) ---
    Anchor("calibration", GPT_OSS_20B, L4, "vLLM 0.15", "decode", 60,
           "devforth.io L4/L40S/H100 gpt-oss-20b"),
    Anchor("calibration", GPT_OSS_20B, H100_SXM, "vLLM 0.15", "decode", 220,
           "devforth.io L4/L40S/H100 gpt-oss-20b"),
    Anchor("calibration", GPT_OSS_120B, PRO6000, "llama.cpp", "decode", 210.1,
           "hardware-corner.net RTX PRO 6000"),
    Anchor("hold-out", GPT_OSS_20B, L40S, "vLLM 0.15", "decode", 160,
           "devforth.io L4/L40S/H100 gpt-oss-20b"),
    Anchor("calibration", GPT_OSS_20B, L4, "vLLM 0.15", "ttft_s", 3.50,
           "devforth.io, ~13k-token prompt"),
    Anchor("calibration", GPT_OSS_20B, L40S, "vLLM 0.15", "ttft_s", 1.17,
           "devforth.io, ~13k-token prompt"),
    Anchor("calibration", GPT_OSS_20B, H100_SXM, "vLLM 0.15", "ttft_s", 0.80,
           "devforth.io, ~13k-token prompt"),
    Anchor("calibration", GPT_OSS_120B, PRO6000, "llama.cpp", "prefill", 4578.6,
           "hardware-corner.net RTX PRO 6000"),
    # --- CPU (AVX-512 path; dense bandwidth eff. 0.55, MoE 0.18) ---
    Anchor("calibration", GPT_OSS_20B, EPYC_9755, "llama.cpp native",
           "prefill", 1187.3, "AMD ZenDNN llama.cpp article, 2026"),
    Anchor("calibration", m("Llama-3.1-8B", 8, "Q4_K_M", 4.92), EPYC_9554, "llama.cpp",
           "decode", 49.8, "ahelpme.com EPYC 9554"),
    Anchor("calibration", m("Qwen3-Coder-30B-A3B", 30.5, "Q4_K_M", 18.55, active=3.3),
           EPYC_9554, "llama.cpp", "decode", 42.2, "ahelpme.com EPYC 9554"),
    Anchor("hold-out", m("QwQ-32B", 32.8, "Q4_K_M", 19.84), EPYC_9554, "llama.cpp",
           "decode", 14.0, "ahelpme.com EPYC 9554"),
    Anchor("hold-out", m("Llama-3.3-70B", 70.6, "Q4_K_M", 42.5), EPYC_9554, "llama.cpp",
           "decode", 7.14, "ahelpme.com EPYC 9554"),
]

def estimate(a: Anchor) -> float:
    calc = PerformanceCalculator(1, calibrated=True)
    tpot = calc.calc_tpot(a.model, a.device)
    prefill = calc.calc_prefill_time_per_token(a.model, a.device)
    if a.metric == "decode":
        return 1000 / tpot
    if a.metric == "ttft_s":
        return (VIA_LIKE_PROMPT * prefill + tpot) / 1000
    return 1000 / prefill

def main() -> None:
    rows = []
    for a in ANCHORS:
        est = estimate(a)
        rows.append({
            "Type": a.kind,
            "Model": a.model.label,
            "Device": a.device.name,
            "Engine": a.engine,
            "Metric": {"decode": "decode tok/s", "prefill": "prefill tok/s",
                       "ttft_s": "TTFT s"}[a.metric],
            "Measured": f"{a.measured:,.2f}",
            "Estimate": f"{est:,.2f}",
            "Estimate / Measured": f"{est / a.measured:.2f}",
            "Source": a.source,
        })
    print(tabulate(rows, headers="keys", tablefmt="orgtbl"))
    for kind in ("calibration", "hold-out"):
        r = [estimate(a) / a.measured for a in ANCHORS if a.kind == kind]
        print(f"{kind:11} estimate/measured: min {min(r):.2f}, max {max(r):.2f} (n={len(r)})")
    print("For throughput rows < 1 means conservative; for TTFT rows > 1 means conservative.")

if __name__ == "__main__":
    main()
