#!/usr/bin/env python3
"""Compare calculator estimates with published batch-1 measurements.

Two comparisons are reported separately and never pooled (some observations appear in
both because the second applies the sources' documented conditions to the same data):

1. Legacy-input comparison: the original anchor inputs, measurements and hold-out
   membership. Decode anchors are context-free and prefill anchors use the projection-only
   cost, so they do not exercise the attention/cache terms.
2. Source-aware comparison: documented prompt/context lengths, KV precision, hardware
   specification and engine, with verified model dimensions. Evaluation only; nothing is
   tuned on it.

GPU MoE decode uses the dense decode factors (0.70 bandwidth efficiency + 0.5 ms). The H100
gpt-oss-20b vLLM result stays visible as an outlier excluded from fitting. Prefill factors
remain provisional. All anchors are public; NDA vendor data is deliberately not included.

Run from the repo root:  python validation/sanity_check.py
"""

import os
import sys
from dataclasses import dataclass, replace
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tabulate import tabulate  # noqa: E402
from configs.cpu_specs import cpu  # noqa: E402
from configs.gpu_specs import GPUSpec  # noqa: E402
from configs.model_specs import MODEL_SPECS, ModelSpec, VIA_MODEL_SPECS  # noqa: E402
from llm_calculator.performance import PerformanceCalculator  # noqa: E402


@dataclass
class Anchor:
    kind: str                       # legacy: calibration|hold-out|outlier; source-aware: evaluation|outlier|validation-only
    model: ModelSpec
    device: GPUSpec
    engine: str
    metric: str                     # "decode" or "prefill" (tok/s, batch 1) or "ttft_s"
    measured: float
    source: str
    prompt_tokens: Optional[int] = None   # documented prompt/prefill length
    output_tokens: Optional[int] = None   # documented generated tokens
    context_tokens: Optional[float] = None  # explicit decode context, overriding the midpoint rule
    source_url: str = ""
    note: str = ""


def m(name, params, precision, weight_gb, active=None):
    # Legacy helper: only params, active params and weight size matter for context-free estimates.
    return ModelSpec(name, params, 4096, 32, 8, 32, 131072, precision=precision,
                     weight_gb_override=weight_gb, active_params_billion=active)


def catalog(name: str, precision: Optional[str] = None) -> ModelSpec:
    return next(s for s in MODEL_SPECS + VIA_MODEL_SPECS
                if s.name == name and (precision is None or s.precision == precision))


H100_SXM = GPUSpec("H100 SXM", 989.5, 80, 3350)
L4 = GPUSpec("L4", 121, 24, 300)
L40S = GPUSpec("L40s", 362, 48, 864)
PRO6000 = GPUSpec("RTX PRO 6000 BW", 500, 96, 1597)            # legacy: Server Edition bandwidth
PRO6000_WS = GPUSpec("RTX PRO 6000 Workstation", 500, 96, 1790)  # Hardware Corner's listed bandwidth
EPYC_9554 = cpu("EPYC 9554 64c / 12ch DDR5-4800", 64, 192, 460.8, amx=False)
EPYC_9755 = cpu("EPYC 9755 128c / 12ch DDR5-6400", 128, 768, 614.4, amx=False)

# Exercise the actual mixed-precision catalog instead of synthetic MoE entries.
GPT_OSS_20B = catalog("gpt-oss-20b")
GPT_OSS_120B = catalog("gpt-oss-120b")
VIA_LIKE_PROMPT = 13_000   # devforth.io used ~10k-word prompts (~13k tokens)

# --------------------------------------------------------------------------------------
# 1. Legacy-input comparison (original inputs, measurements and hold-out membership).
# --------------------------------------------------------------------------------------
LEGACY_ANCHORS: List[Anchor] = [
    # --- GPU, dense ---
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
    # --- GPU, MoE ---
    Anchor("calibration", GPT_OSS_20B, L4, "vLLM 0.15", "decode", 60,
           "devforth.io L4/L40S/H100 gpt-oss-20b"),
    Anchor("outlier", GPT_OSS_20B, H100_SXM, "vLLM 0.15", "decode", 220,
           "devforth.io L4/L40S/H100 gpt-oss-20b",
           note="Excluded from fitting: long context + structured output + small-model serving overhead"),
    Anchor("calibration", GPT_OSS_120B, PRO6000, "llama.cpp", "decode", 210.1,
           "hardware-corner.net RTX PRO 6000"),
    Anchor("hold-out", GPT_OSS_20B, L40S, "vLLM 0.15", "decode", 160,
           "devforth.io L4/L40S/H100 gpt-oss-20b"),
    Anchor("calibration", GPT_OSS_20B, L4, "vLLM 0.15", "ttft_s", 3.50,
           "devforth.io, ~13k-token prompt", prompt_tokens=VIA_LIKE_PROMPT, output_tokens=128),
    Anchor("calibration", GPT_OSS_20B, L40S, "vLLM 0.15", "ttft_s", 1.17,
           "devforth.io, ~13k-token prompt", prompt_tokens=VIA_LIKE_PROMPT, output_tokens=128),
    Anchor("calibration", GPT_OSS_20B, H100_SXM, "vLLM 0.15", "ttft_s", 0.80,
           "devforth.io, ~13k-token prompt", prompt_tokens=VIA_LIKE_PROMPT, output_tokens=128),
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

# --------------------------------------------------------------------------------------
# 2. Source-aware comparison (documented conditions; verified model dimensions).
# --------------------------------------------------------------------------------------
DEVFORTH_URL = ("https://devforth.io/insights/self-hosted-gpt-real-response-time-token-"
                "throughput-and-cost-on-l4-l40s-and-h100-for-gpt-oss-20b/")
HC_URL = "https://www.hardware-corner.net/gpu-llm-benchmarks/rtx-pro-6000-blackwell/"
FLASHFORMER_URL = "https://arxiv.org/html/2505.22758v2"

# Devforth ran vLLM 0.15.1 with --dtype bfloat16 --kv-cache-dtype fp8 on the MXFP4 checkpoint.
GPT_OSS_20B_FP8KV = replace(GPT_OSS_20B, kv_dtype_bytes=1.0)

# Verified against Hugging Face/ModelScope config.json (hidden, layers, heads, KV heads, head_dim).
# Weight sizes are typical Q4_K_M GGUF sizes; Hardware Corner does not state the exact file.
QWEN3_8B = ModelSpec("Qwen3-8B", 8.2, 4096, 32, 8, 36, 40960, precision="Q4_K_M",
                     weight_gb_override=5.0, head_dim=128)
QWEN3_14B = ModelSpec("Qwen3-14B", 14.8, 5120, 40, 8, 40, 40960, precision="Q4_K_M",
                      weight_gb_override=9.0, head_dim=128)
QWEN3_32B = ModelSpec("Qwen3-32B", 32.8, 5120, 64, 8, 64, 40960, precision="Q4_K_M",
                      weight_gb_override=19.8, head_dim=128)
# Qwen3.5-27B's config.json matches Qwen3.8-27B (64 layers: 16 full attention, 48 linear;
# hidden 5120; 24 heads; 4 KV heads; head_dim 256), so reuse that catalog entry.
QWEN35_27B = replace(catalog("Qwen3.8-27B", "Q4_K_M"), name="Qwen3.5-27B", weight_gb_override=17.5)
LLAMA31_8B = catalog("Llama-3.1-8B")

HC_NOTE = ("4K-context tier; llama.cpp build 8288 is listed on the gpt-oss Q8_0 row. KV precision "
           "(assumed FP16) and generation length (assumed: decode evaluated at 4,096-token context) "
           "are not documented; 1,790 GB/s is the page's workstation bandwidth.")


def devforth_decode(kind, model_dev, row):
    """row = (input, thinking, output, effective_seq_len, ttft_s, decode_tok_s)."""
    inp, thinking, out, esl, ttft, tps = row
    return Anchor(kind, GPT_OSS_20B_FP8KV, model_dev, "vLLM 0.15.1 (FP8 KV)", "decode", tps,
                  "devforth.io sequence-length table", prompt_tokens=inp,
                  output_tokens=thinking + out, source_url=DEVFORTH_URL,
                  note=f"Table row: input {inp}, thinking {thinking}, output {out}, "
                       f"effective sequence length {esl}, TTFT {ttft} s.")


SOURCE_ANCHORS: List[Anchor] = [
    # --- Devforth gpt-oss-20b: TTFT at the ~13K-token single-user workload ---
    *[Anchor("evaluation", GPT_OSS_20B_FP8KV, dev, "vLLM 0.15.1 (FP8 KV)", "ttft_s", ttft,
             "devforth.io single-user ~10k-word prompt", prompt_tokens=13000, output_tokens=128,
             source_url=DEVFORTH_URL,
             note="Narrative single-user run (~10k words, assumed ~13K tokens); response length not "
                  "documented (128 assumed for the midpoint context).")
      for dev, ttft in ((L4, 3.50), (L40S, 1.17), (H100_SXM, 0.80))],
    # --- Devforth decode rows at their own prompt lengths (7.6K and 14.2K) ---
    devforth_decode("evaluation", L4, (7591, 144, 63, 7912, 2.926, 62.387)),
    devforth_decode("evaluation", L4, (14203, 205, 67, 14537, 3.160, 60.728)),
    devforth_decode("evaluation", L40S, (7589, 76, 64, 7843, 2.110, 160.183)),
    devforth_decode("evaluation", L40S, (14204, 70, 101, 14437, 1.048, 156.022)),
    devforth_decode("outlier", H100_SXM, (7590, 121, 62, 7887, 2.542, 228.180)),
    devforth_decode("outlier", H100_SXM, (14202, 146, 70, 14480, 1.784, 223.602)),
    # --- Devforth L4 long-context observation: validation only, not tuned ---
    Anchor("validation-only", GPT_OSS_20B_FP8KV, L4, "vLLM 0.15.1 (FP8 KV)", "decode", 54.082,
           "devforth.io sequence-length table (L4, long context)", prompt_tokens=61357,
           output_tokens=62 + 93, source_url=DEVFORTH_URL,
           note="Reported fields: input 61357, thinking 62, output 93, effective sequence length "
                "61266, TTFT 29.654 s, decoding 54.082 tok/s, thinking time 1.103 s. The effective "
                "length is smaller than input_tokens, and the interval behind decoding_tokens/s is not "
                "stated, so the token-counting convention is ambiguous. TTFT is not scored here."),
    # --- Hardware Corner (llama.cpp, RTX PRO 6000 workstation card, 4K tier) ---
    *[Anchor("evaluation", mdl, PRO6000_WS, "llama.cpp", "decode", tps,
             "hardware-corner.net 4K tier", context_tokens=4096, source_url=HC_URL, note=HC_NOTE)
      for mdl, tps in ((QWEN3_8B, 173.7), (QWEN3_14B, 114.4), (QWEN35_27B, 60.9),
                       (QWEN3_32B, 54.9), (GPT_OSS_120B, 210.1))],
    *[Anchor("evaluation", mdl, PRO6000_WS, "llama.cpp", "prefill", tps,
             "hardware-corner.net 4K tier", prompt_tokens=4096, source_url=HC_URL, note=HC_NOTE)
      for mdl, tps in ((QWEN3_8B, 10964.1), (QWEN3_14B, 6865.6), (QWEN35_27B, 3338.5),
                       (QWEN3_32B, 3137.2), (GPT_OSS_120B, 4578.6))],
    # --- FlashFormer Table 6: Llama-3.1-8B BF16, H100 SXM, batch 1, 128 generated tokens, vLLM ---
    *[Anchor("evaluation", LLAMA31_8B, H100_SXM, "vLLM", "decode", tps,
             "FlashFormer Table 6 (vLLM column)", prompt_tokens=prefill, output_tokens=128,
             source_url=FLASHFORMER_URL,
             note="Batch one, one H100 SXM, 128 generated tokens, CUDA graphs enabled; decoding speed only.")
      for prefill, tps in ((128, 146), (1024, 144), (3072, 141), (6144, 138))],
]


def decode_context(a: Anchor) -> float:
    """KV context at which decode is evaluated; zero for legacy context-free anchors."""
    if a.context_tokens is not None:
        return a.context_tokens
    if a.prompt_tokens is not None and a.output_tokens is not None:
        return a.prompt_tokens + (a.output_tokens - 1) / 2
    return 0


def estimate(a: Anchor) -> float:
    calc = PerformanceCalculator(1, calibrated=True)
    if a.metric == "decode":
        return 1000 / calc.calc_tpot(a.model, a.device, decode_context(a))
    if a.metric == "ttft_s":
        return calc.calculate_metrics(a.model, a.device, a.prompt_tokens, a.output_tokens or 128).ttft
    return 1000 / calc.calc_prefill_time_per_token(a.model, a.device, a.prompt_tokens or 0)


METRIC_LABEL = {"decode": "decode tok/s", "prefill": "prefill tok/s", "ttft_s": "TTFT s"}


def table(anchors: List[Anchor], source_aware: bool) -> str:
    rows = []
    for a in anchors:
        est = estimate(a)
        row = {
            "Type": a.kind,
            "Model": a.model.label,
            "Device": a.device.name,
            "Engine": a.engine,
            "Metric": METRIC_LABEL[a.metric],
        }
        if source_aware:
            ctx = decode_context(a) if a.metric == "decode" else (a.prompt_tokens or 0)
            row["Context/prompt"] = f"{ctx:,.0f}"
        row.update({
            "Measured": f"{a.measured:,.2f}",
            "Estimate": f"{est:,.2f}",
            "Estimate / Measured": f"{est / a.measured:.2f}",
            "Source": a.source,
        })
        rows.append(row)
    return tabulate(rows, headers="keys", tablefmt="orgtbl")


def summarize(anchors: List[Anchor], kinds) -> None:
    """Report ratio ranges per type; throughput (<1 conservative) and TTFT (>1 conservative) apart."""
    for kind in kinds:
        for label, metrics in (("throughput", ("decode", "prefill")), ("TTFT", ("ttft_s",))):
            ratios = [estimate(a) / a.measured for a in anchors
                      if a.kind == kind and a.metric in metrics]
            if ratios:
                print(f"{kind:15} {label:10} estimate/measured: min {min(ratios):.2f}, "
                      f"max {max(ratios):.2f} (n={len(ratios)})")


def main() -> None:
    print("GPU MoE decode uses the dense decode factors (0.70 bandwidth efficiency + 0.5 ms/token);")
    print("prefill compute factors are unchanged and provisional. Kimi, GLM and B300 have no anchors.")
    print()
    print("== 1. Legacy-input comparison (original inputs, measurements and hold-out membership) ==")
    print("Decode anchors are context-free and prefill anchors use the projection-only cost; they do not")
    print("validate long-context attention, parallelism or serving throughput.")
    print(table(LEGACY_ANCHORS, source_aware=False))
    summarize(LEGACY_ANCHORS, ("calibration", "hold-out", "outlier"))
    print("The H100 gpt-oss-20b decode result is an outlier excluded from fitting; no hardware-specific")
    print("overhead was added to explain it.")
    print()
    print("== 2. Source-aware comparison (documented conditions; evaluation only, nothing tuned) ==")
    print(table(SOURCE_ANCHORS, source_aware=True))
    summarize(SOURCE_ANCHORS, ("evaluation", "outlier", "validation-only"))
    print("Counts are per comparison; observations shared with the legacy set are not pooled.")
    print()
    print("Sources and conditions (source-aware set):")
    printed = set()
    for a in SOURCE_ANCHORS:
        if a.note.startswith("Table row:"):
            key = (a.source_url, "rows")
            if key not in printed:
                printed.add(key)
                print(f"- {a.source}: {a.source_url}\n    Rows used (input/thinking/output/effective length/TTFT) "
                      f"are recorded in each anchor's note in this file.")
            continue
        key = (a.source_url, a.note)
        if key not in printed:
            printed.add(key)
            print(f"- {a.source}: {a.source_url}\n    {a.note}")
    print("For throughput rows < 1 means conservative; for TTFT rows > 1 means conservative.")


if __name__ == "__main__":
    main()
