#!/usr/bin/env python3
"""
LLM Performance Calculator

This script calculates various performance metrics for LLM inference,
including memory footprint, latency, and throughput for different
GPU and model configurations.
"""

import argparse
from typing import List, Dict, Any

from configs.gpu_specs import GPU_SPECS, VIA_GPU_SPECS, GPUSpec
from configs.cpu_specs import CPU_SPECS, CPUSpec
from configs.model_specs import MODEL_SPECS, VIA_MODEL_SPECS, ALL_MODEL_SPECS, ModelSpec
from llm_calculator.performance import PerformanceCalculator
from llm_calculator.reporting import PerformanceReporter

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate LLM inference performance metrics'
    )
    parser.add_argument(
        '-g', '--num_gpu',
        type=int, default=1,
        help='Number of GPUs'
    )
    parser.add_argument(
        '-p', '--prompt_sz',
        type=int, default=None,
        help='Prompt size in tokens (default 4096; 16000 with --profile via)'
    )
    parser.add_argument(
        '-r', '--response_sz',
        type=int, default=None,
        help='Response size in tokens (default 256; 512 with --profile via)'
    )
    parser.add_argument(
        '-c', '--n_concurrent_req',
        type=int, default=None,
        help='Number of concurrent requests (default 10; 4 with --profile via)'
    )
    parser.add_argument(
        '--profile',
        choices=['default', 'via'], default='default',
        help='via: VMware Intelligent Assist workload (long agentic prompts, short '
             'answers, few users) with the VIA model and GPU lists'
    )
    parser.add_argument(
        '--device',
        choices=['gpu', 'cpu'], default='gpu',
        help='Size GPU devices or CPU-only inference VMs (llama.cpp)'
    )
    parser.add_argument(
        '--models',
        type=str, default=None,
        help='Comma-separated substrings to filter models, e.g. "gpt-oss,Qwen3.8"'
    )
    parser.add_argument(
        '--devices',
        type=str, default=None,
        help='Comma-separated substrings to filter GPUs/CPUs, e.g. "H100,B200"'
    )
    parser.add_argument(
        '--efficiency',
        choices=['theoretical', 'calibrated'], default=None,
        help='GPU efficiency model: theoretical peak (default) or calibrated against '
             'published batch-1 measurements (default with --profile via). '
             'CPU devices always use calibrated values.'
    )
    args = parser.parse_args()

    if args.efficiency is None:
        args.efficiency = 'calibrated' if args.profile == 'via' else 'theoretical'

    defaults = (16000, 512, 4) if args.profile == 'via' else (4096, 256, 10)
    if args.prompt_sz is None:
        args.prompt_sz = defaults[0]
    if args.response_sz is None:
        args.response_sz = defaults[1]
    if args.n_concurrent_req is None:
        args.n_concurrent_req = defaults[2]
    return args

def filter_by_name(items: List[Any], pattern: str, name_of) -> List[Any]:
    """Keep items whose name contains any of the comma-separated substrings."""
    if not pattern:
        return items
    keys = [k.strip().lower() for k in pattern.split(',') if k.strip()]
    return [i for i in items if any(k in name_of(i).lower() for k in keys)]

def select_models_and_devices(args: argparse.Namespace):
    """Pick model and device lists from the profile, device type and filters."""
    if args.profile == 'via':
        models = VIA_MODEL_SPECS
        devices = VIA_GPU_SPECS
    else:
        models = ALL_MODEL_SPECS if args.models else MODEL_SPECS
        devices = GPU_SPECS
    if args.device == 'cpu':
        devices = CPU_SPECS
    models = filter_by_name(models, args.models, lambda m: m.label)
    devices = filter_by_name(devices, args.devices, lambda d: d.name)
    return models, devices

def check_memory_requirements(
    calculator: PerformanceCalculator,
    model: ModelSpec,
    gpu: GPUSpec,
    prompt_size: int,
    response_size: int,
    n_concurrent_request: int
) -> None:
    """Check if memory requirements can be met and print warnings if not."""
    context_window = prompt_size + response_size
    memory_footprint = calculator.calc_memory_footprint(
        model, n_concurrent_request, context_window
    )
    available_memory = calculator.num_gpu * gpu.memory_gb

    if model.weight_gb >= available_memory:
        print(f"\n!!!! Warning {model.label}: weights ({model.weight_gb:.0f} GB) do not fit "
              f"in {calculator.num_gpu}x {gpu.name} ({available_memory:.0f} GB)")
        return

    if memory_footprint > available_memory:
        print(f"\n!!!! Warning {model.label}: n_concurrent_request={n_concurrent_request} "
              f"is TOO Large!!!\nCausing OOM with ISL={prompt_size} and OSL={response_size} "
              f"using {calculator.num_gpu}x {gpu.name}")
        
        kv_cache_size_per_token = calculator.calc_kv_cache_size_per_token(model)
        kv_cache_tokens = calculator.calc_kv_cache_tokens(
            gpu, model, kv_cache_size_per_token
        )
        max_n_concurrent_req = int(kv_cache_tokens // context_window)
        
        print(f"Max number of concurrent requests that can be set for this use case: "
              f"{max_n_concurrent_req}\nIgnore the rows in the following table which "
              f"contains {gpu.name} and rerun the calculator with this number")

def calculate_memory_footprint(
    calculator: PerformanceCalculator,
    models: List[ModelSpec],
    prompt_size: int,
    response_size: int,
    n_concurrent_request: int
) -> List[Dict[str, Any]]:
    """Calculate memory footprint for all models."""
    memory_footprint_table = []
    context_window = prompt_size + response_size
    
    for model in models:
        kv_cache_size_per_token = calculator.calc_kv_cache_size_per_token(model)
        memory_footprint = calculator.calc_memory_footprint(
            model, n_concurrent_request, context_window
        )
        
        row = PerformanceReporter.format_memory_footprint_row(
            model.label,
            prompt_size,
            response_size,
            n_concurrent_request,
            kv_cache_size_per_token,
            memory_footprint,
            model.weight_gb
        )
        memory_footprint_table.append(row)
    
    return memory_footprint_table

def calculate_performance_metrics(
    calculator: PerformanceCalculator,
    models: List[ModelSpec],
    gpus: List[GPUSpec],
    prompt_size: int,
    response_size: int,
    n_concurrent_request: int
) -> List[Dict[str, Any]]:
    """Calculate performance metrics for all model and GPU combinations."""
    performance_table = []
    
    for model in models:
        for gpu in gpus:
            metrics = calculator.calculate_metrics(
                model, gpu, prompt_size, response_size
            )
            
            row = PerformanceReporter.format_performance_row(
                model.label,
                gpu.name,
                prompt_size,
                response_size,
                n_concurrent_request,
                metrics
            )
            if isinstance(gpu, CPUSpec):
                # Which CPU kernel path the estimate assumes for this model
                row['Accel'] = 'AMX' if calculator.uses_amx(model, gpu) else 'AVX-512'
            performance_table.append(row)
    
    return performance_table

def main() -> None:
    """Main execution function."""
    args = parse_args()
    
    print(f" num_gpu = {args.num_gpu}, prompt_size = {args.prompt_sz} tokens, "
          f"response_size = {args.response_sz} tokens")
    print(f" n_concurrent_request = {args.n_concurrent_req}, profile = {args.profile}, "
          f"device = {args.device}, efficiency = {args.efficiency}")

    models, devices = select_models_and_devices(args)

    calculator = PerformanceCalculator(args.num_gpu,
                                       calibrated=(args.efficiency == 'calibrated'))
    reporter = PerformanceReporter()

    # Calculate and report memory footprint
    memory_footprint_table = calculate_memory_footprint(
        calculator,
        models,
        args.prompt_sz,
        args.response_sz,
        args.n_concurrent_req
    )
    reporter.print_table(
        memory_footprint_table,
        "******************** Estimate LLM Memory Footprint ********************"
    )
    memory_csv_file = reporter.save_to_csv(memory_footprint_table, 'llm_memory_footprint')

    # Check memory requirements
    for model in models:
        for gpu in devices:
            check_memory_requirements(
                calculator,
                model,
                gpu,
                args.prompt_sz,
                args.response_sz,
                args.n_concurrent_req
            )

    # Calculate and report performance metrics
    performance_table = calculate_performance_metrics(
        calculator,
        models,
        devices,
        args.prompt_sz,
        args.response_sz,
        args.n_concurrent_req
    )
    reporter.print_table(
        performance_table,
        "******************** Estimate LLM Capacity and Latency ********************"
    )
    perf_csv_file = reporter.save_to_csv(performance_table, 'llm_performance')

    print(f"\nResults saved to CSV files:\n1. {memory_csv_file}\n2. {perf_csv_file}")

if __name__ == '__main__':
    main()