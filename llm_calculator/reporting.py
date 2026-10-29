"""Reporting utilities for LLM performance calculations."""

import csv
from datetime import datetime
from typing import List, Dict, Any
from tabulate import tabulate

class PerformanceReporter:
    """Reporter class for generating and saving performance reports."""

    @staticmethod
    def print_estimate_notes(efficiency: str, device: str) -> None:
        """Explain the effective estimation mode and scope of the output rates."""
        if device == 'cpu':
            print("Estimate mode: calibrated CPU rates (always used for CPU devices).")
        elif efficiency == 'theoretical':
            print("Estimate mode: theoretical (idealized hardware estimate).")
        else:
            print("Estimate mode: calibrated (historical factors; not refitted to corrected work estimates).")
        if device == 'cpu':
            print("CPU efficiency factors are unchanged and have not been refitted to corrected work estimates.")
        print("Single-request output rate includes prefill; single-stream decode rate "
              "excludes prefill. Neither rate represents aggregate serving capacity.")
        print("Concurrency affects memory sizing only; Max Concurrent Req is a "
              "memory limit, not a latency target.")
        print("TPOT uses midpoint decode context. Attention/cache/state work is estimated; "
              "batching, prefix reuse, speculative decoding and queueing are not modeled.")

    @staticmethod
    def save_to_csv(data: List[Dict[str, Any]], filename_prefix: str) -> str:
        """Save data to CSV file with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{filename_prefix}_{timestamp}.csv'
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        return filename

    @staticmethod
    def print_table(data: List[Dict[str, Any]], title: str = "") -> None:
        """Print data in tabulated format."""
        if title:
            print(f"\n{title}")
        print(tabulate(data, headers="keys", tablefmt='orgtbl'))

    @staticmethod
    def format_memory_footprint_row(
        model_name: str,
        prompt_size: int,
        response_size: int,
        n_concurrent_request: int,
        kv_cache_size_per_token: float,
        memory_footprint: float,
        weight_gb: float = None
    ) -> Dict[str, Any]:
        """Format a row for memory footprint report."""
        row = {
            'Model': model_name,
            'Input Size (tokens)': prompt_size,
            'Output Size (tokens)': response_size,
            'Concurrent Requests': n_concurrent_request,
            'KV Cache Size per Token': f"{kv_cache_size_per_token:.6f} GiB/token",
        }
        if weight_gb is not None:
            row['Weights'] = f"{weight_gb:.1f} GB"
        row['Memory Footprint'] = f"{memory_footprint:.2f} GB"
        return row

    @staticmethod
    def format_performance_row(
        model_name: str,
        gpu_name: str,
        prompt_size: int,
        response_size: int,
        n_concurrent_request: int,
        metrics: 'PerformanceMetrics'  # Type hint as string to avoid circular import
    ) -> Dict[str, Any]:
        """Format a row for performance metrics report."""
        return {
            'Model': model_name,
            'GPU': gpu_name,
            'Input Size (tokens)': prompt_size,
            'Output Size (tokens)': response_size,
            'Concurrent Requests': n_concurrent_request,
            'Max # KV Cache Tokens': str(metrics.kv_cache_tokens),
            'Prefill Time': f"{metrics.prefill_time_per_token:.3f} ms" if isinstance(metrics.prefill_time_per_token, float) else metrics.prefill_time_per_token,
            'TPOT (ms)': f"{metrics.tpot:.3f} ms" if isinstance(metrics.tpot, float) else metrics.tpot,
            'TTFT': f"{metrics.ttft:.3f} s" if isinstance(metrics.ttft, float) else metrics.ttft,
            'E2E Latency': f"{metrics.e2e_latency:.1f} s" if isinstance(metrics.e2e_latency, float) else metrics.e2e_latency,
            'Single-request output rate': f"{metrics.throughput:.2f} tokens/sec" if isinstance(metrics.throughput, float) else metrics.throughput,
            'Single-stream decode rate': f"{metrics.decode_throughput:.2f} tokens/sec" if isinstance(metrics.decode_throughput, float) else metrics.decode_throughput,
            'Max Concurrent Req': str(metrics.max_concurrent_requests),
            'Status': metrics.status
        }
