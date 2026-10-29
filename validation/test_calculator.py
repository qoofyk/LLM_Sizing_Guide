"""Regression tests for byte accounting, context work and workload feasibility.

Run: python -m unittest discover -s validation -p 'test_*.py'
"""
import csv
from dataclasses import replace
from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs.gpu_specs import GPUSpec, VIA_GPU_SPECS
from configs.cpu_specs import CPU_SPECS
from configs.model_specs import ModelSpec, VIA_MODEL_SPECS
from llm_calculator.performance import PerformanceCalculator, PerformanceMetrics
from LLM_size_pef_calculator import check_memory_requirements


def model(name, precision=None):
    return next(m for m in VIA_MODEL_SPECS if m.name == name and
                (precision is None or m.precision == precision))


def gpu(name):
    return next(g for g in VIA_GPU_SPECS if g.name == name)


class CalculatorTests(unittest.TestCase):
    def test_mixed_precision_gpt_matrices(self):
        # Independently tabulated matrix storage, including one embedding row.
        expected = 1.9035648 + 1.91102976 + 0.02654208 + 1.15826688 + 0.00000576
        self.assertAlmostEqual(model('gpt-oss-120b').active_weight_gb, expected)
        self.assertGreater(model('Kimi-K3').active_weight_gb, 130)

    def test_decimal_memory_and_sliding_window(self):
        calc = PerformanceCalculator(1)
        m = model('gpt-oss-120b')
        # Growing 18 layers plus 18 layers holding a bounded 128-token window.
        expected = 63 + (131072 + 128) * 18 * 2 * 8 * 64 * 2 / 1e9
        self.assertAlmostEqual(calc.calc_memory_footprint(m, 1, 131072), expected)
        self.assertAlmostEqual(calc.calc_kv_cache_size_per_token(m), 36864 / 2**30)

    def test_state_and_warning_use_same_limit(self):
        calc = PerformanceCalculator(8)
        m, g = model('Kimi-K3'), gpu('B300')
        result = calc.calculate_metrics(m, g, 16000, 512, 2000)
        self.assertEqual(result.max_concurrent_requests, 838)
        self.assertEqual(result.throughput, 'OOM')
        out = io.StringIO()
        with redirect_stdout(out):
            check_memory_requirements(calc, m, g, 16000, 512, 2000)
        self.assertIn('838 concurrent requests', out.getvalue())
        self.assertNotIn('1747', out.getvalue())

    def test_context_oom_with_weights_fitting(self):
        calc = PerformanceCalculator(8)
        result = calc.calculate_metrics(model('GLM-5.2', 'FP8'), gpu('RTX PRO 6000 BW'), 300000, 512)
        self.assertEqual(result.max_concurrent_requests, 0)
        self.assertEqual(result.decode_throughput, 'OOM')
        self.assertEqual(result.ttft, 'OOM')

    def test_weight_oom(self):
        result = PerformanceCalculator(1).calculate_metrics(model('Kimi-K3'), gpu('B300'), 100, 10)
        self.assertEqual(result.decode_throughput, 'OOM')

    def test_context_limit_is_enforced(self):
        result = PerformanceCalculator(8).calculate_metrics(model('Kimi-K3'), gpu('B300'), 1048576, 1)
        self.assertEqual(result.status, 'Context limit exceeded')
        self.assertEqual(result.throughput, 'N/A')

    def test_concurrency_changes_fit_but_not_single_stream_rate(self):
        c, m, g = PerformanceCalculator(8), model('Kimi-K3'), gpu('B300')
        one = c.calculate_metrics(m, g, 16000, 512, 1)
        eight = c.calculate_metrics(m, g, 16000, 512, 8)
        self.assertEqual(one.throughput, eight.throughput)
        self.assertEqual(c.calculate_metrics(m, g, 16000, 512, 839).throughput, 'OOM')

    def test_long_context_costs_more(self):
        c = PerformanceCalculator(8)
        for m in [model('Kimi-K3'), model('GLM-5.2'), model('gpt-oss-120b')]:
            self.assertGreater(c.calc_tpot(m, gpu('B300'), 100000), c.calc_tpot(m, gpu('B300'), 1000))
            self.assertGreater(c.calc_prefill_time_per_token(m, gpu('B300'), 100000),
                               c.calc_prefill_time_per_token(m, gpu('B300'), 1000))

    def test_causal_pair_count(self):
        for n in [1, 32, 128, 129, 4096]:
            for window in [0, 128, 2048]:
                expected = sum(min(i, window) if window else i for i in range(1, n+1))
                self.assertEqual(PerformanceCalculator.causal_pairs(n, window), expected)

    def test_sparse_index_remains_context_dependent(self):
        c, m, g = PerformanceCalculator(8), model('GLM-5.2'), gpu('B300')
        without_index = replace(m, indexer_layers=0, indexer_bytes_per_token=0,
                                kv_bytes_per_token_override=78*576)
        self.assertGreater(c.calc_tpot(m, g, 100000), c.calc_tpot(without_index, g, 100000))
        self.assertEqual(c.calc_tpot(without_index, g, 100000), c.calc_tpot(without_index, g, 10000))

    def test_tensor_communication_and_cache_replication(self):
        m, g = model('Kimi-K3'), gpu('B300')
        ideal = PerformanceCalculator(8)
        tensor = PerformanceCalculator(8, parallelism='tensor', interconnect_gbps=900,
                                       collective_latency_us=5)
        # Two collectives/layer, each with explicit 5us latency plus wire time.
        expected = 186 * (5/1000 + (2*7/8 * 7168*2)/(900*1e9)*1000)
        self.assertAlmostEqual(tensor.communication_ms(m), expected)
        self.assertGreater(tensor.calc_tpot(m, g, 16000), ideal.calc_tpot(m, g, 16000))
        self.assertEqual(tensor.kv_replication(m), 8)
        sharded = PerformanceCalculator(8, parallelism='tensor', interconnect_gbps=900,
                                       kv_cache_sharding='sharded')
        self.assertEqual(sharded.kv_replication(m), 1)
        self.assertLess(tensor.max_concurrent_requests(m,g,16512), sharded.max_concurrent_requests(m,g,16512))

    def test_memory_reserve(self):
        full = PerformanceCalculator(8)
        reserved = PerformanceCalculator(8, memory_utilization=.9)
        m, g = model('Kimi-K3'), gpu('B300')
        self.assertLess(reserved.max_concurrent_requests(m,g,16512), full.max_concurrent_requests(m,g,16512))

    def test_compute_precision_is_explicit(self):
        m, g = model('Kimi-K3'), gpu('B300')
        bf16, fp4 = PerformanceCalculator(8), PerformanceCalculator(8, compute_precision='fp4')
        self.assertLess(fp4.projection_compute_ms(m,g), bf16.projection_compute_ms(m,g))
        self.assertEqual(fp4.calc_tpot(m,g,16000), bf16.calc_tpot(m,g,16000))
        unsupported = fp4.calculate_metrics(model('gpt-oss-20b'), gpu('L4'), 100, 10)
        self.assertEqual(unsupported.throughput, 'N/A')

    def test_cpu_effective_rates_unchanged(self):
        m, d = model('gpt-oss-20b'), CPU_SPECS[-1]
        a, b = PerformanceCalculator(1), PerformanceCalculator(1, calibrated=True)
        self.assertEqual(a.calculate_metrics(m,d,16000,512), b.calculate_metrics(m,d,16000,512))
        self.assertEqual(a.effective_rates(m,d), (64*.066, 614.4*.18))

    def test_metrics_constructor_compatibility(self):
        metrics = PerformanceMetrics(1, 2., 3., 4., 5., 6.)
        self.assertEqual(metrics.decode_throughput, 'OOM')
        self.assertEqual(metrics.status, 'OK')

    def test_invalid_arguments(self):
        for kw in [dict(num_gpu=0), dict(num_gpu=1,memory_utilization=0),
                   dict(num_gpu=1,memory_utilization=math.nan),
                   dict(num_gpu=8,parallelism='tensor')]:
            with self.assertRaises(ValueError):
                PerformanceCalculator(**kw)


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        with tempfile.TemporaryDirectory() as tmp:
            p = subprocess.run([sys.executable, str(ROOT/'LLM_size_pef_calculator.py'), *args],
                               cwd=tmp, capture_output=True, text=True)
            rows = []
            for file in Path(tmp).glob('llm_performance*.csv'):
                with file.open() as stream:
                    rows.extend(csv.DictReader(stream))
            return p, rows

    def test_gpu_defaults_and_csv(self):
        for profile in ['default', 'via']:
            args = ['--profile', profile, '--models', 'gpt-oss-20b', '-c', '1']
            default, a = self.run_cli(*args)
            explicit, b = self.run_cli(*args, '--efficiency', 'theoretical')
            self.assertEqual(default.returncode, 0, default.stderr)
            self.assertEqual(a,b)
            self.assertIn('Single-stream decode rate', a[0])
            self.assertIn('Status', a[0])
            self.assertNotIn('Output Tokens Throughput', a[0])

    def test_cli_oom_suppresses_csv_rates(self):
        p, rows = self.run_cli('--profile','via','--models','Kimi','--devices','B300','-g','8','-c','2000')
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(rows[0]['Single-stream decode rate'],'OOM')
        self.assertEqual(rows[0]['Max Concurrent Req'],'838')
        self.assertIn('838 concurrent requests',p.stdout)

    def test_cpu_reports_effective_mode(self):
        p, rows = self.run_cli('--profile','via','--device','cpu','--models','gpt-oss-20b',
                               '--efficiency','theoretical','-c','1')
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('efficiency = calibrated',p.stdout)

    def test_invalid_cli_and_empty_filter(self):
        for args in [('-g','0'), ('-p','-1'), ('-r','0'), ('-c','0'),
                     ('--models','DeepSeek'), ('-g','8','--parallelism','tensor')]:
            p, rows = self.run_cli(*args)
            self.assertNotEqual(p.returncode,0)
            self.assertEqual(rows,[])


if __name__ == '__main__':
    unittest.main()
