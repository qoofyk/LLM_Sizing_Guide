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
sys.path.insert(0, str(ROOT / 'validation'))
from configs.gpu_specs import GPUSpec, VIA_GPU_SPECS
from configs.cpu_specs import CPU_SPECS
from configs.model_specs import ModelSpec, VIA_MODEL_SPECS
from llm_calculator.performance import GPU_CALIBRATION, PerformanceCalculator, PerformanceMetrics
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
                                        collective_latency_us=5, kv_cache_sharding='sharded')
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

    def test_moe_decode_uses_dense_factors_and_prefill_is_unchanged(self):
        self.assertEqual({k: GPU_CALIBRATION['moe'][k] for k in ('bandwidth', 'overhead_ms')},
                         {k: GPU_CALIBRATION['dense'][k] for k in ('bandwidth', 'overhead_ms')})
        self.assertEqual(GPU_CALIBRATION['moe']['bandwidth'], 0.70)
        self.assertEqual(GPU_CALIBRATION['moe']['overhead_ms'], 0.5)
        self.assertEqual(GPU_CALIBRATION['moe']['compute'], 0.15)
        self.assertEqual(GPU_CALIBRATION['dense']['compute'], 0.38)

    def test_kimi_theoretical_and_recalibrated_rates(self):
        m, g = model('Kimi-K3'), gpu('B300')
        theory = PerformanceCalculator(8).calculate_metrics(m, g, 16000, 512, 1)
        cal = PerformanceCalculator(8, calibrated=True).calculate_metrics(m, g, 16000, 512, 1)
        self.assertAlmostEqual(theory.throughput, 386.62, places=2)
        self.assertAlmostEqual(theory.decode_throughput, 464.11, places=2)
        self.assertAlmostEqual(cal.throughput, 154.84, places=2)
        self.assertAlmostEqual(cal.decode_throughput, 279.39, places=2)

    def test_legacy_gpt_oss_decode_ratios_after_recalibration(self):
        import sanity_check
        ratios = {}
        for a in sanity_check.LEGACY_ANCHORS:
            if a.metric == 'decode' and a.model.name == 'gpt-oss-120b':
                ratios['pro6000'] = sanity_check.estimate(a) / a.measured
            if a.metric == 'decode' and a.model.name == 'gpt-oss-20b':
                ratios[a.device.name] = sanity_check.estimate(a) / a.measured
        self.assertAlmostEqual(ratios['L4'], 0.918, places=3)
        self.assertAlmostEqual(ratios['L40s'], 0.943, places=3)
        self.assertAlmostEqual(ratios['pro6000'], 0.957, places=3)
        hold = [sanity_check.estimate(a) / a.measured for a in sanity_check.LEGACY_ANCHORS
                if a.kind == 'hold-out']
        self.assertEqual(len(hold), 5)
        self.assertAlmostEqual(min(hold), 0.84, places=2)
        self.assertAlmostEqual(max(hold), 1.16, places=2)
        outliers = [a for a in sanity_check.LEGACY_ANCHORS if a.kind == 'outlier']
        self.assertEqual([(a.model.name, a.device.name) for a in outliers], [('gpt-oss-20b', 'H100 SXM')])

    def test_source_aware_anchors_are_separate_and_documented(self):
        import sanity_check
        legacy = {id(a) for a in sanity_check.LEGACY_ANCHORS}
        self.assertFalse(legacy & {id(a) for a in sanity_check.SOURCE_ANCHORS})
        self.assertTrue(all(a.source_url and a.note for a in sanity_check.SOURCE_ANCHORS))
        long_context = [a for a in sanity_check.SOURCE_ANCHORS if a.kind == 'validation-only']
        self.assertEqual([(a.prompt_tokens, a.measured) for a in long_context], [(61357, 54.082)])

    def test_tensor_latency_missing_zero_positive_invalid(self):
        with self.assertRaises(ValueError):
            PerformanceCalculator(8, parallelism='tensor', interconnect_gbps=900)
        # Omission stays valid when no collectives occur.
        PerformanceCalculator(8)
        PerformanceCalculator(1, parallelism='tensor')
        zero = PerformanceCalculator(8, parallelism='tensor', interconnect_gbps=900, collective_latency_us=0)
        ten = PerformanceCalculator(8, parallelism='tensor', interconnect_gbps=900, collective_latency_us=10)
        self.assertEqual(zero.collective_latency_us, 0)
        m, g = model('gpt-oss-120b'), gpu('H100 SXM')
        # Two collectives per layer x 36 layers x 10 us = 0.72 ms of added decode communication.
        self.assertAlmostEqual(ten.communication_ms(m) - zero.communication_ms(m), 0.72)
        self.assertAlmostEqual(ten.calc_tpot(m, g, 16000) - zero.calc_tpot(m, g, 16000), 0.72)
        for bad in (-1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                PerformanceCalculator(8, parallelism='tensor', interconnect_gbps=900,
                                      collective_latency_us=bad)

    def test_direct_python_reserve_default_is_unchanged(self):
        self.assertEqual(PerformanceCalculator(8).memory_utilization, 1.0)
        m, g = model('Kimi-K3'), gpu('B300')
        self.assertEqual(PerformanceCalculator(8).max_concurrent_requests(m, g, 16512), 838)
        self.assertEqual(PerformanceCalculator(8, memory_utilization=.9).max_concurrent_requests(m, g, 16512), 578)

    def test_reserve_changes_capacity_not_latency(self):
        m, g = model('Kimi-K3'), gpu('B300')
        full = PerformanceCalculator(8).calculate_metrics(m, g, 16000, 512, 1)
        reserved = PerformanceCalculator(8, memory_utilization=.9).calculate_metrics(m, g, 16000, 512, 1)
        self.assertEqual((full.ttft, full.tpot, full.throughput), (reserved.ttft, reserved.tpot, reserved.throughput))
        self.assertGreater(full.max_concurrent_requests, reserved.max_concurrent_requests)

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
        self.assertEqual(rows[0]['Max Concurrent Req'],'578')
        self.assertIn('578 concurrent requests',p.stdout)
        p, rows = self.run_cli('--profile','via','--models','Kimi','--devices','B300','-g','8','-c','2000',
                               '--memory-utilization','1')
        self.assertEqual(rows[0]['Max Concurrent Req'],'838')

    def test_cpu_reports_effective_mode(self):
        p, rows = self.run_cli('--profile','via','--device','cpu','--models','gpt-oss-20b',
                               '--efficiency','theoretical','-c','1')
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('efficiency = calibrated',p.stdout)

    def test_via_gpu_reserve_default_and_overrides(self):
        base = ['--models', 'Kimi', '--devices', 'B300', '-g', '8', '-c', '1']
        via, via_rows = self.run_cli('--profile', 'via', *base)
        self.assertIn('Memory reserve: 10% of device memory (profile default)', via.stdout)
        self.assertEqual(via_rows[0]['Max Concurrent Req'], '578')
        # The default profile (and its own catalog) keeps the full memory budget unless overridden.
        plain = ['--profile', 'default', '--models', 'Llama-3.1-8B', '--devices', 'H100 NVL', '-g', '1', '-c', '1']
        default, default_rows = self.run_cli(*plain)
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertIn('Memory reserve: 0% of device memory (profile default)', default.stdout)
        override, override_rows = self.run_cli(*plain, '--memory-utilization', '0.9')
        self.assertIn('Memory reserve: 10% of device memory (explicit)', override.stdout)
        self.assertLess(int(override_rows[0]['Max Concurrent Req']), int(default_rows[0]['Max Concurrent Req']))
        explicit, explicit_rows = self.run_cli('--profile', 'via', '--memory-utilization', '1', *base)
        self.assertIn('Memory reserve: 0% of device memory (explicit)', explicit.stdout)
        self.assertEqual(explicit_rows[0]['Max Concurrent Req'], '838')
        # The reserve changes capacity, not latency, for a workload that fits.
        for key in ('TTFT', 'TPOT (ms)', 'E2E Latency'):
            self.assertEqual(via_rows[0][key], explicit_rows[0][key])

    def test_cpu_keeps_full_memory_budget(self):
        p, rows = self.run_cli('--profile', 'via', '--device', 'cpu', '--models', 'gpt-oss-20b', '-c', '1')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('memory utilization=1', p.stdout)
        self.assertIn('Memory reserve: 0% of device memory (profile default)', p.stdout)

    def test_via_reserve_decides_fit_and_oom(self):
        args = ['--profile', 'via', '--models', 'Kimi', '--devices', 'B300', '-g', '8']
        fits, rows = self.run_cli(*args, '-c', '578')
        self.assertEqual(rows[0]['Status'], 'OK')
        oom, rows = self.run_cli(*args, '-c', '579')
        self.assertTrue(rows[0]['Status'].startswith('OOM'))
        self.assertEqual(rows[0]['Single-stream decode rate'], 'OOM')

    def test_tensor_latency_cli(self):
        base = ['--profile', 'via', '-g', '8', '--parallelism', 'tensor', '--interconnect-gbps', '900',
                '--models', 'gpt-oss-120b', '--devices', 'H100 SXM', '-c', '1']
        missing, rows = self.run_cli(*base)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn('--collective-latency-us', missing.stderr)
        self.assertEqual(rows, [])
        zero, _ = self.run_cli(*base, '--collective-latency-us', '0')
        self.assertEqual(zero.returncode, 0, zero.stderr)
        self.assertIn('lower bound', zero.stdout)
        self.assertIn('Expert parallelism and all-to-all communication are not modeled', zero.stdout)
        ten, _ = self.run_cli(*base, '--collective-latency-us', '10')
        self.assertEqual(ten.returncode, 0, ten.stderr)
        self.assertNotIn('lower bound', ten.stdout)
        self.assertIn('Expert parallelism and all-to-all communication are not modeled', ten.stdout)
        bad, _ = self.run_cli(*base, '--collective-latency-us', '-5')
        self.assertNotEqual(bad.returncode, 0)
        # Ideal mode does not require a latency.
        ideal, _ = self.run_cli('--profile', 'via', '-g', '8', '--models', 'gpt-oss-120b',
                                '--devices', 'H100 SXM', '-c', '1')
        self.assertEqual(ideal.returncode, 0, ideal.stderr)

    def test_invalid_cli_and_empty_filter(self):
        for args in [('-g','0'), ('-p','-1'), ('-r','0'), ('-c','0'),
                     ('--models','DeepSeek'), ('-g','8','--parallelism','tensor')]:
            p, rows = self.run_cli(*args)
            self.assertNotEqual(p.returncode,0)
            self.assertEqual(rows,[])


if __name__ == '__main__':
    unittest.main()
