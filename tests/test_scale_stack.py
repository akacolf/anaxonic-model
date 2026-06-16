import unittest

from anaxonic.scale_stack import load_benchmark_manifest, load_scale_stack, stack_summary


class ScaleStackTest(unittest.TestCase):
    def test_scale_stack_manifest_loads_and_summarizes(self):
        payload = load_scale_stack("configs/scale_stack.json")
        summary = stack_summary(payload)

        self.assertGreaterEqual(summary["components"], 8)
        self.assertIn("containerized_runtime", summary["implemented_or_scaffolded"])
        self.assertIn("distributed_training", summary["planned"])

    def test_benchmark_manifest_tracks_scaffold_dependent_benchmarks(self):
        payload = load_benchmark_manifest("configs/benchmark_manifest.json")
        benchmarks = {benchmark["name"]: benchmark for benchmark in payload["benchmarks"]}

        self.assertEqual(benchmarks["local_prompt_eval"]["adapter_status"], "implemented")
        self.assertEqual(benchmarks["swe_bench_pro_style"]["scaffold_dependency"], "very_high")
        self.assertIn("failure_analysis", payload["reporting_required_fields"])


if __name__ == "__main__":
    unittest.main()
