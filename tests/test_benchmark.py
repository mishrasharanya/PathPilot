import json
import tempfile
import unittest
from pathlib import Path

from pathpilot.controllers import AlwaysReplanController, CachedPathController
from pathpilot.evaluation.benchmark import (
    run_navigation,
    run_paired_benchmark,
    summarize,
    write_results,
)
from pathpilot.evaluation.scenarios import generate_scenario, generate_suite


class ScenarioTests(unittest.TestCase):
    def test_generation_is_reproducible(self) -> None:
        first = generate_scenario(seed=17, dynamic_obstacles=2)
        second = generate_scenario(seed=17, dynamic_obstacles=2)
        self.assertEqual(first, second)

    def test_suite_contains_static_and_dynamic_maps(self) -> None:
        scenarios = generate_suite(
            scenario_count=4,
            seed=42,
            width=12,
            height=8,
            obstacle_density=0.2,
            dynamic_obstacles=2,
        )
        self.assertEqual(
            {scenario.kind for scenario in scenarios},
            {"static", "dynamic"},
        )


class BenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenarios = generate_suite(
            scenario_count=4,
            seed=42,
            width=12,
            height=8,
            obstacle_density=0.15,
            dynamic_obstacles=2,
        )

    def test_paired_runs_use_same_scenarios_and_repetitions(self) -> None:
        records = run_paired_benchmark(self.scenarios, repetitions=2)
        self.assertEqual(len(records), 16)

        pair_keys = {
            (record.scenario, record.repetition)
            for record in records
        }
        for key in pair_keys:
            pair = [
                record
                for record in records
                if (record.scenario, record.repetition) == key
            ]
            self.assertEqual(
                {record.strategy for record in pair},
                {"always_replan", "cached_path"},
            )
            self.assertEqual(pair[0].success, pair[1].success)
            self.assertEqual(pair[0].path_length, pair[1].path_length)

    def test_cached_path_uses_fewer_plans_on_static_map(self) -> None:
        scenario = generate_scenario(seed=9, width=12, height=8)
        baseline = run_navigation(
            scenario,
            AlwaysReplanController(movement_interval_ms=1),
        )
        cached = run_navigation(
            scenario,
            CachedPathController(movement_interval_ms=1),
        )
        self.assertTrue(baseline.success)
        self.assertTrue(cached.success)
        self.assertLess(cached.planning_calls, baseline.planning_calls)
        self.assertLess(cached.nodes_explored, baseline.nodes_explored)

    def test_dynamic_route_blockers_trigger_cached_replanning(self) -> None:
        scenario = generate_scenario(
            seed=9,
            width=12,
            height=8,
            dynamic_obstacles=2,
        )
        cached = run_navigation(
            scenario,
            CachedPathController(movement_interval_ms=1),
        )
        self.assertTrue(cached.success)
        self.assertGreaterEqual(cached.replanning_count, 1)

    def test_summary_contains_required_metrics_and_comparison(self) -> None:
        summary = summarize(
            run_paired_benchmark(self.scenarios, repetitions=1)
        )
        self.assertIn("success_rate", summary["always_replan"])
        self.assertIn("planning_latency_ms_mean", summary["cached_path"])
        self.assertIn("nodes_explored_mean", summary["cached_path"])
        self.assertIn("path_length_mean", summary["cached_path"])
        self.assertIn("replanning_count_mean", summary["cached_path"])
        self.assertIn("latency_reduction_percent", summary["comparison"])

    def test_results_are_written_as_csv_and_json(self) -> None:
        records = run_paired_benchmark(self.scenarios, repetitions=1)
        summary = summarize(records)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            write_results(records, summary, destination)

            csv_text = (destination / "results.csv").read_text()
            parsed_summary = json.loads(
                (destination / "summary.json").read_text()
            )
            self.assertIn("scenario,scenario_kind", csv_text)
            self.assertEqual(parsed_summary, summary)
