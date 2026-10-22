"""Headless paired benchmarks for PathPilot's A* navigation strategies."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

from pathpilot.controllers import (
    AlwaysReplanController,
    CachedPathController,
    NavigationState,
)
from pathpilot.controllers.navigation import TimedNavigationController
from pathpilot.evaluation.scenarios import (
    BenchmarkScenario,
    ObstacleEvent,
    generate_suite,
)


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """Metrics from one strategy on one scenario repetition."""

    scenario: str
    scenario_kind: str
    repetition: int
    strategy: str
    success: bool
    planning_latency_ms: float
    nodes_explored: int
    path_length: int
    planning_calls: int
    replanning_count: int


def run_navigation(
    scenario: BenchmarkScenario,
    controller: TimedNavigationController,
    *,
    max_steps: int | None = None,
) -> BenchmarkRecord:
    """Run one strategy without rendering or real-time waiting."""
    environment = scenario.create_environment()
    controller.start(environment)
    events_by_step = _group_events(scenario.events)
    step_limit = max_steps or scenario.width * scenario.height * 2

    for movement_step in range(step_limit):
        for event in events_by_step.get(movement_step, ()):
            if event.blocked:
                environment.add_obstacle(event.position)
            else:
                environment.remove_obstacle(event.position)
        controller.update(controller.movement_interval_ms, environment)
        if controller.state in {
            NavigationState.SUCCEEDED,
            NavigationState.FAILED,
        }:
            break

    metrics = controller.metrics
    strategy = (
        "always_replan"
        if isinstance(controller, AlwaysReplanController)
        else "cached_path"
    )
    return BenchmarkRecord(
        scenario=scenario.name,
        scenario_kind=scenario.kind,
        repetition=0,
        strategy=strategy,
        success=controller.state is NavigationState.SUCCEEDED,
        planning_latency_ms=metrics.planning_latency_ms,
        nodes_explored=metrics.nodes_explored,
        path_length=metrics.path_length,
        planning_calls=metrics.planning_calls,
        replanning_count=metrics.replanning_count,
    )


def run_paired_benchmark(
    scenarios: Sequence[BenchmarkScenario],
    *,
    repetitions: int = 3,
) -> list[BenchmarkRecord]:
    """Run both strategies repeatedly on identical scenario inputs."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    records: list[BenchmarkRecord] = []
    for repetition in range(repetitions):
        for scenario in scenarios:
            for controller_type in (
                AlwaysReplanController,
                CachedPathController,
            ):
                record = run_navigation(
                    scenario,
                    controller_type(movement_interval_ms=1),
                )
                records.append(replace(record, repetition=repetition))
    return records


def summarize(
    records: Sequence[BenchmarkRecord],
) -> dict[str, dict[str, float]]:
    """Aggregate comparable metrics by strategy and calculate reductions."""
    summary: dict[str, dict[str, float]] = {}
    for strategy in ("always_replan", "cached_path"):
        group = [record for record in records if record.strategy == strategy]
        if not group:
            continue
        summary[strategy] = {
            "runs": float(len(group)),
            "success_rate": mean(record.success for record in group),
            "planning_latency_ms_mean": mean(
                record.planning_latency_ms for record in group
            ),
            "planning_latency_ms_median": median(
                record.planning_latency_ms for record in group
            ),
            "nodes_explored_mean": mean(
                record.nodes_explored for record in group
            ),
            "path_length_mean": mean(record.path_length for record in group),
            "planning_calls_mean": mean(
                record.planning_calls for record in group
            ),
            "replanning_count_mean": mean(
                record.replanning_count for record in group
            ),
        }

    if {"always_replan", "cached_path"}.issubset(summary):
        baseline = summary["always_replan"]
        cached = summary["cached_path"]
        summary["comparison"] = {
            "latency_reduction_percent": _percent_reduction(
                baseline["planning_latency_ms_mean"],
                cached["planning_latency_ms_mean"],
            ),
            "node_reduction_percent": _percent_reduction(
                baseline["nodes_explored_mean"],
                cached["nodes_explored_mean"],
            ),
            "planning_call_reduction_percent": _percent_reduction(
                baseline["planning_calls_mean"],
                cached["planning_calls_mean"],
            ),
        }
    return summary


def write_results(
    records: Sequence[BenchmarkRecord],
    summary: dict[str, dict[str, float]],
    output_directory: Path,
) -> None:
    """Write raw paired observations and aggregate metrics."""
    if not records:
        raise ValueError("Cannot write an empty benchmark")
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(records[0]).keys()),
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    with (output_directory / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def _group_events(
    events: Iterable[ObstacleEvent],
) -> dict[int, tuple[ObstacleEvent, ...]]:
    grouped: dict[int, list[ObstacleEvent]] = {}
    for event in events:
        grouped.setdefault(event.movement_step, []).append(event)
    return {step: tuple(items) for step, items in grouped.items()}


def _percent_reduction(baseline: float, optimized: float) -> float:
    return 0.0 if baseline == 0 else (1 - optimized / baseline) * 100


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--dynamic-obstacles", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("benchmarks"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    scenarios = generate_suite(
        scenario_count=args.scenarios,
        seed=args.seed,
        width=args.width,
        height=args.height,
        obstacle_density=args.density,
        dynamic_obstacles=args.dynamic_obstacles,
    )
    records = run_paired_benchmark(scenarios, repetitions=args.repetitions)
    results = summarize(records)
    write_results(records, results, args.output)
    print(json.dumps(results, indent=2))
    print(f"\nResults written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
