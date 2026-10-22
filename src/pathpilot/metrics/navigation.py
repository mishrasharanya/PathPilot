"""Metrics collected by autonomous navigation controllers."""

from __future__ import annotations

from dataclasses import dataclass

from pathpilot.planning import SearchResult


@dataclass(slots=True)
class NavigationMetrics:
    """Mutable counters owned by one navigation run."""

    planning_latency_ns: int = 0
    nodes_explored: int = 0
    planning_calls: int = 0
    replanning_count: int = 0
    path_length: int = 0

    @property
    def planning_latency_ms(self) -> float:
        return self.planning_latency_ns / 1_000_000

    def record_plan(self, result: SearchResult) -> None:
        """Accumulate one planner invocation."""
        if self.planning_calls > 0:
            self.replanning_count += 1
        self.planning_calls += 1
        self.planning_latency_ns += result.planning_latency_ns
        self.nodes_explored += result.nodes_explored

    def record_movement(self) -> None:
        """Record one successful grid movement."""
        self.path_length += 1
