import unittest

from pathpilot.environment import GridEnvironment
from pathpilot.planning import PathPlanner, SearchResult


class StraightLinePlanner:
    """Small test double proving the structural planner contract."""

    def plan(self, environment, start, goal):
        del environment
        return SearchResult(
            path=(start, goal),
            explored_cells=(start,),
            nodes_explored=1,
            planning_latency_ns=100,
        )


class SearchResultTests(unittest.TestCase):
    def test_success_and_path_length_are_derived_from_path(self) -> None:
        result = SearchResult(
            path=((0, 0), (1, 0), (2, 0)),
            explored_cells=((0, 0), (1, 0)),
            nodes_explored=2,
            planning_latency_ns=100,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.path_length, 2)

    def test_empty_path_represents_failure(self) -> None:
        result = SearchResult((), ((0, 0),), 1, 100)
        self.assertFalse(result.success)
        self.assertEqual(result.path_length, 0)

    def test_inconsistent_exploration_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SearchResult((), ((0, 0),), 2, 100)

    def test_protocol_accepts_compatible_planner(self) -> None:
        planner = StraightLinePlanner()
        self.assertIsInstance(planner, PathPlanner)
        result = planner.plan(GridEnvironment(3, 3), (0, 0), (2, 2))
        self.assertTrue(result.success)
