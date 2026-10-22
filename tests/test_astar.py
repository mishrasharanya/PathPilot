import unittest

from pathpilot.environment import GridEnvironment
from pathpilot.planning import AStarPlanner, PathPlanner


class AStarPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = AStarPlanner()

    def test_astar_implements_planner_contract(self) -> None:
        self.assertIsInstance(self.planner, PathPlanner)

    def test_manhattan_distance(self) -> None:
        self.assertEqual(self.planner.heuristic((1, 2), (4, 6)), 7)
        self.assertEqual(self.planner.heuristic((4, 6), (1, 2)), 7)

    def test_finds_straight_shortest_path(self) -> None:
        environment = GridEnvironment(width=4, height=2, goal=(3, 0))
        result = self.planner.plan(environment, environment.start, environment.goal)

        self.assertTrue(result.success)
        self.assertEqual(result.path, ((0, 0), (1, 0), (2, 0), (3, 0)))
        self.assertEqual(result.path_length, 3)
        self.assertEqual(result.explored_cells[0], environment.start)
        self.assertEqual(result.explored_cells[-1], environment.goal)
        self.assertGreater(result.planning_latency_ns, 0)

    def test_finds_shortest_path_around_obstacle(self) -> None:
        environment = GridEnvironment(
            width=5,
            height=3,
            goal=(4, 0),
            obstacles={(2, 0)},
        )
        result = self.planner.plan(environment, environment.start, environment.goal)

        self.assertTrue(result.success)
        self.assertEqual(result.path_length, 6)
        self.assertNotIn((2, 0), result.path)
        self.assertEqual(result.path[0], environment.start)
        self.assertEqual(result.path[-1], environment.goal)

    def test_reports_failure_when_goal_is_unreachable(self) -> None:
        environment = GridEnvironment(
            width=3,
            height=3,
            obstacles={(1, 0), (0, 1)},
        )
        result = self.planner.plan(environment, environment.start, environment.goal)

        self.assertFalse(result.success)
        self.assertEqual(result.path, ())
        self.assertEqual(result.explored_cells, ((0, 0),))
        self.assertEqual(result.nodes_explored, 1)

    def test_start_equal_to_goal_returns_zero_length_path(self) -> None:
        environment = GridEnvironment(width=3, height=3)
        result = self.planner.plan(environment, (1, 1), (1, 1))

        self.assertTrue(result.success)
        self.assertEqual(result.path, ((1, 1),))
        self.assertEqual(result.path_length, 0)
        self.assertEqual(result.nodes_explored, 1)

    def test_invalid_or_blocked_endpoint_returns_failure(self) -> None:
        environment = GridEnvironment(
            width=3,
            height=3,
            obstacles={(1, 1)},
        )
        blocked = self.planner.plan(environment, (1, 1), environment.goal)
        outside = self.planner.plan(environment, (-1, 0), environment.goal)

        self.assertFalse(blocked.success)
        self.assertFalse(outside.success)
        self.assertEqual(blocked.nodes_explored, 0)
        self.assertEqual(outside.nodes_explored, 0)

    def test_planning_does_not_mutate_environment(self) -> None:
        environment = GridEnvironment(
            width=4,
            height=4,
            obstacles={(1, 1), (2, 1)},
        )
        original_state = (
            environment.start,
            environment.goal,
            environment.agent,
            set(environment.obstacles),
        )

        self.planner.plan(environment, environment.start, environment.goal)

        self.assertEqual(
            (
                environment.start,
                environment.goal,
                environment.agent,
                environment.obstacles,
            ),
            original_state,
        )

    def test_search_is_deterministic_except_for_latency(self) -> None:
        environment = GridEnvironment(width=5, height=5)
        first = self.planner.plan(environment, environment.start, environment.goal)
        second = self.planner.plan(environment, environment.start, environment.goal)

        self.assertEqual(first.path, second.path)
        self.assertEqual(first.explored_cells, second.explored_cells)
        self.assertEqual(first.nodes_explored, second.nodes_explored)
