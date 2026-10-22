import unittest

from pathpilot.environment import Direction, GridEnvironment, MoveResult


class GridEnvironmentTests(unittest.TestCase):
    def test_default_state(self) -> None:
        environment = GridEnvironment(width=5, height=4)
        self.assertEqual(environment.start, (0, 0))
        self.assertEqual(environment.goal, (4, 3))
        self.assertEqual(environment.agent, environment.start)
        self.assertEqual(environment.obstacles, set())
        self.assertFalse(environment.reached_goal)

    def test_valid_movement_updates_agent(self) -> None:
        environment = GridEnvironment(width=3, height=3)
        result = environment.move(Direction.RIGHT)
        self.assertEqual(result, MoveResult.MOVED)
        self.assertEqual(environment.agent, (1, 0))

    def test_neighbors_exclude_obstacles_and_grid_boundaries(self) -> None:
        environment = GridEnvironment(
            width=3,
            height=3,
            obstacles={(1, 0)},
        )
        self.assertEqual(environment.neighbors((0, 0)), ((0, 1),))

    def test_neighbors_have_deterministic_order(self) -> None:
        environment = GridEnvironment(width=3, height=3)
        self.assertEqual(
            environment.neighbors((1, 1)),
            ((2, 1), (1, 2), (0, 1), (1, 0)),
        )

    def test_agent_cannot_leave_grid(self) -> None:
        environment = GridEnvironment(width=3, height=3)
        result = environment.move(Direction.UP)
        self.assertEqual(result, MoveResult.OUT_OF_BOUNDS)
        self.assertEqual(environment.agent, (0, 0))

    def test_obstacle_blocks_movement(self) -> None:
        environment = GridEnvironment(width=3, height=3, obstacles={(1, 0)})
        result = environment.move(Direction.RIGHT)
        self.assertEqual(result, MoveResult.BLOCKED_BY_OBSTACLE)
        self.assertEqual(environment.agent, (0, 0))

    def test_reaching_goal_has_distinct_result(self) -> None:
        environment = GridEnvironment(width=2, height=1)
        result = environment.move(Direction.RIGHT)
        self.assertEqual(result, MoveResult.REACHED_GOAL)
        self.assertTrue(environment.reached_goal)

    def test_obstacles_cannot_cover_reserved_cells(self) -> None:
        environment = GridEnvironment(width=4, height=4)
        self.assertFalse(environment.add_obstacle(environment.start))
        self.assertFalse(environment.add_obstacle(environment.goal))
        self.assertFalse(environment.add_obstacle(environment.agent))
        self.assertEqual(environment.obstacles, set())

    def test_obstacle_can_be_added_and_removed(self) -> None:
        environment = GridEnvironment(width=4, height=4)
        self.assertTrue(environment.add_obstacle((1, 1)))
        self.assertIn((1, 1), environment.obstacles)
        self.assertFalse(environment.add_obstacle((1, 1)))
        self.assertTrue(environment.remove_obstacle((1, 1)))
        self.assertFalse(environment.remove_obstacle((1, 1)))

    def test_setting_start_moves_agent(self) -> None:
        environment = GridEnvironment(width=4, height=4)
        environment.move(Direction.RIGHT)
        self.assertTrue(environment.set_start((2, 2)))
        self.assertEqual(environment.start, (2, 2))
        self.assertEqual(environment.agent, (2, 2))

    def test_endpoints_reject_invalid_cells(self) -> None:
        environment = GridEnvironment(width=4, height=4, obstacles={(1, 1)})
        self.assertFalse(environment.set_start((1, 1)))
        self.assertFalse(environment.set_start(environment.goal))
        self.assertFalse(environment.set_goal((1, 1)))
        self.assertFalse(environment.set_goal(environment.start))

    def test_reset_agent_preserves_board(self) -> None:
        environment = GridEnvironment(width=4, height=4)
        environment.add_obstacle((2, 2))
        environment.move(Direction.RIGHT)
        environment.reset_agent()
        self.assertEqual(environment.agent, environment.start)
        self.assertEqual(environment.obstacles, {(2, 2)})

    def test_reset_restores_clean_initial_board(self) -> None:
        environment = GridEnvironment(width=4, height=4, start=(1, 1), goal=(3, 3))
        environment.set_start((0, 0))
        environment.set_goal((2, 2))
        environment.add_obstacle((1, 0))
        environment.reset()
        self.assertEqual(environment.start, (1, 1))
        self.assertEqual(environment.goal, (3, 3))
        self.assertEqual(environment.agent, (1, 1))
        self.assertEqual(environment.obstacles, set())

    def test_invalid_initial_state_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GridEnvironment(width=2, height=2, start=(0, 0), goal=(0, 0))
        with self.assertRaises(ValueError):
            GridEnvironment(width=2, height=2, obstacles={(5, 5)})
