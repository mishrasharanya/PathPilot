import unittest

from pathpilot.environment import GridEnvironment, MoveResult
from pathpilot.game import apply_grid_tool, movement_status
from pathpilot.pygame_ui import Tool


class GameControlTests(unittest.TestCase):
    def test_obstacle_tool_adds_and_erases(self) -> None:
        environment = GridEnvironment(width=4, height=4)
        self.assertEqual(apply_grid_tool(environment, Tool.OBSTACLES, (1, 1)), "Obstacle added")
        self.assertIn((1, 1), environment.obstacles)
        self.assertEqual(
            apply_grid_tool(environment, Tool.OBSTACLES, (1, 1), erase=True),
            "Obstacle removed",
        )

    def test_start_and_goal_tools_update_environment(self) -> None:
        environment = GridEnvironment(width=4, height=4)
        self.assertEqual(apply_grid_tool(environment, Tool.SET_START, (1, 1)), "Start updated")
        self.assertEqual(environment.start, (1, 1))
        self.assertEqual(environment.agent, (1, 1))
        self.assertEqual(apply_grid_tool(environment, Tool.SET_GOAL, (2, 2)), "Goal updated")
        self.assertEqual(environment.goal, (2, 2))

    def test_play_tool_does_not_edit_grid(self) -> None:
        environment = GridEnvironment(width=4, height=4)
        apply_grid_tool(environment, Tool.PLAY, (1, 1))
        self.assertEqual(environment.obstacles, set())

    def test_every_movement_result_has_status_text(self) -> None:
        for result in MoveResult:
            self.assertTrue(movement_status(result))
