import unittest

from pathpilot.environment import Direction
from pathpilot.learning.actions import NavigationAction


class NavigationActionTests(unittest.TestCase):
    def test_actions_map_to_environment_directions(self) -> None:
        self.assertEqual(NavigationAction.UP.direction, Direction.UP)
        self.assertEqual(NavigationAction.RIGHT.direction, Direction.RIGHT)

    def test_transition_maps_to_action(self) -> None:
        self.assertEqual(
            NavigationAction.from_transition((2, 2), (2, 3)),
            NavigationAction.DOWN,
        )

    def test_non_adjacent_transition_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            NavigationAction.from_transition((0, 0), (2, 0))
        with self.assertRaises(ValueError):
            NavigationAction.from_transition((1, 1), (1, 1))
