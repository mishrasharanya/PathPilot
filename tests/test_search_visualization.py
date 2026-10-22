import unittest

from pathpilot.controllers import (
    SearchVisualizationController,
    VisualizationPhase,
)
from pathpilot.planning import SearchResult


class SearchVisualizationControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = SearchResult(
            path=((0, 0), (1, 0), (2, 0)),
            explored_cells=((0, 0), (0, 1), (1, 0), (2, 0)),
            nodes_explored=4,
            planning_latency_ns=100,
        )
        self.controller = SearchVisualizationController(step_interval_ms=10)

    def test_reveals_exploration_before_path(self) -> None:
        self.controller.start(self.result)
        self.controller.update(20)

        self.assertEqual(
            self.controller.frame.explored_cells,
            ((0, 0), (0, 1)),
        )
        self.assertEqual(self.controller.frame.path, ())
        self.assertEqual(
            self.controller.frame.phase,
            VisualizationPhase.EXPLORING,
        )

    def test_completes_exploration_then_reveals_path(self) -> None:
        self.controller.start(self.result)
        self.controller.update(50)

        self.assertEqual(self.controller.frame.explored_cells, self.result.explored_cells)
        self.assertEqual(self.controller.frame.path, ((0, 0),))
        self.assertEqual(self.controller.frame.phase, VisualizationPhase.PATH)

    def test_pause_prevents_progress(self) -> None:
        self.controller.start(self.result)
        self.assertTrue(self.controller.toggle_pause())
        self.controller.update(100)

        self.assertEqual(self.controller.frame.explored_cells, ())
        self.assertTrue(self.controller.frame.paused)

        self.assertFalse(self.controller.toggle_pause())
        self.controller.update(10)
        self.assertEqual(self.controller.frame.explored_cells, ((0, 0),))

    def test_stop_clears_visible_data(self) -> None:
        self.controller.start(self.result)
        self.controller.update(30)
        self.controller.stop()

        self.assertEqual(self.controller.frame.explored_cells, ())
        self.assertEqual(self.controller.frame.path, ())
        self.assertEqual(self.controller.frame.phase, VisualizationPhase.IDLE)

    def test_failed_result_still_animates_exploration(self) -> None:
        failed = SearchResult(
            path=(),
            explored_cells=((0, 0), (1, 0)),
            nodes_explored=2,
            planning_latency_ns=100,
        )
        self.controller.start(failed)
        self.controller.update(20)

        self.assertEqual(self.controller.frame.explored_cells, failed.explored_cells)
        self.assertEqual(self.controller.frame.phase, VisualizationPhase.COMPLETE)

    def test_rejects_invalid_time_values(self) -> None:
        with self.assertRaises(ValueError):
            SearchVisualizationController(step_interval_ms=0)
        with self.assertRaises(ValueError):
            self.controller.update(-1)
