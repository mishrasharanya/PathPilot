import unittest

from pathpilot.ui import (
    AIAction,
    Tool,
    ai_button_bounds,
    calculate_layout,
    grid_position_at,
    toolbar_button_bounds,
)


class LayoutTests(unittest.TestCase):
    def test_grid_uses_square_cells_and_fits_left_of_sidebar(self) -> None:
        layout = calculate_layout(20, 15)
        self.assertEqual(layout.grid_width, layout.cell_size * 20)
        self.assertEqual(layout.grid_height, layout.cell_size * 15)
        self.assertLessEqual(layout.grid_left + layout.grid_width, layout.sidebar_left)
        self.assertLessEqual(layout.grid_top + layout.grid_height, layout.window_height)

    def test_pixel_coordinates_map_to_grid_cells(self) -> None:
        layout = calculate_layout(20, 15)
        self.assertEqual(grid_position_at(layout, (layout.grid_left, layout.grid_top)), (0, 0))
        self.assertEqual(
            grid_position_at(
                layout,
                (layout.grid_left + layout.grid_width - 1, layout.grid_top + layout.grid_height - 1),
            ),
            (19, 14),
        )
        self.assertIsNone(grid_position_at(layout, (layout.grid_left - 1, layout.grid_top)))

    def test_toolbar_buttons_do_not_overlap(self) -> None:
        buttons = toolbar_button_bounds(calculate_layout(20, 15))
        self.assertEqual(tuple(button.tool for button in buttons), tuple(Tool))
        for current, following in zip(buttons, buttons[1:]):
            self.assertLessEqual(current.top + current.height, following.top)

    def test_ai_buttons_fit_sidebar_without_overlap(self) -> None:
        layout = calculate_layout(20, 15)
        buttons = ai_button_bounds(layout)
        self.assertEqual(tuple(button.action for button in buttons), tuple(AIAction))
        self.assertLessEqual(buttons[0].left + buttons[0].width, buttons[1].left)
        self.assertLessEqual(buttons[1].left + buttons[1].width, buttons[2].left)
        self.assertLessEqual(buttons[0].top + buttons[0].height, buttons[3].top)
        self.assertLessEqual(buttons[3].left + buttons[3].width, buttons[4].left)
        self.assertLessEqual(buttons[-1].top + buttons[-1].height, layout.window_height)

