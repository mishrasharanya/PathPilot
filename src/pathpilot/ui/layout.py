"""Pygame-independent layout geometry and hit-testing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pathpilot.environment import Position

WINDOW_SIZE = (1200, 840)
WINDOW_MARGIN = 32
HEADER_HEIGHT = 76
SIDEBAR_WIDTH = 250
CONTENT_GAP = 28


class Tool(Enum):
    """User-selectable modes and immediate interface actions."""

    OBSTACLES = "Obstacles"
    SET_START = "Set Start"
    SET_GOAL = "Set Goal"
    PLAY = "Play"
    RESET_AGENT = "Reset Agent"
    CLEAR_GRID = "Clear Grid"


class AIAction(Enum):
    """Immediate controls for A* visualization playback."""

    START = "Plan A*"
    RUN_BASELINE = "Baseline"
    RUN_CACHED = "Cached"
    PAUSE = "Pause"
    STOP = "Stop"


SELECTABLE_TOOLS = frozenset(
    {Tool.OBSTACLES, Tool.SET_START, Tool.SET_GOAL, Tool.PLAY}
)


@dataclass(frozen=True, slots=True)
class Layout:
    """Pixel geometry derived from the window and grid dimensions."""

    window_width: int
    window_height: int
    grid_left: int
    grid_top: int
    grid_width: int
    grid_height: int
    cell_size: int
    sidebar_left: int
    sidebar_top: int
    sidebar_height: int


@dataclass(frozen=True, slots=True)
class ButtonBounds:
    """Pixel bounds for one control-panel action."""

    tool: Tool
    left: int
    top: int
    width: int
    height: int

    def contains(self, point: tuple[int, int]) -> bool:
        x, y = point
        return (
            self.left <= x < self.left + self.width
            and self.top <= y < self.top + self.height
        )


@dataclass(frozen=True, slots=True)
class AIButtonBounds:
    """Pixel bounds for an A* playback control."""

    action: AIAction
    left: int
    top: int
    width: int
    height: int

    def contains(self, point: tuple[int, int]) -> bool:
        x, y = point
        return (
            self.left <= x < self.left + self.width
            and self.top <= y < self.top + self.height
        )


def calculate_layout(
    columns: int,
    rows: int,
    window_size: tuple[int, int] = WINDOW_SIZE,
) -> Layout:
    """Fit a square-cell grid beside a fixed-width control panel."""
    if columns <= 0 or rows <= 0:
        raise ValueError("Grid dimensions must be positive")

    window_width, window_height = window_size
    content_top = WINDOW_MARGIN + HEADER_HEIGHT + 20
    content_bottom = window_height - WINDOW_MARGIN
    available_height = content_bottom - content_top
    available_width = (
        window_width - 2 * WINDOW_MARGIN - SIDEBAR_WIDTH - CONTENT_GAP
    )
    cell_size = min(available_width // columns, available_height // rows)
    if cell_size < 1:
        raise ValueError("Window is too small to display this grid")

    grid_width = cell_size * columns
    grid_height = cell_size * rows
    grid_left = WINDOW_MARGIN + (available_width - grid_width) // 2
    grid_top = content_top + (available_height - grid_height) // 2
    sidebar_left = window_width - WINDOW_MARGIN - SIDEBAR_WIDTH
    return Layout(
        window_width=window_width,
        window_height=window_height,
        grid_left=grid_left,
        grid_top=grid_top,
        grid_width=grid_width,
        grid_height=grid_height,
        cell_size=cell_size,
        sidebar_left=sidebar_left,
        sidebar_top=content_top,
        sidebar_height=available_height,
    )


def toolbar_button_bounds(layout: Layout) -> tuple[ButtonBounds, ...]:
    """Calculate non-overlapping bounds for every control-panel action."""
    button_left = layout.sidebar_left + 18
    button_width = SIDEBAR_WIDTH - 36
    button_height = 42
    gap = 8
    first_top = layout.sidebar_top + 82
    return tuple(
        ButtonBounds(
            tool=tool,
            left=button_left,
            top=first_top + index * (button_height + gap),
            width=button_width,
            height=button_height,
        )
        for index, tool in enumerate(Tool)
    )


def ai_button_bounds(layout: Layout) -> tuple[AIButtonBounds, ...]:
    """Calculate bounds for start, pause, and stop visualization actions."""
    left = layout.sidebar_left + 18
    usable_width = SIDEBAR_WIDTH - 36
    section_top = layout.sidebar_top + 420
    half_width = (usable_width - 8) // 2
    third_width = (usable_width - 16) // 3
    return (
        AIButtonBounds(AIAction.START, left, section_top, third_width, 40),
        AIButtonBounds(
            AIAction.RUN_BASELINE,
            left + third_width + 8,
            section_top,
            third_width,
            40,
        ),
        AIButtonBounds(
            AIAction.RUN_CACHED,
            left + 2 * (third_width + 8),
            section_top,
            third_width,
            40,
        ),
        AIButtonBounds(AIAction.PAUSE, left, section_top + 48, half_width, 38),
        AIButtonBounds(
            AIAction.STOP,
            left + half_width + 8,
            section_top + 48,
            half_width,
            38,
        ),
    )


def grid_position_at(layout: Layout, point: tuple[int, int]) -> Position | None:
    """Translate a screen pixel into a grid position, if it hits the board."""
    x, y = point
    if not (
        layout.grid_left <= x < layout.grid_left + layout.grid_width
        and layout.grid_top <= y < layout.grid_top + layout.grid_height
    ):
        return None
    return (
        (x - layout.grid_left) // layout.cell_size,
        (y - layout.grid_top) // layout.cell_size,
    )
