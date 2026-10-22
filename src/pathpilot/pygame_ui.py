"""Modern Pygame presentation and hit-testing for PathPilot."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from pathpilot.environment import GridEnvironment, Position

if TYPE_CHECKING:
    import pygame


WINDOW_SIZE = (1200, 840)
WINDOW_MARGIN = 32
HEADER_HEIGHT = 76
SIDEBAR_WIDTH = 250
CONTENT_GAP = 28


class Tool(Enum):
    OBSTACLES = "Obstacles"
    SET_START = "Set Start"
    SET_GOAL = "Set Goal"
    PLAY = "Play"
    RESET_AGENT = "Reset Agent"
    CLEAR_GRID = "Clear Grid"


SELECTABLE_TOOLS = {Tool.OBSTACLES, Tool.SET_START, Tool.SET_GOAL, Tool.PLAY}


@dataclass(frozen=True)
class Layout:
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


@dataclass(frozen=True)
class ButtonBounds:
    tool: Tool
    left: int
    top: int
    width: int
    height: int

    def contains(self, point: tuple[int, int]) -> bool:
        x, y = point
        return self.left <= x < self.left + self.width and self.top <= y < self.top + self.height


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
    available_width = window_width - 2 * WINDOW_MARGIN - SIDEBAR_WIDTH - CONTENT_GAP
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
    button_left = layout.sidebar_left + 18
    button_width = SIDEBAR_WIDTH - 36
    button_height = 48
    gap = 10
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


class PygameUI:
    """Draw environment state and expose control hit-testing."""

    BACKGROUND = (8, 13, 26)
    SURFACE = (17, 25, 43)
    SURFACE_LIGHT = (25, 36, 58)
    CELL = (232, 238, 246)
    CELL_HOVER = (216, 229, 247)
    GRID_LINE = (190, 202, 219)
    OBSTACLE = (38, 49, 68)
    START = (22, 181, 126)
    GOAL = (139, 92, 246)
    AGENT = (255, 176, 46)
    ACCENT = (63, 140, 255)
    DANGER = (225, 82, 104)
    TEXT = (245, 248, 252)
    MUTED = (153, 167, 190)

    def __init__(self, surface: pygame.Surface, environment: GridEnvironment) -> None:
        import pygame

        self._pygame = pygame
        self.surface = surface
        self.layout = calculate_layout(environment.width, environment.height, surface.get_size())
        self.title_font = pygame.font.SysFont("Avenir Next", 34, bold=True)
        self.heading_font = pygame.font.SysFont("Avenir Next", 20, bold=True)
        self.body_font = pygame.font.SysFont("Avenir Next", 17)
        self.small_font = pygame.font.SysFont("Avenir Next", 14)
        self.marker_font = pygame.font.SysFont("Avenir Next", max(16, self.layout.cell_size // 2), bold=True)

    def tool_at(self, point: tuple[int, int]) -> Tool | None:
        for bounds in toolbar_button_bounds(self.layout):
            if bounds.contains(point):
                return bounds.tool
        return None

    def grid_position_at(self, point: tuple[int, int]) -> Position | None:
        return grid_position_at(self.layout, point)

    def draw(
        self,
        environment: GridEnvironment,
        active_tool: Tool,
        status: str,
        mouse_position: tuple[int, int] | None = None,
    ) -> None:
        self.surface.fill(self.BACKGROUND)
        self._draw_header(environment, status)
        self._draw_board(environment, active_tool, mouse_position)
        self._draw_sidebar(active_tool)

    def _draw_header(self, environment: GridEnvironment, status: str) -> None:
        pygame = self._pygame
        header = pygame.Rect(WINDOW_MARGIN, WINDOW_MARGIN, self.layout.window_width - 2 * WINDOW_MARGIN, HEADER_HEIGHT)
        pygame.draw.rect(self.surface, self.SURFACE, header, border_radius=18)
        pygame.draw.circle(self.surface, self.ACCENT, (header.left + 28, header.centery), 10)

        title = self.title_font.render("PathPilot", True, self.TEXT)
        self.surface.blit(title, (header.left + 48, header.top + 12))
        subtitle = self.small_font.render("Interactive navigation lab", True, self.MUTED)
        self.surface.blit(subtitle, (header.left + 50, header.top + 48))

        status_color = self.START if environment.reached_goal else self.ACCENT
        status_text = "Goal reached" if environment.reached_goal else status
        rendered = self.body_font.render(status_text, True, self.TEXT)
        chip = pygame.Rect(0, 0, rendered.get_width() + 34, 38)
        chip.midright = (header.right - 18, header.centery)
        pygame.draw.rect(self.surface, status_color, chip, border_radius=19)
        self.surface.blit(rendered, rendered.get_rect(center=chip.center))

    def _draw_board(
        self,
        environment: GridEnvironment,
        active_tool: Tool,
        mouse_position: tuple[int, int] | None,
    ) -> None:
        pygame = self._pygame
        padding = 14
        card = pygame.Rect(
            self.layout.grid_left - padding,
            self.layout.grid_top - padding,
            self.layout.grid_width + padding * 2,
            self.layout.grid_height + padding * 2,
        )
        shadow = card.move(0, 7)
        pygame.draw.rect(self.surface, (4, 8, 17), shadow, border_radius=18)
        pygame.draw.rect(self.surface, self.SURFACE, card, border_radius=18)

        hovered = grid_position_at(self.layout, mouse_position) if mouse_position else None
        for row in range(environment.height):
            for column in range(environment.width):
                position = (column, row)
                cell = pygame.Rect(
                    self.layout.grid_left + column * self.layout.cell_size,
                    self.layout.grid_top + row * self.layout.cell_size,
                    self.layout.cell_size,
                    self.layout.cell_size,
                )
                if position in environment.obstacles:
                    color = self.OBSTACLE
                elif position == hovered and active_tool is not Tool.PLAY:
                    color = self.CELL_HOVER
                else:
                    color = self.CELL
                pygame.draw.rect(self.surface, color, cell)
                pygame.draw.rect(self.surface, self.GRID_LINE, cell, width=1)

        self._draw_endpoint(environment.start, self.START, "S")
        self._draw_endpoint(environment.goal, self.GOAL, "G")
        self._draw_agent(environment.agent)

    def _cell_center(self, position: Position) -> tuple[int, int]:
        column, row = position
        return (
            self.layout.grid_left + column * self.layout.cell_size + self.layout.cell_size // 2,
            self.layout.grid_top + row * self.layout.cell_size + self.layout.cell_size // 2,
        )

    def _draw_endpoint(self, position: Position, color: tuple[int, int, int], label: str) -> None:
        pygame = self._pygame
        center = self._cell_center(position)
        radius = max(8, self.layout.cell_size // 3)
        pygame.draw.circle(self.surface, color, center, radius)
        text = self.marker_font.render(label, True, self.TEXT)
        self.surface.blit(text, text.get_rect(center=center))

    def _draw_agent(self, position: Position) -> None:
        pygame = self._pygame
        center = self._cell_center(position)
        radius = max(6, self.layout.cell_size // 5)
        pygame.draw.circle(self.surface, self.AGENT, center, radius)
        pygame.draw.circle(self.surface, self.BACKGROUND, center, radius, width=2)

    def _draw_sidebar(self, active_tool: Tool) -> None:
        pygame = self._pygame
        panel = pygame.Rect(
            self.layout.sidebar_left,
            self.layout.sidebar_top,
            SIDEBAR_WIDTH,
            self.layout.sidebar_height,
        )
        pygame.draw.rect(self.surface, self.SURFACE, panel, border_radius=18)
        heading = self.heading_font.render("Controls", True, self.TEXT)
        self.surface.blit(heading, (panel.left + 20, panel.top + 20))
        mode = self.small_font.render(f"Active mode: {active_tool.value}", True, self.MUTED)
        self.surface.blit(mode, (panel.left + 20, panel.top + 51))

        mouse = pygame.mouse.get_pos()
        for index, bounds in enumerate(toolbar_button_bounds(self.layout), start=1):
            rectangle = pygame.Rect(bounds.left, bounds.top, bounds.width, bounds.height)
            is_selected = bounds.tool == active_tool and bounds.tool in SELECTABLE_TOOLS
            is_hovered = bounds.contains(mouse)
            if is_selected:
                color = self.ACCENT
            elif is_hovered:
                color = (51, 67, 96)
            else:
                color = self.SURFACE_LIGHT
            pygame.draw.rect(self.surface, color, rectangle, border_radius=11)
            if is_selected:
                pygame.draw.rect(
                    self.surface,
                    self.TEXT,
                    (rectangle.left, rectangle.top + 10, 4, rectangle.height - 20),
                    border_radius=2,
                )
            shortcut = str(index) if index <= 4 else ""
            text = self.body_font.render(bounds.tool.value, True, self.TEXT)
            self.surface.blit(text, (rectangle.left + 17, rectangle.centery - text.get_height() // 2))
            if shortcut:
                key = self.small_font.render(shortcut, True, self.MUTED if not is_selected else self.TEXT)
                self.surface.blit(key, (rectangle.right - 22, rectangle.centery - key.get_height() // 2))

        legend_top = panel.top + 456
        legend = self.small_font.render("LEGEND", True, self.MUTED)
        self.surface.blit(legend, (panel.left + 20, legend_top))
        self._draw_legend_item(panel.left + 20, legend_top + 28, self.AGENT, "Agent")
        self._draw_legend_item(panel.left + 120, legend_top + 28, self.START, "Start")
        self._draw_legend_item(panel.left + 20, legend_top + 58, self.GOAL, "Goal")
        self._draw_legend_item(panel.left + 120, legend_top + 58, self.OBSTACLE, "Obstacle")

        hint_lines = ("Play: arrows or WASD", "Obstacles: drag to paint", "Right-drag to erase")
        for offset, line in enumerate(hint_lines):
            hint = self.small_font.render(line, True, self.MUTED)
            self.surface.blit(hint, (panel.left + 20, panel.bottom - 76 + offset * 19))

    def _draw_legend_item(self, x: int, y: int, color: tuple[int, int, int], label: str) -> None:
        pygame = self._pygame
        pygame.draw.circle(self.surface, color, (x + 6, y + 7), 6)
        text = self.small_font.render(label, True, self.TEXT)
        self.surface.blit(text, (x + 17, y))
