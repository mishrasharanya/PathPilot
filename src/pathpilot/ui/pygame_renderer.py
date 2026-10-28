"""Pygame renderer for PathPilot environment state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pathpilot.environment import GridEnvironment, Position
from pathpilot.ui.layout import (
    AIAction,
    HEADER_HEIGHT,
    SELECTABLE_TOOLS,
    SIDEBAR_WIDTH,
    WINDOW_MARGIN,
    Tool,
    ai_button_bounds,
    calculate_layout,
    grid_position_at,
    toolbar_button_bounds,
)

if TYPE_CHECKING:
    import pygame


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
    TEXT = (245, 248, 252)
    MUTED = (153, 167, 190)

    def __init__(self, surface: pygame.Surface, environment: GridEnvironment) -> None:
        import pygame

        self._pygame = pygame
        self.surface = surface
        self.layout = calculate_layout(
            environment.width,
            environment.height,
            surface.get_size(),
        )
        self.title_font = pygame.font.SysFont("Avenir Next", 34, bold=True)
        self.heading_font = pygame.font.SysFont("Avenir Next", 20, bold=True)
        self.body_font = pygame.font.SysFont("Avenir Next", 17)
        self.small_font = pygame.font.SysFont("Avenir Next", 14)
        marker_size = max(16, self.layout.cell_size // 2)
        self.marker_font = pygame.font.SysFont(
            "Avenir Next", marker_size, bold=True
        )

    def tool_at(self, point: tuple[int, int]) -> Tool | None:
        """Return the control at a screen point, if one was hit."""
        for bounds in toolbar_button_bounds(self.layout):
            if bounds.contains(point):
                return bounds.tool
        return None

    def grid_position_at(self, point: tuple[int, int]) -> Position | None:
        """Return the grid cell at a screen point, if one was hit."""
        return grid_position_at(self.layout, point)

    def ai_action_at(self, point: tuple[int, int]) -> AIAction | None:
        """Return the A* playback action at a screen point, if one was hit."""
        for bounds in ai_button_bounds(self.layout):
            if bounds.contains(point):
                return bounds.action
        return None

    def draw(
        self,
        environment: GridEnvironment,
        active_tool: Tool,
        status: str,
        mouse_position: tuple[int, int] | None = None,
        explored_cells: tuple[Position, ...] = (),
        path: tuple[Position, ...] = (),
        visualization_status: str = "Ready",
        navigation_metrics: tuple[tuple[str, str], ...] = (),
    ) -> None:
        """Render a complete frame from the current application state."""
        self.surface.fill(self.BACKGROUND)
        self._draw_header(environment, status)
        self._draw_board(
            environment,
            active_tool,
            mouse_position,
            frozenset(explored_cells),
            frozenset(path),
        )
        self._draw_sidebar(
            active_tool,
            visualization_status,
            navigation_metrics,
        )

    def _draw_header(self, environment: GridEnvironment, status: str) -> None:
        pygame = self._pygame
        header = pygame.Rect(
            WINDOW_MARGIN,
            WINDOW_MARGIN,
            self.layout.window_width - 2 * WINDOW_MARGIN,
            HEADER_HEIGHT,
        )
        pygame.draw.rect(
            self.surface, self.SURFACE, header, border_radius=18
        )
        pygame.draw.circle(
            self.surface, self.ACCENT, (header.left + 28, header.centery), 10
        )

        title = self.title_font.render("PathPilot", True, self.TEXT)
        self.surface.blit(title, (header.left + 48, header.top + 12))
        subtitle = self.small_font.render(
            "Interactive navigation lab", True, self.MUTED
        )
        self.surface.blit(subtitle, (header.left + 50, header.top + 48))

        status_color = self.START if environment.reached_goal else self.ACCENT
        status_text = "Goal reached" if environment.reached_goal else status
        rendered = self.body_font.render(status_text, True, self.TEXT)
        chip = pygame.Rect(0, 0, rendered.get_width() + 34, 38)
        chip.midright = (header.right - 18, header.centery)
        pygame.draw.rect(
            self.surface, status_color, chip, border_radius=19
        )
        self.surface.blit(rendered, rendered.get_rect(center=chip.center))

    def _draw_board(
        self,
        environment: GridEnvironment,
        active_tool: Tool,
        mouse_position: tuple[int, int] | None,
        explored_cells: frozenset[Position],
        path: frozenset[Position],
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
        pygame.draw.rect(
            self.surface, (4, 8, 17), shadow, border_radius=18
        )
        pygame.draw.rect(
            self.surface, self.SURFACE, card, border_radius=18
        )

        hovered = (
            grid_position_at(self.layout, mouse_position)
            if mouse_position
            else None
        )
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
                elif position in path:
                    color = (129, 230, 217)
                elif position in explored_cells:
                    color = (183, 215, 247)
                elif position == hovered and active_tool is not Tool.PLAY:
                    color = self.CELL_HOVER
                else:
                    color = self.CELL
                pygame.draw.rect(self.surface, color, cell)
                pygame.draw.rect(
                    self.surface, self.GRID_LINE, cell, width=1
                )

        self._draw_endpoint(environment.start, self.START, "S")
        self._draw_endpoint(environment.goal, self.GOAL, "G")
        self._draw_agent(environment.agent)

    def _cell_center(self, position: Position) -> tuple[int, int]:
        column, row = position
        return (
            self.layout.grid_left
            + column * self.layout.cell_size
            + self.layout.cell_size // 2,
            self.layout.grid_top
            + row * self.layout.cell_size
            + self.layout.cell_size // 2,
        )

    def _draw_endpoint(
        self,
        position: Position,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
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
        pygame.draw.circle(
            self.surface, self.BACKGROUND, center, radius, width=2
        )

    def _draw_sidebar(
        self,
        active_tool: Tool,
        visualization_status: str,
        navigation_metrics: tuple[tuple[str, str], ...],
    ) -> None:
        pygame = self._pygame
        panel = pygame.Rect(
            self.layout.sidebar_left,
            self.layout.sidebar_top,
            SIDEBAR_WIDTH,
            self.layout.sidebar_height,
        )
        pygame.draw.rect(
            self.surface, self.SURFACE, panel, border_radius=18
        )
        heading = self.heading_font.render("Controls", True, self.TEXT)
        self.surface.blit(heading, (panel.left + 20, panel.top + 20))
        mode = self.small_font.render(
            f"Active mode: {active_tool.value}", True, self.MUTED
        )
        self.surface.blit(mode, (panel.left + 20, panel.top + 51))

        mouse = pygame.mouse.get_pos()
        for index, bounds in enumerate(
            toolbar_button_bounds(self.layout), start=1
        ):
            rectangle = pygame.Rect(
                bounds.left, bounds.top, bounds.width, bounds.height
            )
            is_selected = (
                bounds.tool == active_tool
                and bounds.tool in SELECTABLE_TOOLS
            )
            is_hovered = bounds.contains(mouse)
            if is_selected:
                color = self.ACCENT
            elif is_hovered:
                color = (51, 67, 96)
            else:
                color = self.SURFACE_LIGHT
            pygame.draw.rect(
                self.surface, color, rectangle, border_radius=11
            )
            if is_selected:
                pygame.draw.rect(
                    self.surface,
                    self.TEXT,
                    (
                        rectangle.left,
                        rectangle.top + 10,
                        4,
                        rectangle.height - 20,
                    ),
                    border_radius=2,
                )
            shortcut = str(index) if index <= 4 else ""
            text = self.body_font.render(
                bounds.tool.value, True, self.TEXT
            )
            self.surface.blit(
                text,
                (
                    rectangle.left + 17,
                    rectangle.centery - text.get_height() // 2,
                ),
            )
            if shortcut:
                key_color = self.MUTED if not is_selected else self.TEXT
                key = self.small_font.render(shortcut, True, key_color)
                self.surface.blit(
                    key,
                    (
                        rectangle.right - 22,
                        rectangle.centery - key.get_height() // 2,
                    ),
                )

        ai_label = self.small_font.render("AI NAVIGATION", True, self.MUTED)
        self.surface.blit(ai_label, (panel.left + 20, panel.top + 397))
        for bounds in ai_button_bounds(self.layout):
            rectangle = pygame.Rect(
                bounds.left, bounds.top, bounds.width, bounds.height
            )
            is_hovered = bounds.contains(mouse)
            if bounds.action in {
                AIAction.START,
                AIAction.RUN_BASELINE,
                AIAction.RUN_CACHED,
                AIAction.RUN_CNN,
            }:
                color = self.START if not is_hovered else (29, 202, 143)
            elif is_hovered:
                color = (51, 67, 96)
            else:
                color = self.SURFACE_LIGHT
            pygame.draw.rect(
                self.surface, color, rectangle, border_radius=10
            )
            label_text = {
                AIAction.START: "Plan",
                AIAction.RUN_BASELINE: "V1",
                AIAction.RUN_CACHED: "V2",
                AIAction.RUN_CNN: "CNN",
            }.get(bounds.action, bounds.action.value)
            if bounds.action is AIAction.PAUSE and visualization_status == "Paused":
                label_text = "Resume"
            label = self.small_font.render(label_text, True, self.TEXT)
            self.surface.blit(label, label.get_rect(center=rectangle.center))

        state = self.small_font.render(
            f"State: {visualization_status}", True, self.MUTED
        )
        self.surface.blit(state, (panel.left + 20, panel.top + 512))

        details_top = panel.top + 550
        if navigation_metrics:
            heading = self.small_font.render("LIVE METRICS", True, self.MUTED)
            self.surface.blit(heading, (panel.left + 20, details_top))
            for index, (label, value) in enumerate(navigation_metrics):
                column = index % 2
                row = index // 2
                x = panel.left + 20 + column * 108
                y = details_top + 27 + row * 34
                label_text = self.small_font.render(label, True, self.MUTED)
                value_text = self.body_font.render(value, True, self.TEXT)
                self.surface.blit(label_text, (x, y))
                self.surface.blit(value_text, (x, y + 14))
        else:
            legend = self.small_font.render("LEGEND", True, self.MUTED)
            self.surface.blit(legend, (panel.left + 20, details_top))
            self._draw_legend_item(
                panel.left + 20, details_top + 28, self.AGENT, "Agent"
            )
            self._draw_legend_item(
                panel.left + 120, details_top + 28, self.START, "Start"
            )
            self._draw_legend_item(
                panel.left + 20, details_top + 58, self.GOAL, "Goal"
            )
            self._draw_legend_item(
                panel.left + 120,
                details_top + 58,
                self.OBSTACLE,
                "Obstacle",
            )
            self._draw_legend_item(
                panel.left + 20,
                details_top + 88,
                (183, 215, 247),
                "Explored",
            )
            self._draw_legend_item(
                panel.left + 120,
                details_top + 88,
                (129, 230, 217),
                "A* path",
            )

    def _draw_legend_item(
        self,
        x: int,
        y: int,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        pygame = self._pygame
        pygame.draw.circle(self.surface, color, (x + 6, y + 7), 6)
        text = self.small_font.render(label, True, self.TEXT)
        self.surface.blit(text, (x + 17, y))
