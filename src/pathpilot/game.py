"""Interactive PathPilot Pygame application."""

from __future__ import annotations

from pathpilot.environment import Direction, GridEnvironment, MoveResult, Position
from pathpilot.pygame_ui import PygameUI, SELECTABLE_TOOLS, Tool, WINDOW_SIZE


def apply_grid_tool(
    environment: GridEnvironment,
    tool: Tool,
    position: Position,
    *,
    erase: bool = False,
) -> str:
    """Apply one editing action and return a user-facing status message."""
    if tool is Tool.OBSTACLES:
        changed = environment.remove_obstacle(position) if erase else environment.add_obstacle(position)
        if changed:
            return "Obstacle removed" if erase else "Obstacle added"
        return "That cell cannot be changed"
    if tool is Tool.SET_START:
        return "Start updated" if environment.set_start(position) else "Choose an empty non-goal cell"
    if tool is Tool.SET_GOAL:
        return "Goal updated" if environment.set_goal(position) else "Choose an empty non-start cell"
    return "Select an editing tool to change the grid"


def movement_status(result: MoveResult) -> str:
    return {
        MoveResult.MOVED: "Agent moved",
        MoveResult.REACHED_GOAL: "Goal reached",
        MoveResult.BLOCKED_BY_OBSTACLE: "An obstacle blocks that move",
        MoveResult.OUT_OF_BOUNDS: "The edge of the grid blocks that move",
    }[result]


def main() -> None:
    import pygame

    pygame.init()
    pygame.display.set_caption("PathPilot — Interactive Navigation Lab")
    surface = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    environment = GridEnvironment()
    ui = PygameUI(surface, environment)
    active_tool = Tool.OBSTACLES
    status = "Draw obstacles or choose another tool"
    running = True

    key_directions = {
        pygame.K_UP: Direction.UP,
        pygame.K_w: Direction.UP,
        pygame.K_DOWN: Direction.DOWN,
        pygame.K_s: Direction.DOWN,
        pygame.K_LEFT: Direction.LEFT,
        pygame.K_a: Direction.LEFT,
        pygame.K_RIGHT: Direction.RIGHT,
        pygame.K_d: Direction.RIGHT,
    }
    number_tools = {
        pygame.K_1: Tool.OBSTACLES,
        pygame.K_2: Tool.SET_START,
        pygame.K_3: Tool.SET_GOAL,
        pygame.K_4: Tool.PLAY,
    }

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in number_tools:
                        active_tool = number_tools[event.key]
                        status = f"{active_tool.value} mode selected"
                    elif event.key in key_directions:
                        if active_tool is Tool.PLAY:
                            status = movement_status(environment.move(key_directions[event.key]))
                        else:
                            status = "Select Play before moving the agent"
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    selected = ui.tool_at(event.pos)
                    if selected in SELECTABLE_TOOLS:
                        active_tool = selected
                        status = f"{selected.value} mode selected"
                    elif selected is Tool.RESET_AGENT:
                        environment.reset_agent()
                        status = "Agent returned to start"
                    elif selected is Tool.CLEAR_GRID:
                        environment.reset()
                        active_tool = Tool.OBSTACLES
                        status = "Grid cleared"
                    else:
                        position = ui.grid_position_at(event.pos)
                        if position is not None and event.button in (1, 3):
                            status = apply_grid_tool(
                                environment,
                                active_tool,
                                position,
                                erase=event.button == 3,
                            )
                            if active_tool in {Tool.SET_START, Tool.SET_GOAL} and status.endswith("updated"):
                                active_tool = Tool.OBSTACLES
                                status += " · Obstacles mode restored"
                elif event.type == pygame.MOUSEMOTION and active_tool is Tool.OBSTACLES:
                    position = ui.grid_position_at(event.pos)
                    if position is not None and (event.buttons[0] or event.buttons[2]):
                        status = apply_grid_tool(
                            environment,
                            active_tool,
                            position,
                            erase=bool(event.buttons[2]),
                        )

            ui.draw(environment, active_tool, status, pygame.mouse.get_pos())
            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
