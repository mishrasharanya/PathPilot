"""Interactive PathPilot Pygame application."""

from __future__ import annotations

from pathlib import Path

from pathpilot.controllers import (
    AlwaysReplanController,
    CachedPathController,
    CNNNavigationController,
    NavigationState,
    SearchVisualizationController,
)
from pathpilot.environment import Direction, GridEnvironment, MoveResult, Position
from pathpilot.planning import AStarPlanner
from pathpilot.ui import (
    AIAction,
    PygameUI,
    SELECTABLE_TOOLS,
    Tool,
    WINDOW_SIZE,
)


def apply_grid_tool(
    environment: GridEnvironment,
    tool: Tool,
    position: Position,
    *,
    erase: bool = False,
    toggle: bool = False,
) -> str:
    """Apply one editing action and return a user-facing status message."""
    if tool is Tool.OBSTACLES:
        should_remove = erase or (toggle and position in environment.obstacles)
        changed = (
            environment.remove_obstacle(position)
            if should_remove
            else environment.add_obstacle(position)
        )
        if changed:
            return "Obstacle removed" if should_remove else "Obstacle added"
        return "That cell cannot be changed"
    if tool is Tool.SET_START:
        return (
            "Start updated"
            if environment.set_start(position)
            else "Choose an empty non-goal cell"
        )
    if tool is Tool.SET_GOAL:
        return (
            "Goal updated"
            if environment.set_goal(position)
            else "Choose an empty non-start cell"
        )
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
    planner = AStarPlanner()
    visualization = SearchVisualizationController()
    baseline = AlwaysReplanController()
    cached = CachedPathController()
    cnn: CNNNavigationController | None = None
    cnn_unavailable_reason = "Train the CNN checkpoint first"
    checkpoint_path = Path("artifacts/models/cnn-v1/best_model.pt")
    try:
        from pathpilot.learning.inference import TorchCNNPolicy

        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        cnn = CNNNavigationController(TorchCNNPolicy(checkpoint_path))
        cnn_unavailable_reason = ""
    except ImportError:
        cnn_unavailable_reason = "Install PathPilot with the ML dependencies"
    except FileNotFoundError:
        cnn_unavailable_reason = "Train the CNN checkpoint first"
    except (RuntimeError, ValueError) as error:
        cnn_unavailable_reason = f"CNN unavailable: {error}"

    navigation_controllers = [
        ("Baseline", baseline),
        ("Cached", cached),
    ]
    if cnn is not None:
        navigation_controllers.append(("CNN", cnn))
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
            elapsed_ms = clock.tick(60)
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
                            visualization.stop()
                            for _, controller in navigation_controllers:
                                controller.stop()
                            status = movement_status(environment.move(key_directions[event.key]))
                        else:
                            status = "Select Play before moving the agent"
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    selected = ui.tool_at(event.pos)
                    ai_action = ui.ai_action_at(event.pos)
                    if selected in SELECTABLE_TOOLS:
                        active_tool = selected
                        status = f"{selected.value} mode selected"
                    elif selected is Tool.RESET_AGENT:
                        visualization.stop()
                        for _, controller in navigation_controllers:
                            controller.stop()
                        environment.reset_agent()
                        status = "Agent returned to start"
                    elif selected is Tool.CLEAR_GRID:
                        visualization.stop()
                        for _, controller in navigation_controllers:
                            controller.stop()
                        environment.reset()
                        active_tool = Tool.OBSTACLES
                        status = "Grid cleared"
                    elif ai_action is AIAction.START:
                        for _, controller in navigation_controllers:
                            controller.stop()
                        result = planner.plan(
                            environment,
                            environment.agent,
                            environment.goal,
                        )
                        visualization.start(result)
                        latency_ms = result.planning_latency_ns / 1_000_000
                        status = (
                            f"A*: {result.nodes_explored} nodes · "
                            f"{latency_ms:.3f} ms"
                            if result.success
                            else f"No path · {result.nodes_explored} nodes explored"
                        )
                    elif ai_action is AIAction.RUN_BASELINE:
                        visualization.stop()
                        cached.stop()
                        if cnn is not None:
                            cnn.stop()
                        baseline.start(environment)
                        status = "Always-Replan baseline started"
                    elif ai_action is AIAction.RUN_CACHED:
                        visualization.stop()
                        baseline.stop()
                        if cnn is not None:
                            cnn.stop()
                        cached.start(environment)
                        status = "Cached-Path navigation started"
                    elif ai_action is AIAction.RUN_CNN:
                        visualization.stop()
                        baseline.stop()
                        cached.stop()
                        if cnn is None:
                            status = cnn_unavailable_reason
                        else:
                            try:
                                cnn.start(environment)
                                status = "CNN navigation started"
                            except ValueError as error:
                                status = str(error)
                    elif ai_action is AIAction.PAUSE:
                        active_navigation = next(
                            (
                                (name, controller)
                                for name, controller in navigation_controllers
                                if controller.is_active
                            ),
                            None,
                        )
                        if active_navigation is not None:
                            name, controller = active_navigation
                            paused = controller.toggle_pause()
                            status = (
                                f"{name} paused"
                                if paused
                                else f"{name} resumed"
                            )
                        elif visualization.is_active:
                            paused = visualization.toggle_pause()
                            status = (
                                "Visualization paused"
                                if paused
                                else "Visualization resumed"
                            )
                        else:
                            status = "Start an AI navigation mode first"
                    elif ai_action is AIAction.STOP:
                        visualization.stop()
                        for _, controller in navigation_controllers:
                            controller.stop()
                        status = "AI navigation stopped"
                    else:
                        position = ui.grid_position_at(event.pos)
                        if position is not None and event.button in (1, 3):
                            visualization.stop()
                            status = apply_grid_tool(
                                environment,
                                active_tool,
                                position,
                                erase=event.button == 3,
                                toggle=event.button == 1,
                            )
                            endpoint_changed = (
                                active_tool in {Tool.SET_START, Tool.SET_GOAL}
                                and status.endswith("updated")
                            )
                            if endpoint_changed:
                                for _, controller in navigation_controllers:
                                    controller.stop()
                            if endpoint_changed:
                                active_tool = Tool.OBSTACLES
                                status += " · Obstacles mode restored"
                elif event.type == pygame.MOUSEMOTION and active_tool is Tool.OBSTACLES:
                    position = ui.grid_position_at(event.pos)
                    if position is not None and (event.buttons[0] or event.buttons[2]):
                        visualization.stop()
                        status = apply_grid_tool(
                            environment,
                            active_tool,
                            position,
                            erase=bool(event.buttons[2]),
                        )

            visualization.update(elapsed_ms)
            for name, controller in navigation_controllers:
                previous_calls = (
                    controller.metrics.inference_calls
                    if isinstance(controller, CNNNavigationController)
                    else controller.metrics.planning_calls
                )
                previous_state = controller.state
                controller.update(elapsed_ms, environment)
                if (
                    not isinstance(controller, CNNNavigationController)
                    and controller.metrics.planning_calls > previous_calls
                ):
                    status = (
                        f"{name} plan {controller.metrics.planning_calls} · "
                        f"{controller.metrics.nodes_explored} total nodes"
                    )
                just_succeeded = (
                    controller.state is NavigationState.SUCCEEDED
                    and previous_state is not NavigationState.SUCCEEDED
                )
                just_failed = (
                    controller.state is NavigationState.FAILED
                    and previous_state is not NavigationState.FAILED
                )
                if just_succeeded:
                    status = f"{name} reached the goal"
                elif just_failed:
                    status = (
                        "CNN stopped: invalid action, loop, or step limit"
                        if isinstance(controller, CNNNavigationController)
                        else f"{name} could not find a path"
                    )

            frame = visualization.frame
            current_navigation = next(
                (
                    controller
                    for _, controller in reversed(navigation_controllers)
                    if controller.state is not NavigationState.IDLE
                ),
                None,
            )
            if current_navigation is not None:
                visualization_status = current_navigation.state.value
            else:
                visualization_status = (
                    "Paused" if frame.paused else frame.phase.value
                )
            if current_navigation is not None and current_navigation.latest_result:
                explored_cells = current_navigation.latest_result.explored_cells
                path = (
                    current_navigation.remaining_path
                    if isinstance(current_navigation, CachedPathController)
                    else current_navigation.latest_result.path
                )
            else:
                explored_cells = frame.explored_cells
                path = frame.path
            if isinstance(current_navigation, CNNNavigationController):
                cnn_metrics = current_navigation.metrics
                navigation_metrics = (
                    ("Decisions", str(cnn_metrics.inference_calls)),
                    ("Moves", str(cnn_metrics.path_length)),
                    ("Invalid", str(cnn_metrics.invalid_actions)),
                    ("Encoding", f"{cnn_metrics.encoding_latency_ms:.3f} ms"),
                    ("Inference", f"{cnn_metrics.inference_latency_ms:.3f} ms"),
                )
            elif current_navigation is not None:
                metrics = current_navigation.metrics
                navigation_metrics = (
                    ("Plans", str(metrics.planning_calls)),
                    ("Replans", str(metrics.replanning_count)),
                    ("Nodes", str(metrics.nodes_explored)),
                    ("Moves", str(metrics.path_length)),
                    ("Planning", f"{metrics.planning_latency_ms:.3f} ms"),
                )
            else:
                navigation_metrics = ()
            ui.draw(
                environment,
                active_tool,
                status,
                pygame.mouse.get_pos(),
                explored_cells=explored_cells,
                path=path,
                visualization_status=visualization_status,
                navigation_metrics=navigation_metrics,
            )
            pygame.display.flip()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
