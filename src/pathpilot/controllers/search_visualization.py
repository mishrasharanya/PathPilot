"""Time-based, UI-independent playback of path-planning results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pathpilot.environment import Position
from pathpilot.planning import SearchResult


class VisualizationPhase(Enum):
    """Current stage of search-result playback."""

    IDLE = "Idle"
    EXPLORING = "Exploring"
    PATH = "Showing path"
    COMPLETE = "Complete"


@dataclass(frozen=True, slots=True)
class VisualizationFrame:
    """Visible portion of a search result at one moment."""

    explored_cells: tuple[Position, ...] = ()
    path: tuple[Position, ...] = ()
    phase: VisualizationPhase = VisualizationPhase.IDLE
    paused: bool = False


class SearchVisualizationController:
    """Reveal explored cells, then the path, at a fixed interval."""

    def __init__(self, step_interval_ms: int = 24) -> None:
        if step_interval_ms <= 0:
            raise ValueError("step_interval_ms must be positive")
        self.step_interval_ms = step_interval_ms
        self._result: SearchResult | None = None
        self._explored_count = 0
        self._path_count = 0
        self._elapsed_ms = 0
        self._phase = VisualizationPhase.IDLE
        self._paused = False

    @property
    def result(self) -> SearchResult | None:
        return self._result

    @property
    def is_active(self) -> bool:
        return self._phase in {
            VisualizationPhase.EXPLORING,
            VisualizationPhase.PATH,
        }

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def frame(self) -> VisualizationFrame:
        if self._result is None:
            return VisualizationFrame()
        return VisualizationFrame(
            explored_cells=self._result.explored_cells[: self._explored_count],
            path=self._result.path[: self._path_count],
            phase=self._phase,
            paused=self._paused,
        )

    def start(self, result: SearchResult) -> None:
        """Begin playback from the first explored cell."""
        self._result = result
        self._explored_count = 0
        self._path_count = 0
        self._elapsed_ms = 0
        self._paused = False
        if result.explored_cells:
            self._phase = VisualizationPhase.EXPLORING
        elif result.path:
            self._phase = VisualizationPhase.PATH
        else:
            self._phase = VisualizationPhase.COMPLETE

    def update(self, elapsed_ms: int) -> None:
        """Advance playback according to elapsed real time."""
        if elapsed_ms < 0:
            raise ValueError("elapsed_ms cannot be negative")
        if not self.is_active or self._paused or self._result is None:
            return

        self._elapsed_ms += elapsed_ms
        while self._elapsed_ms >= self.step_interval_ms and self.is_active:
            self._elapsed_ms -= self.step_interval_ms
            self._advance_one_step()

    def toggle_pause(self) -> bool:
        """Toggle active playback and return the new paused state."""
        if not self.is_active:
            return False
        self._paused = not self._paused
        return self._paused

    def stop(self) -> None:
        """Clear the current result and all visible search data."""
        self._result = None
        self._explored_count = 0
        self._path_count = 0
        self._elapsed_ms = 0
        self._phase = VisualizationPhase.IDLE
        self._paused = False

    def _advance_one_step(self) -> None:
        if self._result is None:
            return
        if self._phase is VisualizationPhase.EXPLORING:
            self._explored_count += 1
            if self._explored_count >= len(self._result.explored_cells):
                self._phase = (
                    VisualizationPhase.PATH
                    if self._result.path
                    else VisualizationPhase.COMPLETE
                )
            return

        if self._phase is VisualizationPhase.PATH:
            self._path_count += 1
            if self._path_count >= len(self._result.path):
                self._phase = VisualizationPhase.COMPLETE
