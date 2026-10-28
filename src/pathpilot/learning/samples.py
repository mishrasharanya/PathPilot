"""Typed labeled examples used by supervised navigation datasets."""

from __future__ import annotations

from dataclasses import dataclass

from pathpilot.learning.actions import NavigationAction
from pathpilot.learning.encoding import GridEncoding, validate_encoding


@dataclass(frozen=True, slots=True)
class NavigationSample:
    """One immutable observation paired with the A* teacher's next action."""

    maze_id: str
    step_index: int
    observation: GridEncoding
    action: NavigationAction

    def validate(self, *, width: int, height: int) -> None:
        """Validate identifiers, label type, and observation contents."""
        if not self.maze_id.strip():
            raise ValueError("maze_id cannot be empty")
        if self.step_index < 0:
            raise ValueError("step_index cannot be negative")
        if not isinstance(self.action, NavigationAction):
            raise TypeError("action must be a NavigationAction")
        validate_encoding(self.observation, width=width, height=height)
