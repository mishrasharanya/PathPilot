"""Action labels shared by supervised models and navigation controllers."""

from __future__ import annotations

from enum import IntEnum

from pathpilot.environment import Direction, Position


class NavigationAction(IntEnum):
    """Stable numeric labels for the four valid one-cell movements.

    Explicit values keep saved datasets compatible if the enum declaration is
    reorganized later.
    """

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @property
    def direction(self) -> Direction:
        """Return the environment direction represented by this label."""
        return Direction[self.name]

    @classmethod
    def from_transition(
        cls,
        current: Position,
        next_position: Position,
    ) -> NavigationAction:
        """Convert two adjacent path positions into an action label."""
        delta = (
            next_position[0] - current[0],
            next_position[1] - current[1],
        )
        for action in cls:
            if action.direction.value == delta:
                return action
        raise ValueError(
            f"Positions must be orthogonally adjacent: {current} -> {next_position}"
        )
