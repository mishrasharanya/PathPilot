"""UI-independent conversion of grid state into model-ready channels."""

from __future__ import annotations

from typing import TypeAlias

from pathpilot.environment import GridEnvironment, Position

GridChannel: TypeAlias = tuple[tuple[float, ...], ...]
GridEncoding: TypeAlias = tuple[GridChannel, GridChannel, GridChannel]

OBSTACLE_CHANNEL = 0
AGENT_CHANNEL = 1
GOAL_CHANNEL = 2
CHANNEL_COUNT = 3


def encode_environment(environment: GridEnvironment) -> GridEncoding:
    """Encode obstacles, agent, and goal as three row-major binary channels.

    The returned shape is ``(3, height, width)``. Tuples make an encoded
    observation immutable, so later environment edits cannot corrupt a sample.
    """
    return (
        _binary_channel(environment.width, environment.height, environment.obstacles),
        _binary_channel(environment.width, environment.height, {environment.agent}),
        _binary_channel(environment.width, environment.height, {environment.goal}),
    )


def validate_encoding(
    encoding: GridEncoding,
    *,
    width: int,
    height: int,
) -> None:
    """Raise ``ValueError`` when an encoding has an invalid shape or value."""
    if len(encoding) != CHANNEL_COUNT:
        raise ValueError(f"Encoding must contain {CHANNEL_COUNT} channels")

    for channel in encoding:
        if len(channel) != height:
            raise ValueError(f"Each channel must contain {height} rows")
        for row in channel:
            if len(row) != width:
                raise ValueError(f"Each row must contain {width} values")
            if any(value not in (0.0, 1.0) for value in row):
                raise ValueError("Encoding values must be binary")

    if _active_cell_count(encoding[AGENT_CHANNEL]) != 1:
        raise ValueError("Agent channel must contain exactly one active cell")
    if _active_cell_count(encoding[GOAL_CHANNEL]) != 1:
        raise ValueError("Goal channel must contain exactly one active cell")


def _binary_channel(
    width: int,
    height: int,
    active_positions: set[Position],
) -> GridChannel:
    return tuple(
        tuple(1.0 if (column, row) in active_positions else 0.0 for column in range(width))
        for row in range(height)
    )


def _active_cell_count(channel: GridChannel) -> int:
    return sum(int(value) for row in channel for value in row)
