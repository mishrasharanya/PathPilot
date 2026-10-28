import unittest

from pathpilot.environment import Direction, GridEnvironment
from pathpilot.learning.encoding import encode_environment, validate_encoding


class GridEncodingTests(unittest.TestCase):
    def test_environment_uses_three_row_major_channels(self) -> None:
        environment = GridEnvironment(
            width=3,
            height=2,
            goal=(2, 1),
            obstacles={(1, 0)},
        )

        encoding = encode_environment(environment)

        self.assertEqual(len(encoding), 3)
        self.assertEqual(encoding[0], ((0.0, 1.0, 0.0), (0.0, 0.0, 0.0)))
        self.assertEqual(encoding[1][0][0], 1.0)
        self.assertEqual(encoding[2][1][2], 1.0)
        validate_encoding(encoding, width=3, height=2)

    def test_encoding_is_snapshot_not_live_environment_view(self) -> None:
        environment = GridEnvironment(width=3, height=2)
        encoding = encode_environment(environment)
        environment.move(Direction.RIGHT)
        self.assertEqual(encoding[1][0][0], 1.0)
        self.assertEqual(encoding[1][0][1], 0.0)

    def test_invalid_shape_and_values_are_rejected(self) -> None:
        environment = GridEnvironment(width=3, height=2)
        encoding = encode_environment(environment)
        with self.assertRaises(ValueError):
            validate_encoding(encoding[:2], width=3, height=2)  # type: ignore[arg-type]

        invalid = list(encoding)
        invalid[0] = ((2.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            validate_encoding(tuple(invalid), width=3, height=2)  # type: ignore[arg-type]
