"""Data and timing checks; no Warp or Robotarium installation required."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from utils.replay_robotarium import load_recording, minimum_separation, prepare_replay


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "recording.csv"

    def write(self, rows, fields=("time", "robot", "episode", "x", "y", "z", "theta")):
        with self.path.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(fields)
            writer.writerows(rows)

    def test_warp_axes_and_shared_transform(self):
        self.write([[10, "a", 0, 0, 99, 0, 0], [10, "b", 0, 0, 99, 1, 0],
                    [12, "a", 0, 4, 99, 0, 0], [12, "b", 0, 4, 99, 1, 0]])
        replay = prepare_replay(self.path)
        self.assertAlmostEqual(replay.scale, 0.7)
        np.testing.assert_allclose(replay.positions[0], [[-1.4, -1.4], [-0.35, 0.35]])
        np.testing.assert_allclose(replay.times, [0, 2])
        self.assertEqual(replay.names, ["a", "b"])

    def test_interpolation_and_playback_derivative(self):
        self.write([[5, "a", 0, 0, 0, 0, 0], [7, "a", 0, 1, 0, 0, 0]])
        replay = prepare_replay(self.path, speed_factor=0.1)
        self.assertAlmostEqual(replay.duration, 20)
        p, velocity = replay.sample(5)
        np.testing.assert_allclose(p[:, 0], [-0.25, 0])
        np.testing.assert_allclose(velocity[:, 0], [0.05, 0])
        np.testing.assert_allclose(replay.sample(21)[1], 0)
        np.testing.assert_allclose(replay.sample(21)[0][:, 0], [0.5, 0])

    def test_speed_cap_leaves_feedback_headroom(self):
        self.write([[0, "a", 0, 0, 0, 0, 0], [1, "a", 0, 1, 0, 0, 0]])
        replay = prepare_replay(self.path, max_speed=0.1)
        self.assertAlmostEqual(replay.speed_factor, 0.05)
        self.assertAlmostEqual(replay.duration, 20)

    def test_episode_selection_prevents_reset_jump(self):
        self.write([[0, "a", 0, 0, 0, 0, 0], [1, "a", 0, 1, 0, 0, 0],
                    [1, "a", 1, 0, 0, 0, 0], [2, "a", 1, 1, 0, 0, 0]])
        with self.assertRaisesRegex(ValueError, "multiple episodes"):
            prepare_replay(self.path)
        replay = prepare_replay(self.path, episode=1)
        np.testing.assert_allclose(replay.times, [0, 1])

    def test_bad_timestamps_and_nonfinite_values(self):
        for end, x in [(0, 1), (-1, 1), (1, float("nan")), (1, float("inf"))]:
            with self.subTest(end=end, x=x):
                self.write([[0, "a", 0, 0, 0, 0, 0], [end, "a", 0, x, 0, 0, 0]])
                with self.assertRaises(ValueError):
                    prepare_replay(self.path)

    def test_robot_selection_and_minimal_csv(self):
        self.write([[0, "a", 0, 0], [0, "b", 0, 1], [1, "a", 1, 0], [1, "b", 1, 1]],
                   fields=("time", "robot_id", "x", "z"))
        replay = prepare_replay(self.path, robots=["b", "a"])
        self.assertEqual(replay.names, ["b", "a"])
        np.testing.assert_allclose(replay.headings, 0)
        for names in (["missing"], ["a", "a"]):
            with self.assertRaises(ValueError):
                prepare_replay(self.path, robots=names)

    def test_stationary_robot_and_different_sampling_rates(self):
        self.write([[0, "a", 0, 0, 0, 0, 0], [0, "b", 0, 0, 0, 1, 0],
                    [1, "b", 0, 1, 0, 1, 0], [2, "a", 0, 0, 0, 0, 0],
                    [2, "b", 0, 2, 0, 1, 0]])
        names, times, positions, _ = load_recording(self.path)
        np.testing.assert_allclose(times, [0, 1, 2])
        np.testing.assert_allclose(positions[:, :, names.index("a")], 0)

    def test_incomplete_recording_is_not_extrapolated(self):
        self.write([[0, "a", 0, 0, 0, 0, 0], [0, "b", 0, 0, 0, 1, 0],
                    [1, "a", 0, 1, 0, 0, 0], [2, "b", 0, 1, 0, 1, 0]])
        with self.assertRaisesRegex(ValueError, "same start and end"):
            prepare_replay(self.path)

    def test_collision_between_samples_is_detected(self):
        positions = np.array([[[-1, 1], [0, 0]], [[1, -1], [0, 0]]], dtype=float)
        self.assertAlmostEqual(minimum_separation(positions), 0)
        self.write([[0, "a", 0, -1, 0, 0, 0], [0, "b", 0, 1, 0, 0, 0],
                    [1, "a", 0, 1, 0, 0, 0], [1, "b", 0, -1, 0, 0, 0]])
        with self.assertWarnsRegex(UserWarning, "Collision avoidance"):
            prepare_replay(self.path)

    def test_invalid_start_or_scale_is_rejected(self):
        self.write([[0, "a", 0, 0, 0, 0, 0], [0, "b", 0, 0.1, 0, 0, 0],
                    [1, "a", 0, 1, 0, 0, 0], [1, "b", 0, 1.1, 0, 0, 0]])
        with self.assertRaisesRegex(ValueError, "Starting robots"):
            prepare_replay(self.path)
        for scale in (0, -1, 100, float("nan")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                prepare_replay(self.path, robots=["a"], scale=scale)


if __name__ == "__main__":
    unittest.main()
