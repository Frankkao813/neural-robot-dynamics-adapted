"""Run from the repository root: python -m unittest discover -s tests -v."""

import math
import unittest

import numpy as np
import warp as wp

from envs.warp_sim_envs.env_unicycle import RenderMode, UnicycleEnvironment


class UnicycleTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = UnicycleEnvironment(device="cpu", render_mode=RenderMode.NONE, **kwargs)
        self.addCleanup(env.close)
        return env

    def test_straight_reverse_rotation_and_stop(self):
        env = self.make_env(num_envs=4, frame_dt=0.25)
        env.set_commands([[1.0, 0.0], [-0.5, 0.0], [0.0, 2.0], [0.0, 0.0]])
        for _ in range(4):
            env.update()
        np.testing.assert_allclose(
            env.get_pose(), [[1, 0, 0], [-0.5, 0, 0], [0, 0, 2], [0, 0, 0]], atol=2e-6,
        )
        self.assertAlmostEqual(env.sim_time, 1.0)
        before = env.get_pose()
        env.set_commands(np.zeros((4, 2)))
        env.update()
        np.testing.assert_allclose(env.get_pose(), before, atol=2e-6)
        np.testing.assert_allclose(env.state.body_qd.numpy(), 0.0, atol=2e-6)

    def test_quarter_circle_and_timestep_consistency(self):
        for steps in (1, 40):
            with self.subTest(steps=steps):
                env = self.make_env(frame_dt=math.pi / (2 * steps))
                env.set_commands([[1.0, 1.0]])
                for _ in range(steps):
                    env.update()
                np.testing.assert_allclose(env.get_pose(), [[1, 1, math.pi / 2]], atol=5e-6)

    def test_tiny_and_negative_turn_rates(self):
        env = self.make_env(num_envs=2, frame_dt=1.0)
        env.set_commands([[1.0, 1e-8], [1.0, -1.0]])
        env.update()
        np.testing.assert_allclose(
            env.get_pose(), [[1, 5e-9, 1e-8], [math.sin(1), math.cos(1) - 1, -1]], atol=2e-6,
        )

    def test_limits_and_heading_wrap(self):
        env = self.make_env(max_speed=0.5, max_turn_rate=1.0, frame_dt=1.0,
                            initial_pose=(0, 0, math.pi - 0.1))
        env.set_commands([[10.0, 10.0]])
        np.testing.assert_allclose(env.control.unicycle_commands.numpy(), [[0.5, 1.0]])
        env.update()
        self.assertAlmostEqual(env.get_pose()[0, 2], -math.pi + 0.9, places=5)
        env.set_commands([[-10.0, -10.0]])
        np.testing.assert_allclose(env.control.unicycle_commands.numpy(), [[-0.5, -1.0]])

    def test_offsets_pose_and_body_velocity_are_consistent(self):
        env = self.make_env(num_envs=2, initial_pose=[[1, 2, 0], [-1, 3, math.pi / 2]],
                            env_positions=[(10, 0, 20), (-10, 0, -20)], frame_dt=0.1)
        env.set_commands([[0.8, 0.7], [0.4, -0.3]])
        env.update()
        pose = env.get_pose()
        body = env.state.body_q.numpy()
        velocity = env.state.body_qd.numpy()
        np.testing.assert_allclose(body[:, 0], pose[:, 0] + [10, -10], atol=2e-6)
        np.testing.assert_allclose(body[:, 2], pose[:, 1] + [20, -20], atol=2e-6)
        np.testing.assert_allclose(body[:, 1], env.body_height / 2, atol=2e-6)
        # q and rendered body orientation must both point along the reported heading.
        np.testing.assert_allclose(body[:, 3], 0, atol=2e-6)
        np.testing.assert_allclose(body[:, 5], 0, atol=2e-6)
        np.testing.assert_allclose(-2 * np.arctan2(body[:, 4], body[:, 6]), pose[:, 2], atol=2e-6)
        np.testing.assert_allclose(velocity[:, 1], [-0.7, 0.3], atol=2e-6)
        np.testing.assert_allclose(velocity[:, 3], np.array([0.8, 0.4]) * np.cos(pose[:, 2]), atol=5e-6)
        np.testing.assert_allclose(velocity[:, 5], np.array([0.8, 0.4]) * np.sin(pose[:, 2]), atol=5e-6)

    def test_partial_and_full_reset(self):
        initial = [[1, 2, 0.2], [-1, 3, -0.4]]
        env = self.make_env(num_envs=2, initial_pose=initial)
        env.set_commands([[1, 0.5], [0.5, -0.5]])
        env.update()
        before = env.get_pose()
        env.reset_envs(wp.array([True, False], dtype=wp.bool, device=env.device))
        np.testing.assert_allclose(env.get_pose()[0], initial[0], atol=2e-6)
        np.testing.assert_allclose(env.get_pose()[1], before[1], atol=2e-6)
        np.testing.assert_allclose(env.control.unicycle_commands.numpy(), [[0, 0], [0.5, -0.5]])
        env.update()  # The swapped state buffer must not resurrect the reset pose/command.
        np.testing.assert_allclose(env.get_pose()[0], initial[0], atol=2e-6)
        env.reset()
        np.testing.assert_allclose(env.get_pose(), initial, atol=2e-6)
        np.testing.assert_allclose(env.state.body_qd.numpy(), 0.0, atol=2e-6)
        np.testing.assert_allclose(env.control.unicycle_commands.numpy(), 0.0)
        self.assertEqual(env.sim_time, 0)

    def test_invalid_inputs(self):
        for kwargs in ({"frame_dt": 0}, {"max_speed": -1}, {"max_turn_rate": float("nan")},
                       {"num_envs": 0}, {"initial_pose": [1, 2]}, {"requires_grad": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make_env(**kwargs)
        env = self.make_env()
        for commands in ([1, 0], [[float("nan"), 0]], [[0, float("inf")]]):
            with self.subTest(commands=commands), self.assertRaises(ValueError):
                env.set_commands(commands)


if __name__ == "__main__":
    unittest.main()
