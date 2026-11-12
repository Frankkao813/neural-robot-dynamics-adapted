"""CPU checks of the real ANYmal checkpoint and synchronized unicycle rollout."""

import math
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from utils.run_two_robots import (
    DEFAULT_CONFIG, DEFAULT_POLICY, SharedScene, TwoRobotSimulation,
    footprints_overlap, load_anymal_config, parse_arguments,
)
from envs.neural_environment import NeuralEnvironment


class TwoRobotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = load_anymal_config(DEFAULT_CONFIG, DEFAULT_POLICY, 0)
        cls.sim = TwoRobotSimulation(cfg, DEFAULT_POLICY, device="cpu")

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def setUp(self):
        self.sim.speed, self.sim.turn_rate = 1.0, 0.5
        self.sim.elapsed = 0.0
        self.sim.episode = 0
        self.sim.reset()

    def test_real_policy_and_synchronized_motion(self):
        before = self.sim.anymal.states.clone()
        self.sim.step()
        self.assertTrue(torch.isfinite(self.sim.anymal.states).all())
        self.assertFalse(torch.equal(before, self.sim.anymal.states))
        dt = self.sim.control_dt
        np.testing.assert_allclose(self.sim.unicycle.get_pose()[0],
                                   [2 * math.sin(0.5 * dt), 2 + 2 * (1 - math.cos(0.5 * dt)), 0.5 * dt], atol=2e-6)
        self.assertAlmostEqual(self.sim.anymal.env.sim_time, self.sim.unicycle.sim_time)
        self.assertAlmostEqual(self.sim.elapsed, dt)
        self.assertEqual(self.sim.obs.shape, (1, self.sim.anymal.observation_dim))

    def test_unicycle_commands_do_not_change_anymal(self):
        for _ in range(3):
            self.sim.step()
        reference = self.sim.anymal.states.clone()
        self.sim.speed, self.sim.turn_rate = -0.2, -0.3
        self.sim.reset()
        for _ in range(3):
            self.sim.step()
        torch.testing.assert_close(self.sim.anymal.states, reference, rtol=1e-5, atol=1e-6)

    def test_anymal_joint_targets_scale_actions_once(self):
        actions = torch.full((1, self.sim.anymal.action_dim), 0.2)
        baseline = self.sim.anymal.env.default_joint_q.numpy()[7:].copy()
        self.sim.anymal.step(actions)
        # A 0.2 policy action requests a 0.1-radian offset with scale=0.5.
        np.testing.assert_allclose(self.sim.anymal.env.target_joint_q.numpy(),
                                   baseline + 0.1, atol=1e-6)

    def test_anymal_policy_walks_for_ten_seconds(self):
        initial_x = float(self.sim.anymal.states[0, 0])
        for step in range(600):
            self.assertFalse(self.sim.step(), f"ANYmal terminated at step {step + 1}")
        self.assertTrue(torch.isfinite(self.sim.anymal.states).all())
        self.assertGreater(float(self.sim.anymal.states[0, 0]) - initial_x, 5.0)

    def test_ant_torque_scaling_is_preserved(self):
        ant = NeuralEnvironment(
            env_name="Ant", num_envs=1, device="cpu", render=False,
            default_env_mode="ground-truth",
            warp_env_cfg={"random_reset": False, "setup_renderer": False,
                          "action_scale": 0.5},
        )
        try:
            ant.reset()
            ant.step(torch.full((1, ant.action_dim), 0.25))
            np.testing.assert_allclose(ant.joint_control.joint_act.numpy(),
                                       np.asarray(ant.control_gains) * 0.25 * 0.5)
        finally:
            ant.close()

    def test_shared_renderer_receives_both_robot_poses(self):
        self.sim.step()
        scene = SharedScene.__new__(SharedScene)
        scene.sim = self.sim
        scene.renderer = Mock()
        scene.render()
        scene.renderer.begin_frame.assert_called_once_with(self.sim.elapsed)
        scene.renderer.render.assert_called_once_with(self.sim.anymal.sim_states)
        box = scene.renderer.render_box.call_args.kwargs
        uni = self.sim.unicycle.state.body_q.numpy()[0]
        np.testing.assert_allclose(box["pos"], uni[:3])
        np.testing.assert_allclose(box["rot"], uni[3:])
        nose = scene.renderer.render_sphere.call_args.kwargs["pos"]
        theta = self.sim.unicycle.get_pose()[0, 2]
        expected = uni[:3] + np.array([math.cos(theta), 0, math.sin(theta)]) * self.sim.unicycle.body_length / 2
        np.testing.assert_allclose(nose, expected, atol=1e-6)
        scene.renderer.end_frame.assert_called_once()

    def test_reset_and_csv_records(self):
        self.sim.step()
        rows = list(self.sim.records(1))
        self.assertEqual([row[3] for row in rows], ["anymal", "unicycle"])
        self.assertEqual(len(rows[0]), 17)
        self.assertEqual(rows[0][1], rows[1][1])
        self.assertEqual(rows[0][12:14], ["", ""])
        np.testing.assert_allclose(rows[1][12:14], [1.0, 0.5])
        self.assertEqual(rows[0][14], 0)
        self.assertEqual(rows[0][15:], rows[1][15:])
        elapsed = self.sim.elapsed
        self.sim.reset()
        self.assertEqual(self.sim.episode_step, 0)
        self.assertEqual(self.sim.elapsed, elapsed)
        np.testing.assert_allclose(self.sim.unicycle.get_pose(), [[0, 2, 0]])

    def test_default_paths_and_headless_validation(self):
        args = parse_arguments(["--headless", "--device", "cpu", "--steps", "2"])
        self.assertTrue(args.playback.is_absolute())
        self.assertTrue(args.rl_cfg.is_absolute())
        with self.assertRaises(SystemExit):
            parse_arguments(["--headless", "--export-video-path", "two_robots.gif"])


class FootprintTests(unittest.TestCase):
    def test_contact_overlap_and_near_miss(self):
        self.assertTrue(footprints_overlap((0, 0), (0.75, 0), 0.5, 0.25))
        self.assertTrue(footprints_overlap((2, 3), (2, 3), 0.5, 0.25))
        self.assertFalse(footprints_overlap((0, 0), (0, 0.751), 0.5, 0.25))

    def test_invalid_radii(self):
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parse_arguments(["--anymal-radius", value])

    def test_collision_interrupts_control_repeat_at_frame_boundary(self):
        sim = TwoRobotSimulation.__new__(TwoRobotSimulation)
        sim.collision = None
        sim.anymal, sim.unicycle, sim.player = Mock(), Mock(), Mock()
        sim.elapsed, sim.episode_step, sim.horizon = 0.0, 0, 1000
        sim.frame_dt, sim.control_steps = 0.02, 5
        sim.obs = Mock()
        sim.player.get_action.return_value = torch.zeros((1, 12))
        sim._check_collision = Mock(side_effect=[False, True])
        from unittest.mock import patch
        with patch("utils.run_two_robots.wp.to_torch", return_value=torch.zeros((1, 37))):
            self.assertTrue(sim.step())
        self.assertEqual(sim.anymal.step.call_count, 2)
        self.assertEqual(sim.unicycle.update.call_count, 2)
        self.assertAlmostEqual(sim.elapsed, 0.04)
        self.assertEqual(sim.episode_step, 1)


class CollisionRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = load_anymal_config(DEFAULT_CONFIG, DEFAULT_POLICY, 0)
        cls.sim = TwoRobotSimulation(cfg, DEFAULT_POLICY, device="cpu",
                                    unicycle_pose=(4, 0, math.pi), turn_rate=0,
                                    stop_on_collision=True)

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def setUp(self):
        self.sim.stop_on_collision = True
        self.sim.elapsed = 0.0
        self.sim.reset()

    def test_head_on_stops_at_first_sampled_contact_and_latches(self):
        previous = list(self.sim.records(0))[0]
        for step in range(1, 601):
            if self.sim.step():
                break
            previous = list(self.sim.records(step))[0]
        self.assertIsNotNone(self.sim.collision)
        self.assertLess(step, 600)
        self.assertGreater(previous[15], previous[16])
        final = list(self.sim.records(step))[0]
        self.assertEqual(final[14], 1)
        self.assertLessEqual(final[15], final[16])
        before = self.sim.anymal.states.clone()
        pose, elapsed = self.sim.unicycle.get_pose(), self.sim.elapsed
        self.assertTrue(self.sim.step())
        torch.testing.assert_close(self.sim.anymal.states, before)
        np.testing.assert_array_equal(self.sim.unicycle.get_pose(), pose)
        self.assertEqual(self.sim.elapsed, elapsed)
        self.sim.reset()
        self.assertIsNone(self.sim.collision)

    def test_initial_overlap_stops_without_advancing(self):
        cfg = load_anymal_config(DEFAULT_CONFIG, DEFAULT_POLICY, 0)
        sim = TwoRobotSimulation(cfg, DEFAULT_POLICY, device="cpu",
                                 unicycle_pose=(0, 0, 0), stop_on_collision=True)
        try:
            self.assertIsNotNone(sim.collision)
            self.assertTrue(sim.step())
            self.assertEqual(sim.elapsed, 0.0)
            self.assertEqual(sim.episode_step, 0)
        finally:
            sim.close()

    def test_detection_disabled_preserves_pass_through(self):
        self.sim.stop_on_collision = False
        self.sim.reset()
        for _ in range(200):
            self.assertFalse(self.sim.step())
        self.assertIsNone(self.sim.collision)
        bodies, uni = self.sim.transforms()
        self.assertGreater(bodies[0, 0], uni[0])


if __name__ == "__main__":
    unittest.main()
