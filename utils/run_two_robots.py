"""Visualize a policy-controlled ANYmal and a kinematic unicycle together.

Both robots share world coordinates, simulation time, ground visualization,
and one camera. Their simulation models remain independent, with no physical
robot-to-robot contact response. Optional planar footprint detection stops
the rollout on contact. ANYmal uses the existing ground-truth
NeuralEnvironment path and its unchanged policy observation/action interface.

From the repository root:
    python utils/run_two_robots.py --device cuda:0
    python utils/run_two_robots.py --device cpu --headless --steps 10
    xvfb-run -a python utils/run_two_robots.py --device cuda:0 \
        --export-video-path outputs/two_robots.gif --csv-path outputs/two_robots.csv
"""

import argparse
import csv
import faulthandler
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
START_TIME = time.monotonic()


def progress(message):
    print(f"[two robots +{time.monotonic() - START_TIME:.1f}s] {message}", flush=True)
    if __name__ == "__main__":
        faulthandler.dump_traceback_later(120, repeat=True)


if __name__ == "__main__":
    faulthandler.enable()
    progress("Importing simulation and policy libraries")

import numpy as np
import torch
import warp as wp
import warp.sim.render
import yaml
from rl_games.algos_torch.players import PpoPlayerContinuous

from envs.neural_environment import NeuralEnvironment
from envs.warp_sim_envs.env_unicycle import RenderMode, UnicycleEnvironment
from envs.warp_sim_envs.wrapper_utils import get_action_space, get_observation_space
from utils.run_unicycle import encode_gif


DEFAULT_CONFIG = ROOT / "eval/eval_rl/cfg/Anymal/anymal_forward.yaml"
DEFAULT_POLICY = ROOT / "pretrained_models/RL_policies/Anymal/forward_walk/0/nn/AnymalPPO.pth"


def footprints_overlap(anymal_position, unicycle_position, anymal_radius, unicycle_radius):
    """Check touching/overlapping circular footprints in the (X, Z) plane."""
    return math.hypot(anymal_position[0] - unicycle_position[0],
                      anymal_position[1] - unicycle_position[1]) <= anymal_radius + unicycle_radius


def load_anymal_config(config_path, checkpoint_path, seed):
    """Read a playback configuration without importing the RL training stack."""
    with Path(config_path).open() as source:
        cfg = yaml.safe_load(source)
    if cfg["env"]["env_name"] not in ("Anymal", "AnymalJointPositionControl"):
        raise ValueError("--rl-cfg must describe an ANYmal environment.")
    cfg["seed"] = seed
    cfg["env"].update(num_envs=1, env_mode="ground-truth")
    if "waypoint_mode" in cfg["env"].get("warp_env_cfg", {}):
        raise ValueError("For this runner, provide explicit 'waypoints' in the config instead of 'waypoint_mode'.")
    return cfg


def load_policy(cfg, env, checkpoint_path, device):
    # Same actor/normalization architecture as run_rl.construct_rlg_config,
    # without importing training-only modules (which are version-sensitive).
    params = {
        "model": {"name": "continuous_a2c_logstd"},
        "network": {
            "name": "actor_critic", "separate": False,
            "space": {"continuous": {
                "mu_activation": "None", "sigma_activation": "None",
                "mu_init": {"name": "default"},
                "sigma_init": {"name": "const_initializer", "val": 0},
                "fixed_sigma": cfg["rl"].get("other_params", {}).get("fixed_sigma", True),
            }},
            **cfg["rl"]["network"],
        },
        "config": {**cfg["rl"]["config"], "env_name": "warp", "num_actors": 1},
    }
    params["config"].update(
        device_name=str(device), device=str(device),
        env_info={"observation_space": get_observation_space(env, use_gymnasium=False),
                  "action_space": get_action_space(env, use_gymnasium=False)},
    )
    player = PpoPlayerContinuous(params)
    # Explicit map_location permits CPU checks of checkpoints saved on CUDA.
    checkpoint = torch.load(str(checkpoint_path), map_location=str(device))
    weights = {key.replace("_orig_mod.", "", 1) if key.startswith("_orig_mod.") else key: value
               for key, value in checkpoint["model"].items()}
    player.model.load_state_dict(weights)
    if player.normalize_input and "running_mean_std" in checkpoint:
        player.model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
    player.has_batch_dimension = True
    player.reset()
    return player


class TwoRobotSimulation:
    """One rollout clock with independent ANYmal and unicycle dynamics."""

    def __init__(self, cfg, checkpoint, device="cpu", anymal_position=(0.0, 0.0),
                 unicycle_pose=(0.0, 2.0, 0.0), speed=1.0, turn_rate=0.5,
                 stop_on_collision=False, anymal_radius=0.6, unicycle_radius=0.25):
        if not all(math.isfinite(r) and r > 0 for r in (anymal_radius, unicycle_radius)):
            raise ValueError("Footprint radii must be finite and positive.")
        self.stop_on_collision = stop_on_collision
        self.anymal_radius, self.unicycle_radius = anymal_radius, unicycle_radius
        self.collision = None
        self.anymal = None
        self.unicycle = None
        self.elapsed = 0.0
        self.episode_step = 0
        self.episode = 0
        self.speed, self.turn_rate = speed, turn_rate
        self.control_steps = int(cfg["env"].get("control_steps", 1))
        self.horizon = int(cfg["env"]["max_episode_length"])
        if self.control_steps < 1 or self.horizon < 1:
            raise ValueError("control_steps and max_episode_length must be positive.")
        try:
            warp_cfg = dict(cfg["env"].get("warp_env_cfg", {}))
            warp_cfg.update(
                seed=cfg["seed"], random_reset=False, camera_tracking=False,
                setup_renderer=False, render_mode=RenderMode.NONE,
                env_positions=[(anymal_position[0], 0.0, anymal_position[1])],
            )
            progress("Creating ANYmal with ground-truth dynamics")
            self.anymal = NeuralEnvironment(
                env_name=cfg["env"]["env_name"], num_envs=1, warp_env_cfg=warp_cfg,
                default_env_mode="ground-truth", device=device, render=False,
            )
            # Match every ANYmal frame, including configs that repeat actions.
            self.frame_dt = self.anymal.frame_dt
            self.control_dt = self.frame_dt * self.control_steps
            progress("Creating unicycle")
            self.unicycle = UnicycleEnvironment(
                num_envs=1, initial_pose=unicycle_pose, frame_dt=self.frame_dt,
                device=device, render_mode=RenderMode.NONE, setup_renderer=False,
            )
            progress(f"Loading ANYmal policy: {checkpoint}")
            self.player = load_policy(cfg, self.anymal, checkpoint, device)
            self.obs = wp.empty((1, self.anymal.observation_dim), dtype=float, device=device)
            self.cost = wp.zeros(1, dtype=float, device=device)
            self.done = wp.zeros(1, dtype=wp.bool, device=device)
            self.reset()
        except Exception:
            self.close()
            raise

    def reset(self):
        """Reset both robots together; elapsed rollout time remains monotonic."""
        self.anymal.reset()
        self.unicycle.reset()
        self.unicycle.set_commands([[self.speed, self.turn_rate]])
        self.player.reset()
        self.episode_step = 0
        self.collision = None
        self._check_collision()

    def _check_collision(self):
        if not self.stop_on_collision:
            return False
        bodies, uni = self.transforms()
        anymal_position = tuple(map(float, bodies[0, [0, 2]]))
        unicycle_position = tuple(map(float, uni[[0, 2]]))
        if footprints_overlap(anymal_position, unicycle_position,
                              self.anymal_radius, self.unicycle_radius):
            self.collision = {
                "time": self.elapsed,
                "anymal_position": anymal_position,
                "unicycle_position": unicycle_position,
                "distance": math.hypot(anymal_position[0] - unicycle_position[0],
                                       anymal_position[1] - unicycle_position[1]),
            }
            return True
        return False

    def step(self):
        # A collision is latched until an explicit reset, including spawn overlap.
        if self.collision is not None:
            return True
        self.anymal.compute_observations(self.obs, self.episode_step, self.horizon)
        actions = self.player.get_action(wp.to_torch(self.obs), is_deterministic=True)
        if not torch.isfinite(actions).all():
            raise RuntimeError("ANYmal policy produced non-finite actions.")
        for _ in range(self.control_steps):
            self.anymal.step(actions)
            self.unicycle.update()
            self.elapsed += self.frame_dt
            if self._check_collision():
                self.episode_step += 1
                return True
        self.episode_step += 1
        self.cost.zero_()
        self.done.zero_()
        self.anymal.compute_cost_termination(self.episode_step, self.horizon, self.cost, self.done)
        return bool(self.done.numpy()[0]) or self.episode_step >= self.horizon

    def transforms(self):
        # Same world frame for both; the unicycle's offset is zero for num_envs=1.
        # Match Environment.render(): Featherstone's generalized coordinates
        # are authoritative, so refresh body poses before drawing or logging.
        state = self.anymal.sim_states
        wp.sim.eval_fk(self.anymal.model, state.joint_q, state.joint_qd, None, state)
        return self.anymal.sim_states.body_q.numpy(), self.unicycle.state.body_q.numpy()[0]

    def records(self, step):
        anymal_bodies, unicycle_body = self.transforms()
        distance = math.hypot(float(anymal_bodies[0, 0] - unicycle_body[0]),
                              float(anymal_bodies[0, 2] - unicycle_body[2]))
        for robot, transform in (("anymal", anymal_bodies[0]), ("unicycle", unicycle_body)):
            x, y, z, qx, qy, qz, qw = map(float, transform)
            # Planar heading from the body's local +X, positive toward world +Z.
            theta = math.atan2(2 * (qx * qz - qw * qy), 1 - 2 * (qy * qy + qz * qz))
            commands = self.unicycle.control.unicycle_commands.numpy()[0] if robot == "unicycle" else ("", "")
            yield [step, self.elapsed, self.episode, robot, x, y, z, theta, qx, qy, qz, qw, *commands,
                   int(self.collision is not None), distance, self.anymal_radius + self.unicycle_radius]

    def close(self):
        if self.unicycle is not None:
            self.unicycle.close()
        if self.anymal is not None:
            self.anymal.close()


class SharedScene:
    """Draw both models in one frame without changing ANYmal's physics model."""

    def __init__(self, simulation):
        self.sim = simulation
        self.renderer = wp.sim.render.SimRendererOpenGL(
            simulation.anymal.model, "ANYmal and unicycle", up_axis="Y",
            scaling=3.0, screen_width=1280, screen_height=720,
            near_plane=0.05, far_plane=1000.0, device=simulation.anymal.device,
        )

    def render(self):
        bodies, uni = self.sim.transforms()
        # Fit both robots, even when their paths diverge.
        positions = np.vstack([bodies[:, :3], uni[:3]]) * 3.0
        center = (positions.min(axis=0) + positions.max(axis=0)) * 0.5
        radius = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))) * 0.5 + 1.5
        distance = max(10.0, radius / math.sin(math.radians(22.5)))
        direction = np.array([0.0, -0.6, -1.0])
        direction /= np.linalg.norm(direction)
        self.renderer.update_view_matrix(cam_pos=center - direction * distance, cam_front=direction)
        self.renderer.begin_frame(self.sim.elapsed)
        self.renderer.render(self.sim.anymal.sim_states)
        unicycle = self.sim.unicycle
        self.renderer.render_box(
            "standalone_unicycle_body", pos=tuple(uni[:3]), rot=tuple(uni[3:]),
            extents=(unicycle.body_length / 2, unicycle.body_height / 2, unicycle.body_width / 2),
            color=(0.1, 0.5, 0.95),
        )
        transform = wp.transform(wp.vec3(*uni[:3]), wp.quat(*uni[3:]))
        nose = wp.transform_point(transform, wp.vec3(unicycle.body_length / 2, 0.0, 0.0))
        self.renderer.render_sphere(
            "standalone_unicycle_nose", pos=tuple(nose), rot=tuple(uni[3:]),
            radius=unicycle.body_width / 5, color=(1.0, 0.25, 0.1),
        )
        self.renderer.end_frame()

    def close(self):
        self.renderer.close()


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-cfg", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--playback", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=600, help="Number of policy control steps.")
    parser.add_argument("--speed", type=float, default=1.0, help="Unicycle forward speed in m/s.")
    parser.add_argument("--turn-rate", type=float, default=0.5, help="Unicycle turning rate in rad/s.")
    parser.add_argument("--anymal-position", type=float, nargs=2, default=(0, 0), metavar=("X", "Z"))
    parser.add_argument("--unicycle-pose", type=float, nargs=3, default=(0, 2, 0), metavar=("X", "Z", "THETA"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop-on-collision", action="store_true",
                        help="Stop on circular footprint contact, even with --reset-on-done.")
    parser.add_argument("--anymal-radius", type=float, default=0.6,
                        help="Approximate ANYmal planar footprint radius in metres (default: 0.6).")
    parser.add_argument("--unicycle-radius", type=float, default=0.25,
                        help="Approximate unicycle planar footprint radius in metres (default: 0.25).")
    parser.add_argument("--headless", action="store_true", help="Disable all rendering (cannot export GIF).")
    parser.add_argument("--reset-on-done", action="store_true", help="Reset both robots on ANYmal termination; otherwise stop.")
    parser.add_argument("--export-video-path", type=Path)
    parser.add_argument("--csv-path", type=Path, help="Log world poses of both robots at each control step.")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not all(math.isfinite(r) and r > 0 for r in (args.anymal_radius, args.unicycle_radius)):
        parser.error("Footprint radii must be finite and positive")
    if not all(math.isfinite(v) for v in [args.speed, args.turn_rate, *args.anymal_position, *args.unicycle_pose]):
        parser.error("Commands and spawn poses must be finite")
    for source in (args.rl_cfg, args.playback):
        if not source.is_file():
            parser.error(f"Missing input file: {source}")
    if args.export_video_path is not None:
        if args.headless or args.export_video_path.suffix.lower() != ".gif":
            parser.error("GIF export requires rendering and an output path ending in .gif")
        if shutil.which("ffmpeg") is None:
            parser.error("FFmpeg is required for GIF export")
    for output in (args.export_video_path, args.csv_path):
        if output is not None and output.exists():
            parser.error(f"Output already exists: {output}")
    return args


def main(argv=None):
    args = parse_arguments(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = load_anymal_config(args.rl_cfg, args.playback, args.seed)
    scene = simulation = frame_folder = csv_file = None
    try:
        simulation = TwoRobotSimulation(cfg, args.playback, args.device, args.anymal_position,
                                        args.unicycle_pose, args.speed, args.turn_rate,
                                        args.stop_on_collision, args.anymal_radius, args.unicycle_radius)
        if not args.headless:
            progress("Creating shared OpenGL scene")
            scene = SharedScene(simulation)
        writer = None
        if args.csv_path is not None:
            args.csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = args.csv_path.open("x", newline="")
            writer = csv.writer(csv_file)
            writer.writerow(["step", "time", "episode", "robot", "x", "y", "z", "theta",
                             "qx", "qy", "qz", "qw", "command_v", "command_omega",
                             "collision", "planar_distance", "collision_distance"])
            writer.writerows(simulation.records(0))
        pixels = None
        if args.export_video_path is not None:
            import cv2
            args.export_video_path.parent.mkdir(parents=True, exist_ok=True)
            frame_folder = tempfile.TemporaryDirectory(prefix="two_robot_frames_", dir=str(args.export_video_path.parent))
            pixels = wp.empty((scene.renderer.screen_height, scene.renderer.screen_width, 3), dtype=wp.uint8, device=args.device)
        frame_count = 0
        progress(f"Starting {args.steps} steps; control dt={simulation.control_dt:g} s")
        # Spawn overlap still gets one rendered/exported frame, with no dynamics step.
        for step in range(0 if simulation.collision is not None else 1, args.steps + 1):
            if scene is not None and not scene.renderer.is_running():
                break
            done = simulation.step() if step > 0 else True
            if writer is not None and step > 0:
                writer.writerows(simulation.records(step))
            if scene is not None:
                scene.render()
            if pixels is not None:
                if not scene.renderer.get_pixels(pixels, split_up_tiles=False, mode="rgb", use_uint8=True):
                    raise RuntimeError(f"Failed to capture frame {frame_count}")
                if not cv2.imwrite(str(Path(frame_folder.name) / f"{frame_count:06d}.png"), pixels.numpy()[:, :, ::-1]):
                    raise RuntimeError(f"Failed to save frame {frame_count}")
                frame_count += 1
            if step == 1 or step % 60 == 0 or step == args.steps:
                progress(f"Completed {step}/{args.steps} steps; simulated time={simulation.elapsed:.3f} s")
                if csv_file is not None:
                    csv_file.flush()
            if simulation.collision is not None:
                contact = simulation.collision
                progress(f"Footprint collision at t={contact['time']:.6f} s; "
                         f"ANYmal XZ={contact['anymal_position']}, unicycle XZ={contact['unicycle_position']}; "
                         f"distance={contact['distance']:.6f} m, "
                         f"threshold={simulation.anymal_radius + simulation.unicycle_radius:.6f} m. "
                         "Terminating without reset.")
                break
            if done:
                progress(f"ANYmal episode ended at step {step} (termination or horizon)")
                if not args.reset_on_done:
                    break
                simulation.episode += 1
                simulation.reset()
        progress(f"Final unicycle pose: {simulation.unicycle.get_pose().tolist()}")
        if frame_folder is not None:
            progress("Encoding shared-scene GIF")
            encode_gif(frame_folder.name, args.export_video_path, 1.0 / simulation.control_dt, shutil.which("ffmpeg"))
            progress(f"Saved {frame_count} frames to {args.export_video_path.resolve()}")
        if args.csv_path is not None:
            progress(f"Saved both robot trajectories to {args.csv_path.resolve()}")
    finally:
        if csv_file is not None:
            csv_file.close()
        if scene is not None:
            scene.close()
        if simulation is not None:
            simulation.close()
        if frame_folder is not None:
            frame_folder.cleanup()


if __name__ == "__main__":
    try:
        main()
    finally:
        faulthandler.cancel_dump_traceback_later()
