"""An ideal velocity-controlled unicycle on the X-Z ground plane.

Use this environment directly, without NeuralEnvironment or a trained policy::

    env = UnicycleEnvironment(num_envs=1, device="cpu", render_mode=RenderMode.NONE)
    env.set_commands([[1.0, 0.5]])  # forward m/s, turning rad/s
    env.update()                   # advance one frame_dt
    print(env.get_pose())          # local (x, z, theta), shape (num_envs, 3)
    env.close()

Positive theta turns from +X toward +Z (the negative of Warp's Y-axis yaw).
The free joint is a storage/rendering adapter, not a physical wheel model.
Gravity, contact forces, slip, and joint torques do not affect this robot.
"""

import math

import numpy as np
import warp as wp
import warp.sim

from .environment import Environment, IntegratorType, RenderMode


@wp.func
def unicycle_heading(q: wp.quat):
    forward = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    return wp.atan2(forward[2], forward[0])


@wp.kernel
def assign_unicycle_commands(
    actions: wp.array(dtype=float, ndim=2),
    max_speed: float,
    max_turn_rate: float,
    commands: wp.array(dtype=float, ndim=2),
):
    i = wp.tid()
    commands[i, 0] = wp.clamp(actions[i, 0], -max_speed, max_speed)
    commands[i, 1] = wp.clamp(actions[i, 1], -max_turn_rate, max_turn_rate)


@wp.kernel
def integrate_unicycle(
    joint_q: wp.array(dtype=float),
    commands: wp.array(dtype=float, ndim=2),
    dt: float,
    next_q: wp.array(dtype=float),
    next_qd: wp.array(dtype=float),
):
    i = wp.tid()
    qi = 7 * i
    vi = 6 * i
    q = wp.quat(joint_q[qi + 3], joint_q[qi + 4], joint_q[qi + 5], joint_q[qi + 6])
    theta = unicycle_heading(q)
    v = commands[i, 0]
    omega = commands[i, 1]

    # Exact constant-command arc, expressed using sinc to avoid division by
    # omega and cancellation near straight motion.
    half_turn = 0.5 * omega * dt
    sinc = float(1.0)
    if wp.abs(half_turn) < 1.0e-4:
        sinc = 1.0 - half_turn * half_turn / 6.0
    else:
        sinc = wp.sin(half_turn) / half_turn
    distance = v * dt * sinc
    midpoint_heading = theta + half_turn
    position = wp.vec3(
        joint_q[qi] + distance * wp.cos(midpoint_heading),
        joint_q[qi + 1],
        joint_q[qi + 2] + distance * wp.sin(midpoint_heading),
    )
    new_theta = wp.atan2(wp.sin(theta + 2.0 * half_turn), wp.cos(theta + 2.0 * half_turn))
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -new_theta)
    for j in range(3):
        next_q[qi + j] = position[j]
    for j in range(4):
        next_q[qi + 3 + j] = rotation[j]

    # In Warp 1.8's eval_fk, the free joint stores angular and linear
    # velocity directly in its parent frame (world here). The COM is zero.
    angular = wp.vec3(0.0, -omega, 0.0)
    linear = wp.vec3(v * wp.cos(new_theta), 0.0, v * wp.sin(new_theta))
    for j in range(3):
        next_qd[vi + j] = angular[j]
        next_qd[vi + 3 + j] = linear[j]


@wp.kernel
def observe_unicycle(
    joint_q: wp.array(dtype=float),
    offsets: wp.array(dtype=wp.vec3),
    observations: wp.array(dtype=float, ndim=2),
):
    i = wp.tid()
    qi = 7 * i
    q = wp.quat(joint_q[qi + 3], joint_q[qi + 4], joint_q[qi + 5], joint_q[qi + 6])
    observations[i, 0] = joint_q[qi] - offsets[i][0]
    observations[i, 1] = joint_q[qi + 2] - offsets[i][2]
    observations[i, 2] = unicycle_heading(q)


@wp.kernel
def clear_unicycle_commands(
    reset: wp.array(dtype=wp.bool),
    commands: wp.array(dtype=float, ndim=2),
):
    i = wp.tid()
    if reset:
        if not reset[i]:
            return
    commands[i, 0] = 0.0
    commands[i, 1] = 0.0


class UnicycleEnvironment(Environment):
    """Batched unicycles with commands (v, omega) in m/s and rad/s.

    ``initial_pose`` is either one (x, z, theta) shared by all robots or an
    array of shape (num_envs, 3), relative to each environment's offset.
    ``assign_control`` accepts physical commands, not normalized actions.
    ``update``, ``render``, and ``close`` use the existing Environment API.
    """

    robot_name = "Unicycle"
    sim_name = "env_unicycle"
    num_envs = 1
    env_offset = (2.0, 0.0, 2.0)
    gravity = 0.0
    use_graph_capture = False
    integrator_type = IntegratorType.XPBD
    sim_substeps_xpbd = 1
    sim_substeps_euler = 1
    sim_substeps_featherstone = 1
    opengl_render_settings = dict(scaling=3.0)

    def __init__(
        self,
        initial_pose=(0.0, 0.0, 0.0),
        max_speed=1.0,
        max_turn_rate=math.pi,
        frame_dt=1.0 / 60.0,
        body_length=0.4,
        body_width=0.25,
        body_height=0.15,
        **kwargs,
    ):
        if kwargs.get("requires_grad", False):
            raise ValueError("UnicycleEnvironment currently supports requires_grad=False only.")
        for name, value in (
            ("max_speed", max_speed), ("max_turn_rate", max_turn_rate),
            ("frame_dt", frame_dt), ("body_length", body_length),
            ("body_width", body_width), ("body_height", body_height),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            setattr(self, name, float(value))
        count = kwargs.get("num_envs")
        count = self.num_envs if count is None else count
        if not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("num_envs must be a positive integer.")
        poses = np.asarray(initial_pose, dtype=np.float32)
        if poses.shape == (3,):
            poses = np.tile(poses, (count, 1))
        if poses.shape != (count, 3) or not np.isfinite(poses).all():
            raise ValueError("initial_pose must be finite with shape (3,) or (num_envs, 3).")
        self.initial_pose = poses.copy()
        # These are velocity command channels, not actuated joint indices.
        self.control_limits = [(-self.max_speed, self.max_speed),
                               (-self.max_turn_rate, self.max_turn_rate)]
        self.controllable_dofs = [0, 1]
        self.control_gains = [1.0, 1.0]
        self.custom_dynamics = self.integrate_kinematics
        super().__init__(**kwargs)
        self._pose_buffer = wp.empty((self.num_envs, 3), dtype=float, device=self.device)
        self.reset()

    @property
    def uses_generalized_coordinates(self):
        # Our callback always writes joint_q/qd, regardless of the unused
        # physical integrator constructed by the base class.
        return True

    @property
    def observation_dim(self):
        return 3

    def create_articulation(self, builder):
        builder.add_articulation()
        body = builder.add_body(name="unicycle")
        builder.add_joint_free(child=body, parent=-1)
        builder.add_shape_box(
            body, hx=self.body_length / 2.0, hy=self.body_height / 2.0,
            hz=self.body_width / 2.0, density=100.0,
            has_shape_collision=False, has_ground_collision=False,
        )
        # A small nose marks the local +X (forward) direction. Zero density
        # keeps the center of mass at the body origin.
        builder.add_shape_sphere(
            body, pos=wp.vec3(self.body_length / 2.0, 0.0, 0.0),
            radius=self.body_width / 5.0, density=0.0,
            has_shape_collision=False, has_ground_collision=False,
        )

    def get_env_transform(self, env_id):
        x, z, theta = self.initial_pose[env_id]
        offset = self.env_offsets[env_id]
        return wp.transform(
            wp.vec3(float(offset[0] + x), float(offset[1] + self.body_height / 2.0),
                    float(offset[2] + z)),
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -float(theta)),
        )

    def customize_control(self, control):
        control.unicycle_commands = wp.zeros((self.num_envs, 2), dtype=float, device=self.device)

    def assign_control(self, actions, control, state):
        if actions.shape != (self.num_envs, 2) or actions.dtype != wp.float32:
            raise ValueError("actions must be a float32 Warp array with shape (num_envs, 2).")
        if actions.device != self.device:
            raise ValueError("actions must be on the environment's device.")
        wp.launch(
            assign_unicycle_commands, dim=self.num_envs,
            inputs=[actions, self.max_speed, self.max_turn_rate],
            outputs=[control.unicycle_commands], device=self.device,
        )

    def set_commands(self, commands):
        """Set finite physical velocity commands from a list or NumPy array.

        Commands are clipped to the configured limits and held until replaced
        or reset. For device-resident arrays use ``assign_control`` directly.
        """
        values = np.asarray(commands, dtype=np.float32)
        if values.shape != (self.num_envs, 2) or not np.isfinite(values).all():
            raise ValueError("commands must be finite with shape (num_envs, 2).")
        self.assign_control(wp.array(values, dtype=float, device=self.device), self.control, self.state)

    def integrate_kinematics(self, model, state, next_state, sim_dt, control):
        wp.launch(
            integrate_unicycle, dim=self.num_envs,
            inputs=[state.joint_q, control.unicycle_commands, sim_dt],
            outputs=[next_state.joint_q, next_state.joint_qd], device=self.device,
        )
        wp.sim.eval_fk(model, next_state.joint_q, next_state.joint_qd, None, next_state)

    def compute_observations(self, state, control, observations, step, horizon_length):
        wp.launch(
            observe_unicycle, dim=self.num_envs,
            inputs=[state.joint_q, self.env_offsets_wp], outputs=[observations], device=self.device,
        )

    def get_pose(self):
        """Return local (x, z, theta) as a NumPy copy; synchronizes the device."""
        self.compute_observations(self.state, self.control, self._pose_buffer, 0, 0)
        return self._pose_buffer.numpy().copy()

    def reset_envs(self, env_ids=None, state=None):
        """Reset all robots, or those selected by a device-resident boolean mask."""
        if env_ids is not None:
            if (env_ids.shape != (self.num_envs,) or env_ids.dtype != wp.bool
                    or env_ids.device != self.device):
                raise ValueError("env_ids must be a boolean Warp mask of shape (num_envs,) on this device.")
        state = self.state if state is None else state
        super().reset_envs(env_ids, state)
        for control in self.controls:
            wp.launch(clear_unicycle_commands, dim=self.num_envs,
                      inputs=[env_ids, control.unicycle_commands], device=self.device)
        wp.sim.eval_fk(self.model, state.joint_q, state.joint_qd, None, state)
