"""Track a saved Warp (X, Z) trajectory in the Robotarium Python simulator.

Run from the repository root:
    python utils/replay_robotarium.py outputs/two_robots.csv --dry-run
    python utils/replay_robotarium.py outputs/two_robots.csv --robots anymal

Only NumPy and the official Robotarium simulator are needed; no Warp or policy
imports. See docs/robotarium_replay.md for recording and installation commands.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np


BOUNDARIES = np.array([-1.6, 1.6, -1.0, 1.0])
SAFETY_RADIUS = 0.15
PROJECTION_DISTANCE = 0.05


@dataclass
class Replay:
    names: list
    times: np.ndarray              # Common timeline, starting at zero (seconds).
    positions: np.ndarray          # Samples x 2 planar coordinates x robots.
    headings: np.ndarray           # Initial headings only, radians.
    scale: float
    speed_factor: float            # Source seconds per replay second.

    @property
    def duration(self):
        return float(self.times[-1] / self.speed_factor)

    def sample(self, elapsed):
        """Piecewise-linear position and its feedforward velocity."""
        t = np.clip(elapsed * self.speed_factor, 0.0, self.times[-1])
        i = min(np.searchsorted(self.times, t, side="right") - 1, len(self.times) - 2)
        dt = self.times[i + 1] - self.times[i]
        delta = self.positions[i + 1] - self.positions[i]
        position = self.positions[i] + ((t - self.times[i]) / dt) * delta
        velocity = delta / dt * self.speed_factor
        if elapsed >= self.duration:
            velocity = np.zeros_like(velocity)
        return position, velocity


def load_recording(path, robots=None, episode=None):
    """Read the native run_two_robots CSV, or time,robot_id,x,z[,theta]."""
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or [])
        missing = {"time", "x", "z"} - fields
        if missing:
            raise ValueError("CSV is missing columns: " + ", ".join(sorted(missing)))
        name_column = "robot" if "robot" in fields else "robot_id"
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has no samples.")
    available = list(dict.fromkeys(row.get(name_column, "robot0") for row in rows))
    names = list(robots) if robots else available
    if len(names) != len(set(names)) or set(names) - set(available):
        raise ValueError("Robot selection must be unique and drawn from: " + ", ".join(available))
    rows = [row for row in rows if row.get(name_column, "robot0") in names]
    episodes = sorted({int(row.get("episode", "0")) for row in rows})
    if episode is None:
        if len(episodes) > 1:
            raise ValueError("Recording contains multiple episodes; select --episode from " + str(episodes))
        episode = episodes[0]
    rows = [row for row in rows if int(row.get("episode", "0")) == episode]
    tracks = []
    for name in names:
        selected = [row for row in rows if row.get(name_column, "robot0") == name]
        if len(selected) < 2:
            raise ValueError(f"Robot {name!r}, episode {episode}: need at least two samples.")
        values = np.array([[float(row[k]) for k in ("time", "x", "z")] for row in selected])
        if not np.isfinite(values).all() or np.any(np.diff(values[:, 0]) <= 0):
            raise ValueError(f"Robot {name!r}: values must be finite and timestamps strictly increasing.")
        theta = selected[0].get("theta", "")
        if theta:
            heading = float(theta)
        else:
            differences = np.diff(values[:, 1:], axis=0)
            moving = differences[np.linalg.norm(differences, axis=1) > 1e-9]
            heading = float(np.arctan2(moving[0, 1], moving[0, 0])) if len(moving) else 0.0
        if not np.isfinite(heading):
            raise ValueError(f"Robot {name!r}: initial theta must be finite.")
        tracks.append((values, heading))
    # Do not silently truncate robots' recordings or extrapolate missing endpoints.
    starts = [track[0][0, 0] for track in tracks]
    ends = [track[0][-1, 0] for track in tracks]
    if np.ptp(starts) > 1e-8 or np.ptp(ends) > 1e-8:
        raise ValueError("Selected robots must share the same start and end timestamps.")
    times = np.unique(np.concatenate([track[0][:, 0] for track in tracks]))
    positions = np.stack([
        np.column_stack([np.interp(times, values[:, 0], values[:, k]) for k in (1, 2)])
        for values, _ in tracks
    ], axis=2)
    return names, times - times[0], positions, np.array([heading for _, heading in tracks])


def minimum_separation(positions):
    """Exact minimum for synchronized, piecewise-linear center paths."""
    closest = float("inf")
    for i in range(positions.shape[2]):
        for j in range(i):
            relative = positions[:, :, i] - positions[:, :, j]
            delta = np.diff(relative, axis=0)
            norm2 = np.sum(delta * delta, axis=1)
            fraction = np.clip(-np.sum(relative[:-1] * delta, axis=1) /
                               np.maximum(norm2, 1e-30), 0.0, 1.0)
            closest = min(closest, float(np.min(np.linalg.norm(
                relative[:-1] + fraction[:, None] * delta, axis=1))))
    return closest


def prepare_replay(path, robots=None, episode=None, scale=None, margin=0.2,
                   speed_factor=1.0, max_speed=0.15):
    names, times, positions, headings = load_recording(path, robots, episode)
    if not (np.isfinite(margin) and 0.2 <= margin < 1.0):
        raise ValueError("--margin must be at least 0.2 and less than 1.0 metres.")
    if not (np.isfinite(speed_factor) and speed_factor > 0):
        raise ValueError("--speed-factor must be finite and positive.")
    if not (np.isfinite(max_speed) and 0 < max_speed <= 0.2):
        raise ValueError("--max-speed must be in (0, 0.2] m/s.")
    lower, upper = positions.min(axis=(0, 2)), positions.max(axis=(0, 2))
    available = np.array([3.2, 2.0]) - 2 * margin
    span = upper - lower
    fit = min(available[span > 0] / span[span > 0], default=float("inf"))
    if scale is None:
        scale = min(1.0, fit)  # Shrink when necessary, but do not enlarge by default.
    if not np.isfinite(scale) or scale <= 0 or scale > fit + 1e-12:
        raise ValueError(f"--scale must be positive and fit the arena (maximum {fit:.6g}).")
    positions = (positions - ((lower + upper) / 2)[None, :, None]) * scale
    # Reserve half the command speed for feedback corrections.
    source_speed = np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)[:, None])
    if source_speed > 0:
        speed_factor = min(speed_factor, 0.5 * max_speed / source_speed)
    # The projected-point barrier uses extra clearance. Reject an invalid start
    # before constructing Robotarium (whose constructor may drive to the start).
    starts = positions[0] + PROJECTION_DISTANCE * np.array([np.cos(headings), np.sin(headings)])
    for i in range(len(names)):
        for j in range(i):
            if np.linalg.norm(starts[:, i] - starts[:, j]) <= SAFETY_RADIUS + 2 * PROJECTION_DISTANCE:
                raise ValueError("Starting robots are too close after scaling; select one robot or change the recording.")
    separation = minimum_separation(positions)
    if separation < SAFETY_RADIUS + 2 * PROJECTION_DISTANCE:
        warnings.warn(f"Reference paths approach within {separation:.3f} m. "
                      "Collision avoidance may change the paths or prevent completion.")
    return Replay(names, times, positions, headings, float(scale), float(speed_factor))


def run_replay(replay, args):
    try:
        import rps.robotarium as robotarium
        from rps.utilities import barrier_certificates
        from rps.utilities.transformations import create_si_to_uni_dynamics
    except ImportError as error:
        raise RuntimeError("Install the official robotarium_python_simulator in this Python environment. "
                           "See docs/robotarium_replay.md; --dry-run needs only NumPy.") from error
    # Official simulator releases use either the newer 'uni' or older 'unicycle' name.
    barrier_factory = getattr(barrier_certificates, "create_uni_barrier_certificate_with_boundary", None)
    if barrier_factory is None:
        barrier_factory = barrier_certificates.create_unicycle_barrier_certificate_with_boundary
    barrier = barrier_factory(safety_radius=SAFETY_RADIUS, projection_distance=PROJECTION_DISTANCE,
                              velocity_magnitude_limit=args.max_speed, boundary_points=BOUNDARIES.copy())
    to_unicycle = create_si_to_uni_dynamics()
    n = len(replay.names)
    initial = np.vstack((replay.positions[0], replay.headings))
    # Open the output before creating the simulator; existing logs are never overwritten.
    log = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log = args.log.open("x", newline="")
    r = None
    pose_pending = False
    try:
        writer = csv.writer(log) if log else None
        if writer:
            writer.writerow(["time", "robot", "reference_x", "reference_y", "x", "y", "theta", "error_m"])
        r = robotarium.Robotarium(number_of_robots=n, show_figure=not args.headless,
                                 sim_in_real_time=not args.fast, initial_conditions=initial)
        dt = float(getattr(r, "TIME_STEP", getattr(r, "time_step", 0.033)))
        base_length = float(getattr(r, "BASE_LENGTH", getattr(r, "base_length", 0.105)))
        wheel_limit = float(getattr(r, "MAX_LINEAR_VELOCITY", getattr(r, "max_linear_velocity", 0.2)))
        ids = np.arange(n)
        squared_error, samples, peak_error = 0.0, 0, 0.0
        converged = False
        for step in range(int(np.ceil((replay.duration + args.settle_time) / dt)) + 1):
            elapsed = step * dt
            poses = r.get_poses()
            pose_pending = True
            reference, feedforward = replay.sample(elapsed)
            error = reference - poses[:2]
            distances = np.linalg.norm(error, axis=0)
            squared_error += float(np.sum(distances**2))
            samples += n
            peak_error = max(peak_error, float(distances.max()))
            if writer:
                for i, name in enumerate(replay.names):
                    writer.writerow([elapsed, name, *reference[:, i], *poses[:, i], distances[i]])
            converged = elapsed >= replay.duration and np.all(distances <= args.tolerance)
            desired = feedforward + args.gain * error
            desired *= np.minimum(1.0, args.max_speed / np.maximum(np.linalg.norm(desired, axis=0), 1e-12))
            commands = barrier(to_unicycle(desired, poses), poses)
            if not np.isfinite(commands).all():
                raise RuntimeError("Controller returned non-finite commands.")
            # Uniform scaling preserves relative robot velocities while respecting
            # wheel-speed limits, including simultaneous translation and rotation.
            wheel_speed = np.max(np.abs(commands[0]) + base_length / 2 * np.abs(commands[1]))
            commands *= min(1.0, 0.95 * wheel_limit / max(float(wheel_speed), 1e-12))
            r.set_velocities(ids, np.zeros((2, n)) if converged else commands)
            r.step()
            pose_pending = False
            if converged:
                break
        print(f"Tracking error: RMS {np.sqrt(squared_error / samples):.4f} m; peak {peak_error:.4f} m.")
        print("Final position error: " + ", ".join(f"{name}={value:.4f} m" for name, value in zip(replay.names, distances)))
        if not converged:
            raise RuntimeError("Replay timed out before reaching the final positions. "
                               "Inspect the paths/log or increase --settle-time; collision avoidance may cause deadlock.")
    finally:
        try:
            if r is not None:
                if not pose_pending:
                    r.get_poses()
                r.set_velocities(np.arange(n), np.zeros((2, n)))
                r.step()
                finish = getattr(r, "call_at_scripts_end", None) or getattr(r, "debug", None)
                if finish:
                    finish()
        finally:
            if log:
                log.close()


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--robots", nargs="+", help="CSV robot names, in Robotarium index order; default: all.")
    parser.add_argument("--episode", type=int, help="Required when the recording has multiple episodes.")
    parser.add_argument("--scale", type=float, help="Metres per Warp metre; default: shrink only as needed.")
    parser.add_argument("--margin", type=float, default=0.2, help="Path clearance from arena walls, metres.")
    parser.add_argument("--speed-factor", type=float, default=1.0, help="Requested playback rate; automatically reduced for speed limits.")
    parser.add_argument("--max-speed", type=float, default=0.15, help="Planar command limit in m/s (at most 0.2).")
    parser.add_argument("--gain", type=float, default=2.0, help="Position feedback gain, 1/s.")
    parser.add_argument("--settle-time", type=float, default=10.0, help="Extra seconds to reach final positions.")
    parser.add_argument("--tolerance", type=float, default=0.03, help="Final position tolerance in metres.")
    parser.add_argument("--log", type=Path, help="Write reference/actual positions and error to a new CSV.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without importing Robotarium.")
    parser.add_argument("--headless", action="store_true", help="Disable simulator visualization.")
    parser.add_argument("--fast", action="store_true", help="Run local simulation without real-time throttling.")
    args = parser.parse_args(argv)
    for name in ("gain", "settle_time", "tolerance"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error("--" + name.replace("_", "-") + " must be finite and positive")
    return args


def main(argv=None):
    args = parse_arguments(argv)
    try:
        replay = prepare_replay(args.csv_path, args.robots, args.episode, args.scale,
                                args.margin, args.speed_factor, args.max_speed)
        print("Robotarium IDs: " + ", ".join(f"{i}={name}" for i, name in enumerate(replay.names)))
        print(f"Spatial scale: {replay.scale:.6g}; effective speed factor: {replay.speed_factor:.6g}")
        print(f"Source duration: {replay.times[-1]:.3f} s; replay duration: {replay.duration:.3f} s (+ settling).")
        if not args.dry_run:
            run_replay(replay, args)
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
