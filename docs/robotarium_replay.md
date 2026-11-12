# Replay a Warp recording in Robotarium

`utils/replay_robotarium.py` tracks recorded planar paths with wheeled robots.
It imports no Warp, PyTorch, policy, or NeRD modules. All commands below run
from the `neural-robot-dynamics` directory.

## 1. Record in Warp

Use the existing Warp environment and recording script:

```sh
python utils/run_two_robots.py --device cpu --headless --steps 600 --csv-path outputs/two_robots.csv
```

Use `--device cuda:0` instead if desired. The recorder refuses to overwrite an
existing output. It may stop before the requested number of steps if ANYmal
terminates. No video or renderer is needed to record positions.

The replay script directly accepts this recorder's CSV columns:
`time, episode, robot, x, y, z, theta` (other columns are ignored).
Warp's ground plane `(x, z)` becomes Robotarium's `(x, y)`; Warp's `y` is height.
The minimal supported format is `time,robot_id,x,z`, with optional `theta` and
`episode`. Without a robot column the file describes a single robot `robot0`.
Time is in seconds, position in metres, and heading in radians from +X toward
+Z. Initial heading is inferred from the first moving segment when omitted.

## 2. Install the official simulator in a separate environment

The current official simulator requires Python 3.10 or newer. Keep it separate
from the project's older Warp environment. For example, in PowerShell:

```powershell
py -3.11 -m venv .venv-robotarium
.\.venv-robotarium\Scripts\Activate.ps1
python -m pip install "https://github.com/robotarium/robotarium_python_simulator/archive/refs/heads/master.zip"
```

On Linux/macOS use `python3 -m venv .venv-robotarium` and
`source .venv-robotarium/bin/activate`. An existing environment with the official
simulator installed also works. Do not install the old, differently named
`robotarium-python-simulator` repository.

## 3. Check and replay

```sh
# Check data, spatial scale, duration, and path separation; only NumPy needed.
python utils/replay_robotarium.py outputs/two_robots.csv --dry-run

# Replay both recorded robots in the visible local simulator.
python utils/replay_robotarium.py outputs/two_robots.csv

# Replay ANYmal's planar path using one wheeled robot and log tracking errors.
python utils/replay_robotarium.py outputs/two_robots.csv --robots anymal --log outputs/anymal_replay.csv

# Choose an episode if the recording contains resets.
python utils/replay_robotarium.py outputs/two_robots.csv --episode 0

# Request slower playback, or run a local check without a figure or wall-clock wait.
python utils/replay_robotarium.py outputs/two_robots.csv --speed-factor 0.1
python utils/replay_robotarium.py outputs/two_robots.csv --headless --fast
```

CSV robot names map to Robotarium IDs in first-appearance order, or in the order
supplied to `--robots anymal unicycle`. The script prints the mapping.

All selected paths share one translation and uniform spatial scale, preserving
relative geometry. The default centers their bounding box and shrinks it only
when needed to fit the 3.2 by 2.0 m arena with a 0.2 m wall margin. `--scale 0.5`
requests an explicit scale and still centers the paths; an out-of-bounds scale
is rejected. Robot bodies do not shrink with the trajectories.

The requested `--speed-factor` means source seconds per replay second. It is
automatically reduced to reserve half of `--max-speed` (default 0.15 m/s) for
feedback corrections. For example, 0.1 replays 10 recorded seconds in 100
seconds, before final settling. All robots share this clock. The script
interpolates timestamps at the simulator timestep; it does not assume Warp
and Robotarium have the same timestep. `--fast` changes wall-clock execution
only, not trajectory timing.

The simulator receives the first recorded poses as `initial_conditions`.
Current releases drive the robots to those poses during initialization;
older releases may place them there directly. During replay a planar velocity
feedforward plus position feedback controller generates unicycle commands,
then a boundary/collision barrier filters them. A common scaling of commands
limits wheel speeds. The script tracks **positions**; recorded headings are
used only for initialization. It does not reproduce gait or guarantee heading
tracking. It calls the simulator's final diagnostics and commands zero velocity
at the end, on timeout, and on controller exceptions after construction.

At the last timestamp the reference stops, and the controller gets up to
`--settle-time 10` additional seconds to reach `--tolerance 0.03` metres.
Failure to reach the endpoint is reported as an error. Optional `--log` creates
a new CSV of reference and actual positions, headings, and position error;
existing logs are not overwritten. RMS, peak, and final errors are printed.

Invalid/nonfinite data, duplicate or reversed timestamps, missing robot
endpoints, ambiguous episodes, and overlapping starts are rejected. Close
approaches anywhere along the interpolated reference produce a warning.
Collision avoidance can change the recorded motion or create a deadlock when
paths cross. The existing `outputs/pass_through_check.csv` intentionally has a
close encounter and is therefore unsuitable for expecting exact two-robot replay.

This is a local simulator entry point. Physical-testbed submission also needs
the recording packaged with the experiment and the current Robotarium submission
requirements checked. The script has not been verified on physical robots.

## Checks and API references

```sh
python -m unittest discover -s tests -p test_robotarium_replay.py -v
```

- [Official simulator and installation](https://github.com/robotarium/robotarium_python_simulator)
- [Robotarium loop and initialization](https://github.com/robotarium/robotarium_python_simulator/blob/master/rps/robotarium.py)
- [Boundary and collision certificates](https://github.com/robotarium/robotarium_python_simulator/blob/master/rps/utilities/barrier_certificates.py)
- [Velocity transformations](https://github.com/robotarium/robotarium_python_simulator/blob/master/rps/utilities/transformations.py)

The script accepts both `create_uni_barrier_certificate_with_boundary` and the
older `create_unicycle_barrier_certificate_with_boundary` factory name, and
both `debug()` and `call_at_scripts_end()` diagnostics. Older releases are not
otherwise guaranteed compatible.
