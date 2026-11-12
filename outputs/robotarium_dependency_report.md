# ANYmal projection: Robotarium dependency audit

Audited 2026-11-02. Scope: the current `utils/run_two_robots.py` loading the
pretrained ANYmal forward policy, advancing ground-truth dynamics on CPU,
checking circular footprints, and reading projected positions. No OpenGL
renderer, video export, policy training, or learned NeRD dynamics was used.

## Verification

A fresh process in the existing `nerd` environment successfully loaded the
robot and policy, advanced three frames, and generated position records.
Imported modules and installed distribution metadata were recorded in
`robotarium_dependency_audit.json`. Source inspection distinguished model
loading from rendering and declared installation dependencies from playback.

This verifies the existing Windows/Python 3.8.20 environment, with PyTorch
2.2.2, NumPy 1.24.3, and Gym 0.23.1. It does not verify Robotarium's Linux
environment, PyTorch 2.10.0, NumPy 2.2.6, or Gym 0.26.2. An import trace also
loads optional environment-specific packages; it is not a minimal lockfile.

## Additional package request

All eight entries below are absent from the published Robotarium list checked
on the audit date. Exact versions are those installed locally, not a promise
of compatibility with every Robotarium Python/core-library version.

| Package | Local version | Why include it |
| --- | --- | --- |
| warp-lang | 1.8.0 | Dynamics, contacts, generalized states; code uses legacy `warp.sim`. |
| rl-games | 1.6.5 | PPO actor construction, normalization, inference wrapper. |
| h5py | 3.11.0 | Unconditional import in `envs/neural_environment.py`, even without HDF5 logging. |
| tqdm | 4.67.1 | Environment construction imports and uses `trange`. |
| trimesh | 4.7.1 | Warp's URDF loader loads ANYmal meshes even with rendering disabled. |
| pycollada | 0.9.2 | Reads the `.dae` meshes through trimesh; imported as `collada`. |
| gymnasium[classic-control] | 0.29.1 | RL Games declares this dependency; its compatibility layer uses Gymnasium on Python >=3.9. Gym and Gymnasium are separate packages. |
| psutil | 5.9.8 | RL Games declares `>=5.9,<6`; also observed in the headless process. |

Use `robotarium_additional_requirements.txt` with normal dependency resolution.
The Robotarium packages already cover torch, numpy, PyYAML, gym, scipy,
opencv-python, matplotlib, pillow, networkx, python-dateutil and several other
support dependencies. Do not replace their entire library list with the
repository's older requirements file.

## Dependencies brought in by installation

`rl-games==1.6.5` also declares the following non-optional dependencies absent
from Robotarium's published list:

- `setproctitle>=1.2.2,<2.0.0`
- `tensorboard>=2.8.0,<3.0.0`
- `tensorboardX>=2.5,<3.0`
- `watchdog>=2.1.9`

The Gymnasium classic-control extra brings in `pygame>=2.1.3` and base
Gymnasium brings in `farama-notifications>=0.0.1`. Those are also absent from
the published list. TensorBoard and other packages have further transitive
dependencies, which the package resolver must install for the target Python
version. Do not treat the eight explicit pins as the entire installed closure.

These training/logging/utility packages were not needed by the observed
playback calls, but are part of a normal supported installation of RL Games.
Removing them would require a deliberately reduced inference implementation,
not simply an incomplete installation.

## Not required for this headless projection path

- `pyglet`, an OpenGL display, Xvfb, and FFmpeg: used for the existing 3D/video
  route, not advancing the policy-controlled robot and extracting its position.
- `wandb`, `ipdb`, Ray, Isaac Gym, MuJoCo, and Atari extras: not used by this
  playback path; do not request them based on the training repository alone.
- Windows/Conda packages and opportunistic imports such as `colorama`,
  `mkl-service`, `gmpy2`, and setuptools' vendored helpers should not be copied
  into a Linux requirements list solely because they appear in the trace.

## Before deployment

1. Confirm Robotarium's Python version and install wheels compatible with it.
2. Verify Warp 1.8.0 specifically; a newer version is not an automatic substitute.
3. Confirm writable kernel-cache space and permission to compile/load Warp's
   CPU kernels. GPU access is not needed for the tested path; CUDA package
   installation alone does not establish GPU access.
4. Test checkpoint loading under PyTorch 2.10. The current code leaves
   `weights_only` unspecified; PyTorch changed its default in version 2.6.
   A tensor-only inference checkpoint is one potential adaptation if needed.
5. Validate numerical behavior with NumPy 2.2.6 and the target Gym versions.
6. Include the policy `.pth`, selected YAML, ANYmal URDF/mesh assets and
   required local Python modules. They are project assets, not pip packages.
7. Run the packaged headless experiment in isolation and measure runtime.
   Package availability does not establish server time-limit suitability.

The current loader parses visual meshes even when no renderer is created;
showing only a dot does not yet eliminate trimesh/pycollada. Likewise, h5py
and tqdm could be removed by a targeted deployment refactor, but are required
by the code as currently imported.

## Sources

- Robotarium supported libraries: https://raw.githubusercontent.com/robotarium/robotarium_python_simulator/master/libraries.txt
- RL Games 1.6.5 published package metadata: https://pypi.org/pypi/rl-games/1.6.5/json
- PyTorch 2.10 checkpoint-loading behavior: https://docs.pytorch.org/docs/2.10/notes/serialization.html#torch-load-with-weights-only-true
- Local repository imports, installed package source/metadata and successful
  headless rollout; raw results are in the accompanying audit JSON.
