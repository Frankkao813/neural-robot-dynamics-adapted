# Neural Robot Dynamics (NeRD)

This repository contains the implementation for the paper [Neural Robot Dynamics](https://neural-robot-dynamics.github.io/) (CoRL 2025).

[***Neural Robot Dynamics***](https://neural-robot-dynamics.github.io/) <br/>
[Jie Xu](https://people.csail.mit.edu/jiex), [Eric Heiden](https://eric-heiden.com/), [Iretiayo Akinola](https://research.nvidia.com/person/iretiayo-akinola), [Dieter Fox](https://homes.cs.washington.edu/~fox/), [Miles Macklin](https://blog.mmacklin.com/about/), [Yashraj Narang](https://research.nvidia.com/person/yashraj-narang) <br/>
***CoRL 2025***

In this paper, we propose NeRD (Neural Robot Dynamics), learned robot-specific dynamics models for predicting future states for articulated rigid bodies under contact constraints. NeRD uniquely replaces the low-level dynamics and contact solvers in an analytical simulator and employs a robot-centric and spatially-invariant simulation state representation. In this repository, we demonstrate how to integrate NeRD as an interchangeable backend solver within [NVIDIA Warp](https://developer.nvidia.com/warp-python). 

> [!Note]
> See the work-in-progress branch ([nerd_newton_dev](https://github.com/NVlabs/neural-robot-dynamics/tree/nerd_newton_dev)) for ongoing integration of NeRD as a backend solver within [Newton](https://github.com/newton-physics/newton) physics.


<p align="center">
    <img src="figures/overview.png" alt="overview" width="800" />
</p>


## Installation
- The code has been tested on Ubuntu 20.04 with Python 3.8.20 and CUDA 12.9.
- `git clone git@github.com:NVlabs/neural-robot-dynamics.git`
- Create an Anaconda virtual environment (recommended)
  ```
  conda create -n nerd python=3.8.20
  conda activate nerd
  ```
- Install [PyTorch 2.2.2](https://pytorch.org/get-started/previous-versions/#linux-and-windows-14) (the tested version)
- Install other depandent packages:
  ```
  pip install -r requirements.txt
  ```
- Test the installation
  ```
  cd utils
  python visualize_env.py --env-name Cartpole --num-envs 1
  ```
  You are expected to see a passive Cartpole motion in a visualization UI.

## Test NeRD Models Without Training

We released pretrained NeRD models to quickly test out neural dynamics without training from scratch. We also released the RL policies that are trained in the NeRD-integrated simulator and are used in the Experiment section in the paper. Both pretrained NeRD models and RL policies are in the [`pretrained_models`](pretrained_models/) folder.

> [!Note]
> If you noticed the neural models under `pretrained_models` folders are not properly cloned, you may need to use `git lfs` for those files by running 
> ```
> git lfs install
> git lfs pull
> ```

We also provide evaluation scripts of the metrics used in the Experiment section of the paper.

### Passive Motion Evaluation
The script [`eval/eval_passive/eval_passive_motion.py`](eval/eval_passive/eval_passive_motion.py) conducts the long-horizon passive motion evaluation in Section 5.1. The script below evaluates and renders the 100-step passive motion for Cartpole.
```
cd eval/eval_passive
python eval_passive_motion.py --env-name Cartpole --model-path ../../pretrained_models/NeRD_models/Cartpole/model/nn/model.pt --env-mode neural --num-envs 1 --num-rollouts 5 --rollout-horizon 100 --render
```
To get the metrics averaged from 2048 trajectories, modify the script as follow:
```
python eval_passive_motion.py --env-name Cartpole --model-path ../../pretrained_models/NeRD_models/Cartpole/model/nn/model.pt --env-mode neural --num-envs 2048 --num-rollouts 2048 --rollout-horizon 100 --seed 500
```

To change the ground configuration for the double pendulum as done in Table 3 in Section C.2, you can manually specify the contact configuration ID in [`envs/warp_sim_envs/env_pendulum_with_contact.py`](envs/warp_sim_envs/env_pendulum_with_contact.py?ref_type=heads#L86), then run the script:
```
python eval_passive_motion.py --env-name PendulumWithContact --model-path ../../pretrained_models/NeRD_models/Pendulum/model/nn/model.pt --env-mode neural --num-envs 2048 --num-rollouts 2048 --rollout-horizon 100 --seed 500
```


### RL Policy Evaluation
You can test an individual RL policy using the [`run_rl.py`](eval/eval_rl/run_rl.py) script:
```
cd eval/eval_rl
python run_rl.py --rl-cfg ./cfg/Anymal/anymal_forward.yaml --playback ../../pretrained_models/RL_policies/Anymal/forward_walk/0/nn/AnymalPPO.pth --num-envs 1 --num-games 2 --env-mode [neural|ground-truth] [--render]
```

To render two ANYmal robots that walk along their individual facing
directions, run:

```bash
python run_rl.py --rl-cfg ./cfg/Anymal/anymal_two_directions.yaml --playback ../../pretrained_models/RL_policies/Anymal/forward_walk/0/nn/AnymalPPO.pth --render
```
<!-- 
To place two ANYmal robots in the same collision scene, facing one another,
run the ground-truth-only interaction example:

```bash
python run_rl.py --rl-cfg ./cfg/Anymal/anymal_two_robot_interaction.yaml --playback ../../pretrained_models/RL_policies/Anymal/forward_walk/0/nn/AnymalPPO.pth --render
```

This enables robot--robot collisions in Warp. NeRD mode does not yet support
these dynamic contacts; its learned contact representation is limited to
robot--ground contacts. -->

Warp uses Y as the up axis in this project, so `0` degrees faces world +X and
`-90` degrees faces world +Z (the second horizontal, or requested Y, direction).
The environment converts each robot's observations and walking reward into its
own heading frame, allowing the same forward-walking policy to control both.
Exact spawn locations can be configured with one Warp `(X, Y, Z)` position per
robot under `warp_env_cfg.env_positions`; Y is the vertical coordinate.
where `--env-mode` specifies to use NeRD dynamics or ground-truth analytical dynamics.

To inspect one complete transition for either robot, add
`--trace-step --trace-step-env 0` (or use environment `1`). The JSON trace
prints the current pose, heading-frame velocity target and tracking error,
policy action, target joint positions, applied joint torques, finite-difference
acceleration, and next state. The configured position and heading are spawn
conditions; the walking task tracks a velocity rather than a destination.
The same flags support ANT run, spin, and spin-tracking policies; ANT traces
show its direct action-to-torque mapping instead of ANYmal's joint PD targets.

To evaluate a batch of policies with different seeds in both ground-truth dynamics and NeRD dynamics (as done in Table 1 in the paper), you can run the batch evaluation script with the batch evaluation config file:
```
python batch_eval_policy.py --num-envs 2048 --num-games 2048 --eval-cfg ./eval_cfg/Anymal/eval_cfg_side.yaml
```

> [!Note]
> Some released NeRD models were trained using the data generated from an older version of Warp (i.e., Warp 1.5.1), thus the motion produced by NeRD models might not match the ground-truth dynamics generated with the latest version of Warp. While it does not affect the behavior of the NeRD models' prediction, to fully reproduce the numbers reported in the paper, please use Warp 1.5.1 for *Ant* and *Pendulum* and use Warp 1.8.0 for *ANYmal*, otherwise you can also train your own NeRD models by following the instructions shown in the next section.


### ANYmal and unicycle pass-through baseline

From the repository root, run the existing shared-scene runner with a head-on
encounter. ANYmal starts at ground-plane `(X, Z) = (0, 0)` and its default
forward-walking policy moves toward +X. The unicycle starts at `(4, 0)`, faces
-X (`theta = pi` radians), and drives straight at 1 m/s:

```bash
python utils/run_two_robots.py --device cuda:0 --steps 600 --seed 0 --speed 1.0 --turn-rate 0 --anymal-position 0 0 --unicycle-pose 4 0 3.141592653589793 --export-video-path outputs/pass_through.gif --csv-path outputs/pass_through.csv
```

This uses the existing independent dynamics: there is no robot-to-robot
collision detection, response, or avoidance. The expected result is that the
robots approach, overlap in the ground-plane projection, and continue past
each other. The low unicycle may appear to pass underneath ANYmal's torso;
planar overlap does not necessarily imply intersecting 3D body geometry.
ANYmal's actual lateral drift and encounter time depend on the policy rollout.
The run ends after 600 control steps (10 seconds with the default config), or
earlier if ANYmal terminates. Do not use `--reset-on-done` for this experiment,
so resets cannot be mistaken for uninterrupted pass-through motion.

For a trajectory-only run, replace `--export-video-path outputs/pass_through.gif`
with `--headless`; `--device cpu` is also supported. The CSV records both robots
at matching times, including their initial poses. Check that their X ordering
reverses while their Z positions remain close, then inspect the GIF around
their closest approach. Choose fresh output filenames for repeat runs.

### Footprint collision detection and termination

To repeat the head-on encounter and stop at contact, enable `--stop-on-collision`:

```bash
python utils/run_two_robots.py --device cuda:0 --steps 600 --seed 0 --speed 1.0 --turn-rate 0 --anymal-position 0 0 --unicycle-pose 4 0 3.141592653589793 --stop-on-collision --anymal-radius 0.6 --unicycle-radius 0.25 --export-video-path outputs/collision.gif --csv-path outputs/collision.csv
```

The detector compares circular footprints centered on each robot's body in the
X-Z plane: contact occurs when center distance is at most the sum of radii.
The default radii (ANYmal: 0.60 m, unicycle: 0.25 m) are configurable approximations,
not exact 3D body/leg geometry or an avoidance safety margin. Height is ignored.
Checks run at reset and after every paired simulation frame, including frames
inside repeated policy actions. Contact is sampled at frame boundaries (default
1/60 s); there is no swept collision test or exact time-of-impact interpolation.

On contact, both robots stop advancing, the final state is logged and rendered,
the GIF is encoded, and the run exits without resetting, even with
`--reset-on-done`. Spawn overlap terminates at time zero and exports one frame.
There is no bounce or contact-force response. The CSV retains its original
columns and appends `collision` (0/1), `planar_distance` and `collision_distance`
(sum of radii), with distances in metres. The console also reports the contact
time and both X-Z positions. Without `--stop-on-collision`, the original
pass-through behavior remains available; `collision` then stays zero because
detection is disabled.

On the cluster, `two_robots_video.sbatch` now enables collision detection with
these same encounter settings and exports `outputs/collision_<job-id>.gif`
and the matching CSV. Remove `--stop-on-collision` and choose pass-through
output names to reproduce the baseline. Submit
from the directory containing that script:

```bash
mkdir -p logs
sbatch two_robots_video.sbatch
```

## Train NeRD Models

Training a NeRD model involves two steps: (1) generating datasets and (2) training a NeRD model using the generated datasets. We provided data generation and training scripts, as well as training configuration files for the examples in the paper. Below shows example training commands for *Cartpole* and *Ant*, and you can try other examples using our provided scripts.

### Generate Dataset

We need to generate the training and validation datasets for NeRD training. Each dataset consists of a set of random motion trajectories. We generate datasets with 1M transitions in the examples here.

#### Cartpole
```
cd generate
python generate_dataset_contact_free.py --env-name Cartpole --num-transitions 1000000 --dataset-name trajectory_len-100_1M_train.hdf5 --trajectory-length 100 --num-envs 2048 --seed 0
python generate_dataset_contact_free.py --env-name Cartpole --num-transitions 1000000 --dataset-name trajectory_len-100_valid.hdf5 --trajectory-length 100 --num-envs 2048 --seed 10
```

#### Ant
```
cd generate
python generate_dataset_ant.py --env-name Ant --num-transitions 1000000 --dataset-name trajectory_len-100_1M_train.hdf5 --trajectory-length 100 --num-envs 2048 --seed 0
python generate_dataset_ant.py --env-name Ant --num-transitions 1000000 --dataset-name trajectory_len-100_valid.hdf5 --trajectory-length 100 --num-envs 2048 --seed 10
```

### Training

#### Cartpole
```
cd train
python train.py --cfg ./cfg/Cartpole/transformer.yaml --logdir ../../data/trained_models/Cartpole/
```

#### Ant
```
cd train
python train.py --cfg ./cfg/Ant/transformer.yaml --logdir ../../data/trained_models/Ant/
```

## Train RL Policies in a NeRD-Integrated Simulator
We use [rl-games](https://github.com/Denys88/rl_games) to train RL policies within a NeRD-integrated simulator. Thanks to the seamless design of the NeRD integrator within Warp simulator, we can turn on the using of NeRD dynamics models by simply call the [`set_env_mode`](envs/neural_environment.py#L314) function in `NeuralEnvironment` class to switch between using the NeRD dynamics and using the analytical dynamics. Below shows an example to train the RL policy for the *ANYmal* forward walking task.
```
cd eval/eval_rl
python run_rl.py --rl-cfg ./cfg/Anymal/anymal_forward.yaml --env-mode neural --nerd-model-path ../../pretrained_models/NeRD_models/Anymal/model/nn/model.pt 
```

## Implementation Details
We implemented NeRD as a class of [Integrator](integrators/integrator_neural.py?ref_type=heads#L127) in Warp Sim, which shares the same [`simulate`](integrators/integrator_neural.py?ref_type=heads#L329) function interfaces as other default integrators such as Featherstone and XPBD integrators so that the specific integrator type is transparent to the environments implemented on top of it. The [`AbstractContactEnvironment`](envs/abstract_contact_environment.py#L190) is a wrapper around any existing robotic environment that pre-extract the contact pair lists for NeRD. [`NeuralEnvironment`](https://github.com/NVlabs/neural-robot-dynamics/blob/main/envs/neural_environment.py#L41) is a wrapper around `AbstractContactEnvironment` to construct the NeRD integrator and provide a unified view of different environments to top-level applications such as RL training and NeRD evaluators.

## Citation

If you find our paper or code useful, please consider citing:
```
@inproceedings{
  xu2025neural,
  title={Neural Robot Dynamics},
  author={Jie Xu and Eric Heiden and Iretiayo Akinola and Dieter Fox and Miles Macklin and Yashraj Narang},
  booktitle={9th Annual Conference on Robot Learning},
  year={2025}
}
```
