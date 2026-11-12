#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=two_robots_video
#SBATCH --output=logs/two_robots_%j.out
#SBATCH --error=logs/two_robots_%j.err

# Create logs/ in the submission directory BEFORE running sbatch.
set -euo pipefail
trap 'status=$?; echo "Job finished at $(date); exit code: $status"' EXIT

module purge
module load GCCcore/11.3.0
module load GCC/11.3.0
module load OpenMPI/4.1.4
module load CUDA/11.8.0
module load Xvfb/21.1.3
module load FFmpeg/5.0.1

source ~/load_anaconda.sh
conda activate nerd

TARGET_DIR=/scratch/user/franklin.kao/neuralRobotDynamics-adapated/neural-robot-dynamics-adapted
test -d "$TARGET_DIR" || { echo "Missing directory: $TARGET_DIR" >&2; exit 1; }
cd "$TARGET_DIR"
mkdir -p outputs
echo "Job ${SLURM_JOB_ID} started at $(date) on $(hostname)"
python --version
nvidia-smi

# Defaults select the provided ANYmal forward-walking policy and config.
# Collision experiment: ANYmal walks along +X from (0, 0); the unicycle
# starts at (4, 0), faces -X (theta=pi), and drives straight toward ANYmal.
# The independent models have no robot-to-robot collision response.
# Stop at the first planar footprint contact, recording the final state.
xvfb-run -a -s "-screen 0 1280x720x24" \
    python -u utils/run_two_robots.py \
    --device cuda:0 \
    --steps 600 \
    --seed 0 \
    --stop-on-collision \
    --anymal-radius 0.6 \
    --unicycle-radius 0.25 \
    --speed 1.0 \
    --turn-rate 0 \
    --anymal-position 0 0 \
    --unicycle-pose 4 0 3.141592653589793 \
    --export-video-path "outputs/collision_${SLURM_JOB_ID}.gif" \
    --csv-path "outputs/collision_${SLURM_JOB_ID}.csv"
