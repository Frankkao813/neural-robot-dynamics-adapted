#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=unicycle_video
#SBATCH --output=logs/unicycle_video_%j.out
#SBATCH --error=logs/unicycle_video_%j.err

# Run mkdir -p logs BEFORE sbatch; Slurm opens the logs before this script runs.
set -euo pipefail
trap 'status=$?; echo "Job finished at $(date); exit code: $status"' EXIT

# Module versions and remote path are preserved from the supplied working job.
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
test -f utils/run_unicycle.py || { echo "Upload the updated unicycle code first." >&2; exit 1; }
mkdir -p outputs

echo "Job started at $(date) on $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "Working directory: $PWD"
command -v python
python --version
nvidia-smi
command -v xvfb-run
command -v ffmpeg

OUTPUT_PATH="$TARGET_DIR/outputs/unicycle_${SLURM_JOB_ID}.gif"

# The virtual display is needed for frame capture. Do not add --headless.
xvfb-run -a -s "-screen 0 1280x720x24" \
    python -u utils/run_unicycle.py \
    --device cuda:0 \
    --num-envs 1 \
    --speed 1.0 \
    --turn-rate 0.5 \
    --steps 600 \
    --export-video-path "$OUTPUT_PATH"

echo "Video saved to: $OUTPUT_PATH"
