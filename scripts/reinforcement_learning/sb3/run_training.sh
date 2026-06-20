#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting SAC training for Isaac-Velocity-Rough-Unitree-A1-v0..."

python scripts/reinforcement_learning/sb3/train_mod.py \
    --task Isaac-Velocity-Rough-Unitree-A1-v0 \
    --algorithm sac \
    --wandb_name "sbx_sac_a1_rough_sbxt+_action_1" \
    --headless \
    --action 1

echo "Training run completed or exited."
