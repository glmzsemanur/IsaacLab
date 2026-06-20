#!/bin/bash

# Use this script to run multiple training sessions for different action configurations in parallel.


# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting SAC training for Isaac-Velocity-Flat-Unitree-A1-v0 with action 2..."
python scripts/reinforcement_learning/sb3/train_mod.py \
    --task Isaac-Velocity-Flat-Unitree-A1-v0 \
    --algorithm sac \
    --wandb_name "sbx_sac_a1_flat_sbxt+_action_3" \
    --headless \
    --action 3

echo "Starting SAC training for Isaac-Velocity-Flat-Unitree-A1-v0 with action 2..."
python scripts/reinforcement_learning/sb3/train_mod.py \
    --task Isaac-Velocity-Flat-Unitree-A1-v0 \
    --algorithm sac \
    --wandb_name "sbx_sac_a1_flat_sbxt+_action_4" \
    --headless \
    --action 4

echo "Starting SAC training for Isaac-Velocity-Flat-Unitree-A1-v0 with action 2..."
python scripts/reinforcement_learning/sb3/train_mod.py \
    --task Isaac-Velocity-Flat-Unitree-A1-v0 \
    --algorithm sac \
    --wandb_name "sbx_sac_a1_flat_sbxt+_action_5" \
    --headless \
    --action 5




echo "Training runs completed or exited."
