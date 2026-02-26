import subprocess

# seeds you want to test
seeds = [0, 1, 2, 3, 4]

for seed in seeds:
    print(f"\n=== Starting training with seed {seed} ===\n")

    cmd = [
        "python",
        "train.py",
        "--task=Isaac-Velocity-Flat-Unitree-A1-v0",
        "--headless",
        f"--seed={seed}"
    ]

    # This BLOCKS until training finishes
    subprocess.run(cmd, check=True)

    print(f"\n=== Finished seed {seed} ===\n")

print("\nAll trainings completed.")