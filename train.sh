#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

python -m rl.train_policy "$@"
