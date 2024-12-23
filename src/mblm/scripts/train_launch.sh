#!/bin/bash

if [ -z "$1" ]; then
  echo "Error: No config path provided"
  echo "Usage: $0 <train-config.yml>"
  exit 1
fi

echo "Launching MBLM training from config file:"
echo $1

export JOB_ID=$(date +%s)
export DISPLAY_PROGRESS=1

# add the --standalone flag for single-node
OMP_NUM_THREADS=1 uv run torchrun \
    --standalone \
    --nproc_per_node=gpu \
    src/mblm/scripts/train_mblm.py \
    -c $1