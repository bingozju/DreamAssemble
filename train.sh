#!/bin/bash
# DreamAssemble training entry script.
#
# Usage:
#   ./train.sh -w <workspace> -c <config.yaml> [-g <gpu_id>] [-s <seed>] [-d]
#
# Options:
#   -w  workspace name (results are saved under results/<workspace>)
#   -c  path to a yaml config under trainfiles/
#   -g  CUDA visible devices, e.g. "0" or "0,1" (default: 0)
#   -s  random seed (default: 42)
#   -d  enable DMTet refinement stage (mesh fine-tuning)
#
# Examples:
#   # NeRF coarse training
#   ./train.sh -w boy_girl_tiger -c trainfiles/boy_girl_tiger.yaml -g 0
#
#   # DMTet refinement (requires a finished coarse stage)
#   ./train.sh -w boy_girl_tiger -c trainfiles/boy_girl_tiger.yaml -g 0 -d

set -e

WORKSPACE=""
CONFIG=""
GPU_IDS="0"
SEED=42
USE_DMTET=0

while getopts "w:c:g:s:d" opt; do
    case $opt in
        w) WORKSPACE="$OPTARG" ;;
        c) CONFIG="$OPTARG" ;;
        g) GPU_IDS="$OPTARG" ;;
        s) SEED="$OPTARG" ;;
        d) USE_DMTET=1 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
        :)  echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
    esac
done

if [ -z "$WORKSPACE" ] || [ -z "$CONFIG" ]; then
    echo "Error: WORKSPACE (-w) and CONFIG (-c) are required." >&2
    echo "Usage: $0 -w <workspace> -c <config.yaml> [-g <gpu_id>] [-s <seed>] [-d]" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

if [ "$USE_DMTET" -eq 0 ]; then
    # Stage 1: coarse NeRF training with multi-density field
    python main.py \
        --workspace "$WORKSPACE" \
        --config "$CONFIG" \
        --seed "$SEED" \
        --iters 10000 \
        --train_all
else
    # Stage 2.1: DMTet geometry warm-up
    python main.py \
        --workspace "$WORKSPACE" \
        --config "$CONFIG" \
        --seed "$SEED" \
        --iters 5000 \
        --dmtet \
        --lambda_mesh_normal 5000 \
        --train_all

    # Stage 2.2: DMTet refinement
    python main.py \
        --workspace "$WORKSPACE" \
        --config "$CONFIG" \
        --seed "$SEED" \
        --iters 10000 \
        --dmtet
fi
