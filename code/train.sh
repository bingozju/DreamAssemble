#!/bin/bash

WORKSPACE=""
CONFIG=""
GPU_IDS="0"  # default GPU 0
DMTET=0 # default not use DMTet

while getopts "w:c:g:d" opt; do
    case $opt in
        w) WORKSPACE="$OPTARG" ;;
        c) CONFIG="$OPTARG" ;;
        g) GPU_IDS="$OPTARG" ;;
        d) DMTET=1
        \?) echo "Invalid options：-$OPTARG" >&2; exit 1 ;;
        :) echo "options -$OPTARG need" >&2; exit 1 ;;
    esac
done

if [ -z "$WORKSPACE" ] || [ -z "$CONFIG" ]; then
    echo "E：WORKSPACE和CONFIG must be given!" >&2
    echo "Usage：$0 -w <WORKSPACE> -c <CONFIG.yaml> [-g GPU_ID(s)] [-d]" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

# Training
python main.py --workspace "$WORKSPACE" --config="$CONFIG" --iters=10000

# DMTet
if [ "$DMTET" -eq 1 ]; then
    python main.py --workspace "$WORKSPACE" --config="$CONFIG" --iters=5000 --dmtet --lambda_mesh_normal 5000
    python main.py --workspace "$WORKSPACE" --config="$CONFIG" --iters=10000 --dmtet
fi