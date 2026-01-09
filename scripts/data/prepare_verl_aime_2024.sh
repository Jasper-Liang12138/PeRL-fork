#!/bin/bash
set -euox pipefail

python3 data/aime2024.py\
    --local_dataset_path "/mnt/llm-train/users/explore-train/qingyu/.cache/aime_2024" \
    --local_dir "data/aime_2024-verl"