#!/bin/bash

uv run src/mblm/scripts/clevr_generation.py \
    --model-dir ../../disk/inference/mmb_clevr \
    --clevr-dir ../../disk/data/clevr \
    --out-file misc/clevr_qa_pt3.jsonl \
    -m 8k_2mi_1b_2d_s_pt3_disc \
    -m 8k_2mi_1b_2d_s_pt3_jpeg \
    -m 8k_2mi_1b_2d_s_pt3_nodisc \
    -n 300