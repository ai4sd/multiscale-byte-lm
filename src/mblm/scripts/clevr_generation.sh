#!/bin/bash

uv run src/mblm/scripts/clevr_generation.py \
    --model-dir ../../disk/inference/mmb_clevr \
    --clevr-dir ../../disk/data/clevr \
    --out-file misc/clevr_qa_pt1.jsonl \
    -m 8k_7mi_360m_1d_s_pt1_ft \
    -m 8k_7mi_360m_1d_s_pt1_nft \
    -m 8k_7mi_360m_1d_t_pt1_ft \
    -m 8k_7mi_360m_1d_t_pt1_nft \
    -n 300
# uv run src/mblm/scripts/clevr_generation.py \
#     --model-dir ../../disk/inference/mmb_clevr \
#     --clevr-dir ../../disk/data/clevr \
#     --out-file misc/clevr_qa_pt2.jsonl \
#     -m 8k_2mi_360m_1d_s_pt2_disc \
#     -m 8k_2mi_360m_1d_s_pt2_nodisc \
#     -m 8k_2mi_360m_1d_s_pt2_jpeg \
#     -m 8k_2mi_360m_1d_t_pt2_disc \
#     -m 8k_2mi_360m_1d_t_pt2_nodisc \
#     -m 8k_2mi_360m_1d_t_pt2_jpeg \
#     -n 300