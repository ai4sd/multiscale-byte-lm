#!/bin/bash

# we check the following model ids:
# 8k_30b_360m_1d_ssm
# 8k_30b_360m_1d_t_nopos
# 100k_200b_360m_2d_ss
# 100k_200b_360m_2d_st

# for a quick debugging run, you can execute:
# RUN_PREFIX=test CTX_LEN="8192 32768" NUM_SAMPLES=1 GEN_LEN=3 bash src/multiscale_mambabyte/scripts/pg19_generation.sh 8k_30b_360m_1d_ssm

# check if at least one model ID is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model_id1> [<model_id2> ...]"
    exit 1
fi

# read model IDs from command line arguments
MODEL_IDS=("$@")

# defaults that can be overwritten via env variables
RUN_PREFIX=${RUN_PREFIX:-""}
NUM_SAMPLES=${NUM_SAMPLES:-10} 
GEN_LEN=${GEN_LEN:-512}
# these need to be sorted in ascending order - as soon as we run oom for a ctx
# len, we assume the larger ctx lens are also oom. calculated with:
# np.concat(
#     [
#         np.array([8192]) * np.arange(1, 5, step=1) ** 2,
#         np.array([8192]) * np.arange(5, 12, step=2) ** 2,
#     ]
# )
#
CTX_LEN=${CTX_LEN:-"8192 32768 73728 131072 204800 401408 663552 991232"}

echo "--------------------------"
echo "Model ids: ${MODEL_IDS[@]}"
echo "Run prefix: ${RUN_PREFIX}"
echo "Num samples: ${NUM_SAMPLES}"
echo "Ctx lens: ${CTX_LEN}"
echo "Generation len samples: ${GEN_LEN}"
echo "--------------------------"

if [[ -n "$RUN_PREFIX" ]]; then
    # add underscore if run prefix is set
    RUN_PREFIX="${RUN_PREFIX}_"
fi

for model_id in "${MODEL_IDS[@]}"; do
    uv run \
        src/mblm/scripts/pg19_generation.sh \
        --model-dir ../../disk/inference/mmb_pg19 \
        --pg19-dir ../../disk/data/pg19 \
        --out-file misc/gen_pg19/${RUN_PREFIX}${model_id}.jsonl \
        --model-id $model_id \
        --max-num-samples ${NUM_SAMPLES} \
        --generation-len ${GEN_LEN} \
        --ctx-len ${CTX_LEN}

    if [ $? -ne 0 ]; then
        echo "$model_id ran out of memory, continuing with next model"
        continue
    fi
done
