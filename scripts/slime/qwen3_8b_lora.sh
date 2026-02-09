#!/bin/bash

# usage: bash examples/on_policy_distillation/run-qwen3-8B-opd.sh

set -ex

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

FIXED_PROJECT_NAME="slime-opd-40b-experiments"
PROJECT_DIR=/mnt/llm-train/users/explore-train/qingyu/slime # /root/slime is mounted from the docker env
SCRIPT_DIR=${PROJECT_DIR}/scripts # where is the scripts
LOCAL_IP=$(hostname -I | awk '{print $1}') # get master node local ip for submitting
TIMESTAMP=$(date +%Y%m%d_%H%M%S) # timestamp for naming
SAVE_DIR=/mnt/llm-train/users/explore-train/qingyu/ckpt/${TIMESTAMP}_Qwen3-8B-LoRA
LOG_DIR=$SAVE_DIR/output.log # where to save log
mkdir -p ${SAVE_DIR} # create save dir

WANDB_HOST="http://11.71.1.218:8082"
export WANDB_API_KEY=local-b0d90ad40bfaa2dd58fa4525f18c82ccb8aca2c6 
export WANDB_ENTITY=automl 
export WANDB_PROJECT=${FIXED_PROJECT_NAME} 
export WANDB_NAME="${TIMESTAMP}_Qwen3-8B-LoRA"

wandb login --relogin --host=http://11.71.1.218:8082 ${WANDB_API_KEY}
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"RAY_ENABLE_OPENTELEMETRY\": \"0\",
    \"RAY_DISABLE_METRICS_COLLECTION\": \"1\",
    \"RAY_USAGE_STATS_DISABLED\": \"1\",
    \"GRPC_ENABLE_FORK_SUPPORT\": \"0\",
    \"WANDB_MODE\": \"online\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
    \"WANDB_BASE_URL\": \"${WANDB_HOST}\",
    \"WANDB_PROJECT\": \"${FIXED_PROJECT_NAME}\", 
    \"WANDB_NAME\": \"${RUN_NAME}\",
    \"WANDB_START_METHOD\": \"thread\",
    \"WANDB_INIT_TIMEOUT\": \"300\"
  }
}"


CKPT_ARGS=(
   --hf-checkpoint /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-8B-ODA-Math-460k
   --ref-load /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-8B-ODA-Math-460k
   --load /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-8B-ODA-Math-460k
   --save ${SAVE_DIR}
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template

   --rollout-shuffle
   --num-rollout 500

   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 16384
   --rollout-temperature 1

   --over-sampling-batch-size 64
   --global-batch-size 32
   --balance-data

   # --partial-rollout # aka pipelineRL
   # --rollout-function-path examples.fully_async.fully_async_rollout.generate_rollout_fully_async
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   # --mask-offpolicy-in-partial-rollout # only use on policy data for training

)

RM_ARGS=(
   --rm-type deepscaler
)

EVAL_ARGS=(
   # --eval-interval 20
   # --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   # --n-samples-per-eval-prompt 16
   # --eval-max-response-len 16384
   # --eval-top-p 1
)

PERF_ARGS=(

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20000

   --train-backend fsdp
   --update-weight-buffer-size 536870912
   --gradient-checkpointing
   --attn-implementation flash_attention_3
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'

   --lora-rank 32
   --lora-alpha 16
   --target-modules all-linear

   --actor-num-nodes 1
   --actor-num-gpus-per-node 2
   --rollout-num-gpus 6

)

GRPO_ARGS=(
   --advantage-estimator gspo
   --use-kl-loss
   --kl-loss-coef 0.00 # no kl loss
   --kl-loss-type low_var_kl
   --entropy-coef 0.00 # no entropy loss
   --eps-clip 3e-4
   --eps-clip-high 4e-4 # clip ratio higher
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style linear
   --lr-warmup-fraction 0.01
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project ${FIXED_PROJECT_NAME}
   --wandb-group language-rl
   --wandb-key "local-b0d90ad40bfaa2dd58fa4525f18c82ccb8aca2c6" 
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.9
)


MISC_ARGS=(
   # --attention-dropout 0.0
   # --hidden-dropout 0.0
   # --accumulate-allreduce-grads-in-fp32
   # --attention-softmax-in-fp32
   # --attention-backend flash
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${PROJECT_DIR}/train.py \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]} 2>&1 | tee ${LOG_DIR}

# ####clear after training
# pkill -9 sglang
# sleep 3
# ray stop --force
# pkill -9 ray
# pkill -9 python
# sleep 3
# pkill -9 ray
# pkill -9 python


