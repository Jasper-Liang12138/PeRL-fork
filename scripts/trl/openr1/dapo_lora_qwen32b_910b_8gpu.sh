#!/bin/bash
# CTyunOS 22.06.2 训练脚本 - Qwen2.5-32B-Instruct + 8张华为昇腾910B
# 系统：CTyunOS 22.06.2@ascend-910b 64位
# 硬件：8*HuaweiAscend 910B
# 基础模型：Qwen/Qwen2.5-32B-Instruct

# ============================================================
# 【使用前必读】
# 1. 先下载模型（如果还未下载）：
#      python scripts/download_qwen25_32b_modelscope.py \
#          --save_dir /mnt/nvme0/models/Qwen2.5-32B-Instruct
# 2. 修改下方 MODEL_PATH 为你的实际模型路径
# 3. 32B 模型显存需求较大，建议调小 per_device_train_batch_size
# ============================================================

# ---------- 路径配置（按需修改）----------
MODEL_PATH="${MODEL_PATH:-/mnt/nvme0/models/Qwen2.5-32B-Instruct}"

unset WANDB_DISABLED
OUTPUT_DIR=/mnt/nvme0/output/grpo_lora_qwen25_32b_ctyunos_910b_$(date +%Y%m%d_%H%M%S)
LOG_FILE=${OUTPUT_DIR}/output.log

mkdir -p "${OUTPUT_DIR}"

# 设置NPU环境变量（CTyunOS专用）
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export COMBINED_ENABLE=1       # 启用混合精度优化
export TASK_QUEUE_ENABLE=1     # 启用任务队列优化

# Qwen2.5-32B 显存优化：启用梯度检查点相关优化
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

echo "[INFO] 使用模型路径: ${MODEL_PATH}"
echo "[INFO] 输出目录: ${OUTPUT_DIR}"
echo "[INFO] 开始训练..."

# 启动训练
ACCELERATE_LOG_LEVEL=info \
    accelerate launch \
    --main_process_port 29502 \
    --config_file scripts/trl/accelerate/ds_zero2_8gpu_npu.yaml \
    run.py train \
    --config.common.seed 42 \
    --config.common.debug false \
    --config.model.model_name_or_path "${MODEL_PATH}" \
    --config.model.dtype "bfloat16" \
    --config.model.use_npu true \
    --config.peft.use_peft true \
    --config.peft.type "lora" \
    --config.peft.task_type "CAUSAL_LM" \
    --config.peft.r 16 \
    --config.peft.lora_alpha 32 \
    --config.peft.lora_dropout 0.05 \
    --config.peft.total_step 1000 \
    --config.peft.target_modules '["q_proj","v_proj","k_proj","o_proj","up_proj","down_proj","gate_proj"]' \
    --config.training.learning_rate 1e-5 \
    --config.training.beta 0.0 \
    --config.training.output_dir "${OUTPUT_DIR}" \
    --config.training.run_name "${OUTPUT_DIR}" \
    --config.training.remove_unused_columns false \
    --config.training.gradient_accumulation_steps 16 \
    --config.training.num_train_epochs 1 \
    --config.training.max_completion_length 512 \
    --config.training.num_generations 2 \
    --config.training.warmup_ratio 0.0 \
    --config.training.max_prompt_length 512 \
    --config.training.logging_steps 1 \
    --config.training.per_device_train_batch_size 1 \
    --config.training.save_strategy "steps" \
    --config.training.save_steps 64 \
    --config.training.max_steps 10 \
    --config.training.use_vllm false \
    --config.training.top_entropy_quantile 1.0 \
    --config.training.epsilon_high 0.28 \
    --config.training.lr_scheduler_type "constant" \
    --config.training.lr_scheduler_kwargs.min_lr_rate 0.1 \
    --config.training.use_liger_kernel false \
    --config.training.loss_type "dapo" \
    --config.training.report_to '["wandb"]' \
    --config.logging.trackio_space_id "Open-Tinker/Open-Tinker" \
    --config.logging.trackio_project "grpo-lora-qwen25-32b-ctyunos-910b" \
    --config.logging.wandb_project "grpo-lora-qwen25-32b-ctyunos-910b" \
    --config.dataset.dataset_name_or_path "/root/PERL-FORK/ft-dataset/kicad_sft_dataset_590.json" \
    --config.dataset.example_numbers 1000000000 \
    2>&1 | tee "${LOG_FILE}"
