# 华为昇腾NPU训练快速开始

## 一、环境准备

### 1. 检查NPU环境
```bash
# 运行环境检查脚本
bash scripts/check_npu_env.sh
```

### 2. 安装依赖
```bash
# 使用pip安装
pip install -r requirements_npu.txt

# 或使用uv（更快）
uv pip install -r requirements_npu.txt
```

## 二、开始训练

### 方式1：直接运行（推荐）
```bash
bash scripts/trl/openr1/dapo_lora_npu.sh
```

### 方式2：自定义参数
```bash
# 修改脚本中的参数，例如：
# - 模型路径：--config.model.model_name_or_path
# - LoRA rank：--config.peft.r
# - Batch size：--config.training.per_device_train_batch_size
# - 学习率：--config.training.learning_rate
```

## 三、监控训练

### 查看日志
```bash
# 实时查看训练日志
tail -f outputs/grpo_lora_qwen2_5_7b_npu_*/output.log
```

### 查看NPU使用情况
```bash
# 实时监控NPU
watch -n 1 npu-smi info
```

### Wandb监控
训练会自动上传到Wandb，可以在网页端查看详细指标。

## 四、常见问题

### 显存不足（OOM）
修改 `dapo_lora_npu.sh` 中的参数：
```bash
--config.training.per_device_train_batch_size 1  # 减小batch size
--config.training.gradient_accumulation_steps 16  # 增加梯度累积
--config.training.max_completion_length 4096     # 减小序列长度
```

### torch_npu导入失败
```bash
# 重新安装torch_npu
pip uninstall torch-npu
pip install torch-npu
```

### 训练速度慢
- 确认使用了4张NPU卡
- 检查日志中的 `num_processes: 4`
- 使用 `npu-smi info` 确认4张卡都在工作

## 五、关键配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| model_name_or_path | Qwen/Qwen2.5-7B-Instruct | 基础模型 |
| use_npu | true | 启用NPU支持 |
| use_vllm | false | NPU不支持vLLM |
| r | 16 | LoRA rank |
| lora_alpha | 32 | LoRA alpha |
| per_device_train_batch_size | 2 | 每卡batch size |
| gradient_accumulation_steps | 8 | 梯度累积步数 |
| max_completion_length | 8192 | 最大生成长度 |
| learning_rate | 1e-5 | 学习率 |
| max_steps | 1024 | 最大训练步数 |

## 六、预期结果

- **训练时间**：1024步约需10-20小时
- **显存占用**：每张卡约20-30GB
- **训练速度**：约1-2 steps/s

## 七、完整文档

详细文档请参考：`doc/NPU_TRAINING_GUIDE.md`
