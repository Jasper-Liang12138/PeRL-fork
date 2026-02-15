# 华为昇腾NPU + vLLM 训练评估指南

本指南介绍如何在华为昇腾NPU上进行GRPO训练，并使用vLLM加速训练后的模型评估和推理。

## 重要说明

**当前配置策略**：
- ✅ **训练阶段**：使用标准 `model.generate()`（不使用vLLM）
- ✅ **评估/推理阶段**：使用 vLLM 加速（提速2-5倍）

**原因**：TRL 0.14.0（稳定版本）在训练时不支持vLLM。虽然TRL 0.18.0+支持vLLM，但仅支持vLLM 0.10-0.12版本，与NPU所需的vllm-ascend 0.13.0（需要vLLM 0.13.0+）不兼容。

## 一、环境准备

### 1.1 检查NPU环境

```bash
# 运行环境检查脚本
bash scripts/check_npu_env.sh

# 手动检查NPU状态
npu-smi info
```

### 1.2 安装依赖

```bash
# 安装基础依赖
pip install -r requirements_vllm_npu.txt

# 或使用uv（更快）
uv pip install -r requirements_vllm_npu.txt

# 安装vLLM支持（如果未安装）
pip install vllm==0.15.1
pip install vllm-ascend==0.13.0
```

### 1.3 验证vLLM + NPU环境

```bash
# 检查vLLM是否正确安装
python -c "from vllm import LLM; print('✅ vLLM installed')"

# 检查NPU是否被vLLM识别
python -c "from vllm.platforms import current_platform; print('Platform:', current_platform)"

# 检查torch-npu
python -c "import torch; import torch_npu; print('✅ NPU available:', torch.npu.is_available())"
```

预期输出：
- ✅ vLLM installed
- Platform: <NPUPlatform object>
- ✅ NPU available: True

## 二、训练模式（不使用vLLM）

### 2.1 为什么训练时不用vLLM

由于版本兼容性限制，当前训练配置不使用vLLM：

| 组件 | 版本要求 | 冲突说明 |
|------|---------|---------|
| TRL 0.18.0+ | 支持vLLM 0.10-0.12 | 与vllm-ascend不兼容 |
| vllm-ascend 0.13.0 | 需要vLLM 0.13.0+ | 与TRL支持的版本不兼容 |
| TRL 0.14.0 | 不支持vLLM | 稳定，用于训练 |

### 2.2 训练配置

当前训练脚本已配置为使用标准生成：

```bash
--config.training.use_vllm false  # 禁用vLLM
--config.training.num_generations 2  # GRPO需要至少2个生成
--config.training.per_device_train_batch_size 1
--config.training.gradient_accumulation_steps 16
```

## 三、开始训练（标准模式）

### 3.1 运行训练

```bash
# 直接运行训练脚本
bash scripts/trl/openr1/dapo_lora_npu.sh
```

训练使用标准的 `model.generate()` 进行生成，不依赖vLLM。

### 3.2 关键训练参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| model_name_or_path | 基础模型路径 | /work/mount/qwen7b/Qwen/Qwen2___5-7B |
| use_vllm | 启用vLLM加速 | false（训练时） |
| r | LoRA rank | 16 |
| lora_alpha | LoRA alpha | 32 |
| per_device_train_batch_size | 每卡批次大小 | 1 |
| gradient_accumulation_steps | 梯度累积步数 | 16 |
| max_completion_length | 最大生成长度 | 256 |
| num_generations | 每个prompt生成数 | 2（GRPO最小要求） |
| learning_rate | 学习率 | 1e-5 |
| max_steps | 最大训练步数 | 1024 |

## 四、监控训练

### 4.1 查看训练日志

```bash
# 实时查看日志
tail -f outputs/grpo_lora_qwen2_5_7b_npu_*/output.log

# 查看最近的日志
ls -lt outputs/ | head -5
```

### 4.2 监控NPU使用情况

```bash
# 实时监控NPU
watch -n 1 npu-smi info

# 查看NPU内存使用
npu-smi info -t usages

# 查看所有NPU状态
npu-smi info -l
```

### 4.3 Wandb监控

训练会自动上传到Wandb，查看详细指标：
- 访问 https://wandb.ai
- 查看项目：grpo-lora-qwen2-5-7b-npu
- 监控：loss、reward、learning_rate等

## 五、使用vLLM进行模型评估和推理

训练完成后，使用vLLM加速模型评估和推理，速度提升2-5倍。

```bash
# 评估训练好的模型
python perl/eval.py \
    --model /path/to/trained/model \
    --adapter /path/to/lora/adapter \
    --dataset "open-r1/DAPO-Math-17k-Processed" \
    --result_dir results/ \
    --dp_size 4 \
    --use_vllm  # 启用vLLM加速
```

## 六、性能对比

### 6.1 训练速度（标准模式 vs 理论vLLM模式）

| 模式 | 每步耗时 | 1024步总时间 | 说明 |
|------|----------|--------------|------|
| 标准生成（当前） | ~9分钟 | ~15-20小时 | 稳定可靠 |
| vLLM加速（理论） | ~5-6分钟 | ~9-12小时 | 需要版本兼容 |

**注**：由于版本兼容性问题，当前无法在训练中使用vLLM。

### 6.2 评估/推理速度

| 模式 | 吞吐量 (tokens/s) | 加速比 |
|------|-------------------|--------|
| model.generate() | ~50-80 | 1.0x |
| vLLM | ~150-300 | 2-5x |

**vLLM在评估和推理阶段可以显著提速！**

## 七、常见问题

### 7.1 NPU显存不足（OOM）

**症状**：`NPU out of memory. Tried to allocate XXX MiB`

**解决方案**：
```bash
# 方案1：减小批次大小
--config.training.per_device_train_batch_size 1

# 方案2：增加梯度累积
--config.training.gradient_accumulation_steps 32

# 方案3：减小序列长度
--config.training.max_completion_length 128
--config.training.max_prompt_length 256

# 方案4：降低vLLM内存占用
--config.training.vllm_gpu_memory_utilization 0.2
```

### 7.2 vLLM导入失败

**症状**：`ModuleNotFoundError: No module named 'vllm'`

**解决方案**：
```bash
# 重新安装vLLM
pip install vllm==0.15.1
pip install vllm-ascend==0.13.0

# 验证安装
python -c "from vllm import LLM; print('OK')"
```

### 7.3 torch-npu版本冲突

**症状**：版本警告但不影响使用

**说明**：以下冲突可以忽略
- torch 2.9.1 vs torch-npu要求2.9.0（高度兼容）
- vllm-ascend要求torch 2.8.0（插件仍可工作）

**如果确实有问题**：
```bash
pip install torch==2.9.1 torch-npu==2.9.0 --force-reinstall
pip install "numpy<2.0.0" --force-reinstall
```

### 7.4 训练速度慢

**检查清单**：
1. 确认使用了4张NPU卡
   ```bash
   # 查看日志中的 num_processes: 4
   grep "num_processes" outputs/*/output.log
   ```

2. 确认4张卡都在工作
   ```bash
   npu-smi info
   # 应该看到4张卡的利用率都很高
   ```

3. 确认启用了vLLM
   ```bash
   grep "use_vllm" scripts/trl/openr1/dapo_lora_npu.sh
   # 应该显示 use_vllm true
   ```

### 7.5 vLLM权重同步失败

**症状**：训练过程中vLLM生成结果异常

**解决方案**：
```bash
# 启用重要性采样校正
--config.training.vllm_importance_sampling_correction true
--config.training.vllm_importance_sampling_mode "token_mask"
```

### 7.6 Wandb认证失败

**症状**：`wandb: ERROR Error while calling W&B API`

**解决方案**：
```bash
# 登录wandb
wandb login

# 或设置API key
export WANDB_API_KEY="your_api_key"

# 或禁用wandb
export WANDB_DISABLED=true
```

## 八、依赖版本说明

### 8.1 核心依赖

| 包名 | 版本 | 说明 |
|------|------|------|
| torch | 2.9.1 | PyTorch核心 |
| torch-npu | 2.9.0 | 华为NPU适配 |
| trl | 0.28.0 | 支持vLLM的版本 |
| vllm | 0.15.1 | vLLM推理引擎 |
| vllm-ascend | 0.13.0 | NPU插件 |
| transformers | 4.57.6 | Hugging Face模型库 |
| accelerate | latest | 分布式训练 |
| deepspeed | latest | ZeRO优化 |
| peft | latest | LoRA支持 |

### 8.2 版本兼容性

- ✅ torch 2.9.1 + torch-npu 2.9.0：完全兼容
- ⚠️ vllm-ascend 0.13.0要求torch 2.8.0：警告可忽略
- ✅ TRL 0.18.0+：支持vLLM集成
- ✅ numpy < 2.0：vllm-ascend要求

## 九、最佳实践

### 9.1 训练配置建议

**小模型（7B）**：
```bash
--config.training.per_device_train_batch_size 2
--config.training.gradient_accumulation_steps 8
--config.training.max_completion_length 512
--config.training.vllm_gpu_memory_utilization 0.3
```

**大模型（14B+）**：
```bash
--config.training.per_device_train_batch_size 1
--config.training.gradient_accumulation_steps 16
--config.training.max_completion_length 256
--config.training.vllm_gpu_memory_utilization 0.2
```

### 9.2 内存优化策略

1. **优先调整vLLM内存占用**：从0.3降到0.2
2. **其次减小批次大小**：从2降到1
3. **然后减小序列长度**：从512降到256
4. **最后增加梯度累积**：从8增到16或32

### 9.3 训练流程建议

1. **小规模测试**（10步）：
   ```bash
   --config.training.max_steps 10
   ```
   验证配置正确，无OOM

2. **中等规模验证**（100步）：
   ```bash
   --config.training.max_steps 100
   ```
   验证训练稳定，loss下降

3. **完整训练**（1000+步）：
   ```bash
   --config.training.max_steps 1024
   ```
   正式训练，保存checkpoint

## 十、参考资料

- [TRL vLLM集成文档](https://huggingface.co/docs/trl/main/vllm_integration)
- [Co-located vLLM博客](https://huggingface.co/blog/vllm-colocate)
- [GRPO Trainer文档](https://huggingface.co/docs/trl/main/grpo_trainer)
- [vLLM官方文档](https://docs.vllm.ai/)
- [华为昇腾文档](https://www.hiascend.com/document)

## 十一、快速命令参考

```bash
# 环境检查
bash scripts/check_npu_env.sh
npu-smi info

# 安装依赖
pip install -r requirements_npu.txt

# 开始训练（vLLM加速）
bash scripts/trl/openr1/dapo_lora_npu.sh

# 监控训练
tail -f outputs/grpo_lora_qwen2_5_7b_npu_*/output.log
watch -n 1 npu-smi info

# 验证vLLM
python -c "from vllm import LLM; print('OK')"
python -c "from vllm.platforms import current_platform; print(current_platform)"

# 登录wandb
wandb login
```
