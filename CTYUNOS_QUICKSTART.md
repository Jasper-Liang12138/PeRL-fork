# CTyunOS 910B 训练快速开始

本文档提供CTyunOS 22.06.2 + 8张华为昇腾910B环境的快速启动指南。

## 快速开始（3步）

### 1. 运行环境检查
```bash
bash scripts/check_ctyunos_910b_env.sh
```

### 2. 安装依赖（如果检查未通过）
```bash
# 创建虚拟环境
python -m venv .venv_ctyunos
source .venv_ctyunos/bin/activate

# 安装依赖
pip install -r requirements_vllm_npu.txt
```

### 3. 开始训练
```bash
bash scripts/trl/openr1/dapo_lora_ctyunos_910b_8gpu.sh
```

## 文件说明

### 训练脚本
- `scripts/trl/openr1/dapo_lora_ctyunos_910b_8gpu.sh` - 8卡910B训练脚本
- `scripts/trl/accelerate/ds_zero2_8gpu_npu.yaml` - 8卡DeepSpeed配置

### 检查脚本
- `scripts/check_ctyunos_910b_env.sh` - 环境快速检查

### 文档
- `doc/CTYUNOS_910B_SETUP.md` - 详细环境准备指南

## 训练配置说明

默认配置（可在训练脚本中修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 模型 | DeepSeek-R1-Distill-Qwen-1.5B | 基础模型 |
| NPU卡数 | 8 | 使用全部8张910B |
| LoRA rank | 16 | LoRA秩 |
| batch_size | 2 | 每卡批次大小 |
| gradient_accumulation | 8 | 梯度累积步数 |
| max_steps | 1024 | 最大训练步数 |
| learning_rate | 1e-5 | 学习率 |
| max_completion_length | 512 | 最大生成长度 |

**有效批次大小** = 8卡 × 2 batch × 8累积 = 128

## 性能预估

- **训练速度**: ~1-2 steps/s（取决于序列长度）
- **显存占用**: 每卡约15-25GB
- **总训练时间**: 1024步约需8-15小时

## 监控训练

### 查看日志
```bash
tail -f outputs/grpo_lora_ctyunos_910b_*/output.log
```

### 监控NPU
```bash
watch -n 1 npu-smi info
```

### Wandb监控
如果启用了Wandb，访问 https://wandb.ai 查看项目 `grpo-lora-ctyunos-910b-8gpu`

## 常见调整

### 显存不足（OOM）
```bash
# 在训练脚本中修改：
--config.training.per_device_train_batch_size 1  # 减小batch size
--config.training.gradient_accumulation_steps 16  # 增加梯度累积
--config.training.max_completion_length 256  # 减小序列长度
```

### 加快训练
```bash
# 在训练脚本中修改：
--config.peft.r 8  # 减小LoRA rank
--config.training.num_generations 2  # 保持最小生成数
--config.training.max_completion_length 256  # 减小序列长度
```

### 使用其他模型
```bash
# 在训练脚本中修改model_name_or_path：
--config.model.model_name_or_path "Qwen/Qwen2.5-7B"
# 或本地路径：
--config.model.model_name_or_path "/path/to/local/model"
```

## 故障排除

### NPU不可用
```bash
# 检查驱动
npu-smi info

# 检查环境变量
echo $ASCEND_HOME
source ~/.bashrc

# 重新安装torch-npu
pip install torch-npu==2.9.0 --force-reinstall
```

### 训练速度慢
```bash
# 确认8张卡都在工作
npu-smi info

# 检查日志中的num_processes
grep "num_processes" outputs/*/output.log
# 应该显示: num_processes: 8
```

### 依赖冲突
```bash
# 重新安装所有依赖
pip install -r requirements_vllm_npu.txt --force-reinstall
```

## 获取帮助

- 详细环境准备: `doc/CTYUNOS_910B_SETUP.md`
- NPU训练指南: `doc/NPU_TRAINING_GUIDE.md`
- 项目主文档: `README.md`

## 环境要求总结

### 必需
- ✅ CTyunOS 22.06.2 或兼容系统
- ✅ 8张华为昇腾910B NPU
- ✅ CANN 8.0+
- ✅ Python 3.8-3.11
- ✅ torch==2.9.0 + torch-npu==2.9.0
- ✅ 100GB+ 磁盘空间

### 可选
- Wandb账号（用于训练监控）
- HuggingFace Token（用于下载模型）
- 代理配置（如果网络受限）

---

**准备好了吗？运行环境检查开始吧！**

```bash
bash scripts/check_ctyunos_910b_env.sh
```
