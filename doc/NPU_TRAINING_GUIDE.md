# NPU环境训练指南

## 环境信息
- 系统：ubuntu22.04-teleformers-cann8.2.rc1-npu:v0.2.0.post1.ssh
- 硬件：4张华为昇腾910B NPU
- 基础模型：Qwen2.5-7B-Instruct

## 环境配置

### 1. 安装依赖
```bash
# 使用NPU专用的requirements
pip install -r requirements_npu.txt

# 或使用uv（推荐）
uv pip install -r requirements_npu.txt
```

### 2. 验证NPU环境
```bash
# 检查NPU设备
npu-smi info

# 验证torch_npu
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
```

## 训练启动

### 运行LoRA训练
```bash
cd /path/to/PeRL-FORK
bash scripts/trl/openr1/dapo_lora_npu.sh
```

## 关键修改说明

### 1. 训练脚本修改（dapo_lora_npu.sh）
- **设备指定**：使用 `ASCEND_VISIBLE_DEVICES` 替代 `CUDA_VISIBLE_DEVICES`
- **模型路径**：改为 `Qwen/Qwen2.5-7B-Instruct`
- **禁用vLLM**：`--config.training.use_vllm false`
- **禁用Liger Kernel**：`--config.training.use_liger_kernel false`
- **调整batch size**：`per_device_train_batch_size` 从4降到2（7B模型更大）
- **调整序列长度**：`max_completion_length` 从16384降到8192（节省显存）

### 2. 代码修改
- **config.py**：添加 `use_npu` 配置项
- **train.py**：
  - 自动导入 `torch_npu`
  - 禁用 `flash_attention_2`（NPU不支持）
  - 条件性加载attention实现

### 3. Accelerate配置（ds_zero2_4gpu_npu.yaml）
- 保持DeepSpeed ZeRO-2配置
- 4个进程对应4张NPU卡

## 性能优化建议

### 显存优化
如果遇到OOM（显存不足），可以调整：
```bash
# 减小batch size
--config.training.per_device_train_batch_size 1

# 增加梯度累积
--config.training.gradient_accumulation_steps 16

# 减小序列长度
--config.training.max_completion_length 4096
```

### 训练速度优化
```bash
# 调整LoRA rank（更小的rank训练更快）
--config.peft.r 8

# 减少生成数量
--config.training.num_generations 4
```

## 常见问题

### 1. torch_npu导入失败
```bash
# 检查CANN版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# 重新安装torch_npu
pip uninstall torch-npu
pip install torch-npu
```

### 2. DeepSpeed初始化失败
```bash
# 检查HCCL环境变量
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
```

### 3. 显存不足
- 降低batch size到1
- 使用更小的LoRA rank（r=8或r=4）
- 减小max_completion_length

### 4. 训练速度慢
- 确认NPU驱动和固件版本
- 检查是否正确使用了4张卡（查看日志中的num_processes）
- 考虑使用更激进的梯度累积策略

## 监控训练

### 查看实时日志
```bash
tail -f outputs/grpo_lora_qwen2_5_7b_npu_*/output.log
```

### 查看NPU使用情况
```bash
watch -n 1 npu-smi info
```

### Wandb监控
训练会自动上传到Wandb，项目名称：`grpo-lora-qwen2-5-7b-npu`

## 预期性能

- **训练速度**：约 1-2 steps/s（取决于序列长度和batch size）
- **显存占用**：每张卡约 20-30GB（7B模型 + LoRA）
- **总训练时间**：1024 steps 约需 10-20小时

## 参考资料

- [华为昇腾文档](https://www.hiascend.com/document)
- [torch_npu GitHub](https://github.com/Ascend/pytorch)
- [CANN开发指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/softwareinstall/instg/instg_0000.html)
