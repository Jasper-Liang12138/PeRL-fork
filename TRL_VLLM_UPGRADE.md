# TRL vLLM 升级指南

## 概述

TRL 从 v0.18.0 开始支持 vLLM 集成，可以在训练的生成阶段使用 vLLM 加速推理，提升训练速度最多 1.73 倍。

**📖 完整使用指南请查看：[NPU_VLLM_GUIDE.md](NPU_VLLM_GUIDE.md)**

## 快速升级

### 1. 升级 TRL 版本

```bash
# 卸载旧版本
pip uninstall trl -y

# 安装支持 vLLM 的新版本
pip install trl==0.28.0
```

### 2. 验证安装

```bash
# 验证vLLM
python -c "from vllm import LLM; print('✅ vLLM OK')"

# 验证NPU支持
python -c "from vllm.platforms import current_platform; print('✅ Platform:', current_platform)"
```

### 3. 开始训练

```bash
# 使用vLLM加速训练
bash scripts/trl/openr1/dapo_lora_npu.sh
```

## 已更新的文件

以下文件已自动更新以支持 vLLM：

- ✅ `requirements_npu.txt`: TRL 版本从 0.14.0 升级到 0.28.0
- ✅ `perl/train.py`: 移除了 vLLM 参数过滤，允许传递给 GRPOConfig
- ✅ `scripts/trl/openr1/dapo_lora_npu.sh`: 启用 vLLM 参数

## vLLM 配置说明

训练脚本中的 vLLM 参数：

```bash
--config.training.use_vllm true                      # 启用 vLLM
--config.training.vllm_mode colocate                 # 协同模式（推荐）
--config.training.vllm_gpu_memory_utilization 0.3    # vLLM 使用 30% GPU 内存
```

## 预期效果

- ✅ 生成阶段速度提升：1.5-1.7 倍
- ✅ 总训练时间缩短：约 30-40%
- ✅ 内存使用：vLLM 占用 30%，训练占用 70%

## 更多信息

详细的训练、评估、故障排除等信息，请查看：

**📖 [NPU_VLLM_GUIDE.md](NPU_VLLM_GUIDE.md) - 完整的NPU + vLLM训练评估指南**

## 参考资料

- [TRL vLLM 集成文档](https://huggingface.co/docs/trl/main/vllm_integration)
- [Co-located vLLM 博客](https://huggingface.co/blog/vllm-colocate)
- [GRPO Trainer 文档](https://huggingface.co/docs/trl/main/grpo_trainer)
