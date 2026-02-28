# CTyunOS 22.06.2 + 华为昇腾910B 训练环境准备指南

## 系统信息
- **操作系统**: CTyunOS 22.06.2@ascend-910b 64位
- **硬件**: 8*HuaweiAscend 910B
- **训练框架**: PeRL (基于TRL)
- **训练方法**: GRPO + LoRA

---

## 一、环境准备清单

### 1.1 系统依赖检查

在开始之前，确保系统已安装以下组件：

```bash
# 检查CANN版本（需要8.0+）
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# 检查NPU驱动
npu-smi info

# 检查Python版本（需要3.8-3.11）
python --version
```

**预期输出**：
- CANN版本：8.0.RC1 或更高
- NPU驱动：正常显示8张910B卡
- Python版本：3.8-3.11

### 1.2 必需的系统环境变量

在 `~/.bashrc` 或 `~/.bash_profile` 中添加：

```bash
# CANN环境变量
export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
export PATH=$ASCEND_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ASCEND_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$ASCEND_HOME/python/site-packages:$PYTHONPATH

# NPU优化环境变量
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=1
```

应用环境变量：
```bash
source ~/.bashrc
```

---

## 二、Python环境配置

### 2.1 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv .venv_ctyunos

# 激活虚拟环境
source .venv_ctyunos/bin/activate
```

### 2.2 安装核心依赖

# torch 从 PyPI 安装 CPU 版本（torch-npu 会接管 NPU 后端）
pip install torch==2.8.0

# torch-npu 从 PyPI 正常安装
pip install torch-npu==2.8.0.post2

```bash
# 方式1：使用pip安装（推荐用于CTyunOS）
pip install -r requirements_ctyunos_910b.txt

# 方式2：使用uv安装（更快，如果已安装uv）
uv pip install -r requirements_ctyunos_910b.txt
```

**关键依赖版本**：
- torch==2.8.0
- torch-npu==2.8.0.post2
- transformers==4.57.6
- trl==0.14.0
- deepspeed>=0.14.0
- peft>=0.8.0
- accelerate>=0.20.0

### 2.3 验证安装

```bash
# 验证torch-npu
python -c "import torch; import torch_npu; print('NPU available:', torch.npu.is_available())"

# 验证NPU数量
python -c "import torch; import torch_npu; print('NPU count:', torch.npu.device_count())"

# 验证transformers
python -c "import transformers; print('Transformers version:', transformers.__version__)"

# 验证TRL
python -c "import trl; print('TRL version:', trl.__version__)"
```

**预期输出**：
```
NPU available: True
NPU count: 8
Transformers version: 4.57.6
TRL version: 0.14.0
```

---

## 三、模型准备

### 3.1 下载基础模型

训练脚本默认使用 `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`，你可以：

**选项1：自动下载（需要网络）**
```bash
# 脚本会自动从HuggingFace下载
# 确保设置了HF_TOKEN（如果模型需要授权）
export HF_TOKEN="your_huggingface_token"
```

**选项2：手动下载到本地**
```bash
# 使用huggingface-cli下载
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --local-dir /path/to/models/DeepSeek-R1-Distill-Qwen-1.5B

# 修改训练脚本中的模型路径
# --config.model.model_name_or_path "/path/to/models/DeepSeek-R1-Distill-Qwen-1.5B"
```

**选项3：使用其他模型**
```bash
# 支持的模型系列：
# - Qwen2.5系列：Qwen/Qwen2.5-7B, Qwen/Qwen2.5-14B
# - DeepSeek系列：deepseek-ai/DeepSeek-R1-Distill-*
# - Llama系列：meta-llama/Llama-3-8B

# 修改训练脚本中的模型路径即可
```


**选项4：通过 ModelScope 下载 Qwen2.5-32B（推荐，国内速度快）**
```bash
# 步骤1：下载模型（Python 3.9+ 兼容）
python scripts/download_qwen25_32b_modelscope.py \
    --model_id Qwen/Qwen2.5-32B-Instruct \
    --save_dir /mnt/nvme0/models/Qwen2.5-32B-Instruct

# 步骤2：（可选）指定自定义缓存目录
python scripts/download_qwen25_32b_modelscope.py \
    --save_dir /mnt/nvme0/models/Qwen2.5-32B-Instruct \
    --cache_dir /mnt/nvme0/modelscope_cache

# 步骤3：运行 Qwen2.5-32B 专用训练脚本
MODEL_PATH=/mnt/nvme0/models/Qwen2.5-32B-Instruct \
    bash scripts/trl/openr1/dapo_lora_qwen25_32b_ctyunos_910b_8gpu.sh
```

> **注意（32B 模型显存）**：Qwen2.5-32B 约需 64GB+ 显存。在8张910B（每张32GB）上建议：
> - `per_device_train_batch_size=1`
> - `gradient_accumulation_steps=16`
> - 开启 DeepSpeed ZeRO-2（已在 `ds_zero2_8gpu_npu.yaml` 中配置）

### 3.2 数据集准备

训练脚本默认使用 `open-r1/DAPO-Math-17k-Processed`：

```bash
# 数据集会自动从HuggingFace下载
# 如果需要使用本地数据集，修改脚本中的：
# --config.dataset.dataset_name_or_path "/path/to/local/dataset"
```

---

## 四、训练前检查

### 4.1 运行环境检查脚本

```bash
# 运行NPU环境检查
bash scripts/check_npu_env.sh
```

### 4.2 手动检查NPU状态

```bash
# 查看NPU信息
npu-smi info

# 查看NPU内存使用
npu-smi info -t usages

# 持续监控NPU（每秒刷新）
watch -n 1 npu-smi info
```

**预期输出**：应该看到8张910B卡，状态为Idle或Running，温度正常。

### 4.3 测试小规模训练

在正式训练前，建议先运行10步测试：

```bash
# 编辑训练脚本，将max_steps改为10
# --config.training.max_steps 10

# 运行测试
bash scripts/trl/openr1/dapo_lora_ctyunos_910b_8gpu.sh
```

---

## 五、Wandb配置（可选）

如果需要使用Wandb监控训练：

```bash
# 安装wandb
pip install wandb

# 登录wandb
wandb login

# 或设置API key
export WANDB_API_KEY="your_wandb_api_key"
```

如果不需要wandb，可以禁用：
```bash
# 在训练脚本开头添加
export WANDB_DISABLED=true
```

---

## 六、存储空间检查

### 6.1 磁盘空间要求

- **模型文件**: ~3-15GB（取决于模型大小）
- **数据集**: ~1-5GB
- **训练输出**: ~10-50GB（checkpoints + logs）
- **建议预留**: 至少100GB空闲空间

```bash
# 检查磁盘空间
df -h

# 检查当前目录空间
du -sh .
```

### 6.2 设置输出目录

如果默认输出目录空间不足，可以修改：

```bash
# 在训练脚本中修改OUTPUT_DIR
OUTPUT_DIR=/path/to/large/disk/outputs/grpo_lora_ctyunos_910b_$(date +%Y%m%d_%H%M%S)
```

---

## 七、网络配置

### 7.1 HuggingFace镜像（可选）

如果访问HuggingFace较慢，可以使用镜像：

```bash
# 设置HF镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 7.2 代理配置（如需要）

```bash
# 设置HTTP代理
export http_proxy=http://proxy.example.com:8080
export https_proxy=http://proxy.example.com:8080
```

---

## 八、常见问题预检

### 8.1 NPU不可见

**症状**: `torch.npu.is_available()` 返回 False

**解决方案**:
```bash
# 检查驱动
npu-smi info

# 检查环境变量
echo $ASCEND_HOME
echo $LD_LIBRARY_PATH

# 重新安装torch-npu
pip uninstall torch-npu
pip install torch-npu==2.8.0.post2
```

### 8.2 CANN版本不匹配

**症状**: 版本警告或导入错误

**解决方案**:
```bash
# 检查CANN版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# 确保torch-npu版本与CANN匹配
# CANN 8.2.RC1 -> torch-npu 2.8.0.post2
```

### 8.3 权限问题

**症状**: 无法访问NPU设备

**解决方案**:
```bash
# 检查用户组
groups

# 添加用户到HwHiAiUser组（需要root权限）
sudo usermod -aG HwHiAiUser $USER

# 重新登录使组权限生效
```

---

## 九、环境准备完成检查清单

在开始训练前，确认以下所有项目：

- [ ] CANN 8.0+ 已安装
- [ ] NPU驱动正常，`npu-smi info` 显示8张910B卡
- [ ] Python 3.8-3.11 已安装
- [ ] 虚拟环境已创建并激活
- [ ] requirements_vllm_npu.txt 依赖已安装
- [ ] `torch.npu.is_available()` 返回 True
- [ ] `torch.npu.device_count()` 返回 8
- [ ] 基础模型已下载或可访问
- [ ] 磁盘空间充足（>100GB）
- [ ] 环境变量已配置（ASCEND_HOME, HCCL等）
- [ ] Wandb已配置（如需要）或已禁用
- [ ] 网络连接正常（如需下载模型/数据）

---

## 十、开始训练

环境准备完成后，运行训练脚本：

```bash
# 进入项目目录
cd /path/to/PeRL-fork

# 激活虚拟环境
source .venv_ctyunos/bin/activate

# 运行训练
bash scripts/trl/openr1/dapo_lora_ctyunos_910b_8gpu.sh
```

训练日志会保存在 `outputs/grpo_lora_ctyunos_910b_*/output.log`

---

## 十一、监控训练

### 实时查看日志
```bash
tail -f outputs/grpo_lora_ctyunos_910b_*/output.log
```

### 监控NPU使用
```bash
watch -n 1 npu-smi info
```

### 查看Wandb（如已配置）
访问 https://wandb.ai 查看训练指标

---

## 十二、参考资料

- [华为昇腾文档](https://www.hiascend.com/document)
- [CANN开发指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/softwareinstall/instg/instg_0000.html)
- [torch-npu GitHub](https://github.com/Ascend/pytorch)
- [PeRL项目文档](../README.md)
- [NPU训练指南](NPU_TRAINING_GUIDE.md)
