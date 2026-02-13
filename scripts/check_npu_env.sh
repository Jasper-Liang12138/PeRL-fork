#!/bin/bash
# NPU环境检查脚本

echo "=========================================="
echo "华为昇腾NPU环境检查"
echo "=========================================="

# 1. 检查NPU设备
echo -e "\n[1] 检查NPU设备..."
if command -v npu-smi &> /dev/null; then
    npu-smi info | grep "NPU" | head -4
    echo "✓ NPU设备检测成功"
else
    echo "✗ npu-smi命令未找到，请检查驱动安装"
    exit 1
fi

# 2. 检查CANN版本
echo -e "\n[2] 检查CANN版本..."
if [ -f "/usr/local/Ascend/ascend-toolkit/latest/version.cfg" ]; then
    cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
    echo "✓ CANN已安装"
else
    echo "✗ CANN未找到"
    exit 1
fi

# 3. 检查Python环境
echo -e "\n[3] 检查Python环境..."
python --version
echo "✓ Python版本检查完成"

# 4. 检查torch和torch_npu
echo -e "\n[4] 检查PyTorch和torch_npu..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')

try:
    import torch_npu
    print(f'torch_npu已安装')
    print(f'NPU可用: {torch.npu.is_available()}')
    if torch.npu.is_available():
        print(f'NPU数量: {torch.npu.device_count()}')
        print('✓ NPU环境正常')
    else:
        print('✗ NPU不可用')
        exit(1)
except ImportError:
    print('✗ torch_npu未安装')
    exit(1)
" || exit 1

# 5. 检查关键依赖
echo -e "\n[5] 检查关键依赖..."
python -c "
import sys
packages = ['transformers', 'accelerate', 'deepspeed', 'peft', 'trl', 'datasets']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'✗ {pkg} 未安装')
        missing.append(pkg)

if missing:
    print(f'\n缺少依赖: {missing}')
    print('请运行: pip install -r requirements_npu.txt')
    sys.exit(1)
"

# 6. 检查环境变量
echo -e "\n[6] 检查环境变量..."
if [ -z "$ASCEND_VISIBLE_DEVICES" ]; then
    echo "⚠ ASCEND_VISIBLE_DEVICES 未设置（将使用所有NPU）"
else
    echo "✓ ASCEND_VISIBLE_DEVICES=$ASCEND_VISIBLE_DEVICES"
fi

echo -e "\n=========================================="
echo "环境检查完成！可以开始训练。"
echo "运行命令: bash scripts/trl/openr1/dapo_lora_npu.sh"
echo "=========================================="
