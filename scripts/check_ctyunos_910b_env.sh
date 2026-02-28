#!/bin/bash
# CTyunOS 910B 环境快速检查脚本

echo "=========================================="
echo "CTyunOS 910B 环境检查"
echo "=========================================="
echo ""

# 1. 检查CANN版本
echo "[1/8] 检查CANN版本..."
if [ -f "/usr/local/Ascend/ascend-toolkit/latest/version.cfg" ]; then
    cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
    echo "✅ CANN已安装"
else
    echo "❌ CANN未找到，请先安装CANN 8.0+"
    exit 1
fi
echo ""

# 2. 检查NPU驱动
echo "[2/8] 检查NPU驱动..."
if command -v npu-smi &> /dev/null; then
    npu_count=$(npu-smi info | grep -c "910B")
    echo "检测到 $npu_count 张910B NPU卡"
    if [ "$npu_count" -eq 8 ]; then
        echo "✅ NPU数量正确（8张）"
    else
        echo "⚠️  NPU数量不是8张，当前: $npu_count"
    fi
else
    echo "❌ npu-smi命令未找到，请检查NPU驱动"
    exit 1
fi
echo ""

# 3. 检查Python版本
echo "[3/8] 检查Python版本..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"
if python -c "import sys; exit(0 if (3,8) <= sys.version_info < (3,12) else 1)"; then
    echo "✅ Python版本符合要求（3.8-3.11）"
else
    echo "⚠️  Python版本可能不兼容，推荐3.8-3.11"
fi
echo ""

# 4. 检查虚拟环境
echo "[4/8] 检查虚拟环境..."
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✅ 虚拟环境已激活: $VIRTUAL_ENV"
else
    echo "⚠️  未检测到虚拟环境，建议使用虚拟环境"
fi
echo ""

# 5. 检查torch-npu
echo "[5/8] 检查torch-npu..."
if python -c "import torch_npu" 2>/dev/null; then
    torch_npu_version=$(python -c "import torch_npu; print(torch_npu.__version__)" 2>/dev/null)
    echo "torch-npu版本: $torch_npu_version"
    echo "✅ torch-npu已安装"
else
    echo "❌ torch-npu未安装，请运行: pip install torch-npu==2.9.0"
    exit 1
fi
echo ""

# 6. 检查NPU可用性
echo "[6/8] 检查NPU可用性..."
npu_available=$(python -c "import torch; import torch_npu; print(torch.npu.is_available())" 2>/dev/null)
if [ "$npu_available" = "True" ]; then
    npu_count_torch=$(python -c "import torch; import torch_npu; print(torch.npu.device_count())" 2>/dev/null)
    echo "NPU可用: $npu_available"
    echo "NPU数量: $npu_count_torch"
    echo "✅ NPU环境正常"
else
    echo "❌ NPU不可用，请检查驱动和环境变量"
    exit 1
fi
echo ""

# 7. 检查关键依赖
echo "[7/8] 检查关键依赖..."
deps=("transformers" "trl" "peft" "accelerate" "deepspeed")
all_deps_ok=true
for dep in "${deps[@]}"; do
    if python -c "import $dep" 2>/dev/null; then
        version=$(python -c "import $dep; print($dep.__version__)" 2>/dev/null)
        echo "  ✅ $dep: $version"
    else
        echo "  ❌ $dep: 未安装"
        all_deps_ok=false
    fi
done
if [ "$all_deps_ok" = false ]; then
    echo "⚠️  部分依赖未安装，请运行: pip install -r requirements_vllm_npu.txt"
fi
echo ""

# 8. 检查磁盘空间
echo "[8/8] 检查磁盘空间..."
available_space=$(df -h . | awk 'NR==2 {print $4}')
echo "当前目录可用空间: $available_space"
available_gb=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$available_gb" -gt 100 ]; then
    echo "✅ 磁盘空间充足"
else
    echo "⚠️  磁盘空间可能不足，建议至少100GB"
fi
echo ""

# 总结
echo "=========================================="
echo "环境检查完成"
echo "=========================================="
echo ""
echo "如果所有检查都通过，可以开始训练："
echo "  bash scripts/trl/openr1/dapo_lora_ctyunos_910b_8gpu.sh"
echo ""
echo "查看详细环境准备指南："
echo "  cat doc/CTYUNOS_910B_SETUP.md"
echo ""
