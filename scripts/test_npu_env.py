#!/usr/bin/env python3
"""
NPU环境测试脚本
测试模型加载和简单推理是否正常
"""

import torch
import sys

def test_npu_basic():
    """测试NPU基础功能"""
    print("=" * 50)
    print("测试1: NPU基础功能")
    print("=" * 50)

    try:
        import torch_npu
        print("✓ torch_npu导入成功")
    except ImportError:
        print("✗ torch_npu导入失败")
        return False

    if not torch.npu.is_available():
        print("✗ NPU不可用")
        return False

    print(f"✓ NPU可用")
    print(f"✓ NPU数量: {torch.npu.device_count()}")

    # 测试简单的tensor操作
    try:
        x = torch.randn(3, 3).npu()
        y = torch.randn(3, 3).npu()
        z = x + y
        print(f"✓ NPU tensor操作正常")
        print(f"  设备: {z.device}")
    except Exception as e:
        print(f"✗ NPU tensor操作失败: {e}")
        return False

    return True


def test_model_loading():
    """测试模型加载"""
    print("\n" + "=" * 50)
    print("测试2: 模型加载")
    print("=" * 50)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # 使用小模型测试
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"加载测试模型: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✓ Tokenizer加载成功")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="npu:0"
        )
        print("✓ 模型加载成功")
        print(f"  模型设备: {next(model.parameters()).device}")

        # 测试简单推理
        text = "你好"
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to("npu:0") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ 推理测试成功")
        print(f"  输入: {text}")
        print(f"  输出: {result}")

        return True

    except Exception as e:
        print(f"✗ 模型加载/推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_peft():
    """测试PEFT功能"""
    print("\n" + "=" * 50)
    print("测试3: PEFT (LoRA)")
    print("=" * 50)

    try:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        # 创建小模型
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="npu:0"
        )

        # 应用LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)
        print("✓ LoRA应用成功")

        model.print_trainable_parameters()

        return True

    except Exception as e:
        print(f"✗ PEFT测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 50)
    print("华为昇腾NPU环境测试")
    print("=" * 50 + "\n")

    results = []

    # 测试1: NPU基础功能
    results.append(("NPU基础功能", test_npu_basic()))

    # 测试2: 模型加载
    results.append(("模型加载", test_model_loading()))

    # 测试3: PEFT
    results.append(("PEFT (LoRA)", test_peft()))

    # 总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)

    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")

    all_passed = all(r for _, r in results)

    if all_passed:
        print("\n✓ 所有测试通过！环境配置正确。")
        print("可以开始训练: bash scripts/trl/openr1/dapo_lora_npu.sh")
        return 0
    else:
        print("\n✗ 部分测试失败，请检查环境配置。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
