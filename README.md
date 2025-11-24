# PERL: Parameter-Efficient Reinforcement Learning  
> A minimal, modular, and lightning-fast framework for fine-tuning language models with PEFT + RL.

---

## 🧩 Supported Parameter-Efficient Methods

| Method        | Status | Notes |
|---------------|--------|-------|
| LoRA          | ✅     | Fully tested |
| DoRA          | ✅     | Weight-decomposed LoRA |
| MiSS          | ✅     | Mixture of Sub-Spaces |
| VeRA          | ✅     | Vector-based Random Adaptation |
| PiSSA         | ✅     | Principal Singular values & Singular vectors Adaptation |
| AdaLoRA       | ❌     | Rank allocation unstable under RL |
| RandLoRA      | 🔄     | Coming soon |
| P-Tuning v2   | 🔄     | Prefix tuning variant |
| LayerNorm Tuning | 🔄  | Efficient bias-only |
| DeLoRA        | 🔄     | Dynamic expansion |
| X-LoRA        | 🔄     | Cross-layer routing |
| LoKr          | 🔄     | Kronecker-product adaptation |

> Full list & references: [Awesome-LoRA](https://github.com/Yuheng2000/Awesome-LoRA)

---

## ⚙️ Environment Setup

```
pip install -r requirements.txt
pip install vllm --no-build-isolation # vllm for trl rollout
```

### Flash Attention

```
uv pip install flash-attn --no-cache-dir --no-build-isolation
python -c "import flash_attn" # verify
```

### Liger-Kernel for faster training

```
pip install liger-kernel --no-build-isolation
```