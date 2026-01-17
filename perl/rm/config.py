from dataclasses import dataclass


@dataclass(frozen=True)
class RMConfig:
    """Centralized runtime settings for RM and SGLang."""

    model_path: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    sglang_tp_size: int = 1
    sglang_dp_size: int = 1
    sglang_trust_remote_code: bool = True
    rm_server_port: int = 8000
