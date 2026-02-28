#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通过 ModelScope 下载 Qwen2.5-32B-Instruct 模型
兼容 Python 3.9+

用法:
    python scripts/download_qwen25_32b_modelscope.py
    python scripts/download_qwen25_32b_modelscope.py --model_id Qwen/Qwen2.5-32B-Instruct --save_dir /path/to/models
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional  # 新增：导入兼容3.9的Optional类型


def check_python_version():
    """确保 Python >= 3.9"""
    if sys.version_info < (3, 9):
        print(
            "[ERROR] 需要 Python 3.9 或更高版本，当前版本: {}.{}".format(
                sys.version_info.major, sys.version_info.minor
            )
        )
        sys.exit(1)


def install_modelscope_if_missing():
    """若未安装 modelscope，则自动安装"""
    try:
        import modelscope  # noqa: F401
    except ImportError:
        print("[INFO] 未检测到 modelscope，正在安装...")
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "modelscope", "-q"]
        )
        print("[INFO] modelscope 安装完成")


def download_model(model_id: str, save_dir: str, cache_dir: Optional[str] = None):  # 修改：str | None → Optional[str]
    """
    使用 modelscope 下载模型到本地目录。

    Args:
        model_id: ModelScope 上的模型 ID，如 'Qwen/Qwen2.5-32B-Instruct'
        save_dir: 本地保存路径
        cache_dir: 可选的缓存目录（None 表示使用默认）
    """
    from modelscope import snapshot_download  # type: ignore

    os.makedirs(save_dir, exist_ok=True)

    kwargs: dict = dict(
        model_id=model_id,
        local_dir=save_dir,
    )
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    print("[INFO] 开始下载模型: {}".format(model_id))
    print("[INFO] 保存路径: {}".format(os.path.abspath(save_dir)))
    print("[INFO] 这可能需要较长时间，请耐心等待...\n")

    downloaded_path = snapshot_download(** kwargs)

    print("\n[SUCCESS] 模型下载完成！")
    print("[INFO] 模型路径: {}".format(downloaded_path))
    print("\n[HINT] 在训练脚本中使用以下路径:")
    print("  --config.model.model_name_or_path \"{}\"".format(os.path.abspath(save_dir)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通过 ModelScope 下载 Qwen2.5-32B 模型"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-32B-Instruct",
        help="ModelScope 模型 ID (默认: Qwen/Qwen2.5-32B-Instruct)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/mnt/nvme0/models/Qwen2.5-32B-Instruct",
        help="本地保存目录 (默认: ./models/Qwen2.5-32B-Instruct)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="ModelScope 缓存目录（可选）",
    )
    return parser.parse_args()


def main():
    check_python_version()
    install_modelscope_if_missing()

    args = parse_args()
    download_model(
        model_id=args.model_id,
        save_dir=args.save_dir,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()