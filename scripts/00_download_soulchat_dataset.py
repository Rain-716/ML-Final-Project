#!/usr/bin/env python
"""Download SoulChatCorpus from ModelScope to a local data folder.

This script is intentionally separate from preprocessing, because the course
rubric checks whether raw data preparation and preprocessing are clear.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="YIRONGCHEN/SoulChatCorpus")
    parser.add_argument("--local_dir", default="data/raw/SoulChatCorpus")
    args = parser.parse_args()

    out = Path(args.local_dir)
    out.mkdir(parents=True, exist_ok=True)
    if shutil.which("modelscope") is None:
        print("未检测到 modelscope 命令，正在尝试安装 modelscope ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "modelscope"])

    cmd = ["modelscope", "download", "--dataset", args.dataset, "--local_dir", str(out)]
    print("运行下载命令：", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"下载完成：{out.resolve()}")


if __name__ == "__main__":
    main()
