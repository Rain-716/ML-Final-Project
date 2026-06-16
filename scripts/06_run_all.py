#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd):
    print("\n>>> ", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="一键跑完整流程：预处理 -> 传统ML -> BERT GPU -> 评估 -> 检索索引")
    parser.add_argument("--raw_dir", default="data/raw/SoulChatCorpus")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--skip_bert", action="store_true")
    parser.add_argument("--bert_model", default="hfl/chinese-macbert-base")
    args = parser.parse_args()

    run([sys.executable, "scripts/01_preprocess_soulchat.py", "--raw_dir", args.raw_dir, "--max_samples", str(args.max_samples)])
    run([sys.executable, "scripts/02_train_baselines.py"])
    if not args.skip_bert:
        run([sys.executable, "scripts/03_train_bert_gpu.py", "--pretrained_model", args.bert_model, "--fp16"])
    run([sys.executable, "scripts/04_evaluate.py"])
    run([sys.executable, "scripts/05_build_retrieval_index.py"])
    print("\n全部完成。启动界面：python app.py")


if __name__ == "__main__":
    main()
