#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_DIR, PROCESSED_DIR, RANDOM_SEED  # noqa: E402
from src.retrieval import ResponseRetriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    # Use train.csv by default to keep the demo retrieval index independent from validation/test rows.
    parser.add_argument("--processed_csv", default=str(PROCESSED_DIR / "train.csv"))
    parser.add_argument("--out", default=str(MODEL_DIR / "response_retrieval.joblib"))
    parser.add_argument("--max_rows", type=int, default=100000, help="限制索引规模以加快构建；0 表示全量")
    parser.add_argument("--min_response_len", type=int, default=20)
    args = parser.parse_args()

    src = Path(args.processed_csv)
    out = Path(args.out)
    build_csv = src
    tmp_csv = None

    if args.max_rows and args.max_rows > 0:
        df = pd.read_csv(src)
        if len(df) > args.max_rows:
            df = df.sample(args.max_rows, random_state=RANDOM_SEED).reset_index(drop=True)
            tmp_csv = out.parent / "_retrieval_build_sample.csv"
            tmp_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(tmp_csv, index=False, encoding="utf-8-sig")
            build_csv = tmp_csv
            print(f"已从 {src} 抽样 {len(df)} 行用于构建检索索引。")
        else:
            print(f"{src} 共 {len(df)} 行，不需要抽样。")

    ResponseRetriever.build(build_csv, out, min_response_len=args.min_response_len)
    if tmp_csv and tmp_csv.exists():
        tmp_csv.unlink()
    print(f"回复检索索引已保存到：{out.resolve()}")


if __name__ == "__main__":
    main()
