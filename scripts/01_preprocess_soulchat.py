#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Preprocess SoulChatCorpus multi-turn JSON for this ML course project.

适配目录结构：
    chinese_ai_psych_consultation/data/raw/SoulChatCorpus/
        SoulChatCorpus-sft-multi-Turn.json
        dataset_infos.json

原始数据格式示例：
[
  {
    "id": 0,
    "topic": "成长",
    "messages": [
      {"role": "user", "content": "最近感觉很焦虑，不知道如何缓解。"},
      {"role": "assistant", "content": "我明白你的感受。..."}
    ]
  }
]

输出文件仍与原项目保持一致：
    data/processed/all.csv
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv
    data/processed/preprocess_stats.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR, RANDOM_SEED  # noqa: E402
from src.data_utils import clean_text, ensure_dir  # noqa: E402
from src.label_rules import LABELS_ZH, weak_label  # noqa: E402

PREFERRED_DATA_FILE = "SoulChatCorpus-sft-multi-Turn.json"
METADATA_FILES = {"dataset_infos.json", "README.md", "README.txt"}

# 数据集中偶尔会有不适合课程演示/心理支持系统直接检索的粗俗或攻击性回复。
# 默认删除这些回复，避免 Gradio Demo 检索时把低质量回复展示给老师同学。
BAD_RESPONSE_KEYWORDS = [
    "傻逼", "煞笔", "傻b", "sb", "去死吧", "废物", "活该", "神经病",
]

USER_ROLES = {"user", "human", "client", "patient", "来访者", "用户", "求助者"}
ASSISTANT_ROLES = {"assistant", "gpt", "bot", "counselor", "therapist", "心理咨询师", "咨询师", "助手"}


def normalize_role(role: Any) -> str:
    """Normalize role names from different SFT/chat datasets."""
    role_str = str(role or "").strip().lower()
    if role_str in USER_ROLES:
        return "user"
    if role_str in ASSISTANT_ROLES:
        return "assistant"
    return role_str


def find_data_file(raw_dir: Path, data_file: str = "") -> Path:
    """Find the real corpus JSON file and ignore dataset_infos.json metadata."""
    if data_file:
        p = Path(data_file)
        if not p.is_absolute():
            p = raw_dir / p
        if p.exists():
            return p
        raise FileNotFoundError(f"指定的数据文件不存在：{p}")

    preferred = raw_dir / PREFERRED_DATA_FILE
    if preferred.exists():
        return preferred

    patterns = ["*multi*Turn*.json", "*Multi*Turn*.json", "SoulChatCorpus*.json", "*.jsonl", "*.json"]
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(raw_dir.rglob(pat))

    unique = []
    seen = set()
    for p in candidates:
        if p.name in METADATA_FILES:
            continue
        if p.resolve() not in seen:
            seen.add(p.resolve())
            unique.append(p)

    if not unique:
        raise FileNotFoundError(
            f"在 {raw_dir} 下没有找到真正的数据文件。\n"
            f"请确认存在：{PREFERRED_DATA_FILE}\n"
            "注意：dataset_infos.json 只是元数据说明，不是训练语料。"
        )
    return unique[0]


def iter_json_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield records from JSON/JSONL.

    对大型 JSON 数组，若本机安装了 ijson，会自动流式读取，内存占用更低；
    未安装 ijson 时回退到 json.load，SoulChatCorpus 规模一般也可以正常处理。
    """
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
        return

    if suffix != ".json":
        raise ValueError(f"暂不支持该文件类型：{path}")

    # 优先用 ijson 流式读取 JSON 数组，避免一次性加载超大文件。
    try:
        import ijson  # type: ignore

        with path.open("rb") as f:
            first_items = ijson.items(f, "item")
            yielded = False
            for obj in first_items:
                if isinstance(obj, dict):
                    yielded = True
                    yield obj
            if yielded:
                return
    except Exception:
        pass

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
    elif isinstance(obj, dict):
        # 兼容 {"train": [...]}, {"data": [...]}, {"items": [...]} 这类包装结构。
        yielded = False
        for key in ["train", "data", "items", "examples"]:
            if isinstance(obj.get(key), list):
                yielded = True
                for item in obj[key]:
                    if isinstance(item, dict):
                        yield item
        if not yielded and "messages" in obj:
            yield obj
    else:
        raise ValueError(f"无法识别的 JSON 顶层结构：{type(obj)}")


def format_context(history: List[Tuple[str, str]], max_context_turns: int) -> str:
    """Convert recent turns to readable context text."""
    recent = history[-max_context_turns:] if max_context_turns > 0 else history
    lines = []
    for role, content in recent:
        speaker = "用户" if role == "user" else "心理咨询师"
        lines.append(f"{speaker}：{content}")
    return "\n".join(lines)


def response_is_bad(response: str) -> bool:
    low = response.lower()
    return any(k.lower() in low for k in BAD_RESPONSE_KEYWORDS)


def extract_pairs_from_messages(
    record: Dict[str, Any],
    source_file: str,
    max_context_turns: int = 6,
    drop_bad_response: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Extract one supervised sample for every assistant reply.

    每条样本 = 最近一轮用户输入 text + 助手回复 response + 上下文 context。
    """
    messages = record.get("messages")
    if not isinstance(messages, list):
        return

    topic = clean_text(record.get("topic", "")) or "未知"
    sample_id = record.get("id", "")

    history: List[Tuple[str, str]] = []
    last_user: Optional[str] = None

    for turn_index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = normalize_role(msg.get("role"))
        content = clean_text(msg.get("content", ""))
        if not content:
            continue

        if role == "user":
            last_user = content
            history.append(("user", content))
            continue

        if role == "assistant":
            if last_user:
                response = content
                if drop_bad_response and response_is_bad(response):
                    history.append(("assistant", response))
                    last_user = None
                    continue

                context = format_context(history, max_context_turns)
                yield {
                    "sample_id": sample_id,
                    "topic": topic,
                    "turn_index": turn_index,
                    "context": context,
                    "text": last_user,
                    "response": response,
                    "source_file": source_file,
                }
            history.append(("assistant", content))
            last_user = None


def build_label(row: Dict[str, Any], label_mode: str) -> Tuple[str, str, str, int, str]:
    """Return label, label_zh, risk_level, risk_keyword_count, matched_keywords."""
    if label_mode == "topic":
        topic = clean_text(row.get("topic", "")) or "未知"
        weak = weak_label(row.get("text", ""))
        return topic, topic, weak.risk_level, weak.risk_keyword_count, weak.matched_keywords

    weak = weak_label(row.get("text", ""))
    return (
        weak.label,
        LABELS_ZH.get(weak.label, weak.label),
        weak.risk_level,
        weak.risk_keyword_count,
        weak.matched_keywords,
    )


def enrich_row(row: Dict[str, Any], label_mode: str) -> Dict[str, Any]:
    text = clean_text(row.get("text", ""))
    response = clean_text(row.get("response", ""))
    context = clean_text(row.get("context", ""))
    label, label_zh, risk_level, risk_keyword_count, matched_keywords = build_label(row, label_mode)

    return {
        "sample_id": row.get("sample_id", ""),
        "topic": row.get("topic", "未知"),
        "turn_index": row.get("turn_index", -1),
        "context": context,
        "text": text,
        "response": response,
        "label": label,
        "label_zh": label_zh,
        "risk_level": risk_level,
        "matched_keywords": matched_keywords,
        "text_len": len(text),
        "response_len": len(response),
        "num_turns": max(context.count("用户："), 1),
        "risk_keyword_count": risk_keyword_count,
        "question_mark_count": text.count("?") + text.count("？"),
        "exclamation_count": text.count("!") + text.count("！"),
        "source_file": row.get("source_file", ""),
    }


def merge_rare_labels(df: pd.DataFrame, label_mode: str, min_count: int = 3) -> pd.DataFrame:
    """Avoid stratified split crashes caused by labels with too few samples."""
    df = df.copy()
    vc = df["label"].value_counts()
    rare_labels = set(vc[vc < min_count].index)
    if not rare_labels:
        return df

    target = "other" if label_mode == "weak" else "其他"
    target_zh = LABELS_ZH.get("other", "其他心理支持") if label_mode == "weak" else "其他"
    df.loc[df["label"].isin(rare_labels), "label"] = target
    df.loc[df["label"] == target, "label_zh"] = target_zh
    return df


def random_three_way_split(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    if n == 1:
        return df, df.iloc[0:0].copy(), df.iloc[0:0].copy()

    n_train = max(1, int(n * 0.70))
    n_val = max(1, int(n * 0.15)) if n >= 3 else 0
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    if len(test_df) == 0 and len(val_df) > 0:
        test_df = val_df.tail(1).copy()
        val_df = val_df.iloc[:-1].copy()
    return train_df, val_df, test_df



def _mode_or_first(values: pd.Series) -> str:
    values = values.dropna().astype(str)
    if values.empty:
        return "other"
    mode = values.mode()
    return str(mode.iloc[0]) if not mode.empty else str(values.iloc[0])


def make_group_key(df: pd.DataFrame, group_col: str) -> pd.Series:
    """Build a stable group key to keep one dialogue in only one split."""
    if group_col not in df.columns:
        return pd.Series([f"row_{i}" for i in range(len(df))], index=df.index)

    group_values = df[group_col].fillna("").astype(str).str.strip()
    if "source_file" in df.columns:
        source_values = df["source_file"].fillna("").astype(str).str.strip()
    else:
        source_values = pd.Series([""] * len(df), index=df.index)

    keys = []
    for i, (src, gid) in enumerate(zip(source_values, group_values)):
        if gid:
            keys.append(f"{src}::{gid}")
        else:
            keys.append(f"{src}::row_{i}")
    return pd.Series(keys, index=df.index)


def split_groups_randomly(
    group_df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[set, set, set]:
    """Random group split fallback. Works even for tiny/debug data."""
    rng = random.Random(seed)
    groups = group_df["group_key"].tolist()
    rng.shuffle(groups)

    n_groups = len(groups)
    if n_groups == 1:
        return set(groups), set(), set()

    n_test = max(1, int(round(n_groups * test_size))) if n_groups >= 3 else 1
    n_val = max(1, int(round(n_groups * val_size))) if n_groups >= 4 else 0
    if n_test + n_val >= n_groups:
        n_test = 1
        n_val = 1 if n_groups >= 3 else 0

    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test:n_test + n_val])
    train_groups = set(groups[n_test + n_val:])

    if not train_groups:
        train_groups = set(groups[-1:])
        test_groups.discard(groups[-1])
        val_groups.discard(groups[-1])

    return train_groups, val_groups, test_groups


def split_groups_stratified(
    group_df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[set, set, set, str]:
    """Stratified split on dialogue groups by each group's majority label."""
    n_groups = len(group_df)
    n_classes = int(group_df["group_label"].nunique())
    min_group_count = int(group_df["group_label"].value_counts().min()) if n_classes else 0

    can_stratify_test = (
        n_groups >= 30
        and n_classes >= 2
        and min_group_count >= 2
        and int(n_groups * test_size) >= n_classes
    )

    if not can_stratify_test:
        train_g, val_g, test_g = split_groups_randomly(group_df, test_size, val_size, seed)
        return train_g, val_g, test_g, "group_random_fallback"

    try:
        trainval_gdf, test_gdf = train_test_split(
            group_df,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=group_df["group_label"],
        )

        relative_val = val_size / (1.0 - test_size)
        n_trainval = len(trainval_gdf)
        n_trainval_classes = int(trainval_gdf["group_label"].nunique())
        min_trainval_count = int(trainval_gdf["group_label"].value_counts().min()) if n_trainval_classes else 0
        can_stratify_val = (
            n_trainval >= 20
            and n_trainval_classes >= 2
            and min_trainval_count >= 2
            and int(n_trainval * relative_val) >= n_trainval_classes
        )

        if can_stratify_val:
            train_gdf, val_gdf = train_test_split(
                trainval_gdf,
                test_size=relative_val,
                random_state=seed,
                shuffle=True,
                stratify=trainval_gdf["group_label"],
            )
            method = "stratified_group"
        else:
            train_g, val_g, _ = split_groups_randomly(
                trainval_gdf,
                test_size=0.0,
                val_size=relative_val,
                seed=seed + 17,
            )
            train_gdf = trainval_gdf[trainval_gdf["group_key"].isin(train_g)]
            val_gdf = trainval_gdf[trainval_gdf["group_key"].isin(val_g)]
            method = "test_stratified_group_val_random_group"

        return (
            set(train_gdf["group_key"]),
            set(val_gdf["group_key"]),
            set(test_gdf["group_key"]),
            method,
        )
    except ValueError as exc:
        print(f"[提示] 分组分层划分失败，改用分组随机划分：{exc}")
        train_g, val_g, test_g = split_groups_randomly(group_df, test_size, val_size, seed)
        return train_g, val_g, test_g, "group_random_fallback"


def split_dataset(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
    group_col: str = "sample_id",
    split_by_group: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, Dict[str, int]]:
    """
    Split train/val/test.

    重要修正：
    SoulChatCorpus 一条原始多轮对话会抽取出多个监督样本。如果普通按“行”随机划分，
    同一个 sample_id 的不同轮次会同时出现在 train/val/test，导致对话级数据泄漏，
    基准模型分数会虚高。因此默认按 sample_id 分组划分。
    """
    n = len(df)
    if n == 1:
        empty = df.iloc[0:0].copy()
        return df, empty, empty, "single_row", {"train_val": 0, "train_test": 0, "val_test": 0}

    if split_by_group and group_col in df.columns:
        work = df.copy()
        work["_group_key"] = make_group_key(work, group_col)
        group_df = (
            work.groupby("_group_key", as_index=False)
            .agg(
                group_key=("_group_key", "first"),
                group_label=("label", _mode_or_first),
                rows=("label", "size"),
            )
        )

        train_groups, val_groups, test_groups, method = split_groups_stratified(group_df, test_size, val_size, seed)

        train_df = work[work["_group_key"].isin(train_groups)].drop(columns=["_group_key"]).sample(frac=1, random_state=seed)
        val_df = work[work["_group_key"].isin(val_groups)].drop(columns=["_group_key"]).sample(frac=1, random_state=seed + 1)
        test_df = work[work["_group_key"].isin(test_groups)].drop(columns=["_group_key"]).sample(frac=1, random_state=seed + 2)

        overlaps = {
            "train_val": len(train_groups & val_groups),
            "train_test": len(train_groups & test_groups),
            "val_test": len(val_groups & test_groups),
        }

        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
            method,
            overlaps,
        )

    # Backward-compatible row-level fallback for data without sample_id.
    n_classes = int(df["label"].nunique())
    min_count = int(df["label"].value_counts().min()) if n_classes else 0

    can_stratify = (
        n >= 30
        and n_classes >= 2
        and min_count >= 3
        and int(n * test_size) >= n_classes
        and int(n * val_size) >= n_classes
    )
    if can_stratify:
        try:
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=seed,
                shuffle=True,
                stratify=df["label"],
            )
            relative_val = val_size / (1.0 - test_size)
            train_df, val_df = train_test_split(
                train_df,
                test_size=relative_val,
                random_state=seed,
                shuffle=True,
                stratify=train_df["label"],
            )
            return (
                train_df.reset_index(drop=True),
                val_df.reset_index(drop=True),
                test_df.reset_index(drop=True),
                "row_stratified_no_group",
                {"train_val": 0, "train_test": 0, "val_test": 0},
            )
        except ValueError as exc:
            print(f"[提示] 行级分层划分失败，改用随机划分：{exc}")

    train_df, val_df, test_df = random_three_way_split(df, seed)
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        "row_random_fallback_no_group",
        {"train_val": 0, "train_test": 0, "val_test": 0},
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess SoulChatCorpus-sft-multi-Turn.json")
    parser.add_argument("--raw_dir", default="data/raw/SoulChatCorpus", help="SoulChatCorpus 下载后的本地目录")
    parser.add_argument("--data_file", default="", help="可选：指定数据文件名或绝对路径，默认自动寻找 SoulChatCorpus-sft-multi-Turn.json")
    parser.add_argument("--out_dir", default=str(PROCESSED_DIR))
    parser.add_argument("--label_mode", choices=["weak", "topic"], default="weak", help="weak=关键词弱标签；topic=使用原始 topic 作为分类标签")
    parser.add_argument("--min_text_len", type=int, default=4)
    parser.add_argument("--max_text_len", type=int, default=800)
    parser.add_argument("--min_response_len", type=int, default=10)
    parser.add_argument("--max_response_len", type=int, default=3000)
    parser.add_argument("--max_context_turns", type=int, default=6)
    parser.add_argument("--max_samples", type=int, default=0, help="调试用；0 表示使用全部样本")
    parser.add_argument("--test_size", type=float, default=0.10)
    parser.add_argument("--val_size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--keep_bad_response", action="store_true", help="保留含粗俗/攻击性关键词的助手回复；默认删除")
    parser.add_argument("--split_by_group", action=argparse.BooleanOptionalAction, default=True, help="默认开启：按 sample_id 对话级划分，避免同一对话泄漏到多个集合")
    parser.add_argument("--group_col", default="sample_id", help="对话级分组列名，默认 sample_id")
    args = parser.parse_args()

    random.seed(args.seed)
    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"没有找到原始数据目录：{raw_dir}\n"
            "请把 SoulChatCorpus-sft-multi-Turn.json 放到 data/raw/SoulChatCorpus 下。"
        )

    data_path = find_data_file(raw_dir, args.data_file)
    print(f"[信息] 使用数据文件：{data_path}")
    print("[信息] dataset_infos.json 是元数据文件，本脚本会自动忽略。")

    rows: List[Dict[str, Any]] = []
    raw_pair_count = 0
    filtered_count = 0

    try:
        source_file = str(data_path.relative_to(raw_dir))
    except ValueError:
        source_file = data_path.name

    for record in tqdm(iter_json_records(data_path), desc="抽取多轮对话样本"):
        for base_row in extract_pairs_from_messages(
            record,
            source_file=source_file,
            max_context_turns=args.max_context_turns,
            drop_bad_response=not args.keep_bad_response,
        ):
            raw_pair_count += 1
            row = enrich_row(base_row, args.label_mode)

            if not (args.min_text_len <= row["text_len"] <= args.max_text_len):
                filtered_count += 1
                continue
            if not (args.min_response_len <= row["response_len"] <= args.max_response_len):
                filtered_count += 1
                continue

            rows.append(row)
            if args.max_samples and len(rows) >= args.max_samples:
                break
        if args.max_samples and len(rows) >= args.max_samples:
            break

    if not rows:
        raise RuntimeError(
            "没有抽取到有效样本。请确认 JSON 顶层是列表，且每条数据包含 messages: [{role, content}, ...]。"
        )

    df = pd.DataFrame(rows)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text", "response"]).reset_index(drop=True)
    after_dedup = len(df)

    df = merge_rare_labels(df, label_mode=args.label_mode, min_count=3)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    all_path = out_dir / "all.csv"
    df.to_csv(all_path, index=False, encoding="utf-8-sig")

    train_df, val_df, test_df, split_method, split_group_overlaps = split_dataset(
        df,
        args.test_size,
        args.val_size,
        args.seed,
        group_col=args.group_col,
        split_by_group=args.split_by_group,
    )
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    label_mapping = (
        df[["label", "label_zh"]]
        .drop_duplicates()
        .sort_values("label")
        .reset_index(drop=True)
        .to_dict(orient="records")
    )
    (out_dir / "label_mapping.json").write_text(json.dumps(label_mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {
        "data_file": str(data_path),
        "label_mode": args.label_mode,
        "raw_extracted_pairs": raw_pair_count,
        "length_filtered_pairs": filtered_count,
        "rows_before_dedup": before_dedup,
        "rows_after_dedup": after_dedup,
        "final_rows": int(len(df)),
        "split_method": split_method,
        "split_by_group": bool(args.split_by_group),
        "group_col": args.group_col,
        "split_group_overlaps": split_group_overlaps,
        "warning": "weak 标签由关键词规则生成，分数代表对弱标签的拟合效果，不等同于真实心理诊断能力。" if args.label_mode == "weak" else "",
        "split_sizes": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
        "topic_distribution_top20": df["topic"].value_counts().head(20).to_dict(),
        "label_distribution": df["label"].value_counts().to_dict(),
        "risk_distribution": df["risk_level"].value_counts().to_dict(),
        "columns": list(df.columns),
    }
    (out_dir / "preprocess_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"预处理完成，文件保存在：{out_dir.resolve()}")


if __name__ == "__main__":
    main()
