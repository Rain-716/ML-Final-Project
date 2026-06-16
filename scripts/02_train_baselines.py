#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast baseline training for the Chinese AI psychological consultation project.

核心提速思路：
1. 去掉耗时的 RandomizedSearchCV / K 折交叉验证，改为 train -> val 的快速候选模型选择。
2. 用 HashingVectorizer 代替 TfidfVectorizer，避免在大语料上构建巨大词表。
3. 只训练极快的线性模型 / 朴素贝叶斯模型，跳过慢速 LogisticRegression 和 LinearSVC 大规模 CV。
4. 学习曲线改为可选；默认只做轻量训练与必要图表输出。

保存结果仍兼容原项目：
- models/best_baseline.joblib
- models/dummy_baseline.joblib
- outputs/baseline/baseline_summary.json
- outputs/baseline/model_macro_f1_comparison.png
- outputs/baseline/best_baseline_confusion_matrix.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

# 让 matplotlib 在服务器/Windows 终端中也能保存图片，不弹窗阻塞。
import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    LABEL_COLUMN,
    MODEL_DIR,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    TEXT_COLUMN,
)
from src.metrics_utils import (  # noqa: E402
    compute_metrics,
    plot_bar,
    plot_confusion_matrix,
    save_json,
)

# Columns used only for leakage diagnostics, not as model features.
DIAGNOSTIC_COLUMNS = ["sample_id", "source_file", "turn_index", "topic", "matched_keywords"]


# -----------------------------
# Data loading
# -----------------------------

def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}\n请先运行 scripts/01_preprocess_soulchat.py")

    # 只读训练必需列 + 少量诊断列，减少大 CSV 的读取时间和内存。
    header = pd.read_csv(path, nrows=0).columns.tolist()
    need_cols = [TEXT_COLUMN, LABEL_COLUMN] + [c for c in NUMERIC_FEATURES if c in header]
    need_cols += [c for c in DIAGNOSTIC_COLUMNS if c in header and c not in need_cols]
    df = pd.read_csv(path, usecols=need_cols)

    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0

    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].fillna("other").astype(str)

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    ordered = [TEXT_COLUMN] + NUMERIC_FEATURES + [LABEL_COLUMN]
    ordered += [c for c in DIAGNOSTIC_COLUMNS if c in df.columns and c not in ordered]
    return df[ordered]


def _stratified_limit(df: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    """Stratified sampling for quick runs. max_samples<=0 means no limit."""
    if max_samples is None or max_samples <= 0 or len(df) <= max_samples:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    value_counts = df[LABEL_COLUMN].value_counts()
    pieces = []

    for label, count in value_counts.items():
        group = df[df[LABEL_COLUMN] == label]
        # 至少保留 1 条；按原类别比例抽样。
        n = max(1, int(round(max_samples * count / len(df))))
        n = min(n, len(group))
        pieces.append(group.sample(n=n, random_state=int(rng.integers(0, 1_000_000))))

    sampled = pd.concat(pieces, ignore_index=True)
    if len(sampled) > max_samples:
        sampled = sampled.sample(n=max_samples, random_state=seed)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # 只把 text + 数值特征送入模型，绝不使用 sample_id/topic/matched_keywords 等诊断列。
    x = df[[TEXT_COLUMN] + NUMERIC_FEATURES]
    y = df[LABEL_COLUMN].astype(str)
    return x, y


def _nonempty_set(values: pd.Series) -> set:
    return {
        str(v).strip()
        for v in values.dropna().tolist()
        if str(v).strip() and str(v).strip().lower() not in {"nan", "none"}
    }


def _group_keys(df: pd.DataFrame) -> pd.Series:
    """Use source_file + sample_id as dialogue-level key when available."""
    if "sample_id" not in df.columns:
        return pd.Series([], dtype=str)

    src = df["source_file"].fillna("").astype(str) if "source_file" in df.columns else ""
    sid = df["sample_id"].fillna("").astype(str).str.strip()
    if isinstance(src, pd.Series):
        return (src.str.strip() + "::" + sid).loc[sid != ""]
    return sid.loc[sid != ""]


def _overlap_size(a: pd.Series, b: pd.Series) -> int:
    return len(_nonempty_set(a) & _nonempty_set(b))


def check_dataset_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path,
    allow_leakage: bool,
) -> Dict:
    """
    Detect common leakage caused by row-level splitting.

    For SoulChatCorpus, one original multi-turn conversation becomes several samples.
    If the same sample_id appears in train/val/test, the model sees parts of the same
    conversation during training and evaluation, which makes baseline scores look too good.
    """
    parts = {"train": train, "val": val, "test": test}
    report: Dict = {
        "group_key": "source_file::sample_id" if all("sample_id" in p.columns for p in parts.values()) else None,
        "group_overlap": {},
        "exact_text_overlap": {},
        "has_leakage": False,
    }

    # 1) Dialogue-level overlap.
    if report["group_key"]:
        group_series = {name: _group_keys(df) for name, df in parts.items()}
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            n = _overlap_size(group_series[a], group_series[b])
            report["group_overlap"][f"{a}_{b}"] = n
            if n > 0:
                report["has_leakage"] = True

    # 2) Exact duplicate user utterances across splits.
    text_series = {name: df[TEXT_COLUMN].fillna("").astype(str) for name, df in parts.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        n = _overlap_size(text_series[a], text_series[b])
        report["exact_text_overlap"][f"{a}_{b}"] = n
        # Exact text duplicates also make evaluation optimistic, but in dialogue corpora
        # they can naturally happen. We warn about them but do not hard-stop unless
        # group leakage exists.
        if n > 0:
            report.setdefault("warnings", []).append(f"{a}_{b} 存在 {n} 条完全相同 text")

    report["message"] = (
        "OK：未发现 sample_id 对话级交叉泄漏。"
        if not report["has_leakage"]
        else "发现 sample_id 对话级交叉泄漏：请重新运行修复版 01_preprocess_soulchat.py，使用默认对话级划分。"
    )

    save_json(report, out_dir / "leakage_diagnostics.json")

    if report["has_leakage"] and not allow_leakage:
        raise RuntimeError(
            "\n[错误] 检测到 train/val/test 中存在相同 sample_id 的对话级泄漏。\n"
            "这会导致验证集/测试集 Macro-F1 虚高。\n"
            "请先替换并运行修复版 scripts/01_preprocess_soulchat.py：\n"
            "  python scripts/01_preprocess_soulchat.py --raw_dir data/raw/SoulChatCorpus\n"
            "确认 data/processed/preprocess_stats.json 中 split_group_overlaps 全部为 0 后，"
            "再运行本脚本。\n"
            "如果你只是想复现实验结果，可加 --allow_leakage 强制继续，但不建议写入报告。\n"
        )
    return report


# -----------------------------
# Model building
# -----------------------------

def _ngram_tuple(text: str) -> Tuple[int, int]:
    lo, hi = str(text).split(",")
    return int(lo), int(hi)


def make_features(n_features: int, ngram_range: Tuple[int, int]) -> ColumnTransformer:
    """
    HashingVectorizer 是核心提速点：
    - 不需要扫描全量训练集建立词表；
    - 内存稳定；
    - 对中文使用 char n-gram，不依赖分词。
    """
    text_pipe = Pipeline(
        steps=[
            (
                "hash",
                HashingVectorizer(
                    analyzer="char",
                    ngram_range=ngram_range,
                    n_features=n_features,
                    alternate_sign=False,  # 保证非负，兼容朴素贝叶斯。
                    norm=None,
                    lowercase=False,
                    dtype=np.float32,
                ),
            ),
            ("tfidf", TfidfTransformer(sublinear_tf=True)),
        ]
    )

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            # MaxAbsScaler 适合和稀疏文本特征拼接，比 StandardScaler 更省事。
            ("scaler", MaxAbsScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("text", text_pipe, TEXT_COLUMN),
            ("num", num_pipe, NUMERIC_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        n_jobs=None,
    )


def make_pipeline(clf, n_features: int, ngram_range: Tuple[int, int]) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", make_features(n_features=n_features, ngram_range=ngram_range)),
            ("clf", clf),
        ]
    )


def build_candidates(preset: str, n_jobs: int) -> List[Tuple[str, object, Tuple[int, int]]]:
    """
    返回：候选名称、分类器、字符 n-gram 范围。

    ultra_fast：最快，适合先跑通。
    fast：推荐默认，速度和效果折中。
    accurate：候选更多，但仍比原来的 RandomizedSearchCV 快很多。
    """
    common_sgd = dict(
        random_state=RANDOM_SEED,
        class_weight="balanced",
        max_iter=8,
        tol=1e-3,
        # 关闭 SGD 内部 early_stopping，避免小数据集因内部切分报错；
        # 外部已有 val.csv 负责模型选择。
        early_stopping=False,
        n_iter_no_change=3,
        n_jobs=n_jobs,
    )

    if preset == "ultra_fast":
        return [
            ("sgd_log_loss_alpha_1e-5_ngram_2_3", SGDClassifier(loss="log_loss", alpha=1e-5, **common_sgd), (2, 3)),
            ("complement_nb_alpha_0.2_ngram_2_3", ComplementNB(alpha=0.2), (2, 3)),
        ]

    if preset == "accurate":
        return [
            ("sgd_log_loss_alpha_3e-6_ngram_1_3", SGDClassifier(loss="log_loss", alpha=3e-6, **common_sgd), (1, 3)),
            ("sgd_log_loss_alpha_1e-5_ngram_2_4", SGDClassifier(loss="log_loss", alpha=1e-5, **common_sgd), (2, 4)),
            ("sgd_hinge_alpha_1e-5_ngram_2_4", SGDClassifier(loss="hinge", alpha=1e-5, **common_sgd), (2, 4)),
            ("sgd_modified_huber_alpha_1e-5_ngram_2_4", SGDClassifier(loss="modified_huber", alpha=1e-5, **common_sgd), (2, 4)),
            ("complement_nb_alpha_0.1_ngram_2_4", ComplementNB(alpha=0.1), (2, 4)),
            ("multinomial_nb_alpha_0.1_ngram_2_4", MultinomialNB(alpha=0.1), (2, 4)),
        ]

    # default: fast
    return [
        ("sgd_log_loss_alpha_1e-5_ngram_2_3", SGDClassifier(loss="log_loss", alpha=1e-5, **common_sgd), (2, 3)),
        ("sgd_hinge_alpha_1e-5_ngram_2_3", SGDClassifier(loss="hinge", alpha=1e-5, **common_sgd), (2, 3)),
        ("sgd_modified_huber_alpha_1e-5_ngram_2_3", SGDClassifier(loss="modified_huber", alpha=1e-5, **common_sgd), (2, 3)),
        ("complement_nb_alpha_0.2_ngram_2_3", ComplementNB(alpha=0.2), (2, 3)),
    ]


# -----------------------------
# Evaluation and plotting
# -----------------------------

def evaluate_model(model, x, y, labels: List[str]) -> Dict:
    pred = model.predict(x)
    return compute_metrics(y, pred, labels)


def plot_normalized_confusion_matrix(y_true, y_pred, labels: List[str], out_path: Path, title: str) -> None:
    """Save a row-normalized confusion matrix so minority classes are visible."""
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    denom = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom != 0)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), max(6, len(labels) * 0.7)))
    im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm_norm.shape[1]),
        yticks=np.arange(cm_norm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def evaluate_and_save(name: str, model, x, y, labels: List[str], out_dir: Path) -> Dict:
    pred = model.predict(x)
    metrics = compute_metrics(y, pred, labels)
    save_json(metrics, out_dir / f"{name}_metrics.json")
    plot_confusion_matrix(
        y,
        pred,
        labels,
        out_dir / f"{name}_confusion_matrix.png",
        f"{name} Confusion Matrix",
    )
    plot_normalized_confusion_matrix(
        y,
        pred,
        labels,
        out_dir / f"{name}_confusion_matrix_normalized.png",
        f"{name} Normalized Confusion Matrix",
    )
    return metrics


def make_fast_learning_curve(
    best_pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    out_path: Path,
    train_sizes: Iterable[float] = (0.25, 0.5, 0.75, 1.0),
) -> None:
    """轻量学习曲线：不用 K 折，只在验证集上看不同训练量的 Macro-F1。"""
    import matplotlib.pyplot as plt

    rows = []
    train_df = x_train.copy()
    train_df[LABEL_COLUMN] = y_train.values

    for frac in train_sizes:
        frac = float(frac)
        n = max(2, int(len(train_df) * frac))
        sub = _stratified_limit(train_df, n, RANDOM_SEED + int(frac * 1000))
        sub_x, sub_y = split_xy(sub)

        model = clone(best_pipeline)
        start = time.time()
        model.fit(sub_x, sub_y)
        pred_train = model.predict(sub_x)
        pred_val = model.predict(x_val)
        rows.append(
            {
                "n_samples": len(sub),
                "train_macro_f1": float(f1_score(sub_y, pred_train, average="macro", zero_division=0)),
                "val_macro_f1": float(f1_score(y_val, pred_val, average="macro", zero_division=0)),
                "seconds": round(time.time() - start, 3),
            }
        )

    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve_df["n_samples"], curve_df["train_macro_f1"], marker="o", label="train")
    ax.plot(curve_df["n_samples"], curve_df["val_macro_f1"], marker="o", label="val")
    ax.set_title("Fast learning curve: best baseline")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Macro-F1")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fast traditional ML baseline trainer")
    parser.add_argument("--data_dir", default=str(PROCESSED_DIR), help="processed 数据目录，需包含 train/val/test.csv")
    parser.add_argument("--model_dir", default=str(MODEL_DIR), help="模型输出目录")
    parser.add_argument("--out_dir", default=str(OUTPUT_DIR / "baseline"), help="评估结果输出目录")
    parser.add_argument("--preset", choices=["ultra_fast", "fast", "accurate"], default="fast", help="速度/效果预设")
    parser.add_argument("--n_features", type=int, default=2**19, help="HashingVectorizer 特征桶数量，越大越准但越占内存")
    parser.add_argument("--max_train_samples", type=int, default=80000, help="最多使用多少训练样本；0 表示使用全部")
    parser.add_argument("--max_val_samples", type=int, default=20000, help="最多使用多少验证样本；0 表示使用全部")
    parser.add_argument("--max_test_samples", type=int, default=0, help="最多使用多少测试样本；0 表示使用全部")
    parser.add_argument("--n_jobs", type=int, default=-1, help="CPU 并行数，-1 表示尽量用满")
    parser.add_argument("--learning_curve", action="store_true", help="额外生成轻量学习曲线；会增加训练时间")
    parser.add_argument("--no_refit_trainval", action="store_true", help="不使用 train+val 重新拟合最优模型，进一步提速")
    parser.add_argument("--allow_leakage", action="store_true", help="检测到 sample_id 跨集合泄漏时仍强制训练；仅用于复现实验，不建议用于报告")
    args = parser.parse_args()

    # 避免某些 BLAS 线程过多导致 Windows 电脑卡死；用户显式设置环境变量时尊重用户设置。
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== Loading data =====")
    train_raw = _safe_read_csv(data_dir / "train.csv")
    val_raw = _safe_read_csv(data_dir / "val.csv")
    test_raw = _safe_read_csv(data_dir / "test.csv")

    leakage_report = check_dataset_leakage(
        train_raw,
        val_raw,
        test_raw,
        out_dir=out_dir,
        allow_leakage=args.allow_leakage,
    )
    print(f"Leakage check: {leakage_report['message']}")

    train = _stratified_limit(train_raw, args.max_train_samples, RANDOM_SEED)
    val = _stratified_limit(val_raw, args.max_val_samples, RANDOM_SEED + 1)
    test = _stratified_limit(test_raw, args.max_test_samples, RANDOM_SEED + 2)

    x_train, y_train = split_xy(train)
    x_val, y_val = split_xy(val)
    x_test, y_test = split_xy(test)

    labels = sorted(pd.concat([y_train, y_val, y_test], ignore_index=True).astype(str).unique().tolist())
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,} | Labels: {labels}")
    print(f"Preset: {args.preset} | n_features: {args.n_features:,}")
    if "matched_keywords" in train.columns:
        print("[提示] 当前数据可能使用关键词弱标签；基准分数代表对弱标签任务的拟合，不等同于真实心理诊断准确率。")

    summary: Dict = {
        "mode": "fast_hashing_validation_search",
        "selection_metric": "validation_macro_f1",
        "preset": args.preset,
        "n_features": args.n_features,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "max_test_samples": args.max_test_samples,
        "leakage_diagnostics": leakage_report,
        "results": {},
    }

    # 1) Dummy baseline：作为性能下限，速度极快。
    print("\n===== Training dummy baseline =====")
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
    start = time.time()
    dummy.fit(x_train, y_train)
    dummy_seconds = round(time.time() - start, 3)
    dummy_test = evaluate_and_save("dummy_most_frequent", dummy, x_test, y_test, labels, out_dir)
    dummy_test["fit_seconds"] = dummy_seconds
    dummy_test["note"] = "most frequent label baseline"
    summary["results"]["dummy_most_frequent"] = dummy_test
    joblib.dump({"model": dummy, "labels": labels, "best_name": "dummy_most_frequent"}, model_dir / "dummy_baseline.joblib")

    # 2) 快速候选模型：只在验证集选择最优，不做慢速 CV。
    print("\n===== Fast validation search =====")
    best_name = None
    best_val_score = -1.0
    best_pipeline = None
    candidates = build_candidates(args.preset, args.n_jobs)

    for name, clf, ngram_range in candidates:
        print(f"\n--- Candidate: {name} ---")
        pipe = make_pipeline(clf, n_features=args.n_features, ngram_range=ngram_range)
        start = time.time()
        pipe.fit(x_train, y_train)
        fit_seconds = round(time.time() - start, 3)

        val_metrics = evaluate_model(pipe, x_val, y_val, labels)
        test_metrics = evaluate_model(pipe, x_test, y_test, labels)
        val_macro = val_metrics["macro_f1"]
        test_macro = test_metrics["macro_f1"]

        print(f"fit_seconds={fit_seconds} | val_macro_f1={val_macro:.4f} | test_macro_f1={test_macro:.4f}")

        result = dict(test_metrics)
        result["fit_seconds"] = fit_seconds
        result["val_accuracy"] = val_metrics["accuracy"]
        result["val_macro_f1"] = val_metrics["macro_f1"]
        result["val_weighted_f1"] = val_metrics["weighted_f1"]
        result["ngram_range"] = list(ngram_range)
        result["classifier"] = clf.__class__.__name__
        summary["results"][name] = result
        save_json(result, out_dir / f"{name}_metrics.json")

        if val_macro > best_val_score:
            best_val_score = val_macro
            best_name = name
            best_pipeline = pipe

    if best_pipeline is None or best_name is None:
        raise RuntimeError("没有成功训练任何候选模型。")

    # 3) 最优模型：可选 train+val 重训。默认开启，效果更稳；要极限提速可加 --no_refit_trainval。
    print("\n===== Final best baseline =====")
    final_model = best_pipeline
    final_fit_seconds = 0.0
    if not args.no_refit_trainval:
        trainval = pd.concat([train, val], ignore_index=True)
        x_trainval, y_trainval = split_xy(trainval)
        final_model = clone(best_pipeline)
        start = time.time()
        final_model.fit(x_trainval, y_trainval)
        final_fit_seconds = round(time.time() - start, 3)
        print(f"Refit on train+val finished: {final_fit_seconds}s")
    else:
        print("Skip refit: using the model fitted on train only.")

    best_test_metrics = evaluate_and_save("best_baseline", final_model, x_test, y_test, labels, out_dir)
    best_test_metrics["selected_from"] = best_name
    best_test_metrics["best_val_macro_f1"] = float(best_val_score)
    best_test_metrics["final_refit_seconds"] = final_fit_seconds

    # 兼容旧版 04_evaluate.py：best_macro_f1 表示最终测试集 Macro-F1。
    summary["best_model"] = best_name
    summary["best_macro_f1"] = best_test_metrics["macro_f1"]
    summary["best_test_metrics"] = best_test_metrics

    joblib.dump(
        {
            "model": final_model,
            "labels": labels,
            "best_name": best_name,
            "metadata": {
                "mode": summary["mode"],
                "preset": args.preset,
                "n_features": args.n_features,
                "selected_by": "validation_macro_f1",
                "best_val_macro_f1": float(best_val_score),
                "test_macro_f1": best_test_metrics["macro_f1"],
            },
        },
        model_dir / "best_baseline.joblib",
    )

    # 4) 图表：候选模型验证集对比 + 最终测试集混淆矩阵。
    val_bars = {
        name: item.get("val_macro_f1", item.get("macro_f1", 0.0))
        for name, item in summary["results"].items()
    }
    plot_bar(val_bars, out_dir / "model_macro_f1_comparison.png", "Fast baseline validation Macro-F1", "Macro-F1")

    if args.learning_curve:
        print("\n===== Building fast learning curve =====")
        make_fast_learning_curve(
            final_model,
            x_train,
            y_train,
            x_val,
            y_val,
            out_dir / "best_learning_curve.png",
        )
    else:
        print("\nSkip learning curve by default. Add --learning_curve if the report needs it.")

    save_json(summary, out_dir / "baseline_summary.json")
    print("\n===== Done =====")
    print(json.dumps(
        {
            "best_model": summary["best_model"],
            "best_val_macro_f1": best_test_metrics["best_val_macro_f1"],
            "best_test_macro_f1": summary["best_macro_f1"],
            "saved_model": str(model_dir / "best_baseline.joblib"),
            "saved_summary": str(out_dir / "baseline_summary.json"),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
