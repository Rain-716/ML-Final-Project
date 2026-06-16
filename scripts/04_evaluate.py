#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rich final evaluation and visualization for the Chinese AI psychological consultation project.

替换 scripts/04_evaluate.py 后运行：
    python scripts/04_evaluate.py

它会在 outputs/final/ 下生成更完整的评估表格、图表和 Markdown 索引，尽量复用
02_train_baselines.py 与 03_train_bert_gpu.py 已经保存的 JSON/CSV/PNG，不强制重新训练。
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import LABEL_COLUMN, OUTPUT_DIR, PROCESSED_DIR  # type: ignore
except Exception:
    LABEL_COLUMN = "label"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# -----------------------------
# Basic IO helpers
# -----------------------------

def setup_matplotlib() -> None:
    """Make saved figures readable on Windows and when Chinese labels appear."""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.autolayout"] = False


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        value = float(x)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def sanitize_name(name: str) -> str:
    out = []
    for ch in str(name):
        if ch.isalnum() or ch in "_-.":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")[:120] or "model"


def metric_triplet(item: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        "accuracy": safe_float(item.get("accuracy")),
        "macro_f1": safe_float(item.get("macro_f1")),
        "weighted_f1": safe_float(item.get("weighted_f1")),
    }


def detect_existing_pngs(*dirs: Path) -> List[Path]:
    files: List[Path] = []
    for d in dirs:
        if d.exists():
            files.extend(sorted(d.glob("*.png")))
    return files


# -----------------------------
# Data extraction
# -----------------------------

def extract_baseline_rows(baseline: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}

    best_model = baseline.get("best_model")
    best_metrics = baseline.get("best_test_metrics", {}) or {}
    if not best_metrics and best_model in baseline.get("results", {}):
        best_metrics = baseline["results"][best_model]

    if best_metrics:
        rows.append({
            "model_key": "best_traditional_ml",
            "display_name": f"Best traditional ML\n({best_model})" if best_model else "Best traditional ML",
            "family": "traditional_ml_best",
            "is_selected": True,
            "source_file": "outputs/baseline/baseline_summary.json",
            **metric_triplet(best_metrics),
            "fit_seconds": safe_float(best_metrics.get("final_refit_seconds") or best_metrics.get("fit_seconds")),
            "val_accuracy": safe_float(best_metrics.get("val_accuracy")),
            "val_macro_f1": safe_float(best_metrics.get("best_val_macro_f1") or best_metrics.get("val_macro_f1")),
            "val_weighted_f1": safe_float(best_metrics.get("val_weighted_f1")),
        })
        summary["best_traditional_ml"] = {
            "model": best_model,
            **metric_triplet(best_metrics),
            "selection_metric": baseline.get("selection_metric"),
            "n_features": baseline.get("n_features"),
            "preset": baseline.get("preset"),
        }

    for name, item in (baseline.get("results", {}) or {}).items():
        if not isinstance(item, dict):
            continue
        rows.append({
            "model_key": f"baseline::{name}",
            "display_name": name,
            "family": "traditional_ml_candidate",
            "is_selected": name == best_model,
            "source_file": "outputs/baseline/baseline_summary.json",
            **metric_triplet(item),
            "fit_seconds": safe_float(item.get("fit_seconds")),
            "val_accuracy": safe_float(item.get("val_accuracy")),
            "val_macro_f1": safe_float(item.get("val_macro_f1")),
            "val_weighted_f1": safe_float(item.get("val_weighted_f1")),
            "classifier": item.get("classifier"),
        })

    if baseline.get("selection_metric"):
        summary["baseline_selection_metric"] = baseline.get("selection_metric")
    if baseline.get("max_train_samples") is not None:
        summary["baseline_max_train_samples"] = baseline.get("max_train_samples")
    if baseline.get("n_features") is not None:
        summary["baseline_n_features"] = baseline.get("n_features")
    return rows, summary


def extract_bert_rows(bert: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    tm = bert.get("test_metrics", {}) or {}
    if tm:
        rows.append({
            "model_key": "bert_gpu",
            "display_name": "BERT / MacBERT GPU",
            "family": "bert_gpu",
            "is_selected": True,
            "source_file": "outputs/bert/bert_summary.json",
            **metric_triplet(tm),
            "fit_seconds": None,
        })
        summary["bert_gpu"] = metric_triplet(tm)
    if bert.get("history"):
        hist = bert.get("history") or []
        summary["bert_epochs"] = len(hist)
        if hist:
            summary["bert_best_val_macro_f1"] = max(safe_float(h.get("val_macro_f1")) or 0.0 for h in hist)
            summary["bert_total_train_seconds"] = sum(safe_float(h.get("epoch_seconds")) or 0.0 for h in hist)
    return rows, summary


def per_class_to_df(metrics: Dict[str, Any], model_key: str, display_name: str) -> pd.DataFrame:
    per = metrics.get("per_class", {}) or {}
    rows = []
    for label, vals in per.items():
        if not isinstance(vals, dict):
            continue
        rows.append({
            "model_key": model_key,
            "display_name": display_name,
            "label": str(label),
            "precision": safe_float(vals.get("precision")),
            "recall": safe_float(vals.get("recall")),
            "f1": safe_float(vals.get("f1")),
            "support": safe_float(vals.get("support")),
        })
    return pd.DataFrame(rows)


def collect_per_class_tables(out: Path, baseline: Optional[Dict[str, Any]], bert: Optional[Dict[str, Any]]) -> pd.DataFrame:
    pieces = []
    if baseline:
        best_model = baseline.get("best_model")
        best_metrics = baseline.get("best_test_metrics", {}) or {}
        if not best_metrics and best_model in baseline.get("results", {}):
            best_metrics = baseline["results"][best_model]
        if best_metrics:
            pieces.append(per_class_to_df(best_metrics, "best_traditional_ml", f"Best traditional ML ({best_model})"))
        # Candidate models are useful for the baseline family heatmap / per-class comparison.
        for name, item in (baseline.get("results", {}) or {}).items():
            if isinstance(item, dict) and item.get("per_class"):
                pieces.append(per_class_to_df(item, f"baseline::{name}", name))

    if bert:
        tm = bert.get("test_metrics", {}) or {}
        if tm:
            pieces.append(per_class_to_df(tm, "bert_gpu", "BERT / MacBERT GPU"))

    if pieces:
        return pd.concat(pieces, ignore_index=True)
    return pd.DataFrame(columns=["model_key", "display_name", "label", "precision", "recall", "f1", "support"])


def read_label_distribution(data_dir: Path) -> pd.DataFrame:
    rows = []
    for split in ["train", "val", "test"]:
        path = data_dir / f"{split}.csv"
        if not path.exists():
            continue
        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
            label_col = LABEL_COLUMN if LABEL_COLUMN in header else ("label" if "label" in header else None)
            if label_col is None:
                continue
            vc = pd.read_csv(path, usecols=[label_col])[label_col].astype(str).value_counts().sort_index()
            for label, count in vc.items():
                rows.append({"split": split, "label": str(label), "count": int(count)})
        except Exception as e:
            rows.append({"split": split, "label": f"读取失败：{e}", "count": 0})
    return pd.DataFrame(rows)


# -----------------------------
# Plot helpers
# -----------------------------

def save_no_data_plot(path: Path, title: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def annotate_bars(ax, values: Iterable[float], fmt: str = "{:.3f}", horizontal: bool = False) -> None:
    for patch, val in zip(ax.patches, values):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if horizontal:
            ax.text(patch.get_width() + 0.006, patch.get_y() + patch.get_height() / 2, fmt.format(val), va="center", fontsize=8)
        else:
            ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 0.006, fmt.format(val), ha="center", fontsize=8)


def plot_horizontal_bar(df: pd.DataFrame, label_col: str, value_col: str, path: Path, title: str, xlabel: str, top_n: Optional[int] = None, xlim: Optional[Tuple[float, float]] = None) -> None:
    tmp = df[[label_col, value_col]].dropna().copy()
    if tmp.empty:
        save_no_data_plot(path, title, f"没有可用的 {value_col} 数据")
        return
    tmp[value_col] = tmp[value_col].astype(float)
    tmp = tmp.sort_values(value_col, ascending=True)
    if top_n and len(tmp) > top_n:
        tmp = tmp.tail(top_n)
    fig_h = max(4.5, 0.45 * len(tmp) + 1.2)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(tmp[label_col].astype(str), tmp[value_col])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis="x", alpha=0.25)
    annotate_bars(ax, tmp[value_col].tolist(), horizontal=True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_grouped_core_metrics(df: pd.DataFrame, path: Path) -> None:
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    tmp = df[df["family"].isin(["traditional_ml_best", "bert_gpu"])].copy()
    if tmp.empty:
        save_no_data_plot(path, "核心指标对比", "没有找到传统机器学习/BERT 的最终指标")
        return
    tmp = tmp.drop_duplicates("model_key", keep="first")
    labels = tmp["display_name"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.2), 5.2))
    for i, metric in enumerate(metrics):
        vals = [safe_float(v) or 0.0 for v in tmp[metric].tolist()]
        ax.bar(x + (i - 1) * width, vals, width, label=metric)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("最终模型核心指标对比")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_metric_heatmap(df: pd.DataFrame, path: Path, title: str) -> None:
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    tmp = df.dropna(subset=["macro_f1"]).copy()
    if tmp.empty:
        save_no_data_plot(path, title, "没有可用的模型指标")
        return
    # Avoid duplicated best/candidate row with same display name becoming too noisy.
    tmp = tmp.sort_values("macro_f1", ascending=False).head(20)
    matrix = tmp[metrics].fillna(0).astype(float).to_numpy()
    fig_h = max(4.5, len(tmp) * 0.5 + 1.2)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    ax.set_title(title)
    ax.set_yticks(np.arange(len(tmp)))
    ax.set_yticklabels(tmp["display_name"].astype(str).tolist(), fontsize=8)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] > 0.55 else "black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_per_class_metrics(per_df: pd.DataFrame, model_key: str, path_prefix: Path) -> List[Path]:
    paths: List[Path] = []
    tmp = per_df[per_df["model_key"] == model_key].copy()
    if tmp.empty:
        return paths
    display = str(tmp["display_name"].iloc[0])
    tmp = tmp.sort_values("f1", ascending=True)

    # 1) per-class F1 ranking
    path = path_prefix.with_name(path_prefix.name + "_per_class_f1.png")
    plot_horizontal_bar(tmp, "label", "f1", path, f"{display}：各类别 F1", "F1", xlim=(0, 1.02))
    paths.append(path)

    # 2) grouped precision/recall/F1 per class
    path = path_prefix.with_name(path_prefix.name + "_per_class_prf_grouped.png")
    labels = tmp["label"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.8), 5.2))
    for idx, metric in enumerate(["precision", "recall", "f1"]):
        vals = tmp[metric].fillna(0).astype(float).tolist()
        ax.bar(x + (idx - 1) * width, vals, width, label=metric)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{display}：Precision / Recall / F1 分类别对比")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    # 3) support distribution for this model/test set
    path = path_prefix.with_name(path_prefix.name + "_class_support.png")
    plot_horizontal_bar(tmp.sort_values("support", ascending=True), "label", "support", path, f"{display}：测试集类别样本数", "Support")
    paths.append(path)

    # 4) support vs f1 scatter
    path = path_prefix.with_name(path_prefix.name + "_support_vs_f1.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    xvals = tmp["support"].fillna(0).astype(float)
    yvals = tmp["f1"].fillna(0).astype(float)
    ax.scatter(xvals, yvals, s=60)
    for _, row in tmp.iterrows():
        ax.annotate(str(row["label"]), (float(row["support"]), float(row["f1"])), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Support（log scale）")
    ax.set_ylabel("F1")
    ax.set_title(f"{display}：类别样本量与 F1 关系")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    return paths


def plot_dataset_distribution(dist: pd.DataFrame, path: Path) -> None:
    if dist.empty:
        save_no_data_plot(path, "数据集类别分布", "没有找到 data/processed/train/val/test.csv 或标签列")
        return
    labels = sorted(dist["label"].astype(str).unique().tolist())
    splits = [s for s in ["train", "val", "test"] if s in set(dist["split"])]
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(splits))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.85), 5.6))
    for i, split in enumerate(splits):
        counts = []
        sub = dist[dist["split"] == split].set_index("label")
        for lab in labels:
            counts.append(int(sub.loc[lab, "count"]) if lab in sub.index else 0)
        ax.bar(x + (i - (len(splits)-1)/2) * width, counts, width, label=split)
    ax.set_yscale("log")
    ax.set_ylabel("Count（log scale）")
    ax.set_title("Train / Val / Test 类别分布")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_learning_curve(csv_path: Path, out_path: Path) -> None:
    if not csv_path.exists():
        save_no_data_plot(out_path, "学习曲线", "没有找到 best_learning_curve.csv；运行 02_train_baselines.py 时加 --learning_curve 可生成")
        return
    df = pd.read_csv(csv_path)
    if not {"n_samples", "train_macro_f1", "val_macro_f1"}.issubset(df.columns):
        save_no_data_plot(out_path, "学习曲线", f"{csv_path.name} 缺少 n_samples/train_macro_f1/val_macro_f1 列")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["n_samples"], df["train_macro_f1"], marker="o", label="Train Macro-F1")
    ax.plot(df["n_samples"], df["val_macro_f1"], marker="o", label="Val Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Macro-F1")
    ax.set_title("传统机器学习最佳模型学习曲线")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_learning_curve_gap(csv_path: Path, out_path: Path) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if not {"n_samples", "train_macro_f1", "val_macro_f1"}.issubset(df.columns):
        return
    df["generalization_gap"] = df["train_macro_f1"] - df["val_macro_f1"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["n_samples"], df["generalization_gap"], marker="o")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Train Macro-F1 - Val Macro-F1")
    ax.set_title("传统机器学习泛化差距")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_bert_history(bert_summary: Dict[str, Any], bert_dir: Path, eval_dir: Path) -> List[Path]:
    paths: List[Path] = []
    history = bert_summary.get("history") if bert_summary else None
    hist_df = None
    if history:
        hist_df = pd.DataFrame(history)
    elif (bert_dir / "bert_training_history.csv").exists():
        hist_df = pd.read_csv(bert_dir / "bert_training_history.csv")
    if hist_df is None or hist_df.empty:
        return paths

    hist_df.to_csv(eval_dir / "bert_training_history_copy.csv", index=False, encoding="utf-8-sig")

    if {"epoch", "train_loss"}.issubset(hist_df.columns):
        path = eval_dir / "bert_training_loss_curve.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hist_df["epoch"], hist_df["train_loss"], marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("BERT / MacBERT 训练损失曲线")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    if {"epoch", "val_macro_f1"}.issubset(hist_df.columns):
        path = eval_dir / "bert_val_macro_f1_curve.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hist_df["epoch"], hist_df["val_macro_f1"], marker="o")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Macro-F1")
        ax.set_title("BERT / MacBERT 验证集 Macro-F1 曲线")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    if {"epoch", "epoch_seconds"}.issubset(hist_df.columns):
        path = eval_dir / "bert_epoch_time_bar.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(hist_df["epoch"].astype(str), hist_df["epoch_seconds"].astype(float) / 60.0)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Minutes")
        ax.set_title("BERT / MacBERT 每轮训练耗时")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    return paths


def short_model_name(name: Any) -> str:
    """把很长的模型名压缩成适合画图标注的短名称。"""
    text = str(name)
    mapping = {
        "dummy_most_frequent": "Dummy",
        "complement_nb_alpha_0.2_ngram_2_3": "ComplementNB",
        "sgd_log_loss_alpha_1e-5_ngram_2_3": "SGD-LogLoss",
        "sgd_hinge_alpha_1e-5_ngram_2_3": "SGD-Hinge",
        "sgd_modified_huber_alpha_1e-5_ngram_2_3": "SGD-Huber",
    }
    if text in mapping:
        return mapping[text]
    if text.startswith("Best traditional ML"):
        return "Best traditional ML"
    text = text.replace("baseline::", "")
    text = text.replace("sgd_modified_huber", "SGD-Huber")
    text = text.replace("sgd_log_loss", "SGD-LogLoss")
    text = text.replace("sgd_hinge", "SGD-Hinge")
    text = text.replace("complement_nb", "ComplementNB")
    text = text.replace("dummy_most_frequent", "Dummy")
    text = text.replace("_alpha_1e-5_ngram_2_3", "")
    text = text.replace("_alpha_0.2_ngram_2_3", "")
    text = text.replace("\n", " ")
    return text[:28]


def add_metric_validity_notes(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """不改动真实指标，只额外添加报告解释字段，避免把弱标签高分误解为真实诊断能力。"""
    if metrics_df is None or metrics_df.empty:
        return metrics_df
    df = metrics_df.copy()
    notes = []
    flags = []
    for _, row in df.iterrows():
        macro = safe_float(row.get("macro_f1"))
        acc = safe_float(row.get("accuracy"))
        family = str(row.get("family", ""))
        note_parts = []
        flag = "ok"
        if macro is not None and macro >= 0.98:
            flag = "very_high_metric"
            note_parts.append("指标接近满分；若使用关键词弱标签，应解释为弱标签拟合效果，不代表真实心理诊断准确率")
        elif macro is not None and macro >= 0.94 and "traditional" in family:
            flag = "high_metric"
            note_parts.append("传统模型指标很高；通常说明标签规则/关键词特征较强，报告中需说明任务边界")
        if acc is not None and macro is not None and (acc - macro) > 0.05:
            flag = "class_imbalance_sensitive" if flag == "ok" else flag
            note_parts.append("Accuracy 明显高于 Macro-F1，类别不均衡影响较大")
        if not note_parts:
            note_parts.append("指标正常记录，未做人工改值")
        notes.append("；".join(note_parts))
        flags.append(flag)
    df["validity_flag"] = flags
    df["report_note"] = notes
    return df


def plot_speed_performance(metrics_df: pd.DataFrame, path: Path) -> None:
    """画候选传统模型耗时-性能散点图。

    修复点：
    1) 排除 best_traditional_ml 这类“最终重训汇总行”，因为它和候选模型的 fit_seconds 含义不同，且会和候选点重叠。
    2) 使用短标签、错位标注和更大的画布，避免长模型名糊成一团。
    3) 标题明确说明 fit_seconds 是脚本记录的候选模型拟合耗时，仅供比较，不代表完整项目训练耗时。
    """
    tmp = metrics_df.dropna(subset=["fit_seconds", "macro_f1"]).copy()
    if tmp.empty:
        save_no_data_plot(path, "速度与性能关系", "没有找到 fit_seconds 字段；新版 02_train_baselines.py 会保存训练耗时")
        return

    # best_traditional_ml 是最终重训后的汇总行，不属于候选模型搜索；排除可减少重复点和语义混淆。
    if "family" in tmp.columns:
        tmp = tmp[tmp["family"].astype(str).eq("traditional_ml_candidate")].copy()
    if tmp.empty:
        save_no_data_plot(path, "速度与性能关系", "没有可比较的传统机器学习候选模型")
        return

    tmp["short_name"] = tmp["display_name"].map(short_model_name)
    tmp["fit_seconds"] = tmp["fit_seconds"].astype(float)
    tmp["macro_f1"] = tmp["macro_f1"].astype(float)
    tmp = tmp.sort_values("fit_seconds")

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.scatter(tmp["fit_seconds"], tmp["macro_f1"], s=90, alpha=0.9)

    # 交错标注，尽量避免点聚集时文字重叠。
    offsets = [(9, 12), (9, -18), (-82, 12), (-92, -18), (12, 28), (-95, 28), (12, -32)]
    for i, (_, row) in enumerate(tmp.iterrows()):
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            row["short_name"],
            (row["fit_seconds"], row["macro_f1"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.82),
            arrowprops=dict(arrowstyle="-", lw=0.6, alpha=0.55),
        )

    xmin, xmax = tmp["fit_seconds"].min(), tmp["fit_seconds"].max()
    xpad = max((xmax - xmin) * 0.22, 0.25)
    ax.set_xlim(max(0, xmin - xpad), xmax + xpad)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Fit seconds recorded by baseline script")
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("传统机器学习候选模型：拟合耗时 vs Macro-F1（仅供候选模型间比较）")
    ax.text(
        0.01, -0.18,
        "说明：fit_seconds 只记录传统 ML 候选模型的脚本拟合耗时，不含数据下载、预处理、BERT 训练和可视化时间；不要与 GPU 训练总时长直接比较。",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    ax.grid(alpha=0.25)
    fig.subplots_adjust(bottom=0.24, right=0.96, left=0.08, top=0.88)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_fit_seconds_bar(metrics_df: pd.DataFrame, path: Path) -> None:
    """额外生成一个更适合报告展示的耗时条形图，避免散点标注拥挤。"""
    tmp = metrics_df.dropna(subset=["fit_seconds"]).copy()
    if "family" in tmp.columns:
        tmp = tmp[tmp["family"].astype(str).eq("traditional_ml_candidate")].copy()
    if tmp.empty:
        return
    tmp["short_name"] = tmp["display_name"].map(short_model_name)
    tmp["fit_seconds"] = tmp["fit_seconds"].astype(float)
    tmp = tmp.sort_values("fit_seconds", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.55 * len(tmp) + 1.5)))
    y = np.arange(len(tmp))
    ax.barh(y, tmp["fit_seconds"])
    ax.set_yticks(y)
    ax.set_yticklabels(tmp["short_name"])
    ax.set_xlabel("Fit seconds recorded by baseline script")
    ax.set_title("传统机器学习候选模型拟合耗时（脚本记录）")
    for i, v in enumerate(tmp["fit_seconds"]):
        ax.text(v + max(tmp["fit_seconds"].max() * 0.015, 0.02), i, f"{v:.2f}s", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_macro_weighted_gap(metrics_df: pd.DataFrame, path: Path) -> None:
    tmp = metrics_df.dropna(subset=["macro_f1", "weighted_f1"]).copy()
    if tmp.empty:
        return
    tmp["weighted_minus_macro"] = tmp["weighted_f1"].astype(float) - tmp["macro_f1"].astype(float)
    plot_horizontal_bar(
        tmp.sort_values("weighted_minus_macro"),
        "display_name",
        "weighted_minus_macro",
        path,
        "Weighted-F1 与 Macro-F1 差距（类别不均衡敏感性）",
        "Weighted-F1 - Macro-F1",
        top_n=20,
    )



def df_to_markdown(df: pd.DataFrame, max_rows: int = 30) -> str:
    """Small dependency-free Markdown table writer; avoids requiring tabulate."""
    if df is None or df.empty:
        return "无数据。"
    tmp = df.head(max_rows).copy()
    cols = [str(c) for c in tmp.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in tmp.iterrows():
        vals = []
        for c in tmp.columns:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"\n（仅展示前 {max_rows} 行，共 {len(df)} 行。）")
    return "\n".join(lines)


def write_visualization_report(eval_dir: Path, summary: Dict[str, Any], metrics_df: pd.DataFrame, per_df: pd.DataFrame, generated: List[Path]) -> None:
    lines = [
        "# 最终评估与可视化索引",
        "",
        "本文件由 `scripts/04_evaluate.py` 自动生成，用于课程报告/答辩时快速定位图表。",
        "",
        "## 1. 核心结论摘要",
        "",
    ]
    if not metrics_df.empty:
        compact = metrics_df[["model_key", "display_name", "family", "accuracy", "macro_f1", "weighted_f1"]].dropna(how="all")
        lines.append(df_to_markdown(compact))
    else:
        lines.append("未找到模型指标。")

    lines += ["", "## 2. 每类表现最低的类别", ""]
    best_like = per_df[per_df["model_key"].isin(["best_traditional_ml", "bert_gpu"])].copy()
    if not best_like.empty:
        worst = best_like.dropna(subset=["f1"]).sort_values(["model_key", "f1"]).groupby("model_key").head(3)
        lines.append(df_to_markdown(worst[["model_key", "label", "precision", "recall", "f1", "support"]]))
    else:
        lines.append("未找到 per_class 指标。")

    lines += [
        "",
        "## 3. 已生成图表",
        "",
        "| 文件 | 用途 |",
        "|---|---|",
    ]
    use_hint = {
        "final_core_metrics_grouped_bar.png": "传统 ML 与 BERT 的 Accuracy/Macro-F1/Weighted-F1 对比",
        "all_models_macro_f1_ranking.png": "所有候选模型 Macro-F1 排名",
        "all_models_metric_heatmap.png": "多模型多指标热力图",
        "baseline_learning_curve_enhanced.png": "学习曲线，展示样本量增加后的训练/验证表现",
        "baseline_generalization_gap.png": "训练-验证差距，辅助说明过拟合风险",
        "dataset_label_distribution.png": "训练/验证/测试类别分布，说明样本不均衡",
        "macro_weighted_f1_gap.png": "Weighted-F1 与 Macro-F1 差距，说明类别不均衡影响",
        "baseline_speed_vs_macro_f1.png": "传统 ML 候选模型拟合耗时与 Macro-F1 的关系，已使用短标签防重叠",
        "baseline_fit_seconds_bar.png": "传统 ML 候选模型拟合耗时条形图，比散点图更适合报告展示",
    }
    for p in generated:
        rel = p.relative_to(eval_dir) if p.is_relative_to(eval_dir) else p.name
        hint = use_hint.get(p.name, "模型评估可视化")
        lines.append(f"| `{rel}` | {hint} |")

    lines += [
        "",
        "## 4. 指标异常与展示处理说明",
        "",
        "- 本脚本不会修改 Accuracy、Macro-F1、Weighted-F1 等真实评估数值，只会增加 `validity_flag` 和 `report_note` 字段帮助解释。",
        "- 如果 BERT/MacBERT 指标接近 1.0，而本项目使用的是关键词弱标签，则应表述为：模型很好地拟合了弱标签分类任务，不能表述为真实心理诊断准确率接近 100%。",
        "- `baseline_speed_vs_macro_f1.png` 中的 fit_seconds 仅代表传统 ML 候选模型在脚本中的拟合耗时，不包含数据下载、预处理、BERT GPU 训练和最终评估时间。",
        "- 若散点图仍显拥挤，报告中优先使用 `baseline_fit_seconds_bar.png` 和 `all_models_macro_f1_ranking.png`。",
        "",
        "## 5. 报告写作提醒",
        "",
        "- 分类任务不要只汇报 Accuracy，应同时汇报 Macro-F1、Weighted-F1、Precision、Recall。",
        "- 如果 Weighted-F1 明显高于 Macro-F1，说明大类样本对总体指标影响较大，需要说明类别不均衡。",
        "- 如果训练 Macro-F1 远高于验证 Macro-F1，说明可能存在过拟合，需要在局限性中说明。",
        "- 高风险类别的召回率比普通类别更重要；报告中应单独解释 high_risk 的 Recall。",
    ]
    (eval_dir / "visualization_report.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="生成更丰富的最终评估图表与表格")
    parser.add_argument("--out_dir", default=str(OUTPUT_DIR), help="总输出目录，默认 outputs")
    parser.add_argument("--data_dir", default=str(PROCESSED_DIR), help="processed 数据目录，用于画类别分布")
    parser.add_argument("--eval_subdir", default="final", help="最终评估输出子目录")
    parser.add_argument("--top_n", type=int, default=20, help="排名图最多显示多少个模型")
    parser.add_argument("--copy_existing_png", action="store_true", help="把 baseline/bert 已有 PNG 复制到 outputs/final/existing_assets")
    args = parser.parse_args()

    setup_matplotlib()

    out = Path(args.out_dir)
    baseline_dir = out / "baseline"
    bert_dir = out / "bert"
    eval_dir = out / args.eval_subdir
    eval_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_json(baseline_dir / "baseline_summary.json")
    bert = load_json(bert_dir / "bert_summary.json")

    if not baseline and not bert:
        raise FileNotFoundError(
            "没有找到训练结果。请先运行 02_train_baselines.py 和/或 03_train_bert_gpu.py。\n"
            "需要的文件示例：outputs/baseline/baseline_summary.json 或 outputs/bert/bert_summary.json"
        )

    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}

    if baseline:
        b_rows, b_summary = extract_baseline_rows(baseline)
        rows.extend(b_rows)
        summary.update(b_summary)
        leakage = load_json(baseline_dir / "leakage_diagnostics.json")
        if leakage:
            summary["baseline_leakage_diagnostics"] = leakage

    if bert:
        bert_rows, bert_summary = extract_bert_rows(bert)
        rows.extend(bert_rows)
        summary.update(bert_summary)
        if bert.get("leakage_diagnostics"):
            summary["bert_leakage_diagnostics"] = bert.get("leakage_diagnostics")
        leakage = load_json(bert_dir / "leakage_diagnostics.json")
        if leakage:
            summary["bert_leakage_diagnostics_file"] = leakage

    metrics_df = pd.DataFrame(rows)
    if not metrics_df.empty:
        # Stable ordering: selected/final first, then macro-F1 ranking.
        metrics_df["macro_f1_rank_value"] = metrics_df["macro_f1"].fillna(-1).astype(float)
        metrics_df = metrics_df.sort_values(["is_selected", "macro_f1_rank_value"], ascending=[False, False]).drop(columns=["macro_f1_rank_value"])
        # 只增加解释字段，不改动任何真实评估数值。
        metrics_df = add_metric_validity_notes(metrics_df)

    per_df = collect_per_class_tables(out, baseline, bert)
    dist_df = read_label_distribution(Path(args.data_dir))

    # Save raw tables.
    save_json(summary, eval_dir / "final_evaluation_summary.json")
    metrics_df.to_csv(eval_dir / "final_metrics_table.csv", index=False, encoding="utf-8-sig")
    if not per_df.empty:
        per_df.to_csv(eval_dir / "per_class_metrics_table.csv", index=False, encoding="utf-8-sig")
    if not dist_df.empty:
        dist_df.to_csv(eval_dir / "dataset_label_distribution.csv", index=False, encoding="utf-8-sig")

    generated: List[Path] = []

    # Core model-level figures.
    p = eval_dir / "final_core_metrics_grouped_bar.png"
    plot_grouped_core_metrics(metrics_df, p)
    generated.append(p)

    if not metrics_df.empty:
        p = eval_dir / "all_models_macro_f1_ranking.png"
        plot_horizontal_bar(metrics_df, "display_name", "macro_f1", p, "所有模型 Macro-F1 排名", "Macro-F1", top_n=args.top_n, xlim=(0, 1.02))
        generated.append(p)

        p = eval_dir / "all_models_accuracy_ranking.png"
        plot_horizontal_bar(metrics_df, "display_name", "accuracy", p, "所有模型 Accuracy 排名", "Accuracy", top_n=args.top_n, xlim=(0, 1.02))
        generated.append(p)

        p = eval_dir / "all_models_metric_heatmap.png"
        plot_metric_heatmap(metrics_df, p, "所有模型核心指标热力图")
        generated.append(p)

        p = eval_dir / "baseline_speed_vs_macro_f1.png"
        trad_df = metrics_df[metrics_df["family"].str.contains("traditional", na=False)]
        plot_speed_performance(trad_df, p)
        generated.append(p)

        p = eval_dir / "baseline_fit_seconds_bar.png"
        plot_fit_seconds_bar(trad_df, p)
        if p.exists():
            generated.append(p)

        p = eval_dir / "macro_weighted_f1_gap.png"
        plot_macro_weighted_gap(metrics_df, p)
        generated.append(p)

    # Per-class figures for final models.
    for model_key, prefix in [("best_traditional_ml", "best_traditional_ml"), ("bert_gpu", "bert_gpu")]:
        generated.extend(plot_per_class_metrics(per_df, model_key, eval_dir / prefix))

    # Dataset distribution.
    p = eval_dir / "dataset_label_distribution.png"
    plot_dataset_distribution(dist_df, p)
    generated.append(p)

    # Learning/training curves.
    p = eval_dir / "baseline_learning_curve_enhanced.png"
    plot_learning_curve(baseline_dir / "best_learning_curve.csv", p)
    generated.append(p)

    p = eval_dir / "baseline_generalization_gap.png"
    plot_learning_curve_gap(baseline_dir / "best_learning_curve.csv", p)
    if p.exists():
        generated.append(p)

    if bert:
        generated.extend(plot_bert_history(bert, bert_dir, eval_dir))

    # Copy existing confusion matrices and other generated assets for a single report folder.
    if args.copy_existing_png:
        asset_dir = eval_dir / "existing_assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        for img in detect_existing_pngs(baseline_dir, bert_dir):
            dest = asset_dir / img.name
            try:
                shutil.copy2(img, dest)
                generated.append(dest)
            except Exception:
                pass

    write_visualization_report(eval_dir, summary, metrics_df, per_df, generated)

    # Console summary.
    print("\n[OK] 最终评估和可视化已生成：")
    print(f"  {eval_dir.resolve()}")
    print("\n关键输出：")
    key_files = [
        "final_evaluation_summary.json",
        "final_metrics_table.csv",
        "per_class_metrics_table.csv",
        "visualization_report.md",
        "final_core_metrics_grouped_bar.png",
        "all_models_macro_f1_ranking.png",
        "all_models_metric_heatmap.png",
        "dataset_label_distribution.png",
    ]
    for name in key_files:
        path = eval_dir / name
        if path.exists():
            print(f"  - {path}")

    if not metrics_df.empty:
        print("\n模型指标预览：")
        preview_cols = ["model_key", "accuracy", "macro_f1", "weighted_f1"]
        print(metrics_df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
