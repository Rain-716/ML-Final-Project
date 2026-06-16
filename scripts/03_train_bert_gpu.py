#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LABEL_COLUMN, MODEL_DIR, OUTPUT_DIR, PROCESSED_DIR, RANDOM_SEED, TEXT_COLUMN  # noqa: E402
from src.metrics_utils import compute_metrics, save_json  # noqa: E402


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> Dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def label_distribution(df: pd.DataFrame) -> Dict[str, int]:
    return {str(k): int(v) for k, v in df[LABEL_COLUMN].astype(str).value_counts().sort_index().items()}


def sample_df(df: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    """Stratified-ish fast sampling for debugging. 0 means keep all."""
    if not max_samples or max_samples <= 0 or len(df) <= max_samples:
        return df.reset_index(drop=True)
    parts = []
    labels = sorted(df[LABEL_COLUMN].astype(str).unique().tolist())
    per_label_min = max(1, max_samples // max(len(labels), 1) // 3)
    remain = max_samples
    for lab in labels:
        sub = df[df[LABEL_COLUMN].astype(str) == lab]
        take = min(len(sub), per_label_min)
        if take > 0:
            parts.append(sub.sample(take, random_state=seed))
            remain -= take
    already = pd.concat(parts) if parts else df.iloc[:0]
    rest = df.drop(index=already.index, errors="ignore")
    if remain > 0 and len(rest) > 0:
        parts.append(rest.sample(min(remain, len(rest)), random_state=seed))
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def compute_split_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, group_col: str = "sample_id") -> Dict[str, int | bool | str]:
    if group_col not in train.columns or group_col not in val.columns or group_col not in test.columns:
        return {"ok": False, "reason": f"missing {group_col}", "train_val": -1, "train_test": -1, "val_test": -1}
    tr = set(train[group_col].dropna().astype(str))
    va = set(val[group_col].dropna().astype(str))
    te = set(test[group_col].dropna().astype(str))
    return {
        "ok": True,
        "group_col": group_col,
        "train_groups": len(tr),
        "val_groups": len(va),
        "test_groups": len(te),
        "train_val": len(tr & va),
        "train_test": len(tr & te),
        "val_test": len(va & te),
    }


def require_no_group_leakage(data_dir: Path, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, allow_possible_leakage: bool = False) -> Dict:
    """
    Stop training when train/val/test may contain rows from the same original dialogue.
    This prevents falsely high BERT results after row-level random split.
    """
    stats = read_json(data_dir / "preprocess_stats.json")
    direct_overlap = compute_split_overlap(train, val, test, "sample_id")
    diagnostics = {"preprocess_stats_found": bool(stats), "direct_group_overlap": direct_overlap}

    stats_overlap = stats.get("split_group_overlaps", {}) if isinstance(stats, dict) else {}
    stats_says_clean = (
        stats.get("split_by_group") is True
        and int(stats_overlap.get("train_val", 999999)) == 0
        and int(stats_overlap.get("train_test", 999999)) == 0
        and int(stats_overlap.get("val_test", 999999)) == 0
    )
    direct_says_clean = (
        direct_overlap.get("ok") is True
        and direct_overlap.get("train_val") == 0
        and direct_overlap.get("train_test") == 0
        and direct_overlap.get("val_test") == 0
    )
    diagnostics["stats_says_clean"] = bool(stats_says_clean)
    diagnostics["direct_says_clean"] = bool(direct_says_clean)
    diagnostics["safe_to_train"] = bool(stats_says_clean and direct_says_clean)

    if not diagnostics["safe_to_train"] and not allow_possible_leakage:
        msg = (
            "检测到当前 data/processed 可能不是按 sample_id/原始对话分组划分的数据，"
            "继续训练会导致 train/val/test 泄漏，BERT 分数可能虚高。\n"
            "请先删除 data/processed 后重新运行修复版 01_preprocess_soulchat.py。\n"
            "若只是临时调试旧数据，可加 --allow_possible_leakage，但正式结果不要这样做。\n"
            f"diagnostics={json.dumps(diagnostics, ensure_ascii=False)}"
        )
        raise RuntimeError(msg)
    return diagnostics


class TextClsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, label2id: Dict[str, int], max_length: int):
        self.texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()
        self.labels = [label2id[str(x)] for x in df[LABEL_COLUMN].astype(str).tolist()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def run_eval(model, loader, device, id2label: List[str], out_path: Path | None = None) -> Tuple[Dict, List[str], List[str]]:
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            logits = model(**batch).logits
            pred = logits.argmax(dim=-1).detach().cpu().numpy().tolist()
            preds.extend(pred)
            golds.extend(labels.detach().cpu().numpy().tolist())
    y_true = [id2label[i] for i in golds]
    y_pred = [id2label[i] for i in preds]
    metrics = compute_metrics(y_true, y_pred, id2label)
    if out_path:
        save_json(metrics, out_path)
    return metrics, y_true, y_pred


def plot_confusion(y_true: List[str], y_pred: List[str], labels: List[str], out_path: Path, title: str, normalize: bool = False) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        plot_cm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
    else:
        plot_cm = cm

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), max(6, len(labels) * 0.72)))
    im = ax.imshow(plot_cm, vmin=0, vmax=1 if normalize else None)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = 0.5 if normalize else (plot_cm.max() / 2 if plot_cm.size else 0)
    for i in range(plot_cm.shape[0]):
        for j in range(plot_cm.shape[1]):
            text = f"{plot_cm[i, j]:.2f}" if normalize else f"{int(plot_cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center", color="white" if plot_cm[i, j] > threshold else "black", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(PROCESSED_DIR))
    parser.add_argument("--pretrained_model", default="hfl/chinese-macbert-base", help="可填 HuggingFace 名称或本地模型目录")
    parser.add_argument("--model_dir", default=str(MODEL_DIR / "bert_best"))
    parser.add_argument("--out_dir", default=str(OUTPUT_DIR / "bert"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.08)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=0, help="Windows 建议 0；Linux/服务器可设 2 或 4")
    parser.add_argument("--max_train_samples", type=int, default=0, help="调试用；0 表示全量")
    parser.add_argument("--max_val_samples", type=int, default=0, help="调试用；0 表示全量")
    parser.add_argument("--max_test_samples", type=int, default=0, help="调试用；0 表示全量")
    parser.add_argument("--allow_possible_leakage", action="store_true", help="仅临时调试旧数据使用；正式实验不要开启")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[警告] 未检测到 CUDA，将自动改用 CPU。")
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    model_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")

    leakage_diag = require_no_group_leakage(data_dir, train, val, test, args.allow_possible_leakage)
    save_json(leakage_diag, out_dir / "leakage_diagnostics.json")

    original_sizes = {"train": len(train), "val": len(val), "test": len(test)}
    train = sample_df(train, args.max_train_samples, RANDOM_SEED)
    val = sample_df(val, args.max_val_samples, RANDOM_SEED + 1)
    test = sample_df(test, args.max_test_samples, RANDOM_SEED + 2)

    labels = sorted(train[LABEL_COLUMN].astype(str).unique().tolist())
    missing_in_train = sorted((set(val[LABEL_COLUMN].astype(str)) | set(test[LABEL_COLUMN].astype(str))) - set(labels))
    if missing_in_train:
        raise ValueError(f"val/test 中存在 train 没有的类别，无法训练分类头：{missing_in_train}")
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = labels

    save_json(
        {
            "original_sizes": original_sizes,
            "used_sizes": {"train": len(train), "val": len(val), "test": len(test)},
            "labels": labels,
            "label_distribution": {
                "train": label_distribution(train),
                "val": label_distribution(val),
                "test": label_distribution(test),
            },
            "args": vars(args),
            "leakage_diagnostics": leakage_diag,
        },
        out_dir / "bert_run_config.json",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_model,
        num_labels=len(labels),
        id2label={i: label for i, label in enumerate(labels)},
        label2id=label2id,
    ).to(device)

    pin = device.type == "cuda"
    train_ds = TextClsDataset(train, tokenizer, label2id, args.max_length)
    val_ds = TextClsDataset(val, tokenizer, label2id, args.max_length)
    test_ds = TextClsDataset(test, tokenizer, label2id, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    label_counts = train[LABEL_COLUMN].astype(str).value_counts().to_dict()
    weights = torch.tensor(
        [len(train) / (len(labels) * label_counts.get(label, 1)) for label in labels],
        dtype=torch.float,
        device=device,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped_params, lr=args.lr)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum)))
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar, start=1):
            labels_t = batch.pop("labels").to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                logits = model(**batch).logits
                loss = criterion(logits, labels_t) / max(1, args.grad_accum)
            scaler.scale(loss).backward()
            losses.append(float(loss.detach().cpu()) * max(1, args.grad_accum))

            should_step = (step % max(1, args.grad_accum) == 0) or (step == len(train_loader))
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=float(np.mean(losses[-20:])))

        val_metrics, _, _ = run_eval(model, val_loader, device, id2label)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "epoch_seconds": round(time.time() - start, 3),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            (model_dir / "label2id.json").write_text(json.dumps(label2id, ensure_ascii=False, indent=2), encoding="utf-8")
            save_json(val_metrics, out_dir / "best_val_metrics.json")

    best_model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    test_metrics, y_true, y_pred = run_eval(best_model, test_loader, device, id2label, out_dir / "test_metrics.json")

    plot_confusion(y_true, y_pred, id2label, out_dir / "bert_confusion_matrix.png", "BERT / MacBERT Confusion Matrix", normalize=False)
    plot_confusion(y_true, y_pred, id2label, out_dir / "bert_confusion_matrix_normalized.png", "BERT / MacBERT Normalized Confusion Matrix", normalize=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], marker="o", label="train loss")
    ax.set_xlabel("Epoch")
    ax.set_title("BERT training loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "bert_training_loss.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([h["epoch"] for h in history], [h["val_macro_f1"] for h in history], marker="o", label="val macro-F1")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1.0)
    ax.set_title("BERT validation Macro-F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "bert_val_f1.png", dpi=180)
    plt.close(fig)

    pd.DataFrame(history).to_csv(out_dir / "bert_training_history.csv", index=False, encoding="utf-8-sig")
    save_json({"history": history, "test_metrics": test_metrics, "labels": labels, "leakage_diagnostics": leakage_diag}, out_dir / "bert_summary.json")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(f"BERT 最优模型已保存到：{model_dir.resolve()}")


if __name__ == "__main__":
    main()
