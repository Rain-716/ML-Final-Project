from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import NUMERIC_FEATURES, TEXT_COLUMN
from .label_rules import label_display, weak_label
from .safety import detect_crisis


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


class Predictor:
    def __init__(self, model_dir: str | Path = "models"):
        self.model_dir = Path(model_dir)
        self.kind = "rule"
        self.model = None
        self.tokenizer = None
        self.labels: List[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bert_dir = self.model_dir / "bert_best"
        baseline_path = self.model_dir / "best_baseline.joblib"
        if bert_dir.exists() and (bert_dir / "config.json").exists():
            self.kind = "bert"
            self.tokenizer = AutoTokenizer.from_pretrained(str(bert_dir), local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(str(bert_dir), local_files_only=True).to(self.device)
            self.model.eval()
            label_path = bert_dir / "label2id.json"
            if label_path.exists():
                label2id = json.loads(label_path.read_text(encoding="utf-8"))
                self.labels = [k for k, _ in sorted(label2id.items(), key=lambda kv: kv[1])]
        elif baseline_path.exists():
            self.kind = "baseline"
            payload = joblib.load(baseline_path)
            self.model = payload["model"]
            self.labels = list(payload["labels"])


    @staticmethod
    def _make_frame(text: str) -> pd.DataFrame:
        row = {
            TEXT_COLUMN: text,
            "text_len": len(text),
            "num_turns": 1,
            "risk_keyword_count": 0,
            "question_mark_count": text.count("?") + text.count("？"),
            "exclamation_count": text.count("!") + text.count("！"),
        }
        for col in NUMERIC_FEATURES:
            row.setdefault(col, 0)
        return pd.DataFrame([row])

    def predict(self, text: str) -> Dict:
        text = str(text or "").strip()
        if detect_crisis(text):
            return {"label": "high_risk", "label_zh": label_display("high_risk"), "confidence": 1.0, "kind": self.kind}

        if self.kind == "bert" and self.model and self.tokenizer:
            with torch.no_grad():
                inputs = self.tokenizer(text, max_length=256, truncation=True, padding=True, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits.detach().cpu().numpy()[0]
                probs = softmax(logits)
                idx = int(probs.argmax())
                label = self.labels[idx] if idx < len(self.labels) else str(idx)
                return {"label": label, "label_zh": label_display(label), "confidence": float(probs[idx]), "kind": self.kind}

        if self.kind == "baseline" and self.model:
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(self._make_frame(text))[0]
                idx = int(np.argmax(probs))
                label = self.model.classes_[idx]
                return {"label": str(label), "label_zh": label_display(str(label)), "confidence": float(probs[idx]), "kind": self.kind}
            pred = self.model.predict(self._make_frame(text))[0]
            return {"label": str(pred), "label_zh": label_display(str(pred)), "confidence": 0.65, "kind": self.kind}

        # Safe fallback if no trained model exists yet.
        lr = weak_label(text)
        return {"label": lr.label, "label_zh": label_display(lr.label), "confidence": 0.50, "kind": "rule"}
