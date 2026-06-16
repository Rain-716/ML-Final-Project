from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import LABEL_COLUMN, RESPONSE_COLUMN, TEXT_COLUMN


@dataclass
class RetrievalResult:
    response: str
    score: float
    label: str
    source_text: str


class ResponseRetriever:
    def __init__(self, index_path: Path):
        payload = joblib.load(index_path)
        self.vectorizer: TfidfVectorizer = payload["vectorizer"]
        self.matrix = payload["matrix"]
        self.df: pd.DataFrame = payload["df"]

    @staticmethod
    def build(processed_csv: Path, out_path: Path, min_response_len: int = 20) -> None:
        df = pd.read_csv(processed_csv)
        df = df.dropna(subset=[TEXT_COLUMN, RESPONSE_COLUMN, LABEL_COLUMN]).copy()
        df = df[df[RESPONSE_COLUMN].astype(str).str.len() >= min_response_len]
        # Keep the index compact for local demos.
        keep_cols = [TEXT_COLUMN, RESPONSE_COLUMN, LABEL_COLUMN]
        df = df[keep_cols].drop_duplicates(subset=[TEXT_COLUMN, RESPONSE_COLUMN]).reset_index(drop=True)
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=80000)
        mat = vec.fit_transform(df[TEXT_COLUMN].astype(str).tolist())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": vec, "matrix": mat, "df": df}, out_path)

    def retrieve(self, query: str, label: Optional[str] = None, top_k: int = 5) -> RetrievalResult:
        if not query:
            return RetrievalResult("我在这里。你可以先从最困扰你的那件事说起。", 0.0, label or "other", "")
        qv = self.vectorizer.transform([query])
        candidate_idx = np.arange(len(self.df))
        if label and label in set(self.df[LABEL_COLUMN]):
            same = np.where(self.df[LABEL_COLUMN].values == label)[0]
            if len(same) >= 10:
                candidate_idx = same
        scores = cosine_similarity(qv, self.matrix[candidate_idx]).ravel()
        if scores.size == 0:
            return RetrievalResult("我理解你现在可能不太容易把事情说清楚。我们可以慢一点，从你此刻最明显的感受开始。", 0.0, label or "other", "")
        best_local = int(scores.argmax())
        best_idx = int(candidate_idx[best_local])
        row = self.df.iloc[best_idx]
        return RetrievalResult(
            response=str(row[RESPONSE_COLUMN]),
            score=float(scores[best_local]),
            label=str(row[LABEL_COLUMN]),
            source_text=str(row[TEXT_COLUMN]),
        )
