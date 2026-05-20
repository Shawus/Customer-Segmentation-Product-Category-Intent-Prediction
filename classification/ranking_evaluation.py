"""
Ranking evaluation metrics for LightGBM ranker models.

Implements standard information retrieval metrics:
- Precision@K
- Recall@K
- F1-score@K
- NDCG and NDCG@K
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import ndcg_score


class RankingEvaluation:
    def __init__(self):
        self.current_dir = Path(__file__).parent
        self.results_dir = self.current_dir / "evaluation_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_df(self, group_id_series, count_series, pred_series):
        return pd.DataFrame({
            "group_id": group_id_series,
            "count": count_series,
            "pred_score": pred_series,
        })

    def precision_at_k(self, group_id_series, count_series, pred_series, top_k=None):
        if top_k is None:
            top_k = [1, 3, 5]
        df = self._prepare_df(group_id_series, count_series, pred_series)
        precisions = {k: [] for k in top_k}

        for _, group in df.groupby("group_id"):
            group_sorted = group.sort_values("pred_score", ascending=False)
            relevant = set(group_sorted.index[group_sorted["count"] > 0])
            for k in top_k:
                recommended = set(group_sorted.index[:k])
                precisions[k].append(len(recommended & relevant) / k)

        return {k: np.mean(v) for k, v in precisions.items()}

    def recall_at_k(self, group_id_series, count_series, pred_series, top_k=None):
        if top_k is None:
            top_k = [1, 3, 5]
        df = self._prepare_df(group_id_series, count_series, pred_series)
        recalls = {k: [] for k in top_k}

        for _, group in df.groupby("group_id"):
            group_sorted = group.sort_values("pred_score", ascending=False)
            relevant = set(group_sorted.index[group_sorted["count"] > 0])
            if not relevant:
                continue
            for k in top_k:
                recommended = set(group_sorted.index[:k])
                recalls[k].append(len(recommended & relevant) / len(relevant))

        return {k: np.mean(v) for k, v in recalls.items()}

    def compute_ndcg(self, group_id_series, count_series, pred_series, top_k=None):
        if top_k is None:
            top_k = [1, 3, 5, 10]
        df = self._prepare_df(group_id_series, count_series, pred_series)
        ndcg_at_k = {k: [] for k in top_k}
        ndcg_full_list = []

        for _, group in df.groupby("group_id"):
            y_true = group["count"].to_numpy().reshape(1, -1)
            y_score = group["pred_score"].to_numpy().reshape(1, -1)
            ndcg_full_list.append(ndcg_score(y_true, y_score))
            for k in top_k:
                ndcg_at_k[k].append(ndcg_score(y_true, y_score, k=k))

        return {
            "ndcg_full": np.mean(ndcg_full_list),
            "ndcg_at_k": {k: np.mean(v) for k, v in ndcg_at_k.items()},
        }

    def evaluate_all_metrics(self, group_id_series, count_series, pred_series):
        """Run all evaluation metrics and return combined results."""
        return {
            "precision": self.precision_at_k(group_id_series, count_series, pred_series),
            "recall": self.recall_at_k(group_id_series, count_series, pred_series),
            "ndcg": self.compute_ndcg(group_id_series, count_series, pred_series),
        }
