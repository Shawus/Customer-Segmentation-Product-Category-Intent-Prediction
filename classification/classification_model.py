"""
LightGBM-based Ranking Model for Product Recommendation.

Uses LambdaRank objective to learn product relevance scoring per customer group,
with Word2Vec embeddings as feature representations.
"""
import json
import pandas as pd
from pathlib import Path
import pickle
from pandas import DataFrame

import lightgbm as lgb

from classification.ranking_evaluation import RankingEvaluation


class ClassificationModel:
    def __init__(self):
        self.current_dir = Path(__file__).parent
        self.project_root = self.current_dir.parent
        self.model_dir = self.project_root / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = self.current_dir / "data"
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "learning_rate": 0.01,
            "n_estimators": 1000,
            "max_depth": 6,
            "random_state": 42,
            "n_jobs": -1,
        }
        self.eval_metric = "ndcg"
        self.eval_at = [1, 3, 5, 25]
        self.early_stopping_rounds = 100

    def train(self, model_name, X_train, y_train, X_val, y_val, config=None):
        """
        Train a LightGBM ranking model.

        Args:
            model_name: Identifier for the model.
            X_train: Training feature DataFrame.
            y_train: Training target Series (relevance labels).
            X_val: Validation feature DataFrame.
            y_val: Validation target Series.
            config: Optional hyperparameter overrides.

        Returns:
            artifact: Dict with model, features, best_score, etc.
            results: Training evaluation log.
        """
        print(f"\n{model_name} Training...")

        means_vector_cols = X_train.columns[
            X_train.columns.str.startswith("means_vector_")
        ]

        # Group sizes for LambdaRank (rows sharing the same group vector)
        train_group = (
            X_train.groupby(list(means_vector_cols), sort=False).size().to_numpy()
        )
        val_group = X_val.groupby(list(means_vector_cols), sort=False).size().to_numpy()

        if config is None:
            config = self.params
        ranker = lgb.LGBMRanker(**config)
        results = {}

        model = ranker.fit(
            X_train,
            y_train,
            group=train_group,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_group=[train_group, val_group],
            eval_metric=self.eval_metric,
            eval_at=self.eval_at,
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
                lgb.record_evaluation(results),
            ],
        )

        metric_name = f"{self.eval_metric}@{self.eval_at[-1]}"
        current_score = model.best_score_["valid_1"][metric_name]
        model_path = self.model_dir / f"{model_name}.pkl"
        save_model = False

        if model_path.exists():
            with open(model_path, "rb") as f:
                old_artifact = pickle.load(f)
            old_score = old_artifact["best_score"]
            if current_score > old_score:
                save_model = True
        else:
            save_model = True

        artifact = {
            "model": model,
            "features": list(X_train.columns),
            "best_score": current_score,
            "metric": metric_name,
            "best_iteration": model.best_iteration_,
        }

        if save_model:
            with open(model_path, "wb") as f:
                pickle.dump(artifact, f)
            print(f"Model saved to {model_path}")

        return artifact, results

    def train_all(self, data, config=None):
        """Train ranking models for all sectors and product levels."""
        if config is not None:
            config = {**self.params, **config}

        s_sectors = ["ka", "online", "channel_sf"]
        product_levels = ["product_category_1", "product_category_2"]
        models = {}

        for sector in s_sectors:
            for level in product_levels:
                model_name = f"{sector}_ranking_model_{level}"
                X_train = data[sector]["train_data"][f"X_{level}"]
                y_train = data[sector]["train_data"][f"Y_{level}"]
                X_val = data[sector]["validation_data"][f"X_{level}"]
                y_val = data[sector]["validation_data"][f"Y_{level}"]

                artifact, results = self.train(
                    model_name, X_train, y_train, X_val, y_val, config=config
                )
                models[model_name] = {"artifact": artifact, "results": results}

        return models

    def _create_feature_df_for_inference(self, means_vector, product_level) -> DataFrame:
        """
        Create feature DataFrame by combining group means_vector with product embeddings.

        Args:
            means_vector: Group centroid vector from clustering.
            product_level: 'product_category_1' or 'product_category_2'.

        Returns:
            DataFrame with columns: [means_vector_0..N, embedding_0..M, product_name]
        """
        result_data = []
        json_path = self.data_path / f"{product_level}_embedding.json"

        with open(json_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        for product, embedding in mapping.items():
            combined = means_vector + embedding
            combined.append(product)
            result_data.append(combined)

        columns = (
            [f"means_vector_{i}" for i in range(len(means_vector))]
            + [f"{product_level}_embedding_{i}" for i in range(len(next(iter(mapping.values()))))]
            + [product_level]
        )

        return pd.DataFrame(result_data, columns=columns)

    def _predict_with_model(self, purpose, product_level, data, model_name, artifact=None):
        """
        Predict ranking scores using a trained LightGBM model.

        Args:
            purpose: 'inference' or 'evaluation'.
            product_level: 'product_category_1' or 'product_category_2'.
            data: Input DataFrame.
            model_name: Model identifier for loading.
            artifact: Pre-loaded model artifact (for evaluation).

        Returns:
            DataFrame with ranking scores.
        """
        if purpose == "inference":
            model_path = self.model_dir / f"{model_name}.pkl"
            with open(model_path, "rb") as f:
                artifact = pickle.load(f)
        elif purpose == "evaluation" and artifact is None:
            raise ValueError("artifact must be provided for evaluation")

        model = artifact["model"]
        features = artifact["features"]
        best_iteration = artifact["best_iteration"]

        data_copy = data.copy()
        X = data_copy[features]
        scores = model.predict(X, num_iteration=best_iteration)

        result = data.copy()
        result["ranking_score"] = scores

        if purpose == "evaluation":
            result = result[["ranking_score"]]
        else:
            if product_level in result.columns:
                result = result.sort_values("ranking_score", ascending=False)
                result = result[[product_level, "ranking_score"]]
            else:
                result = result[["ranking_score"]]

        return result

    def predict_for_inference(self, means_vector, s_sector, top_k=5):
        """
        Predict top-K product recommendations for a customer group.

        Args:
            means_vector: Group centroid from spectral clustering.
            s_sector: Business sector ('ka', 'online', 'channel_sf').
            top_k: Number of top products to return.

        Returns:
            Dict with 'product_category_1' and 'product_category_2' DataFrames of ranked predictions.
        """
        results = {}

        for product_level in ["product_category_1", "product_category_2"]:
            feature_df = self._create_feature_df_for_inference(means_vector, product_level)
            model_name = f"{s_sector.lower()}_ranking_model_{product_level}"

            predictions = self._predict_with_model(
                purpose="inference",
                product_level=product_level,
                data=feature_df,
                model_name=model_name,
            )
            results[product_level] = predictions.head(top_k)

        return results

    def evaluate_all(self, data, artifacts):
        """Evaluate all trained models on test data."""
        evaluator = RankingEvaluation()
        evaluation_results = {}

        for s_sector in ["ka", "online", "channel_sf"]:
            test_data = data[s_sector]["testing_data"]
            evaluation_results[s_sector] = {}

            for level in ["product_category_1", "product_category_2"]:
                raw_data = test_data[f"{level}_raw_data"]
                X_test = test_data[f"X_{level}"]
                y_test = test_data[f"Y_{level}"]
                model_name = f"{s_sector}_ranking_model_{level}"
                artifact = artifacts[model_name]["artifact"]

                prediction = self._predict_with_model(
                    purpose="evaluation",
                    product_level=level,
                    data=X_test,
                    model_name=model_name,
                    artifact=artifact,
                )

                evaluation_results[s_sector][level] = evaluator.evaluate_all_metrics(
                    raw_data["group_id"], y_test, prediction["ranking_score"]
                )

        return evaluation_results
