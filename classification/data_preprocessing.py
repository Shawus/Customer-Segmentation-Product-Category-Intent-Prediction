"""
Data preprocessing pipeline for the classification (ranking) model.

Handles:
- Loading clustering results and shipment data
- Product-to-category mapping
- Feature engineering (count quantiles, data augmentation)
- Dataset splitting (train/validation/test)
- Word2Vec embedding generation
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

from sklearn.model_selection import train_test_split, GroupShuffleSplit

from classification.embedding import EmbeddingProcessor


class ClassificationDataPreprocessor:
    def __init__(self):
        self.current_dir = Path(__file__).parent
        self.project_root = self.current_dir.parent
        self.model_mapping_path = self.current_dir / "data" / "complete_model_product_by_name.csv"
        self.category_file = self.current_dir / "data" / "complete_model_info.json"
        self.clustering_data_path = self.project_root / "clustering" / "data"

    def _load_data(self, sector):
        """Load shipment data and cluster mapping, then merge."""
        shippment_data = pd.read_csv(
            self.clustering_data_path / f"{sector}_data_for_classification.csv"
        )

        with open(
            self.clustering_data_path / f"{sector}_cluster_mapping.json",
            "r", encoding="utf-8-sig",
        ) as f:
            cluster_mapping = json.load(f)
        cluster_data = pd.DataFrame(cluster_mapping["records"])

        data = pd.merge(shippment_data, cluster_data, on=["customer", "site"], how="left")
        data.rename(columns={"label": "group_id"}, inplace=True)
        data = data.dropna(subset=["customer", "group_id", "means_vector", "pd", "product_category_2", "product"])

        return data

    def _mapping_product_to_category(self, df):
        """Map products to product categories via model mapping."""
        model_mapping = pd.read_csv(self.model_mapping_path)
        model_mapping = model_mapping.rename(columns={
            "Product": "product",
            "ModelId": "model_id",
            "ModelName": "model_name",
        })

        with open(self.category_file, "r") as f:
            model_info = json.load(f)

        df_cat = pd.DataFrame(model_info)[["model_id", "product_category_1"]].drop_duplicates()
        df_mapping = model_mapping.merge(df_cat, on="model_id", how="left")
        df_mapping = df_mapping.drop_duplicates(subset=["product", "product_category_1"], keep="last")

        df = df.merge(df_mapping, on="product", how="left")
        df["product_category_1"] = df["product_category_1"].replace("", pd.NA)
        df = df.dropna(subset=["model_id", "product_category_1"])

        return df

    def _calculate_additional_features(self, df):
        """Calculate count-based features and quantile labels."""
        df["product_category_1_count"] = df["product_category_1"].map(df["product_category_1"].value_counts())
        df["product_category_2_count"] = df["product_category_2"].map(df["product_category_2"].value_counts())
        df["product_category_1_count_quantile"] = (
            df.groupby("group_id")["product_category_1_count"]
            .transform(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")) + 1
        )
        df["product_category_2_count_quantile"] = (
            df.groupby("group_id")["product_category_2_count"]
            .transform(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")) + 1
        )
        return df

    def _expanding_means_vector(self, df):
        """Expand means_vector list into individual columns."""
        means_vector_cols = [
            f"means_vector_{i}" for i in range(len(df["means_vector"].iloc[0]))
        ]
        df[means_vector_cols] = pd.DataFrame(df["means_vector"].tolist(), index=df.index)
        return df

    def _data_augmentation(self, df, column_name):
        """
        Add negative samples for items not purchased in each group.
        Randomly selects 20-50 unseen items per group to create balanced training data.
        """
        all_item = df[column_name].unique()
        augmented_rows = []

        for group_id, group in df.groupby("group_id"):
            if group.empty:
                continue

            bought = set(group[column_name])
            unseen = list(set(all_item) - bought)

            if len(unseen) == 0:
                continue

            k = min(len(unseen), np.random.randint(20, 51))
            sampled = np.random.choice(unseen, size=k, replace=False)

            for item in sampled:
                base_row = group.sample(1).iloc[0].to_dict()
                base_row[column_name] = item
                base_row[column_name + "_count"] = 0
                base_row[column_name + "_count_quantile"] = 0
                augmented_rows.append(base_row)

        return pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)

    def _split_dataset(self, data):
        """Split into train (70%), validation (15%), test (15%)."""
        train_data, temp = train_test_split(data, test_size=0.3, random_state=42)
        test_data, val_data = train_test_split(temp, test_size=0.5, random_state=42)
        return train_data, val_data, test_data

    def _split_dataset_by_group(self, data):
        """Split by group_id to prevent data leakage across sets."""
        gss_1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, temp_idx = next(gss_1.split(data, groups=data["group_id"]))
        train_data = data.iloc[train_idx]
        temp_data = data.iloc[temp_idx]

        gss_2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        val_idx, test_idx = next(gss_2.split(temp_data, groups=temp_data["group_id"]))
        val_data = temp_data.iloc[val_idx]
        test_data = temp_data.iloc[test_idx]

        return train_data, val_data, test_data

    def _reduce_large_groups(self, df, col, max_rows=10000):
        """Reduce oversized groups while maintaining category distribution."""
        result = []
        group_sizes = df.groupby("group_id").size()

        for group_id, size in group_sizes.items():
            g = df[df["group_id"] == group_id]
            if size <= max_rows:
                result.append(g)
                continue

            ratio = g[col].value_counts(normalize=True)
            keep_counts = (ratio * max_rows).round().astype(int)
            diff = max_rows - keep_counts.sum()
            if diff != 0:
                adjust_labels = keep_counts.sample(abs(diff), random_state=42).index
                keep_counts[adjust_labels] += 1 if diff > 0 else -1

            keep_idx = []
            for label, n_keep in keep_counts.items():
                subset = g[g[col] == label]
                n_sample = min(len(subset), n_keep)
                if n_sample > 0:
                    idx = subset.sample(n_sample, random_state=42).index
                    keep_idx.extend(idx)

            result.append(g.loc[keep_idx])

        return pd.concat(result).reset_index(drop=True)

    def _sort_data_by_group(self, df):
        return df.sort_values(by=["group_id"]).reset_index(drop=True)

    def _select_features(self, df, feature_columns, target_column):
        return df[feature_columns], df[target_column]

    def main_process(self, sector: str) -> Dict:
        """
        Main preprocessing pipeline for one sector.

        Returns dict with train/validation/testing data containing:
        - X_product_category_1, Y_product_category_1: Features and targets for category-1 ranking
        - X_product_category_2, Y_product_category_2: Features and targets for category-2 ranking
        - {level}_raw_data: Full DataFrame for evaluation
        """
        group_data = self._load_data(sector)
        group_data = self._mapping_product_to_category(group_data)
        group_data = self._calculate_additional_features(group_data)

        result = {"train_data": {}, "validation_data": {}, "testing_data": {}}

        # Save group data for agent inference use
        group_data.to_csv(
            self.current_dir / "data" / f"{sector.lower()}_group_data.csv",
            index=False,
        )

        for product_level in ["product_category_1", "product_category_2"]:
            data = self._data_augmentation(group_data, product_level)

            # Generate Word2Vec embeddings
            processor = EmbeddingProcessor()
            data = processor.embedding_products(data, product_level)
            data = processor.expanding_embedding_vector(data, product_level)
            data.drop(columns=[f"{product_level}_embedding"], inplace=True)

            # Expand means_vector
            data = self._expanding_means_vector(data)
            data.drop(columns=["means_vector"], inplace=True)

            train, val, test = self._split_dataset(data)

            feature_columns = [
                col for col in data.columns
                if col.startswith(("means_vector", f"{product_level}_embedding"))
            ]

            for dataset_name, dataset in [("train", train), ("validation", val), ("testing", test)]:
                dataset = self._reduce_large_groups(dataset, product_level)
                dataset = self._sort_data_by_group(dataset)
                X, Y = self._select_features(dataset, feature_columns, f"{product_level}_count_quantile")
                result[f"{dataset_name}_data"][f"{product_level}_raw_data"] = dataset
                result[f"{dataset_name}_data"][f"X_{product_level}"] = X
                result[f"{dataset_name}_data"][f"Y_{product_level}"] = Y

        return result

    def process_data_for_all_sectors(self, s_sectors=None):
        """Process data for all specified sectors."""
        if s_sectors is None:
            s_sectors = ["KA", "Online", "Channel_SF"]

        all_data = {}
        for sector in s_sectors:
            print(f"Processing data for sector: {sector}")
            all_data[sector.lower()] = self.main_process(sector)
        return all_data
