"""
Spectral Clustering Model for Customer Segmentation.

Implements a modified spectral clustering algorithm that handles both
numerical and categorical features through an augmented similarity graph.

Key algorithm steps:
1. Build base similarity graph from numerical features using KNN
2. Construct augmented graph incorporating categorical features via edge weights
3. Compute normalized Laplacian and extract eigenvectors
4. Apply KMeans on the spectral embedding space

Reference: Spectral clustering on mixed-type data (numerical + categorical)
"""
import logging
import json
import os
import pandas as pd
import numpy as np
from typing import List

from scipy.sparse import csr_matrix, diags, vstack, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

logger = logging.getLogger(__name__)


class ClusteringModel:
    """
    Spectral clustering model for customer segmentation.

    Features:
    - Mixed-type data handling (numerical + categorical)
    - KNN-based similarity graph (memory-efficient vs. fully-connected)
    - Configurable category weights (lambda) for domain knowledge injection
    - Automatic best-group-number search via silhouette score
    """

    def __init__(self):
        # ---------------------------------------------------------------
        # NOTE: In production, data should be loaded from certain data management system.
        # Replace with your own data loading strategy.
        # ---------------------------------------------------------------
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")

        # Feature definitions
        self.num_cols = [
            "Frequency", "Monetary",
            "frequency_12m", "frequency_24m", "frequency_36m",
            "average_order_value", "big_order_ratio",
            "Q1", "Q2", "Q3", "Q4",
            "percent_in_highest_quarter", "percent_in_second_highest_quarter",
        ]
        self.cat_cols = [
            "region", "industry_level_1", "industry_level_2", "Top_Topic",
            "SalesForce_Frequency_segment", "Monetary_segment",
            "most_frequency_pd", "highest_quarter", "second_highest_quarter",
        ]
        self.target_cols = [
            "endcustomer", "endcustomername"
        ] + self.num_cols + self.cat_cols

        # Category importance weights (domain knowledge)
        self.cat_cols_lambda = {
            "region": 1.0,
            "industry_level_1": 1.5,
            "industry_level_2": 1.5,
            "SalesForce_Frequency_segment": 1.0,
            "Monetary_segment": 1.0,
            "most_frequency_pd": 1.5,
            "highest_quarter": 1.0,
            "second_highest_quarter": 1.0,
            "Top_Topic": 1.5,
        }

        # Search ranges for optimal cluster count per sector
        self.sector_group_ranges = {
            "KA": np.arange(50, 70, 1),
            "Channel_SF": np.arange(15, 30, 1),
            "Online": np.arange(35, 60, 1),
        }

    def build_base_similarity_graph(
        self, data: pd.DataFrame, num_cols: List[str], cat_cols: List[str], knn_k: int
    ):
        """
        Build augmented similarity graph combining numerical KNN graph
        and categorical edge-weight connections.

        Args:
            data: Customer feature DataFrame.
            num_cols: Numerical feature column names.
            cat_cols: Categorical feature column names.
            knn_k: Number of nearest neighbors for the base graph.

        Returns:
            G_all: Augmented similarity matrix (sparse CSR).
            n: Number of customer nodes.
        """
        n = data.shape[0]
        df = data.copy()
        df[num_cols] = StandardScaler().fit_transform(df[num_cols])

        # --- Numerical features: KNN-based similarity graph ---
        R = len(num_cols)
        if R > 0:
            nn = NearestNeighbors(n_neighbors=knn_k + 1, metric="euclidean", n_jobs=-1)
            nn.fit(df[num_cols])
            dists, idxs = nn.kneighbors(df[num_cols])
            dists = dists[:, 1:]  # Remove self-connections
            idxs = idxs[:, 1:]

            # Gaussian kernel weights: exp(-||xi - xj||^2)
            rows = np.repeat(np.arange(n), knn_k)
            cols = idxs.ravel()
            w_num = np.exp(-(dists.ravel() ** 2))

            # Symmetrize the graph
            G_R = csr_matrix((w_num, (rows, cols)), shape=(n, n))
            G_R = 0.5 * (G_R + G_R.T)
            G_R.setdiag(0.0)
        else:
            G_R = csr_matrix((n, n), dtype=float)

        # --- Categorical features: One-hot edge connections ---
        OneHot_vec = []
        lambda_cols = []

        for c in cat_cols:
            vals = df[c].astype("category")
            codes = vals.cat.codes.to_numpy()
            t_c = len(vals.cat.categories)

            rows_cat = np.arange(n, dtype=int)
            cols_cat = codes
            data_cat = np.ones(n, dtype=float)
            H_c = csr_matrix((data_cat, (rows_cat, cols_cat)), shape=(n, t_c))
            OneHot_vec.append(H_c)

            lambda_c = self.cat_cols_lambda.get(c, 1.0)
            lambda_cols.append(np.full(t_c, lambda_c, dtype=float))

        if OneHot_vec:
            H = hstack(OneHot_vec, format="csr")
            lambda_vec = np.hstack(lambda_cols)
            W_edge = H @ diags(lambda_vec)
            t = W_edge.shape[1]
        else:
            W_edge = csr_matrix((n, 0), dtype=float)
            t = 0

        # --- Construct augmented graph ---
        if t > 0:
            top = hstack([G_R, W_edge], format="csr")
            I_t = diags(np.ones(t), 0, format="csr")
            bottom = hstack([W_edge.T, I_t], format="csr")
            G_all = vstack([top, bottom], format="csr")
            G_all = (G_all + G_all.T) * 0.5
            G_all.setdiag(0)
        else:
            G_all = G_R

        return G_all, n

    def spectral_clustering(self, G_all: csr_matrix, n: int, group: int):
        """
        Perform spectral clustering via normalized Laplacian eigenvectors.

        Steps:
        1. Compute symmetric normalized Laplacian: L = D^(-1/2) (D - G) D^(-1/2)
        2. Extract smallest eigenvectors (excluding trivial)
        3. Normalize eigenvectors row-wise
        4. Apply KMeans on the spectral embedding

        Args:
            G_all: Augmented similarity matrix.
            n: Number of actual data points (excluding augmented category nodes).
            group: Target number of clusters.

        Returns:
            labels: Cluster assignments for each data point.
            U_points: Spectral embedding coordinates (n x group-1).
        """
        d = np.asarray(G_all.sum(axis=1)).ravel().astype(float)
        D = diags(d, 0, format="csr")

        with np.errstate(divide="ignore"):
            inv_sqrt_d = 1.0 / np.sqrt(d)
        inv_sqrt_d[~np.isfinite(inv_sqrt_d)] = 0.0
        D_inv_sqrt = diags(inv_sqrt_d, 0, format="csr")

        L = D_inv_sqrt @ (D - G_all) @ D_inv_sqrt

        # Compute smallest eigenvalues/vectors
        vals, vecs = eigsh(A=L, k=group, which="SA", tol=1e-5, maxiter=50000)

        # Take only data-point rows, skip the trivial first eigenvector
        U_points = vecs[:n, 1:group + 1]
        U_points /= np.linalg.norm(U_points, axis=1, keepdims=True)

        km = KMeans(n_clusters=group, n_init="auto", random_state=42)
        labels = km.fit_predict(U_points)

        return labels, U_points

    def find_best_group(self, data: pd.DataFrame, test_group_range: np.ndarray) -> int:
        """
        Search for optimal cluster count using silhouette score.

        Applies a size-balance constraint: rejects solutions where
        max_cluster/min_cluster ratio exceeds 10.
        """
        logger.info("Searching for best cluster count...")
        best_score = 0.0
        best_group = 0

        for group in test_group_range:
            G_all, n = self.build_base_similarity_graph(
                data, num_cols=self.num_cols, cat_cols=self.cat_cols, knn_k=3
            )
            labels, U_points = self.spectral_clustering(G_all, n=n, group=group)

            unique, counts = np.unique(labels, return_counts=True)
            sizes = counts.astype(float)
            ratio = sizes.max() / sizes.min()

            score = silhouette_score(U_points, labels) if ratio <= 10 else 0.0

            if score > best_score:
                best_score = score
                best_group = group

        logger.info(f"Best group: {best_group}, silhouette: {best_score:.6f}")
        return best_group

    def evaluate_clustering(self, labels, U_points, s_sector, best_group, n_samples):
        """Evaluate clustering quality with multiple metrics."""
        unique, counts = np.unique(labels, return_counts=True)
        sizes = counts.astype(float)

        metrics = {
            "s_sector": s_sector,
            "n_clusters": int(best_group),
            "n_samples": int(n_samples),
            "silhouette_score": float(silhouette_score(U_points, labels)),
            "calinski_harabasz_index": float(calinski_harabasz_score(U_points, labels)),
            "davies_bouldin_index": float(davies_bouldin_score(U_points, labels)),
            "cluster_size_min": int(sizes.min()),
            "cluster_size_max": int(sizes.max()),
            "cluster_size_mean": float(sizes.mean()),
            "imbalance_ratio": float(sizes.max() / sizes.min()),
        }

        logger.info(f"Clustering metrics for {s_sector}: {json.dumps(metrics, indent=2)}")
        return metrics

    def save_clustering_mapping(self, data, s_sector, labels, U_points):
        """
        Save cluster assignments and spectral embeddings.

        Output schema per record:
        {
            "endcustomer": str,
            "region": str,
            "label": int,
            "U_point": list[float],
            "means_vector": list[float]
        }
        """
        data = data.copy()
        data["clustering_group"] = np.asarray(labels, dtype=int)

        embedding_dim = U_points.shape[1]
        data["U_point"] = U_points.tolist()

        # Compute group centroids (means_vector)
        means_vectors = (
            data.groupby("clustering_group")["U_point"]
            .apply(lambda vectors: [
                float(sum(axis_vals)) / len(axis_vals)
                for axis_vals in zip(*vectors)
            ])
            .to_dict()
        )

        records = [
            {
                "endcustomer": row["endcustomer"],
                "region": row["region"],
                "label": int(row["clustering_group"]),
                "U_point": row["U_point"],
                "means_vector": means_vectors[int(row["clustering_group"])],
            }
            for _, row in data.iterrows()
        ]

        # ---------------------------------------------------------------
        # NOTE: In production, this uploads to certain data management system:
        #   upload_data()...
        # For local development, save to file:
        # ---------------------------------------------------------------
        output_path = os.path.join(self.data_dir, f"{s_sector}_cluster_mapping.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"records": records}, f, ensure_ascii=False)

        logger.info(f"Saved {s_sector} cluster mapping ({len(records)} records)")

    def run_clustering_training(self):
        """Run full clustering training pipeline for all sectors."""
        for s_sector, test_range in self.sector_group_ranges.items():
            logger.info(f"Training {s_sector} clustering model...")

            # ---------------------------------------------------------------
            # NOTE: In production, training data is loaded from certain data management system:
            #   load_data()...
            # For local development, load from file:
            # ---------------------------------------------------------------
            data_path = os.path.join(self.data_dir, f"{s_sector}_data_for_clustering.csv")
            df = pd.read_csv(data_path)

            training_data = (
                df[self.target_cols]
                .dropna(subset=self.target_cols)
                .reset_index(drop=True)
                .copy()
            )

            best_group = self.find_best_group(training_data, test_range)

            G_all, n = self.build_base_similarity_graph(
                training_data, num_cols=self.num_cols, cat_cols=self.cat_cols, knn_k=3
            )
            labels, U_points = self.spectral_clustering(G_all, n=n, group=best_group)

            self.evaluate_clustering(labels, U_points, s_sector, best_group, n)
            self.save_clustering_mapping(training_data, s_sector, labels, U_points)

            logger.info(f"{s_sector} clustering training completed")

    # =================== Inference Methods ===================

    def get_customer_vector(self, erp_id: str, s_sector: str):
        """
        Look up customer's cluster assignment and embedding vectors.

        Returns:
            (group_label, means_vector, U_point) or ("unknown", [], [])
        """
        # ---------------------------------------------------------------
        # NOTE: In production, cluster mapping is loaded from cloud storage.
        # For local: load from clustering/data/{s_sector}_cluster_mapping.json
        # ---------------------------------------------------------------
        mapping_path = os.path.join(self.data_dir, f"{s_sector}_cluster_mapping.json")
        if not os.path.exists(mapping_path):
            return "unknown", [], []

        with open(mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f).get("records", [])

        for record in data:
            if record.get("endcustomer") == erp_id:
                return record["label"], record["means_vector"], record["U_point"]

        return "unknown", [], []

    def get_customer_features(self, erp_id: str, s_sector: str, region: str) -> dict:
        """Get customer's feature profile from clustering data."""
        # ---------------------------------------------------------------
        # NOTE: In production, loaded from cloud storage.
        # ---------------------------------------------------------------
        data_path = os.path.join(self.data_dir, f"{s_sector}_data_for_clustering.csv")
        if not os.path.exists(data_path):
            return {}

        df = pd.read_csv(data_path)
        row = df[(df["endcustomer"] == erp_id) & (df["region"] == region)]

        if row.empty:
            return {}

        features = {}
        for col in self.target_cols:
            if col in row.columns:
                features[col] = row.iloc[0][col]
        return features

    def get_group_feature(self, group: str, s_sector: str) -> dict:
        """Get cluster group-level aggregated features."""
        # ---------------------------------------------------------------
        # NOTE: In production, loaded from certain data management system.
        # ---------------------------------------------------------------
        exp_path = os.path.join(self.data_dir, f"{s_sector}_clustering_explanations.json")
        if not os.path.exists(exp_path):
            return {}

        with open(exp_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            if str(item.get("cluster_group")) == str(group):
                return item
        return {}

    def TopK_related_customer(self, erp_id: str, s_sector: str, top_K: int) -> dict:
        """
        Find top-K most similar customers in the same cluster group.

        Uses progressive filter relaxation:
        1. Same group + region + industry + RFM segments
        2. Relaxes filters one by one until enough candidates found
        3. Ranks by Euclidean distance in spectral embedding space
        """
        mapping_path = os.path.join(self.data_dir, f"{s_sector}_cluster_mapping.json")
        data_path = os.path.join(self.data_dir, f"{s_sector}_data_for_clustering.csv")

        if not os.path.exists(mapping_path) or not os.path.exists(data_path):
            return {}

        with open(mapping_path, "r", encoding="utf-8") as f:
            vector_data = json.load(f).get("records", [])
        vector_df = pd.DataFrame(vector_data)

        features_df = pd.read_csv(data_path)
        merged_df = features_df.merge(
            vector_df, on=["endcustomer", "site", "region"], how="inner"
        )

        target_rows = merged_df[merged_df["endcustomer"] == erp_id]
        if target_rows.empty:
            return {}

        target = target_rows.iloc[0]
        target_U_point = target["U_point"]

        # Progressive filter relaxation
        filters = [
            ("label", target.get("label")),
            ("region", target.get("region")),
            ("industry_level_1", target.get("industry_level_1")),
            ("Monetary_segment", target.get("Monetary_segment")),
            ("SalesForce_Frequency_segment", target.get("SalesForce_Frequency_segment")),
        ]

        for depth in range(len(filters), 1, -1):
            filtered = merged_df.copy()
            for col, val in filters[:depth]:
                if col in filtered.columns:
                    filtered = filtered[filtered[col] == val]
            if len(filtered) >= top_K + 1:
                break

        # Compute distances and rank
        candidates = filtered[filtered["endcustomer"] != erp_id].copy()
        candidates["distance"] = candidates["U_point"].apply(
            lambda x: float(np.linalg.norm(
                np.asarray(target_U_point, dtype=float) - np.asarray(x, dtype=float)
            ))
        )
        candidates = candidates.nsmallest(top_K, "distance")

        result = {}
        for _, row in candidates.iterrows():
            cust_id = row["endcustomer"]
            result[cust_id] = self.get_customer_features(cust_id, s_sector, row["region"])

        return result

    def inference(self, erp_id: str, region: str, top_K: int) -> dict:
        """
        Run clustering inference for a customer.

        Checks membership across all sectors (KA, Channel_SF, Online) and returns
        full profile including group assignment, related customers, etc.

        Returns:
            Dict keyed by sector name, each containing:
            - group, means_vector, personal_features, group_features, related_customers
        """
        inference_results = {}

        for s_sector in ["KA", "Channel_SF", "Online"]:
            # Check if customer exists in this sector's data
            data_path = os.path.join(self.data_dir, f"{s_sector}_data_for_clustering.csv")
            if not os.path.exists(data_path):
                continue

            df = pd.read_csv(data_path)
            exists = not df[
                (df["endcustomer"] == erp_id) & (df["region"] == region)
            ].empty

            if not exists:
                continue

            group, means_vector, U_point = self.get_customer_vector(erp_id, s_sector)
            if group == "unknown":
                continue

            inference_results[s_sector] = {
                "erp_id": erp_id,
                "region": region,
                "group": group,
                "personal_vector": U_point,
                "means_vector": means_vector,
                "personal_features": self.get_customer_features(erp_id, s_sector, region),
                "group_features": self.get_group_feature(group, s_sector),
                "related_customers": self.TopK_related_customer(erp_id, s_sector, top_K),
            }

        return inference_results
