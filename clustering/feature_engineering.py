"""
Feature engineering module for clustering pipeline.

Transforms raw shipment data into customer-level features for spectral clustering:
- RFM (Recency, Frequency, Monetary) segmentation
- Temporal purchase patterns (quarterly distribution)
- Product preference features (top topics, most frequent products)
- Industry classification
"""
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ClusteringFeatureEngineering:
    """
    Builds customer-level features from transaction-level shipment data.

    Input: Preprocessed financial data (one row per order line item)
    Output: One row per customer with aggregated features

    Feature categories:
    1. RFM metrics and segments
    2. Frequency breakdowns (12m, 24m, 36m)
    3. Order value statistics (average, big_order_ratio)
    4. Temporal patterns (Q1-Q4 distribution, highest/second-highest quarter)
    5. Product preferences (most_frequency_pd, Top_Topic)
    6. Industry classification
    """

    def __init__(self):
        self.today = datetime.now().strftime("%Y%m")

    def compute_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute RFM (Recency, Frequency, Monetary) metrics per customer.

        Args:
            df: Transaction data with columns [, , , ]

        Returns:
            DataFrame with columns: endcustomer, Recency, Frequency, Monetary + segments
        """
        today_int = int(self.today)

        rfm = df.groupby("endcustomer").agg(
            Recency=("orderym", lambda x: today_int - x.max()),
            Frequency=("orderno", "nunique"),
            Monetary=("sales_revenue", "sum"),
        ).reset_index()

        # Segment assignment
        rfm["Recency_segment"] = pd.cut(
            rfm["Recency"],
            bins=[-1, 3, 6, 12, float("inf")],
            labels=["Active Buyer", "Semi-Active Buyer", "Semi-Dormant Buyer", "Dormant Buyer"],
        )

        rfm["SalesForce_Frequency_segment"] = rfm["Frequency"].apply(
            self._frequency_segment
        )

        rfm["Monetary_segment"] = pd.cut(
            rfm["Monetary"],
            bins=[-1, 6000, 10000, 20000, float("inf")],
            labels=["Low-Value Customer", "Mid-Value Customer", "Upper-Mid Value Customer", "High-Value Customer"],
        )

        return rfm

    def _frequency_segment(self, freq: int) -> str:
        if freq >= 4:
            return "High Frequency Buyer"
        elif freq >= 2:
            return "Medium Frequency Buyer"
        elif freq == 1:
            return "Low Frequency Buyer"
        else:
            return "One-Time Buyer"

    def compute_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute quarterly purchase distribution per customer.

        Returns DataFrame with: Q1, Q2, Q3, Q4, highest_quarter, second_highest_quarter,
        percent_in_highest_quarter, percent_in_second_highest_quarter
        """
        df = df.copy()
        df["quarter"] = ((df["orderym"] % 100 - 1) // 3) + 1

        quarterly = df.groupby(["endcustomer", "quarter"]).agg(
            amount=("sales_revenue", "sum")
        ).reset_index()

        pivot = quarterly.pivot_table(
            index="endcustomer", columns="quarter", values="amount", fill_value=0
        )
        pivot.columns = ["Q1", "Q2", "Q3", "Q4"]
        pivot = pivot.reset_index()

        # Determine highest and second-highest quarters
        q_cols = ["Q1", "Q2", "Q3", "Q4"]
        total = pivot[q_cols].sum(axis=1)

        pivot["highest_quarter"] = pivot[q_cols].idxmax(axis=1)
        pivot["percent_in_highest_quarter"] = pivot[q_cols].max(axis=1) / total.replace(0, 1)

        # Second highest
        for idx, row in pivot.iterrows():
            sorted_q = row[q_cols].sort_values(ascending=False)
            pivot.at[idx, "second_highest_quarter"] = sorted_q.index[1]
            pivot.at[idx, "percent_in_second_highest_quarter"] = (
                sorted_q.iloc[1] / total.iloc[idx] if total.iloc[idx] > 0 else 0
            )

        return pivot

    def compute_order_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute average order value and big-order ratio per customer."""
        order_values = df.groupby(["endcustomer", "orderno"])["sales_revenue"].sum().reset_index()
        avg_order = order_values.groupby("endcustomer")["sales_revenue"].mean()
        overall_avg = order_values["sales_revenue"].mean()

        big_order_ratio = order_values.groupby("endcustomer").apply(
            lambda g: (g["sales_revenue"] > overall_avg).sum() / len(g) * 100
        )

        result = pd.DataFrame({
            "endcustomer": avg_order.index,
            "average_order_value": avg_order.values,
            "big_order_ratio": big_order_ratio.values,
        })

        return result

    def compute_frequency_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute transaction frequency for 12m, 24m, 36m windows."""
        today_int = int(self.today)

        result = df.groupby("endcustomer").apply(
            lambda g: pd.Series({
                "frequency_12m": g[g["orderym"] >= today_int - 100]["orderno"].nunique(),
                "frequency_24m": g[g["orderym"] >= today_int - 200]["orderno"].nunique(),
                "frequency_36m": g["orderno"].nunique(),
            })
        ).reset_index()

        return result

    def compute_product_preferences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute most frequent product division and top topic per customer."""
        most_freq_pd = (
            df.groupby("endcustomer")["pd"]
            .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else None)
            .reset_index()
            .rename(columns={"pd": "most_frequency_pd"})
        )

        # Top_Topic is derived from the most purchased product line (product_category_2)
        top_topic = (
            df.groupby("endcustomer")["product_category_2"]
            .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else None)
            .reset_index()
            .rename(columns={"product_category_2": "Top_Topic"})
        )

        return most_freq_pd.merge(top_topic, on="endcustomer", how="outer")

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run full feature engineering pipeline.

        Args:
            df: Cleaned shipment data.

        Returns:
            Customer-level feature DataFrame ready for clustering.
        """
        rfm = self.compute_rfm(df)
        temporal = self.compute_temporal_features(df)
        order_value = self.compute_order_value_features(df)
        freq_windows = self.compute_frequency_windows(df)
        product_prefs = self.compute_product_preferences(df)

        # Merge all features
        result = rfm
        for feat_df in [temporal, order_value, freq_windows, product_prefs]:
            result = result.merge(feat_df, on="endcustomer", how="left")

        logger.info(f"Feature engineering complete: {len(result)} customers, {len(result.columns)} features")
        return result
