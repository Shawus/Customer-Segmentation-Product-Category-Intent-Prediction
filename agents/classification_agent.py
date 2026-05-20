import os, json, re
import time
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from langchain_openai import AzureChatOpenAI
from langgraph.types import Command

from agents.base import BaseAgent
from classification.classification_model import ClassificationModel


class ClassificationAgent(BaseAgent):
    """
    Uses LightGBM ranking models to predict recommended products (category level)
    and builds recommendation bundles using collaborative filtering and co-purchase graphs.
    """

    def __init__(self, llm_client=None):
        self.classification_model = ClassificationModel()
        self.llm = (
            llm_client.llm
            if llm_client
            else AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
                temperature=0,
            )
        )
        current_dir = Path(__file__).parent
        self.project_root = current_dir.parent
        self.classification_data_dir = self.project_root / "classification" / "data"

    def _safe_parse_json(self, text: str) -> dict:
        if isinstance(text, dict):
            return text
        if not text or not isinstance(text, str):
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            cleaned = re.sub(r"```json|```", "", text).strip()
            m = re.search(r"\{[\s\S]*\}", cleaned)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    return {}
            return {}

    def _convert_predictions_to_recommendations(
        self, predictions, product_level, top_n=5
    ) -> List[Dict]:
        """
        Convert LightGBM ranking DataFrame to structured recommendation list.

        Args:
            predictions: DataFrame with product and ranking_score columns.
            product_level: Column name for product identifier ('product_category_1' or 'product_category_2').
            top_n: Number of top recommendations to return.

        Returns:
            List of recommendation dicts with name, rank, and score.
        """
        df = predictions.copy()
        df = df.sort_values("ranking_score", ascending=False).reset_index(drop=True)
        df_top_n = df.head(top_n)

        return [
            {
                "name": row[product_level],
                "rank": i + 1,
                "score": float(row["ranking_score"]),
            }
            for i, (_, row) in enumerate(df_top_n.iterrows())
        ]

    def _precision_at_k(self, prediction, relevant_items, top_k=5, product_level="product_category_1") -> float:
        """Compute Precision@K for model evaluation."""
        recommend_items = prediction[product_level].head(top_k).tolist()
        num_relevant = len(set(recommend_items) & set(relevant_items))
        return num_relevant / top_k

    def _create_recommendation_set(self, sector, region, customer_id, recommended_product_category_2):
        """
        Create recommendation bundles based on recommended PDLs.

        This method integrates:
        1. Collaborative filtering (user-to-item recommendations from recommendation API)
        2. Co-purchase graph analysis (product families frequently bought together)
        3. Model-level product information enrichment

        Args:
            sector: Business sector.
            region: Customer region.
            customer_id: Customer ERP identifier.
            recommended_product_category_2: List of recommended product lines.

        Returns:
            Dict containing graph-based and reco-based decision units.
        """
        # ---------------------------------------------------------------
        # NOTE: The following logic has been simplified for the public repo.
        # In production, this method:
        #   1. Calls a collaborative filtering API to get user-specific
        #      product recommendations (user_to_items service)
        #   2. Enriches product info using internal product information system
        #   3. Builds "decision units" by matching recommended PDLs against
        #      a co-purchase graph (family bundle store) to form complete
        #      solution packages
        #   4. Falls back to model-name-aggregated recommendation CSV when
        #      graph-based bundles are unavailable
        # ---------------------------------------------------------------
        return {
            "graph_based_decision_units": [],
            "reco_based_decision_units": [],
            "user_recommended_items_top25": [],
        }

    def run(self, state: Dict[str, Any]) -> Command:
        classification_start = time.time()

        customer_id = state.get("customer_id", "")
        clustering_response = state.get("clustering_response", {})
        customer_profile = state.get("customer_profile", {})
        group_id = clustering_response.get("group", "")
        means_vector = clustering_response.get("means_vector", [])
        sector = clustering_response.get("sector", "")
        region = customer_profile.get("region", "")

        if not customer_id or not group_id or not means_vector or not sector or not region:
            return Command(update={
                "classification_response": {
                    "success": False,
                    "message": "Not enough information for classification.",
                }
            })

        # LightGBM ranking model inference
        predictions = self.classification_model.predict_for_inference(
            means_vector=means_vector, s_sector=sector, top_k=5
        )

        product_category_1_predictions = predictions.get("product_category_1") if predictions else None
        product_category_2_predictions = predictions.get("product_category_2") if predictions else None

        if product_category_1_predictions is None or product_category_2_predictions is None:
            return Command(update={
                "classification_response": {
                    "success": False,
                    "message": "Failed to get predictions from classification model.",
                }
            })

        # Load ground truth data for precision calculation
        data_path = self.classification_data_dir / f"{sector.lower()}_group_data.csv"
        so_data = pd.read_csv(data_path)
        relevant_items_product_category_1 = (
            so_data[(so_data["group_id"] == group_id) & (so_data["product_category_1_count"] > 0)]["product_category_1"]
            .unique()
            .tolist()
        )

        product_category_1_precision = self._precision_at_k(product_category_1_predictions, relevant_items_product_category_1)
        product_category_1_precision = min(product_category_1_precision, 0.8)  # Cap to avoid overstating confidence

        top_product_category_2_list = product_category_2_predictions["product_category_2"].head(5).tolist()

        response = {
            "success": True,
            "recommended_product_category_2": {
                "product_category_2_list": product_category_2_predictions,
            },
            "recommended_product_category_1": {
                "precision": product_category_1_precision,
                "recommended_products": self._convert_predictions_to_recommendations(
                    product_category_1_predictions, product_level="product_category_1", top_n=5
                ),
            },
            "recommended_set": self._create_recommendation_set(
                sector, region, customer_id, top_product_category_2_list
            ),
        }

        classification_end = time.time()
        print(f"Classification process took {classification_end - classification_start:.2f}s")

        return Command(update={"classification_response": response})
