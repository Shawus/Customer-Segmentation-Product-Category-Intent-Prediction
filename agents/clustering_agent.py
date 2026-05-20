import os, json, re
import time
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.types import Command

from agents.base import BaseAgent
from agents.prompts import CLUSTERING_AGENT_PROMPT
from clustering.clustering_model import ClusteringModel


class ClusteringAgent(BaseAgent):
    """
    Runs spectral clustering inference to find customer's group,
    personal profile, and similar customers.
    """

    def __init__(self, llm_client=None):
        self.clustering_model = ClusteringModel()
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

        self.rfm_definitions_mapper = {
            # Recency Segment
            "Active Buyer": "Last transaction within 3 months",
            "Semi-Active Buyer": "Last transaction between 3 to 6 months ago",
            "Semi-Dormant Buyer": "Last transaction between 6 months to 1 year ago",
            "Dormant Buyer": "Last transaction over 1 year ago",
            # Frequency Segment
            "High Frequency Buyer": "4+ transactions in the past year, with prior history",
            "Medium Frequency Buyer": "2-3 transactions in the past year, with prior history",
            "Low Frequency Buyer": "1 transaction in the past year, with prior history",
            "One-Time Buyer": "No transactions in the past year, but prior history exists",
            "New Active or Return Customer": "Transactions in the past year, no prior history",
            "Long-Time Dormant Buyer": "Transaction history dates back over 2 years",
            # Monetary Segment
            "High-Value Customer": "Total transaction amount exceeds $20,000 in 3 years",
            "Upper-Mid Value Customer": "Total transaction amount $10,000-$20,000 in 3 years",
            "Mid-Value Customer": "Total transaction amount $6,000-$10,000 in 3 years",
            "Low-Value Customer": "Total transaction amount below $6,000 in 3 years",
        }

        # BOR (Big Order Ratio) x Frequency strategy mapping
        # Used to generate sales approach strategies based on customer purchase behavior
        self.BOR_strategies = {
            ("B3", "F0"): "High order value but extremely low frequency — likely project-based purchasing. Recommend tracking project cycles and introducing maintenance/accessory products to increase repeat purchases.",
            ("B3", "F1"): "High order value with low frequency — possibly project-driven. Focus on project timelines, bundled solutions, and converting one-time purchases into ongoing service contracts.",
            ("B3", "F2"): "High order value with moderate frequency — growth potential. Plan quarterly/annual demand reviews and use contract pricing to improve continuity.",
            ("B3", "F3"): "High order value and high frequency — strategic key account. Manage with framework agreements, dedicated support, and cross-product-line solutions.",
            ("B3", "F4"): "Flagship customer with highest value and frequency. Establish joint business plans, priority resources, and long-term centralized procurement agreements.",
            ("B2", "F0"): "Occasionally high orders but extremely low frequency. Maintain low-cost contact; when project signals appear, quickly provide evaluation and solution packages.",
            ("B2", "F1"): "Occasionally high orders with low frequency. Conduct demand reviews around purchase motivations and propose phased procurement plans.",
            ("B2", "F2"): "Occasionally high orders with moderate frequency — cultivation value. Establish regular visits and use combo/upgrade offers to improve order continuity.",
            ("B2", "F3"): "Occasionally high orders with high frequency — stable growth type. Cross-sell and upgrade to increase order value while consolidating engagement.",
            ("B2", "F4"): "Extremely high frequency with occasionally high orders. Promote centralized procurement, multi-category bundles, and supply guarantees.",
            ("B1", "F0"): "Rarely high orders with extremely low frequency — low engagement. Maintain awareness via content/events; invest resources when project signals emerge.",
            ("B1", "F1"): "Rarely high orders with low frequency. Strengthen demand exploration and decision-chain building through small campaigns.",
            ("B1", "F2"): "Rarely high orders with moderate frequency. Focus on repurchase and upgrade pathways for existing categories.",
            ("B1", "F3"): "Rarely high orders but high frequency — maintenance/replenishment type. Use bundles and threshold promotions to increase per-order value.",
            ("B1", "F4"): "Rarely high orders but extremely high frequency. Propose contract pricing and identify upgrade/substitute opportunities to raise average order value.",
            ("B0", "F0"): "Low value and extremely low frequency — long-tail segment. Maintain minimal contact; invest resources only when clear demand signals appear.",
            ("B0", "F1"): "Low value with low frequency. Use targeted outreach for replacement/upgrade opportunities; avoid over-investing resources.",
            ("B0", "F2"): "Low value with moderate frequency. Gradually increase order value through bundles, upgrades, and threshold promotions.",
            ("B0", "F3"): "Low value but high frequency — possibly cost-sensitive. Promote centralized procurement and standardized product SKUs to reduce transaction costs.",
            ("B0", "F4"): "Low value but extremely high frequency. Prioritize process optimization (fixed product lists, contract pricing, automated reordering) and design upgrade paths.",
        }

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

    def get_BOR_explanation(self, big_order_ratio: float, frequency: int) -> str:
        """
        Generate a strategy explanation based on Big Order Ratio and Frequency.

        Args:
            big_order_ratio: Percentage (0-100) of orders above average value.
            frequency: Total transaction count over 3 years.
        """
        if frequency <= 2:
            f = "F0"
        elif frequency <= 6:
            f = "F1"
        elif frequency <= 18:
            f = "F2"
        elif frequency <= 36:
            f = "F3"
        else:
            f = "F4"

        if big_order_ratio < 20:
            BOR = "B0"
        elif big_order_ratio < 45:
            BOR = "B1"
        elif big_order_ratio < 70:
            BOR = "B2"
        else:
            BOR = "B3"

        return self.BOR_strategies.get((BOR, f), "")

    def run(self, state: Dict[str, Any]) -> Command:
        clustering_start = time.time()

        customer_profile = {}

        if state.get("region") == "unknown":
            return Command(update={
                "clustering_response": {
                    "customer_id": state.get("customer_id"),
                    "group": "unknown",
                    "sector": "unknown",
                    "means_vector": [],
                    "feature": {},
                    "success": False,
                    "personal_information": "",
                    "group_information": "",
                },
                "customer_profile": customer_profile,
            })

        # Run clustering model inference
        inference_result = self.clustering_model.inference(
            customer_id=state.get("customer_id"),
            region=state.get("region"),
            top_K=3,
        )

        # Check results across business sectors (KA, Channel_SF, Online)
        sector_found = None
        for sector in ["KA", "Channel_SF", "Online"]:
            if inference_result.get(sector):
                sector_found = sector
                break

        if not sector_found:
            return Command(update={
                "clustering_response": {
                    "customer_id": state.get("customer_id"),
                    "group": "unknown",
                    "sector": "unknown",
                    "means_vector": [],
                    "feature": {},
                    "success": False,
                    "personal_information": "",
                    "group_information": "",
                },
                "customer_profile": customer_profile,
            })

        sector_data = inference_result[sector_found]
        clustering_response = {
            "customer_id": state.get("customer_id"),
            "group": sector_data.get("group"),
            "sector": sector_found,
            "means_vector": sector_data.get("means_vector"),
            "feature": sector_data.get("group_features"),
            "success": True,
            "personal_information": "",
            "group_information": "",
        }
        customer_profile = sector_data.get("personal_features")
        related_customers = sector_data.get("related_customers")

        # Build personal information summary
        big_order_ratio_explanation = self.get_BOR_explanation(
            big_order_ratio=customer_profile.get("big_order_ratio"),
            frequency=customer_profile.get("Frequency"),
        )

        personal_information_template = (
            f"Target customer: {customer_profile.get('customername')} in {state.get('region')}.\n"
            f"Industry (ISIC classification): {customer_profile.get('industry')}.\n"
            f"Recent interest topics: {customer_profile.get('Top_Topic')}.\n"
            f"Average order value: ${customer_profile.get('average_order_value')} USD, "
            f"with {customer_profile.get('big_order_ratio')}% of orders above average. "
            f"{big_order_ratio_explanation}\n"
            f"- Recency: {customer_profile.get('Recency_segment')} — "
            f"{self.rfm_definitions_mapper.get(customer_profile.get('Recency_segment'), '')}\n"
            f"- Frequency: {customer_profile.get('SalesForce_Frequency_segment')} — "
            f"{self.rfm_definitions_mapper.get(customer_profile.get('SalesForce_Frequency_segment'), '')}\n"
            f"- Monetary: {customer_profile.get('Monetary_segment')} — "
            f"{self.rfm_definitions_mapper.get(customer_profile.get('Monetary_segment'), '')}\n"
        )

        # Use LLM to generate group-level insights
        prompt = CLUSTERING_AGENT_PROMPT.format(
            personal_information=customer_profile,
            topK_customer_profile=related_customers,
            rfm_definitions=self.rfm_definitions_mapper,
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        content = self._safe_parse_json(response.content)
        group_information = content.get("group_information", "")

        clustering_response["personal_information"] = personal_information_template
        clustering_response["group_information"] = group_information

        clustering_end = time.time()
        print(f"Clustering process took {clustering_end - clustering_start:.2f}s")

        return Command(update={
            "clustering_response": clustering_response,
            "customer_profile": customer_profile,
        })
