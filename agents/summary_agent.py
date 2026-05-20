import os
import re
import time
import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.types import Command

from agents.base import BaseAgent
from agents.prompts import SUMMARY_AGENT_PROMPT, SELECTION_HELPER_PROMPT
from dotenv import load_dotenv

load_dotenv()


class SummaryAgent(BaseAgent):
    """
    Synthesizes final sales brief from clustering and classification results
    using a reasoning-capable LLM.
    """

    def __init__(self, llm_client=None) -> None:
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_REASONING_DEPLOYMENT", "gpt-5.1"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            temperature=1,
            max_retries=3,
        )

    def _load_solution_information(self) -> dict:
        """
        Integrate the resource from internal recommendation module
        """
        # ---------------------------------------------------------------
        # NOTE: In production, this loads enterprise-developed recommendation module
        # ---------------------------------------------------------------
        return {}

    def _stringify(self, obj: Any) -> str:
        if isinstance(obj, (dict, list)):
            return json.dumps(obj, ensure_ascii=False, indent=2)
        return str(obj)

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

    def run(self, state: Dict[str, Any]) -> Command:
        summary_start = time.time()

        original_query = state.get("origin_query", "")
        clustering_response = state.get("clustering_response", {})
        classification_response = state.get("classification_response", {})

        if clustering_response.get("success") and classification_response.get("success"):
            # Load product catalog data for solution composition
            solution_information_content = self._load_solution_information()

            # Step 1: Use LLM to select optimal product solution set
            selection_helper_prompt = SELECTION_HELPER_PROMPT.format(
                recommended_product_category_2=classification_response["recommended_product_category_2"]["product_category_2_list"],
                candidate_text=solution_information_content,
            )

            recommended_set_and_reason = self.llm.invoke([
                SystemMessage(content=selection_helper_prompt),
                HumanMessage(content="Generate a solution recommendation package based on the customer info and data island content."),
            ])

            # Step 2: Generate final summary brief
            prompt = SUMMARY_AGENT_PROMPT.format(
                original_query=original_query,
                recommended_set=self._safe_parse_json(str(recommended_set_and_reason.content)),
                recommended_product_category_1=classification_response["recommended_product_category_1"],
                personal_information=clustering_response["personal_information"],
                group_information=clustering_response["group_information"],
            )

            resp = self.llm.invoke([HumanMessage(content=prompt)])
            summary_text = str(resp.content).strip()
        else:
            summary_text = (
                "This question is not aligned with the system's intended use case. "
                "Please revise your question and try again."
                if state.get("next_action") == "other"
                else "Sorry, an error occurred. Please retry or contact the owner."
            )

        summary_end = time.time()
        print(f"Summary process took {summary_end - summary_start:.2f}s")

        return Command(update={
            "summarization_response": summary_text,
            "completed": True,
        })
