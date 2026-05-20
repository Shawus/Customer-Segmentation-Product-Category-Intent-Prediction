import os, json, re
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.types import Command

from agents.base import BaseAgent
from agents.prompts import INTENT_AGENT_PROMPT


class IntentAgent(BaseAgent):
    """Gatekeeper agent: determines if the query aligns with the system goal."""

    def __init__(self, llm_client=None):
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
        prompt = INTENT_AGENT_PROMPT.format(origin_query=state.get("origin_query", ""))
        response = self.llm.invoke([HumanMessage(content=prompt)])
        content = self._safe_parse_json(response.content)

        response_content = content.get(
            "response",
            "Sorry, an error occurred. Please retry or contact the owner.",
        )
        next_action = content.get("next_action", "")

        return Command(update={
            "origin_query": state.get("origin_query", ""),
            "response_content_intent": response_content or "",
            "next_action": next_action or "",
        })
