"""
LangGraph Multi-Agent Orchestration

Pipeline: User Query → Intent → Extraction → Clustering → Classification → Summary → Response
"""
import time
from typing import List, Dict, Any, TypedDict

from langgraph.graph import StateGraph, END, START

from agents.intent_agent import IntentAgent
from agents.extraction_agent import ExtractionAgent
from agents.clustering_agent import ClusteringAgent
from agents.classification_agent import ClassificationAgent
from agents.summary_agent import SummaryAgent
from utils.llm_client import LLMClient
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict, total=False):
    origin_query: str
    request: List[str]
    customer_id: str
    region: str
    next_action: str
    response_content_intent: str
    clustering_response: Dict[str, Any]
    customer_profile: Dict[str, Any]
    classification_response: Dict[str, Any]
    summarization_response: str
    completed: bool


def run_agent(query: str) -> str:
    """
    Build and execute the multi-agent graph for customer analysis.

    Args:
        query: Natural language query from the user.

    Returns:
        Summarized response string.
    """
    llm = LLMClient()

    intent_agent = IntentAgent(llm_client=llm)
    extraction_agent = ExtractionAgent(llm_client=llm)
    clustering_agent = ClusteringAgent(llm_client=llm)
    classification_agent = ClassificationAgent(llm_client=llm)
    summary_agent = SummaryAgent(llm_client=llm)

    def _intent_router(state: AgentState) -> str:
        return "summary" if state.get("next_action") == "other" else "extraction"

    def _check_parameter_router(state: AgentState) -> str:
        return (
            "clustering"
            if state.get("customer_id") != "unknown" and state.get("region") != "unknown"
            else "summary"
        )

    def _check_clustering_router(state: AgentState) -> str:
        return (
            "classification"
            if state.get("clustering_response", {}).get("success")
            else "summary"
        )

    graph = StateGraph(state_schema=AgentState)
    graph.add_node("intent", intent_agent)
    graph.add_node("extraction", extraction_agent)
    graph.add_node("clustering", clustering_agent)
    graph.add_node("classification", classification_agent)
    graph.add_node("summary", summary_agent)

    graph.add_edge(START, "intent")
    graph.add_conditional_edges("intent", _intent_router, ["extraction", "summary"])
    graph.add_conditional_edges("extraction", _check_parameter_router, ["clustering", "summary"])
    graph.add_conditional_edges("clustering", _check_clustering_router, ["classification", "summary"])
    graph.add_edge("classification", "summary")
    graph.add_edge("summary", END)

    agent_graph = graph.compile()

    init_state: AgentState = {
        "origin_query": query,
        "request": [],
        "customer_id": "",
        "region": "",
        "next_action": "",
        "response_content_intent": "",
        "clustering_response": {},
        "classification_response": {},
        "summarization_response": "",
        "completed": False,
    }

    result = agent_graph.invoke(init_state, config={"recursion_limit": 50})
    return result.get("summarization_response", "")


if __name__ == "__main__":
    start_time = time.time()
    response = run_agent("Analyze customer T90371143 in Taiwan")
    print(f"\n=== RESULT ===\n{response}")
    print(f"Total process time: {time.time() - start_time:.2f}s")
