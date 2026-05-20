import os
import logging
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper around Azure OpenAI for LangChain-based agent interactions."""

    def __init__(self) -> None:
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
            temperature=0,
            verbose=False,
            timeout=120,
            max_retries=2,
        )
        self.messages = []

    def init_messages(self):
        self.messages = []

    def add_system_message(self, content: str):
        self.messages.append(SystemMessage(content=content))

    def add_user_message(self, content: str):
        self.messages.append(HumanMessage(content=content))

    def add_bot_message(self, content: str):
        self.messages.append(AIMessage(content=content))

    def get_response(self) -> str:
        response = self.llm.invoke(self.messages)
        return response.content

    def invoke(self, messages):
        return self.llm.invoke(messages)