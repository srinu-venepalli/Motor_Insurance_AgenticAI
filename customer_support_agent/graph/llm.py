"""Chat LLM factory -- points at the IITM-provided OpenAI-compatible
endpoint via settings.openai_base_url, same as the embeddings client in
integrations/rag.py.
"""

from langchain_openai import ChatOpenAI

from customer_support_agent.core.settings import settings


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=temperature,
    )
