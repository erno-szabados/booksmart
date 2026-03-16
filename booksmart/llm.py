from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import AppConfig


def build_chat_model(config: AppConfig, model: str | None = None, base_url: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or config.model,
        base_url=base_url or config.llm_base_url,
        api_key=config.api_key,
        temperature=config.temperature,
    )


def build_embeddings(config: AppConfig) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.embedding_model,
        base_url=config.embedding_base_url,
        api_key=config.api_key,
    )
