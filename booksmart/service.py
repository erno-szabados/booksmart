from __future__ import annotations

import socket
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.prompts import ChatPromptTemplate

from .config import AppConfig
from .llm import build_chat_model
from .pipeline import IngestionPipeline
from .retrieval import BookContextRetriever
from .storage import BookStore

QA_PROMPT = ChatPromptTemplate.from_template(
    """
You are a local book analyst assistant.
Answer the user's question using only the supplied book context.
If the context is insufficient, say so plainly.
Always answer in English.
Include a short 'References' section listing the cited chapter and chunk ids you used.

Question:
{question}

Global summary:
{global_summary}

Relevant chapter summaries:
{chapter_summaries}

Nearby chunk summaries:
{nearby_chunk_summaries}

Retrieved chunk excerpts:
{retrieved_chunks}

Full section context:
{full_section_text}
""".strip()
)


class EndpointUnavailableError(RuntimeError):
    pass


class BookAnalystService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.store = BookStore(config.books_dir)
        self.ingestion = IngestionPipeline(config, self.store)
        self.retriever = BookContextRetriever(config, self.store)
        self.llm = build_chat_model(config)

    def ingest_from_path(self, path: str | Path, force_rebuild: bool = True, progress=None):
        self._assert_ingestion_dependencies()
        return self.ingestion.ingest_path(Path(path), force_rebuild=force_rebuild, progress=progress)

    def ingest_uploaded_text(self, title: str, text: str, force_rebuild: bool = True, progress=None):
        self._assert_ingestion_dependencies()
        return self.ingestion.ingest_text(title=title, source_name=title, text=text, force_rebuild=force_rebuild, progress=progress)

    def list_books(self):
        return self.store.list_books()

    def whole_book_summary(self, slug: str) -> str:
        return self.store.load_text(self.store.global_summary_path(slug))

    def chapter_summary(self, slug: str, chapter_ref: str) -> str:
        for summary in self.store.load_chapter_summaries(slug):
            if chapter_ref.strip().lower() in {str(summary.chapter_index), summary.chapter_title.lower()}:
                return summary.summary
        raise ValueError(f"Unknown chapter reference: {chapter_ref}")

    def nearby_chunk_summaries(self, slug: str, query: str, chapter_ref: str | None = None) -> list[str]:
        self._assert_embedding_dependency()
        context = self.retriever.build_context(slug, query, chapter_ref)
        return context.nearby_chunk_summaries

    def full_section(self, slug: str, query: str, chapter_ref: str | None = None) -> str:
        self._assert_embedding_dependency()
        context = self.retriever.build_context(slug, query, chapter_ref)
        return context.full_section_text

    def answer_question(self, slug: str, question: str, chapter_ref: str | None = None) -> str:
        self._assert_analysis_dependencies()
        context = self.retriever.build_context(slug, question, chapter_ref)
        chapter_summaries = "\n\n".join(
            f"{summary.chapter_title}: {summary.summary}" for summary in context.chapter_summaries
        )
        nearby_chunk_summaries = "\n\n".join(context.nearby_chunk_summaries)
        retrieved_chunks = "\n\n".join(
            f"[{doc.metadata['chapter_title']} / chunk {doc.metadata['chunk_index']}]\n{doc.page_content}"
            for doc in context.retrieved_chunks
        )
        prompt = QA_PROMPT.format_messages(
            question=question,
            global_summary=context.global_summary,
            chapter_summaries=chapter_summaries,
            nearby_chunk_summaries=nearby_chunk_summaries,
            retrieved_chunks=retrieved_chunks,
            full_section_text=context.full_section_text,
        )
        response = self.llm.invoke(prompt)
        references = "\n".join(f"- {citation}" for citation in context.citations)
        return f"{response.content}\n\nReferences\n{references}"

    def _assert_ingestion_dependencies(self) -> None:
        self._assert_chat_dependency()
        self._assert_embedding_dependency()

    def _assert_analysis_dependencies(self) -> None:
        self._assert_chat_dependency()
        self._assert_embedding_dependency()

    def _assert_chat_dependency(self) -> None:
        _assert_endpoint_available(self.config.llm_base_url, "LLM")

    def _assert_embedding_dependency(self) -> None:
        _assert_endpoint_available(self.config.embedding_base_url, "embedding")


def _assert_endpoint_available(base_url: str, label: str) -> None:
    parsed = urlparse(base_url)
    host = parsed.hostname
    port = parsed.port
    if not host or not port:
        raise EndpointUnavailableError(f"Invalid {label} endpoint URL: {base_url}")

    try:
        with socket.create_connection((host, port), timeout=2):
            return
    except OSError as exc:
        raise EndpointUnavailableError(
            f"Cannot reach the configured {label} endpoint at {base_url}. "
            f"Check that the local llama.cpp service is running and that host/port are correct."
        ) from exc
