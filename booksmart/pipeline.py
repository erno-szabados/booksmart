from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from langchain_classic.chains.summarize import load_summarize_chain
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .chunking import build_chunks, slugify, split_into_chapters, stable_hash, title_from_source_name
from .config import AppConfig
from .llm import build_chat_model, build_embeddings
from .schemas import BookManifest, ChapterRecord, ChapterSummaryRecord, MapSummaryRecord
from .storage import BookStore

logger = logging.getLogger(__name__)

MAP_PROMPT = ChatPromptTemplate.from_template(
    """
You are summarizing one chunk from a book.
Write a concise summary focused on plot progression, character development, and other durable details.
Avoid speculation and avoid repeating exact phrasing from the text.

Book chunk:
{context}
""".strip()
)

REDUCE_PROMPT = ChatPromptTemplate.from_template(
    """
You are combining partial book summaries into a single coherent narrative summary.
Preserve chronology, highlight character arcs, and keep only details that matter for downstream analysis.

Partial summaries:
{context}
""".strip()
)

CHAPTER_REDUCE_PROMPT = ChatPromptTemplate.from_template(
    """
Combine the chunk summaries for one chapter into a concise chapter summary.
Keep the summary chronological and analysis-ready.

Chapter title: {chapter_title}
Chunk summaries:
{chunk_summaries}
""".strip()
)

ProgressCallback = Callable[[str], None] | None


class IngestionPipeline:
    def __init__(self, config: AppConfig, store: BookStore) -> None:
        self.config = config
        self.store = store
        self.llm = build_chat_model(config)
        self.embeddings = build_embeddings(config)

    def ingest_path(self, source_path: Path, force_rebuild: bool = True, progress: ProgressCallback = None) -> BookManifest:
        text = source_path.read_text(encoding="utf-8")
        title = title_from_source_name(source_path.name)
        return self.ingest_text(title=title, source_name=source_path.name, text=text, force_rebuild=force_rebuild, progress=progress)

    def ingest_text(
        self,
        title: str,
        source_name: str,
        text: str,
        force_rebuild: bool = True,
        progress: ProgressCallback = None,
    ) -> BookManifest:
        slug = slugify(title)
        if force_rebuild:
            logger.info("event=ingest.reset slug=%s", slug)
            self.store.clear_book(slug)

        self._notify(progress, f"Preparing chapters for '{title}'.")
        chapters = split_into_chapters(text)
        chunks = build_chunks(chapters, self.config.chunk_size_chars, self.config.chunk_overlap_chars)
        source_hash = stable_hash(text)

        manifest = BookManifest(
            slug=slug,
            title=title,
            source_name=source_name,
            source_hash=source_hash,
            ingested_at=datetime.now(UTC).isoformat(),
            total_chars=len(text),
            chapter_count=len(chapters),
            chunk_count=len(chunks),
        )

        book_dir = self.store.ensure_book_dir(slug)
        self.store.save_text(self.store.source_path(slug), text)
        self.store.save_manifest(manifest)
        self.store.save_chapters(slug, chapters)
        self.store.save_chunks(slug, chunks)

        self._notify(progress, f"Prepared {len(chapters)} chapters and {len(chunks)} chunks.")
        self._notify(progress, f"Running map-reduce summarization over {len(chunks)} chunks.")
        documents = [
            Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "chapter_index": chunk.chapter_index,
                    "chapter_title": chunk.chapter_title,
                    "chunk_index": chunk.chunk_index,
                },
            )
            for chunk in chunks
        ]
        summary_chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=MAP_PROMPT,
            combine_prompt=REDUCE_PROMPT,
            map_reduce_document_variable_name="context",
            combine_document_variable_name="context",
            return_intermediate_steps=True,
            token_max=self.config.reduce_max_tokens,
        )
        summary_result = summary_chain.invoke({"input_documents": documents})
        intermediate_steps = summary_result.get("intermediate_steps", [])
        map_summaries = [
            MapSummaryRecord(
                chunk_id=chunk.chunk_id,
                chapter_index=chunk.chapter_index,
                chapter_title=chunk.chapter_title,
                chunk_index=chunk.chunk_index,
                summary=summary,
            )
            for chunk, summary in zip(chunks, intermediate_steps, strict=False)
        ]
        self.store.save_map_summaries(slug, map_summaries)
        self.store.save_text(self.store.global_summary_path(slug), summary_result["output_text"])
        self._notify(progress, f"Completed chunk summarization for {len(map_summaries)} chunks.")

        self._notify(progress, f"Reducing chunk summaries into {len(chapters)} chapter summaries.")
        chapter_summaries = self._build_chapter_summaries(chapters=chapters, map_summaries=map_summaries, progress=progress)
        self.store.save_chapter_summaries(slug, chapter_summaries)
        self._notify(progress, f"Completed chapter summarization for {len(chapter_summaries)} chapters.")

        self._notify(progress, "Building the local Chroma index.")
        self._build_vector_store(slug, documents)
        logger.info("event=ingest.complete slug=%s chapters=%s chunks=%s dir=%s", slug, manifest.chapter_count, manifest.chunk_count, book_dir)
        self._notify(progress, f"Book '{title}' is ready.")
        return manifest

    def _build_chapter_summaries(
        self,
        chapters: list[ChapterRecord],
        map_summaries: list[MapSummaryRecord],
        progress: ProgressCallback = None,
    ) -> list[ChapterSummaryRecord]:
        summaries_by_chapter: dict[int, list[str]] = {}
        for summary in map_summaries:
            summaries_by_chapter.setdefault(summary.chapter_index, []).append(summary.summary)

        chapter_summaries: list[ChapterSummaryRecord] = []
        total_chapters = len(chapters)
        for index, chapter in enumerate(chapters, start=1):
            self._notify(progress, f"Summarizing chapter {index}/{total_chapters}: {chapter.title}")
            joined = "\n\n".join(summaries_by_chapter.get(chapter.index, []))
            prompt = CHAPTER_REDUCE_PROMPT.format_messages(chapter_title=chapter.title, chunk_summaries=joined)
            response = self.llm.invoke(prompt)
            chapter_summaries.append(
                ChapterSummaryRecord(
                    chapter_index=chapter.index,
                    chapter_title=chapter.title,
                    summary=_message_text(response.content),
                )
            )
        return chapter_summaries

    def _build_vector_store(self, slug: str, documents: list[Document]) -> None:
        chroma_dir = self.store.chroma_dir(slug)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        vector_store = Chroma(
            collection_name=f"booksmart-{slug}",
            persist_directory=str(chroma_dir),
            embedding_function=self.embeddings,
        )
        vector_store.reset_collection()
        vector_store.add_documents(documents)

    @staticmethod
    def _notify(progress: ProgressCallback, message: str) -> None:
        if progress:
            progress(message)


def _message_text(content: str | list[str | dict[object, object]]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        else:
            parts.append(str(item))
    return "\n".join(parts)
