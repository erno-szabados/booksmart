from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import AppConfig
from .llm import build_embeddings
from .schemas import ChapterSummaryRecord, ChunkRecord, MapSummaryRecord
from .storage import BookStore


@dataclass(slots=True)
class ContextBundle:
    global_summary: str
    chapter_summaries: list[ChapterSummaryRecord]
    nearby_chunk_summaries: list[str]
    retrieved_chunks: list[Document]
    full_section_text: str
    citations: list[str]


class BookContextRetriever:
    def __init__(self, config: AppConfig, store: BookStore) -> None:
        self.config = config
        self.store = store
        self.embeddings = build_embeddings(config)

    def build_context(self, slug: str, query: str, chapter_ref: str | None = None) -> ContextBundle:
        global_summary = self.store.load_text(self.store.global_summary_path(slug))
        chapter_summaries = self.store.load_chapter_summaries(slug)
        chunks = self.store.load_chunks(slug)
        map_summaries = self.store.load_map_summaries(slug)

        target_chapter = _match_chapter_ref(chapter_summaries, chapter_ref)
        vector_store = Chroma(
            collection_name=f"booksmart-{slug}",
            persist_directory=str(self.store.chroma_dir(slug)),
            embedding_function=self.embeddings,
        )
        search_filter = cast(dict[str, str] | None, {"chapter_index": target_chapter.chapter_index} if target_chapter else None)
        retrieved_chunks = vector_store.similarity_search(query, k=self.config.retrieval_k, filter=search_filter)

        relevant_chapter_summaries = [target_chapter] if target_chapter else _top_chapter_summaries(chapter_summaries, retrieved_chunks)
        nearby_chunk_summaries = _nearby_summaries(map_summaries, retrieved_chunks)
        full_section_text = _build_full_section_text(chunks, retrieved_chunks, self.config.section_char_budget)
        citations = _citations(retrieved_chunks)

        return ContextBundle(
            global_summary=global_summary,
            chapter_summaries=relevant_chapter_summaries,
            nearby_chunk_summaries=nearby_chunk_summaries,
            retrieved_chunks=retrieved_chunks,
            full_section_text=full_section_text,
            citations=citations,
        )


def _match_chapter_ref(
    chapter_summaries: list[ChapterSummaryRecord],
    chapter_ref: str | None,
) -> ChapterSummaryRecord | None:
    if not chapter_ref:
        return None
    normalized = chapter_ref.strip().lower()
    for summary in chapter_summaries:
        if normalized == str(summary.chapter_index) or normalized == summary.chapter_title.lower():
            return summary
    raise ValueError(f"Unknown chapter reference: {chapter_ref}")


def _top_chapter_summaries(
    chapter_summaries: list[ChapterSummaryRecord],
    retrieved_chunks: list[Document],
) -> list[ChapterSummaryRecord]:
    chapter_indexes = {_chapter_index(doc.metadata) for doc in retrieved_chunks}
    selected = [summary for summary in chapter_summaries if summary.chapter_index in chapter_indexes]
    return selected or chapter_summaries[:1]


def _nearby_summaries(map_summaries: list[MapSummaryRecord], retrieved_chunks: list[Document]) -> list[str]:
    indexed = {(item.chapter_index, item.chunk_index): item.summary for item in map_summaries}
    collected: list[str] = []
    seen: set[tuple[int, int]] = set()
    for doc in retrieved_chunks:
        chapter_index = _chapter_index(doc.metadata)
        chunk_index = _chunk_index(doc.metadata)
        for offset in (-1, 0, 1):
            key = (chapter_index, chunk_index + offset)
            if key in indexed and key not in seen:
                seen.add(key)
                collected.append(indexed[key])
    return collected


def _build_full_section_text(
    chunks: list[ChunkRecord],
    retrieved_chunks: list[Document],
    char_budget: int,
) -> str:
    if not retrieved_chunks:
        return ""

    by_chapter: dict[int, list[ChunkRecord]] = {}
    for chunk in chunks:
        by_chapter.setdefault(chunk.chapter_index, []).append(chunk)

    best = retrieved_chunks[0]
    chapter_chunks = by_chapter.get(_chapter_index(best.metadata), [])
    target_chunk_index = _chunk_index(best.metadata)

    section_parts: list[str] = []
    used_chars = 0
    for chunk in chapter_chunks:
        if abs(chunk.chunk_index - target_chunk_index) > 3:
            continue
        if used_chars + chunk.char_count > char_budget:
            break
        section_parts.append(chunk.text)
        used_chars += chunk.char_count
    return "\n\n".join(section_parts)


def _citations(retrieved_chunks: list[Document]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for doc in retrieved_chunks:
        citation = f"{doc.metadata['chapter_title']} / chunk {doc.metadata['chunk_index']}"
        if citation not in seen:
            seen.add(citation)
            result.append(citation)
    return result


def _chapter_index(metadata: dict[str, Any]) -> int:
    return int(metadata["chapter_index"])


def _chunk_index(metadata: dict[str, Any]) -> int:
    return int(metadata["chunk_index"])
