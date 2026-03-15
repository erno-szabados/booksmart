from __future__ import annotations

import hashlib
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .schemas import ChapterRecord, ChunkRecord

CHAPTER_HEADING_RE = re.compile(r"^(?:chapter|prologue|epilogue|part)\b.*$", re.IGNORECASE | re.MULTILINE)
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    cleaned = NON_ALNUM_RE.sub("-", value.lower()).strip("-")
    return cleaned or "book"


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def title_from_source_name(source_name: str) -> str:
    return Path(source_name).stem.replace("_", " ").strip() or "Untitled Book"


def split_into_chapters(text: str) -> list[ChapterRecord]:
    matches = list(CHAPTER_HEADING_RE.finditer(text))
    if not matches:
        return [ChapterRecord(index=1, title="Full Text", text=text.strip())]

    chapters: list[ChapterRecord] = []
    for index, match in enumerate(matches, start=1):
        start = match.start()
        end = matches[index].start() if index < len(matches) else len(text)
        chunk = text[start:end].strip()
        heading = match.group(0).strip()
        chapters.append(ChapterRecord(index=index, title=heading, text=chunk))
    return chapters


def build_chunks(
    chapters: list[ChapterRecord],
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> list[ChunkRecord]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=chunk_overlap_chars,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[ChunkRecord] = []
    for chapter in chapters:
        texts = splitter.split_text(chapter.text)
        for chunk_index, text in enumerate(texts, start=1):
            chunk_id = f"c{chapter.index:03d}-p{chunk_index:03d}"
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    chapter_index=chapter.index,
                    chapter_title=chapter.title,
                    chunk_index=chunk_index,
                    text=text,
                    char_count=len(text),
                )
            )
    return chunks
