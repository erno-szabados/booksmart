from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    chapter_index: int
    chapter_title: str
    chunk_index: int
    text: str
    char_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MapSummaryRecord:
    chunk_id: str
    chapter_index: int
    chapter_title: str
    chunk_index: int
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChapterSummaryRecord:
    chapter_index: int
    chapter_title: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChapterRecord:
    index: int
    title: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BookManifest:
    slug: str
    title: str
    source_name: str
    source_hash: str
    ingested_at: str
    total_chars: int
    chapter_count: int
    chunk_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
