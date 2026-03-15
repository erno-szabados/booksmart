from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .schemas import BookManifest, ChapterRecord, ChapterSummaryRecord, ChunkRecord, MapSummaryRecord


class BookStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def book_dir(self, slug: str) -> Path:
        return self.root_dir / slug

    def clear_book(self, slug: str) -> None:
        book_dir = self.book_dir(slug)
        if book_dir.exists():
            shutil.rmtree(book_dir)

    def ensure_book_dir(self, slug: str) -> Path:
        book_dir = self.book_dir(slug)
        book_dir.mkdir(parents=True, exist_ok=True)
        return book_dir

    def source_path(self, slug: str) -> Path:
        return self.book_dir(slug) / "source.txt"

    def manifest_path(self, slug: str) -> Path:
        return self.book_dir(slug) / "manifest.json"

    def chapters_path(self, slug: str) -> Path:
        return self.book_dir(slug) / "chapters.json"

    def chunks_path(self, slug: str) -> Path:
        return self.book_dir(slug) / "chunks.json"

    def map_summaries_path(self, slug: str) -> Path:
        return self.book_dir(slug) / "map_summaries.json"

    def chapter_summaries_path(self, slug: str) -> Path:
        return self.book_dir(slug) / "chapter_summaries.json"

    def global_summary_path(self, slug: str) -> Path:
        return self.book_dir(slug) / "global_summary.md"

    def chroma_dir(self, slug: str) -> Path:
        return self.book_dir(slug) / "chroma"

    def save_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def load_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def save_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_json(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    def save_manifest(self, manifest: BookManifest) -> None:
        self.save_json(self.manifest_path(manifest.slug), manifest.to_dict())

    def load_manifest(self, slug: str) -> BookManifest:
        return BookManifest(**self.load_json(self.manifest_path(slug)))

    def save_chapters(self, slug: str, chapters: list[ChapterRecord]) -> None:
        self.save_json(self.chapters_path(slug), [chapter.to_dict() for chapter in chapters])

    def load_chapters(self, slug: str) -> list[ChapterRecord]:
        payload = self.load_json(self.chapters_path(slug))
        return [ChapterRecord(**item) for item in payload]

    def save_chunks(self, slug: str, chunks: list[ChunkRecord]) -> None:
        self.save_json(self.chunks_path(slug), [chunk.to_dict() for chunk in chunks])

    def load_chunks(self, slug: str) -> list[ChunkRecord]:
        payload = self.load_json(self.chunks_path(slug))
        return [ChunkRecord(**item) for item in payload]

    def save_map_summaries(self, slug: str, summaries: list[MapSummaryRecord]) -> None:
        self.save_json(self.map_summaries_path(slug), [summary.to_dict() for summary in summaries])

    def load_map_summaries(self, slug: str) -> list[MapSummaryRecord]:
        payload = self.load_json(self.map_summaries_path(slug))
        return [MapSummaryRecord(**item) for item in payload]

    def save_chapter_summaries(self, slug: str, summaries: list[ChapterSummaryRecord]) -> None:
        self.save_json(self.chapter_summaries_path(slug), [summary.to_dict() for summary in summaries])

    def load_chapter_summaries(self, slug: str) -> list[ChapterSummaryRecord]:
        payload = self.load_json(self.chapter_summaries_path(slug))
        return [ChapterSummaryRecord(**item) for item in payload]

    def list_books(self) -> list[BookManifest]:
        manifests: list[BookManifest] = []
        for child in sorted(self.root_dir.iterdir()):
            manifest_path = child / "manifest.json"
            if manifest_path.exists():
                manifests.append(BookManifest(**self.load_json(manifest_path)))
        return manifests
