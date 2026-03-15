from __future__ import annotations

import asyncio
from pathlib import Path

import chainlit as cl

from booksmart.config import AppConfig
from booksmart.logging_utils import setup_logging
from booksmart.service import BookAnalystService

WORKSPACE_DIR = Path(__file__).parent
CONFIG = AppConfig.load(WORKSPACE_DIR)
setup_logging(CONFIG.logs_dir)
SERVICE = BookAnalystService(CONFIG)

HELP_TEXT = """
Commands:
- /books : list processed books
- /use <book-slug> : continue working with an existing book without rebuilding
- /ingest <path-to-book.txt> : ingest or re-ingest a text file from disk
- /upload : upload a .txt file and rebuild that book's artifacts
- /summary : show the whole-book summary for the active book
- /chapter <number-or-title> : show a chapter summary
- /nearby <question> : show nearby chunk summaries relevant to a question
- /section <question> : show the full local section context used for analysis
- any other message: ask a question about the active book
""".strip()


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content=(
            "BookSmart is ready. Ingest a book to rebuild fresh artifacts, or select an existing book to continue from persisted state.\n\n"
            f"{HELP_TEXT}"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    content = message.content.strip()
    if not content:
        return

    if content == "/books":
        manifests = await cl.make_async(SERVICE.list_books)()
        if not manifests:
            await cl.Message(content="No processed books are available yet.").send()
            return
        lines = [f"- {manifest.slug}: {manifest.title} ({manifest.chapter_count} chapters, {manifest.chunk_count} chunks)" for manifest in manifests]
        await cl.Message(content="\n".join(lines)).send()
        return

    if content.startswith("/use "):
        slug = content.removeprefix("/use ").strip()
        cl.user_session.set("active_book", slug)
        await cl.Message(content=f"Active book set to {slug}. Existing artifacts will be reused.").send()
        return

    if content.startswith("/ingest "):
        path = content.removeprefix("/ingest ").strip()
        await _ingest_path(path)
        return

    if content == "/upload":
        files = await cl.AskFileMessage(
            content="Upload a plain text book file. Re-uploading the same book rebuilds all artifacts.",
            accept=["text/plain", ".txt", ".md"],
            max_files=1,
            timeout=180,
        ).send()
        if not files:
            await cl.Message(content="Upload cancelled.").send()
            return
        uploaded = files[0]
        text = Path(uploaded.path).read_text(encoding="utf-8")
        await _ingest_text(uploaded.name, text)
        return

    if content == "/summary":
        slug = _require_active_book()
        summary = await cl.make_async(SERVICE.whole_book_summary)(slug)
        await cl.Message(content=summary).send()
        return

    if content.startswith("/chapter "):
        slug = _require_active_book()
        chapter_ref = content.removeprefix("/chapter ").strip()
        summary = await cl.make_async(SERVICE.chapter_summary)(slug, chapter_ref)
        await cl.Message(content=summary).send()
        return

    if content.startswith("/nearby "):
        slug = _require_active_book()
        query = content.removeprefix("/nearby ").strip()
        summaries = await cl.make_async(SERVICE.nearby_chunk_summaries)(slug, query)
        await cl.Message(content="\n\n".join(f"- {summary}" for summary in summaries)).send()
        return

    if content.startswith("/section "):
        slug = _require_active_book()
        query = content.removeprefix("/section ").strip()
        section = await cl.make_async(SERVICE.full_section)(slug, query)
        await cl.Message(content=section or "No relevant section was found.").send()
        return

    slug = _require_active_book()
    answer = await cl.make_async(SERVICE.answer_question)(slug, content)
    await cl.Message(content=answer).send()


async def _ingest_path(path: str) -> None:
    loop = asyncio.get_running_loop()

    def progress(update: str) -> None:
        future = asyncio.run_coroutine_threadsafe(cl.Message(content=update).send(), loop)
        future.result()

    manifest = await cl.make_async(SERVICE.ingest_from_path)(path, True, progress)
    cl.user_session.set("active_book", manifest.slug)
    await cl.Message(content=f"Active book set to {manifest.slug}.").send()


async def _ingest_text(title: str, text: str) -> None:
    loop = asyncio.get_running_loop()

    def progress(update: str) -> None:
        future = asyncio.run_coroutine_threadsafe(cl.Message(content=update).send(), loop)
        future.result()

    manifest = await cl.make_async(SERVICE.ingest_uploaded_text)(title, text, True, progress)
    cl.user_session.set("active_book", manifest.slug)
    await cl.Message(content=f"Active book set to {manifest.slug}.").send()


def _require_active_book() -> str:
    slug = cl.user_session.get("active_book")
    if not slug:
        raise ValueError("No active book selected. Use /books, /use <slug>, /ingest <path>, or /upload first.")
    return slug
