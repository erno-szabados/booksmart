from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import AppConfig
from .logging_utils import setup_logging
from .service import BookAnalystService, EndpointUnavailableError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BookSmart local book analysis pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest and summarize a book from a text file")
    ingest_parser.add_argument("path", type=Path)
    ingest_parser.add_argument("--no-rebuild", action="store_true", help="Reuse an existing book directory if present")

    list_parser = subparsers.add_parser("list", help="List processed books")
    list_parser.set_defaults(command="list")

    summary_parser = subparsers.add_parser("summary", help="Print the whole-book summary")
    summary_parser.add_argument("book")

    chapter_parser = subparsers.add_parser("chapter", help="Print one chapter summary")
    chapter_parser.add_argument("book")
    chapter_parser.add_argument("chapter")

    answer_parser = subparsers.add_parser("ask", help="Ask a question about a processed book")
    answer_parser.add_argument("book")
    answer_parser.add_argument("question")
    answer_parser.add_argument("--chapter")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = AppConfig.load()
    setup_logging(config.logs_dir)
    service = BookAnalystService(config)

    try:
        if args.command == "ingest":
            manifest = service.ingest_from_path(
                args.path,
                force_rebuild=not args.no_rebuild,
                progress=lambda message: print(message, flush=True),
            )
            print(f"Ingested {manifest.title} as {manifest.slug}")
            return

        if args.command == "list":
            for manifest in service.list_books():
                print(f"{manifest.slug}\t{manifest.title}\t{manifest.chapter_count} chapters\t{manifest.chunk_count} chunks")
            return

        if args.command == "summary":
            print(service.whole_book_summary(args.book))
            return

        if args.command == "chapter":
            print(service.chapter_summary(args.book, args.chapter))
            return

        if args.command == "ask":
            print(service.answer_question(args.book, args.question, chapter_ref=args.chapter))
            return

        parser.error(f"Unknown command: {args.command}")
    except (EndpointUnavailableError, ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
