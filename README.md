## BookSmart

BookSmart is a local-first book analysis pipeline for long books that do not fit into a single model context window. It uses hierarchical summarization with LangChain map-reduce chains, a local llama.cpp-compatible chat endpoint, a local embedding endpoint, Chroma for retrieval, and Chainlit for the UI.

### What It Supports

- Whole-book summaries.
- Chapter summaries.
- Question answering over a processed book.
- Question answering scoped by chapter.
- Nearby chunk summaries for a query.
- A larger local section view assembled around the most relevant chunk.

### Processing Model

1. Split a text book into chapters when chapter headings are present.
2. Split large chapters into context-safe chunks.
3. Run LangChain map-reduce summarization over the chunks.
4. Persist chunk summaries, chapter summaries, and a whole-book summary.
5. Build a local Chroma index from the source chunks.
6. Answer questions using global summary + chapter summaries + retrieved chunks + a larger local section.

### Rebuild Policy

- Re-ingesting or uploading a book rebuilds its artifacts so stale outputs are not reused.
- Continuing with an existing processed book reuses the persisted artifacts already on disk.

### Configuration

Copy `booksmart.example.toml` to `booksmart.toml` or set environment variables directly.

Default endpoints:

- Chat LLM: `http://localhost:8001/v1`
- Embeddings: `http://localhost:8002/v1`

Config values:

- `data_dir`
- `llm_base_url`
- `embedding_base_url`
- `api_key`
- `model`
- `embedding_model`
- `temperature`
- `chunk_size_chars`
- `chunk_overlap_chars`
- `embedding_chunk_size_chars`
- `embedding_chunk_overlap_chars`
- `chat_llm_base_url`
- `chat_model`
- `retrieval_k`
- `section_char_budget`
- `reduce_max_tokens`

### Install

```bash
uv sync
```

### CLI Usage

Ingest a book from a text file:

```bash
uv run booksmart ingest /path/to/book.txt
```

List processed books:

```bash
uv run booksmart list
```

Show a whole-book summary:

```bash
uv run booksmart summary <book-slug>
```

Ask a question:

```bash
uv run booksmart ask <book-slug> "What drives the protagonist's decision near the end?"
```

### Chainlit UI

Run the UI locally:

```bash
uv run chainlit run chainlit_app.py
```

Useful commands inside the chat:

- `/books`
- `/use <book-slug>`
- `/ingest <path-to-book.txt>`
- `/upload`
- `/summary`
- `/chapter <number-or-title>`
- `/nearby <question>`
- `/section <question>`

### Data Layout

Each processed book is stored under `data/books/<slug>/` with:

- `source.txt`
- `manifest.json`
- `chapters.json`
- `chunks.json`
- `map_summaries.json`
- `chapter_summaries.json`
- `global_summary.md`
- `chroma/`

### Notes

- Input support is intentionally focused on plain text for v1.
- The implementation assumes OpenAI-compatible llama.cpp endpoints.
- The section context budget is character-based, which is a pragmatic approximation for local models.
- Summarization and embeddings now use separate chunk budgets. Keep `chunk_size_chars` large enough for efficient map-reduce summarization, and keep `embedding_chunk_size_chars` small enough for your embedding endpoint's token limit.
