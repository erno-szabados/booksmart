from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _load_dotenv_file() -> None:
    """Load a simple .env file into os.environ for missing keys.

    Looks for a path specified by BOOKSMART_DOTENV_PATH (relative or absolute),
    otherwise searches for `.env` in the current working directory and the
    repository root (two parents up from this file).
    """
    env_name = os.environ.get("BOOKSMART_DOTENV_PATH", ".env")
    candidates = [Path(env_name), Path.cwd() / env_name, Path(__file__).resolve().parents[2] / env_name]
    for p in candidates:
        try:
            p = Path(p)
        except Exception:
            continue
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().lstrip("\"").lstrip("'\"").rstrip("\"").rstrip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val
        break


# Load .env early so the DEFAULT_* values below can read from environment.
_load_dotenv_file()


DEFAULT_LLM_BASE_URL = os.environ.get("BOOKSMART_LLM_BASE_URL", "http://localhost:8001/v1")
DEFAULT_EMBEDDING_BASE_URL = os.environ.get("BOOKSMART_EMBEDDING_BASE_URL", "http://localhost/v1")
DEFAULT_API_KEY = os.environ.get("BOOKSMART_API_KEY", "local-token")
DEFAULT_MODEL = os.environ.get("BOOKSMART_MODEL", "local-chat-model")
DEFAULT_EMBEDDING_MODEL = os.environ.get("BOOKSMART_EMBEDDING_MODEL", "local-embedding-model")
DEFAULT_TEMPERATURE = 0.1
DEFAULT_CHUNK_SIZE_CHARS = 12000
DEFAULT_CHUNK_OVERLAP_CHARS = 1500
DEFAULT_EMBEDDING_CHUNK_SIZE_CHARS = 1500
DEFAULT_EMBEDDING_CHUNK_OVERLAP_CHARS = 150
DEFAULT_RETRIEVAL_K = 4
DEFAULT_SECTION_CHAR_BUDGET = 120000
DEFAULT_REDUCE_MAX_TOKENS = 12000


@dataclass(slots=True)
class AppConfig:
    data_dir: Path
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL
    api_key: str = DEFAULT_API_KEY
    model: str = DEFAULT_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS
    embedding_chunk_size_chars: int = DEFAULT_EMBEDDING_CHUNK_SIZE_CHARS
    embedding_chunk_overlap_chars: int = DEFAULT_EMBEDDING_CHUNK_OVERLAP_CHARS
    retrieval_k: int = DEFAULT_RETRIEVAL_K
    section_char_budget: int = DEFAULT_SECTION_CHAR_BUDGET
    reduce_max_tokens: int = DEFAULT_REDUCE_MAX_TOKENS

    @property
    def books_dir(self) -> Path:
        return self.data_dir / "books"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @classmethod
    def load(cls, workspace_dir: str | Path | None = None) -> "AppConfig":
        workspace = Path(workspace_dir or Path.cwd()).resolve()
        config_path = Path(os.environ.get("BOOKSMART_CONFIG", workspace / "booksmart.toml"))
        file_values = _load_config_file(config_path) if config_path.exists() else {}

        data_dir = Path(_env_or_config("BOOKSMART_DATA_DIR", file_values, "data_dir", workspace / "data"))
        return cls(
            data_dir=Path(data_dir),
            llm_base_url=str(_env_or_config("BOOKSMART_LLM_BASE_URL", file_values, "llm_base_url", DEFAULT_LLM_BASE_URL)),
            embedding_base_url=str(_env_or_config("BOOKSMART_EMBEDDING_BASE_URL", file_values, "embedding_base_url", DEFAULT_EMBEDDING_BASE_URL)),
            api_key=str(_env_or_config("BOOKSMART_API_KEY", file_values, "api_key", DEFAULT_API_KEY)),
            model=str(_env_or_config("BOOKSMART_MODEL", file_values, "model", DEFAULT_MODEL)),
            embedding_model=str(_env_or_config("BOOKSMART_EMBEDDING_MODEL", file_values, "embedding_model", DEFAULT_EMBEDDING_MODEL)),
            temperature=float(_env_or_config("BOOKSMART_TEMPERATURE", file_values, "temperature", DEFAULT_TEMPERATURE)),
            chunk_size_chars=int(_env_or_config("BOOKSMART_CHUNK_SIZE_CHARS", file_values, "chunk_size_chars", DEFAULT_CHUNK_SIZE_CHARS)),
            chunk_overlap_chars=int(_env_or_config("BOOKSMART_CHUNK_OVERLAP_CHARS", file_values, "chunk_overlap_chars", DEFAULT_CHUNK_OVERLAP_CHARS)),
            embedding_chunk_size_chars=int(
                _env_or_config(
                    "BOOKSMART_EMBEDDING_CHUNK_SIZE_CHARS",
                    file_values,
                    "embedding_chunk_size_chars",
                    DEFAULT_EMBEDDING_CHUNK_SIZE_CHARS,
                )
            ),
            embedding_chunk_overlap_chars=int(
                _env_or_config(
                    "BOOKSMART_EMBEDDING_CHUNK_OVERLAP_CHARS",
                    file_values,
                    "embedding_chunk_overlap_chars",
                    DEFAULT_EMBEDDING_CHUNK_OVERLAP_CHARS,
                )
            ),
            retrieval_k=int(_env_or_config("BOOKSMART_RETRIEVAL_K", file_values, "retrieval_k", DEFAULT_RETRIEVAL_K)),
            section_char_budget=int(_env_or_config("BOOKSMART_SECTION_CHAR_BUDGET", file_values, "section_char_budget", DEFAULT_SECTION_CHAR_BUDGET)),
            reduce_max_tokens=int(_env_or_config("BOOKSMART_REDUCE_MAX_TOKENS", file_values, "reduce_max_tokens", DEFAULT_REDUCE_MAX_TOKENS)),
        )


def _load_config_file(config_path: Path) -> dict[str, Any]:
    if config_path.suffix == ".toml":
        with config_path.open("rb") as handle:
            return tomllib.load(handle)
    if config_path.suffix in {".yaml", ".yml"}:
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    raise ValueError(f"Unsupported config format: {config_path}")


def _env_or_config(env_name: str, file_values: dict[str, Any], key: str, default: Any) -> Any:
    if env_name in os.environ:
        return os.environ[env_name]
    return file_values.get(key, default)
