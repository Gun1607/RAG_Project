from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
	"""Application configuration loaded from environment variables."""

	model_config = SettingsConfigDict(
		env_file=".env",
		env_file_encoding="utf-8",
		case_sensitive=False,
		extra="ignore",
	)

	app_name: str = "RAG-Based Intelligent Document Assistant"
	app_version: str = "0.1.0"
	debug: bool = False

	api_prefix: str = "/api"

	data_dir: Path = Field(default=PROJECT_ROOT / "data")
	uploads_dir: Path = Field(default=PROJECT_ROOT / "data" / "uploads")
	faiss_index_path: Path = Field(default=PROJECT_ROOT / "data" / "index" / "faiss.index")
	faiss_metadata_path: Path = Field(default=PROJECT_ROOT / "data" / "index" / "metadata.json")

	embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
	embedding_batch_size: int = 32

	chunk_size: int = 800
	chunk_overlap: int = 120
	min_chunk_chars: int = 50

	retrieval_top_k: int = 4

	openai_api_key: str | None = None
	openai_model: str = "gpt-4o-mini"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
	"""Return cached settings instance for reuse across the app."""
	settings = Settings()

	settings.data_dir.mkdir(parents=True, exist_ok=True)
	settings.uploads_dir.mkdir(parents=True, exist_ok=True)
	settings.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)

	return settings
