from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingError(Exception):
	"""Raised when embedding generation fails."""


@dataclass(slots=True)
class EmbeddingService:
	model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
	batch_size: int = 32
	model: Any = field(init=False, repr=False)

	def __post_init__(self) -> None:
		try:
			self.model = SentenceTransformer(self.model_name)
		except Exception as exc:  # noqa: BLE001
			raise EmbeddingError(f"Failed to load embedding model '{self.model_name}': {exc}") from exc

	def embed_documents(self, texts: list[str]) -> np.ndarray:
		"""Return L2-friendly float32 embeddings for a list of text chunks."""
		cleaned = [t.strip() for t in texts if t and t.strip()]
		if not cleaned:
			raise EmbeddingError("No valid texts provided for document embeddings.")

		try:
			vectors = self.model.encode(
				cleaned,
				batch_size=self.batch_size,
				show_progress_bar=False,
				convert_to_numpy=True,
				normalize_embeddings=True,
			)
		except Exception as exc:  # noqa: BLE001
			raise EmbeddingError(f"Failed to generate document embeddings: {exc}") from exc

		return np.asarray(vectors, dtype=np.float32)

	def embed_query(self, query: str) -> np.ndarray:
		"""Return a single normalized query embedding as shape (1, dim)."""
		if not query or not query.strip():
			raise EmbeddingError("Query cannot be empty for embedding.")

		try:
			vector = self.model.encode(
				[query.strip()],
				show_progress_bar=False,
				convert_to_numpy=True,
				normalize_embeddings=True,
			)
		except Exception as exc:  # noqa: BLE001
			raise EmbeddingError(f"Failed to generate query embedding: {exc}") from exc

		return np.asarray(vector, dtype=np.float32)
