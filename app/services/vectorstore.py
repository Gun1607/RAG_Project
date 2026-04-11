import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class VectorStoreError(Exception):
	"""Raised when vector store operations fail."""


class VectorStoreService:
	"""FAISS-backed vector store for chunk embeddings and their metadata."""

	def __init__(self, index_path: str | Path, metadata_path: str | Path) -> None:
		self.index_path = Path(index_path)
		self.metadata_path = Path(metadata_path)
		self.index: faiss.Index | None = None
		self.metadata: list[dict[str, Any]] = []

		self.index_path.parent.mkdir(parents=True, exist_ok=True)
		self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

	@property
	def is_ready(self) -> bool:
		return self.index is not None and self.index.ntotal > 0 and len(self.metadata) > 0

	@property
	def size(self) -> int:
		if self.index is None:
			return 0
		return int(self.index.ntotal)

	def _ensure_float32_2d(self, vectors: np.ndarray) -> np.ndarray:
		array = np.asarray(vectors, dtype=np.float32)
		if array.ndim != 2:
			raise VectorStoreError("Embeddings must be a 2D array of shape (n, dim).")
		if array.shape[0] == 0 or array.shape[1] == 0:
			raise VectorStoreError("Embeddings array cannot be empty.")
		return np.ascontiguousarray(array)

	def _ensure_query_vector(self, query_vector: np.ndarray) -> np.ndarray:
		array = np.asarray(query_vector, dtype=np.float32)
		if array.ndim == 1:
			array = np.expand_dims(array, axis=0)
		if array.ndim != 2 or array.shape[0] != 1:
			raise VectorStoreError("Query embedding must have shape (1, dim) or (dim,).")
		return np.ascontiguousarray(array)

	def _create_index(self, embedding_dim: int) -> None:
		# Embeddings are normalized in embedding service, so inner product works as cosine similarity.
		self.index = faiss.IndexFlatIP(embedding_dim)

	def add_embeddings(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> int:
		"""Append new embeddings with metadata and return number of vectors added."""
		vectors = self._ensure_float32_2d(embeddings)

		if len(metadata) != vectors.shape[0]:
			raise VectorStoreError(
				"Metadata count must match number of embeddings "
				f"({len(metadata)} != {vectors.shape[0]})."
			)

		if self.index is None:
			self._create_index(vectors.shape[1])
		elif self.index.d != vectors.shape[1]:
			raise VectorStoreError(
				"Embedding dimension mismatch with existing index "
				f"({vectors.shape[1]} != {self.index.d})."
			)

		self.index.add(vectors)
		self.metadata.extend(metadata)

		if self.size != len(self.metadata):
			raise VectorStoreError("Index size and metadata size diverged after add operation.")

		return vectors.shape[0]

	def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[dict[str, Any]]:
		"""Search nearest chunks and return metadata with similarity scores."""
		if not self.is_ready:
			raise VectorStoreError("Vector store is empty or not initialized.")

		if top_k <= 0:
			raise VectorStoreError("top_k must be > 0.")

		query_vector = self._ensure_query_vector(query_embedding)
		if query_vector.shape[1] != self.index.d:
			raise VectorStoreError(
				"Query embedding dimension mismatch with index "
				f"({query_vector.shape[1]} != {self.index.d})."
			)

		k = min(top_k, self.size)
		scores, indices = self.index.search(query_vector, k)

		results: list[dict[str, Any]] = []
		for score, idx in zip(scores[0], indices[0], strict=False):
			if idx < 0:
				continue
			meta = dict(self.metadata[int(idx)])
			meta["score"] = float(score)
			meta["vector_id"] = int(idx)
			results.append(meta)

		return results

	def save(self) -> None:
		"""Persist FAISS index and metadata to disk."""
		if self.index is None:
			raise VectorStoreError("Cannot save an uninitialized index.")

		try:
			faiss.write_index(self.index, str(self.index_path))
			with self.metadata_path.open("w", encoding="utf-8") as f:
				json.dump(self.metadata, f, ensure_ascii=True, indent=2)
		except Exception as exc:  # noqa: BLE001
			raise VectorStoreError(f"Failed to save vector store: {exc}") from exc

	def load(self) -> bool:
		"""Load FAISS index and metadata from disk. Returns True if loaded."""
		if not self.index_path.exists() or not self.metadata_path.exists():
			self.index = None
			self.metadata = []
			return False

		try:
			self.index = faiss.read_index(str(self.index_path))
			with self.metadata_path.open("r", encoding="utf-8") as f:
				loaded_metadata = json.load(f)
		except Exception as exc:  # noqa: BLE001
			raise VectorStoreError(f"Failed to load vector store: {exc}") from exc

		if not isinstance(loaded_metadata, list):
			raise VectorStoreError("Loaded metadata must be a list.")

		self.metadata = loaded_metadata

		if self.size != len(self.metadata):
			raise VectorStoreError(
				"Loaded index size does not match metadata size "
				f"({self.size} != {len(self.metadata)})."
			)

		return True

	def clear(self, delete_from_disk: bool = False) -> None:
		"""Reset in-memory store and optionally remove persisted files."""
		self.index = None
		self.metadata = []

		if delete_from_disk:
			if self.index_path.exists():
				self.index_path.unlink()
			if self.metadata_path.exists():
				self.metadata_path.unlink()
