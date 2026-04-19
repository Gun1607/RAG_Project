from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.services.chunking import TextChunk, chunk_text
from app.services.embedding import EmbeddingService
from app.services.llm_service import LLMService
from app.services.pdf_parser import extract_text_from_pdf_bytes, extract_text_from_pdf_path
from app.services.vectorstore import VectorStoreError, VectorStoreService


class RAGPipelineError(Exception):
	"""Raised when the RAG pipeline cannot complete a requested operation."""


@dataclass(slots=True)
class RetrievalResult:
	question: str
	answer: str
	sources: list[dict[str, Any]]


class RAGPipeline:
	"""Coordinates ingestion and retrieval-augmented answer generation."""

	def __init__(self, settings: Settings | None = None) -> None:
		self.settings = settings or get_settings()
		self.llm_service = LLMService(self.settings)
		self.embedding_service = EmbeddingService(
			model_name=self.settings.embedding_model_name,
			batch_size=self.settings.embedding_batch_size,
		)
		self.vectorstore = VectorStoreService(
			index_path=self.settings.faiss_index_path,
			metadata_path=self.settings.faiss_metadata_path,
		)
		self.vectorstore.load()

	def ingest_pdf_file(self, pdf_path: str | Path, source_name: str | None = None) -> dict[str, Any]:
		"""Ingest a PDF file from disk into the vector store."""
		path = Path(pdf_path)
		text = extract_text_from_pdf_path(path)
		source = source_name or path.name
		return self._ingest_text(text=text, source_name=source)

	def ingest_pdf_bytes(self, pdf_bytes: bytes, source_name: str, replace_existing: bool = False) -> dict[str, Any]:
		"""Ingest a PDF represented as bytes into the vector store."""
		if not source_name:
			raise RAGPipelineError("source_name is required when ingesting PDF bytes.")
		text = extract_text_from_pdf_bytes(pdf_bytes)
		return self._ingest_text(text=text, source_name=source_name, replace_existing=replace_existing)

	def answer_question(self, question: str, top_k: int | None = None) -> RetrievalResult:
		"""Retrieve relevant chunks and generate an answer for the question."""
		if not question or not question.strip():
			raise RAGPipelineError("Question cannot be empty.")

		if not self.vectorstore.is_ready:
			self.vectorstore.load()

		if not self.vectorstore.is_ready:
			raise RAGPipelineError("No indexed content available. Upload and process documents first.")

		k = top_k or self.settings.retrieval_top_k
		query_vector = self.embedding_service.embed_query(question)

		try:
			retrieved_chunks = self.vectorstore.search(query_embedding=query_vector, top_k=k)
		except VectorStoreError as exc:
			raise RAGPipelineError(f"Retrieval failed: {exc}") from exc

		if not retrieved_chunks:
			raise RAGPipelineError("No relevant chunks were retrieved for this question.")

		context = self._build_context(retrieved_chunks)
		answer = self.llm_service.generate_answer(question=question, context=context)

		sources = [
			{
				"source": item.get("source", "unknown"),
				"chunk_id": item.get("chunk_id"),
				"score": item.get("score"),
				"text": item.get("text", ""),
			}
			for item in retrieved_chunks
		]

		return RetrievalResult(question=question.strip(), answer=answer, sources=sources)

	def get_index_status(self) -> dict[str, Any]:
		"""Return simple diagnostics about indexed data availability."""
		return {
			"ready": self.vectorstore.is_ready,
			"vector_count": self.vectorstore.size,
			"index_path": str(self.settings.faiss_index_path),
			"metadata_path": str(self.settings.faiss_metadata_path),
		}

	def _ingest_text(self, text: str, source_name: str, replace_existing: bool = False) -> dict[str, Any]:
		if replace_existing:
			self.vectorstore.clear(delete_from_disk=True)

		chunks = chunk_text(
			text=text,
			chunk_size=self.settings.chunk_size,
			chunk_overlap=self.settings.chunk_overlap,
			min_chunk_chars=self.settings.min_chunk_chars,
		)

		chunk_texts = [chunk.content for chunk in chunks]
		embeddings = self.embedding_service.embed_documents(chunk_texts)
		metadata = self._build_chunk_metadata(source_name=source_name, chunks=chunks)

		added_count = self.vectorstore.add_embeddings(embeddings=embeddings, metadata=metadata)
		self.vectorstore.save()

		return {
			"source": source_name,
			"index_mode": "replaced" if replace_existing else "appended",
			"chunks_created": len(chunks),
			"vectors_added": added_count,
			"total_vectors": self.vectorstore.size,
		}

	def _build_chunk_metadata(self, source_name: str, chunks: list[TextChunk]) -> list[dict[str, Any]]:
		records: list[dict[str, Any]] = []
		for chunk in chunks:
			records.append(
				{
					"source": source_name,
					"chunk_id": chunk.chunk_id,
					"start_char": chunk.start_char,
					"end_char": chunk.end_char,
					"text": chunk.content,
				}
			)
		return records

	def _build_context(self, retrieved_chunks: list[dict[str, Any]]) -> str:
		sections: list[str] = []
		for idx, item in enumerate(retrieved_chunks, start=1):
			source = item.get("source", "unknown")
			chunk_id = item.get("chunk_id", -1)
			text = item.get("text", "")
			sections.append(f"[{idx}] source={source} chunk={chunk_id}\n{text}")
		return "\n\n".join(sections)


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
	"""Return a shared, lazily initialized RAG pipeline instance."""
	return RAGPipeline(get_settings())
