from dataclasses import dataclass


class ChunkingError(Exception):
	"""Raised when chunking input is invalid."""


@dataclass(slots=True)
class TextChunk:
	chunk_id: int
	content: str
	start_char: int
	end_char: int


def chunk_text(
	text: str,
	chunk_size: int = 800,
	chunk_overlap: int = 120,
	min_chunk_chars: int = 50,
) -> list[TextChunk]:
	"""Split text into overlapping chunks suitable for embedding and retrieval."""
	if not text or not text.strip():
		raise ChunkingError("Cannot chunk empty text.")

	if chunk_size <= 0:
		raise ChunkingError("chunk_size must be > 0.")

	if chunk_overlap < 0:
		raise ChunkingError("chunk_overlap must be >= 0.")

	if chunk_overlap >= chunk_size:
		raise ChunkingError("chunk_overlap must be smaller than chunk_size.")

	normalized = " ".join(text.split())
	if len(normalized) <= chunk_size:
		return [TextChunk(chunk_id=0, content=normalized, start_char=0, end_char=len(normalized))]

	chunks: list[TextChunk] = []
	step = chunk_size - chunk_overlap
	start = 0
	chunk_id = 0

	while start < len(normalized):
		end = min(start + chunk_size, len(normalized))

		# Move end to the nearest whitespace to avoid cutting words abruptly.
		if end < len(normalized):
			while end > start and not normalized[end - 1].isspace():
				end -= 1
			if end == start:
				end = min(start + chunk_size, len(normalized))

		chunk_content = normalized[start:end].strip()
		if len(chunk_content) >= min_chunk_chars:
			chunks.append(
				TextChunk(
					chunk_id=chunk_id,
					content=chunk_content,
					start_char=start,
					end_char=end,
				)
			)
			chunk_id += 1

		if end >= len(normalized):
			break

		start += step

	if not chunks:
		raise ChunkingError("No valid chunks were produced from the text.")

	return chunks
