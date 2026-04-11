import pytest

from app.services.chunking import ChunkingError, chunk_text


def test_chunk_text_returns_chunks_for_long_text() -> None:
    text = " ".join(["token"] * 600)

    chunks = chunk_text(text=text, chunk_size=120, chunk_overlap=20, min_chunk_chars=10)

    assert len(chunks) > 1
    assert chunks[0].chunk_id == 0
    assert all(chunk.content for chunk in chunks)


def test_chunk_text_single_chunk_for_short_text() -> None:
    text = "This is a short sentence."

    chunks = chunk_text(text=text, chunk_size=200, chunk_overlap=20, min_chunk_chars=5)

    assert len(chunks) == 1
    assert chunks[0].content == text


def test_chunk_text_raises_for_invalid_overlap() -> None:
    with pytest.raises(ChunkingError):
        chunk_text(text="abc", chunk_size=50, chunk_overlap=50)


def test_chunk_text_raises_for_empty_input() -> None:
    with pytest.raises(ChunkingError):
        chunk_text(text="   ")
