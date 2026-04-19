from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
	message: str = Field(..., description="Human-readable ingestion status message")
	filename: str = Field(..., description="Stored filename for the uploaded PDF")
	source: str = Field(..., description="Source identifier used in vector metadata")
	index_mode: str = Field(..., description="Whether the upload replaced the existing index or appended to it")
	chunks_created: int = Field(..., ge=0)
	vectors_added: int = Field(..., ge=0)
	total_vectors: int = Field(..., ge=0)


class QueryRequest(BaseModel):
	question: str = Field(..., min_length=1, description="Natural-language question about uploaded documents")
	top_k: int | None = Field(default=None, ge=1, le=20, description="Number of relevant chunks to retrieve")


class SourceChunk(BaseModel):
	source: str
	chunk_id: int | None = None
	score: float | None = None
	text: str


class QueryResponse(BaseModel):
	question: str
	answer: str
	sources: list[SourceChunk]


class ErrorResponse(BaseModel):
	detail: str
