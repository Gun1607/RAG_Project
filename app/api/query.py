from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import ErrorResponse, QueryRequest, QueryResponse, SourceChunk
from app.services.rag_pipeline import RAGPipelineError, get_rag_pipeline


router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
	"",
	response_model=QueryResponse,
	responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def ask_question(payload: QueryRequest) -> QueryResponse:
	"""Ask a natural-language question against indexed document content."""
	question = payload.question.strip()
	if not question:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")

	try:
		result = get_rag_pipeline().answer_question(question=question, top_k=payload.top_k)
	except RAGPipelineError as exc:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected query error: {exc}") from exc

	sources = [SourceChunk(**source) for source in result.sources]
	return QueryResponse(question=result.question, answer=result.answer, sources=sources)


@router.get("/status")
def query_status() -> dict[str, Any]:
	"""Return ingestion/index readiness details for quick diagnostics."""
	return get_rag_pipeline().get_index_status()
