from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.models.schemas import ErrorResponse, UploadResponse
from app.services.rag_pipeline import RAGPipelineError, get_rag_pipeline


router = APIRouter(prefix="/upload", tags=["Upload"])

ALLOWED_EXTENSIONS = {".pdf"}
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024


def _validate_pdf_file(file: UploadFile, payload_size: int) -> None:
	filename = file.filename or ""
	ext = Path(filename).suffix.lower()
	if ext not in ALLOWED_EXTENSIONS:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Only PDF files are supported.",
		)

	if payload_size == 0:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Uploaded file is empty.",
		)

	if payload_size > MAX_UPLOAD_SIZE_BYTES:
		raise HTTPException(
			status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
			detail="File exceeds 20MB upload limit.",
		)


@router.post(
	"",
	response_model=UploadResponse,
	responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def upload_document(
	file: UploadFile = File(...),
	replace_existing: bool = Form(default=False),
) -> UploadResponse:
	"""Upload and ingest a PDF document into the FAISS vector store."""
	try:
		file_bytes = await file.read()
		_validate_pdf_file(file=file, payload_size=len(file_bytes))

		settings = get_settings()
		original_name = Path(file.filename or "document.pdf").name
		stored_name = f"{uuid4().hex}_{original_name}"
		output_path = settings.uploads_dir / stored_name

		output_path.write_bytes(file_bytes)

		try:
			result = get_rag_pipeline().ingest_pdf_bytes(
				pdf_bytes=file_bytes,
				source_name=stored_name,
				replace_existing=replace_existing,
			)
		except RAGPipelineError as exc:
			raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

		return UploadResponse(
			message="Document uploaded and indexed successfully." if not replace_existing else "Document uploaded and indexed successfully after clearing previous documents.",
			filename=stored_name,
			source=result["source"],
			index_mode=result["index_mode"],
			chunks_created=result["chunks_created"],
			vectors_added=result["vectors_added"],
			total_vectors=result["total_vectors"],
		)
	finally:
		await file.close()
