from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.query import router as query_router
from app.api.upload import router as upload_router
from app.core.config import get_settings


settings = get_settings()


app = FastAPI(
	title=settings.app_name,
	version=settings.app_version,
	description="RAG assistant API for PDF upload, indexing, retrieval, and question answering.",
	debug=settings.debug,
)

# Permissive CORS for local frontend development. Tighten in production.
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(upload_router, prefix=settings.api_prefix)
app.include_router(query_router, prefix=settings.api_prefix)


@app.get("/")
def root() -> dict[str, str]:
	return {
		"message": "RAG-Based Intelligent Document Assistant API",
		"docs": "/docs",
		"openapi": "/openapi.json",
	}


@app.get("/health")
def health() -> dict[str, str]:
	return {"status": "ok", "service": settings.app_name, "version": settings.app_version}
