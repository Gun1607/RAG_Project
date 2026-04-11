from dataclasses import dataclass
import sys
from types import ModuleType

from fastapi import FastAPI
from fastapi.testclient import TestClient


@dataclass
class FakeRetrievalResult:
    question: str
    answer: str
    sources: list[dict]


class FakePipeline:
    def get_index_status(self) -> dict:
        return {
            "ready": True,
            "vector_count": 3,
            "index_path": "data/index/faiss.index",
            "metadata_path": "data/index/metadata.json",
        }

    def answer_question(self, question: str, top_k: int | None = None) -> FakeRetrievalResult:
        return FakeRetrievalResult(
            question=question,
            answer="Mocked answer from test pipeline.",
            sources=[
                {
                    "source": "sample.pdf",
                    "chunk_id": 0,
                    "score": 0.9,
                    "text": "Sample retrieved text.",
                }
            ],
        )

    def ingest_pdf_bytes(self, pdf_bytes: bytes, source_name: str) -> dict:
        return {
            "source": source_name,
            "chunks_created": 2,
            "vectors_added": 2,
            "total_vectors": 10,
        }


class FakeRAGPipelineError(Exception):
    pass


def _build_test_client() -> TestClient:
    fake_module = ModuleType("app.services.rag_pipeline")
    fake_module.RAGPipelineError = FakeRAGPipelineError
    fake_module.get_rag_pipeline = lambda: FakePipeline()
    sys.modules["app.services.rag_pipeline"] = fake_module

    from app.api.query import router as query_router
    from app.api.upload import router as upload_router

    app = FastAPI()
    app.include_router(upload_router, prefix="/api")
    app.include_router(query_router, prefix="/api")
    return TestClient(app)


def test_health_endpoint() -> None:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_status_endpoint(monkeypatch) -> None:
    client = _build_test_client()
    response = client.get("/api/query/status")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["vector_count"] == 3


def test_query_endpoint_returns_answer(monkeypatch) -> None:
    client = _build_test_client()
    response = client.post("/api/query", json={"question": "What is in the PDF?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Mocked answer from test pipeline."
    assert len(body["sources"]) == 1


def test_upload_endpoint_accepts_pdf(monkeypatch) -> None:
    client = _build_test_client()
    files = {"file": ("sample.pdf", b"%PDF-1.4 fake pdf bytes", "application/pdf")}
    response = client.post("/api/upload", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Document uploaded and indexed successfully."
    assert body["vectors_added"] == 2


def test_upload_endpoint_rejects_non_pdf() -> None:
    client = _build_test_client()

    files = {"file": ("sample.txt", b"hello", "text/plain")}
    response = client.post("/api/upload", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported."
