from pathlib import Path

import fitz


class PDFParserError(Exception):
	"""Raised when PDF parsing fails."""


def _normalize_text(text: str) -> str:
	# Normalize repeated whitespace while preserving paragraph separation.
	lines = [line.strip() for line in text.splitlines()]
	cleaned_lines = [line for line in lines if line]
	return "\n".join(cleaned_lines)


def extract_text_from_pdf_path(pdf_path: str | Path) -> str:
	"""Extract all text from a PDF file path."""
	path = Path(pdf_path)
	if not path.exists() or not path.is_file():
		raise PDFParserError(f"PDF file not found: {path}")

	try:
		with fitz.open(path) as doc:
			pages = [page.get_text("text") for page in doc]
	except Exception as exc:  # noqa: BLE001
		raise PDFParserError(f"Failed to parse PDF at {path}: {exc}") from exc

	text = "\n\n".join(pages).strip()
	if not text:
		raise PDFParserError("No extractable text found in PDF.")

	return _normalize_text(text)


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
	"""Extract all text from in-memory PDF bytes."""
	if not pdf_bytes:
		raise PDFParserError("Empty PDF bytes provided.")

	try:
		with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
			pages = [page.get_text("text") for page in doc]
	except Exception as exc:  # noqa: BLE001
		raise PDFParserError(f"Failed to parse PDF bytes: {exc}") from exc

	text = "\n\n".join(pages).strip()
	if not text:
		raise PDFParserError("No extractable text found in PDF.")

	return _normalize_text(text)
