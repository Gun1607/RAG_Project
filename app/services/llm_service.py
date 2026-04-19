from app.core.config import Settings, get_settings


class LLMServiceError(Exception):
	"""Raised when LLM response generation fails."""


class LLMService:
	"""Handles answer generation using an LLM with a robust fallback mode."""

	def __init__(self, settings: Settings | None = None) -> None:
		self.settings = settings or get_settings()

	def generate_answer(self, question: str, context: str) -> str:
		"""Generate a context-grounded answer for the given question."""
		question_clean = question.strip()
		if not question_clean:
			raise LLMServiceError("Question cannot be empty.")

		context_clean = context.strip()
		if not context_clean:
			raise LLMServiceError("Context cannot be empty.")

		if not self.settings.gemini_api_key:
			return self._extractive_fallback_answer(
				question=question_clean,
				context=context_clean,
				reason="GEMINI_API_KEY is missing.",
			)

		answer, error_reason = self._generate_with_gemini(question=question_clean, context=context_clean)
		if answer:
			return answer

		return self._extractive_fallback_answer(
			question=question_clean,
			context=context_clean,
			reason=error_reason,
		)

	def _generate_with_gemini(self, question: str, context: str) -> tuple[str | None, str | None]:
		try:
			from google import genai
			from google.genai import types

			client = genai.Client(api_key=self.settings.gemini_api_key)
			prompt = (
				"You are a helpful document assistant. "
				"Use only the provided context. "
				"If information is missing in context, clearly say so.\n\n"
				f"Question:\n{question}\n\n"
				f"Context:\n{context}\n\n"
				"Answer concisely and include short source markers such as [1], [2]."
			)
			candidate_models = [
				self.settings.gemini_model,
				"gemini-flash-latest",
				"gemini-2.0-flash",
			]
			last_error: str | None = None

			for model_name in dict.fromkeys(candidate_models):
				try:
					response = client.models.generate_content(
						model=model_name,
						contents=prompt,
						config=types.GenerateContentConfig(temperature=0.1),
					)
				except Exception as exc:  # noqa: BLE001
					last_error = f"Gemini request failed on model '{model_name}': {type(exc).__name__}: {exc}"
					continue

				content = getattr(response, "text", None)
				if content and content.strip():
					return content.strip(), None

				last_error = f"Gemini model '{model_name}' returned an empty response."

			return None, last_error or "Gemini returned an empty response."
		except Exception as exc:  # noqa: BLE001
			return None, f"Gemini initialization failed: {type(exc).__name__}: {exc}"

	def _extractive_fallback_answer(self, question: str, context: str, reason: str | None = None) -> str:
		context_preview = context[:1200].strip()
		if not context_preview:
			return "I could not find enough relevant context to answer this question."

		reason_text = f"\n\nReason: {reason}" if reason else ""

		return (
			"LLM API is not configured or unavailable, so this is a context-grounded fallback answer.\n\n"
			f"Configured model: {self.settings.gemini_model}"
			+ reason_text
			+ "\n\n"
			f"Question: {question}\n\n"
			"Most relevant retrieved content:\n"
			f"{context_preview}"
		)
