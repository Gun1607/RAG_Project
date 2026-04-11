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

		if self.settings.openai_api_key:
			answer = self._generate_with_openai(question=question_clean, context=context_clean)
			if answer:
				return answer

		return self._extractive_fallback_answer(question=question_clean, context=context_clean)

	def _generate_with_openai(self, question: str, context: str) -> str | None:
		try:
			from openai import OpenAI

			client = OpenAI(api_key=self.settings.openai_api_key)
			response = client.chat.completions.create(
				model=self.settings.openai_model,
				messages=[
					{
						"role": "system",
						"content": (
							"You are a helpful document assistant. "
							"Use only the provided context. "
							"If information is missing in context, clearly say so."
						),
					},
					{
						"role": "user",
						"content": (
							f"Question:\n{question}\n\n"
							f"Context:\n{context}\n\n"
							"Answer concisely and include short source markers such as [1], [2]."
						),
					},
				],
				temperature=0.1,
			)
		except Exception:
			return None

		content = response.choices[0].message.content
		if content and content.strip():
			return content.strip()
		return None

	def _extractive_fallback_answer(self, question: str, context: str) -> str:
		context_preview = context[:1200].strip()
		if not context_preview:
			return "I could not find enough relevant context to answer this question."

		return (
			"LLM API is not configured or unavailable, so this is a context-grounded fallback answer.\n\n"
			f"Question: {question}\n\n"
			"Most relevant retrieved content:\n"
			f"{context_preview}"
		)
