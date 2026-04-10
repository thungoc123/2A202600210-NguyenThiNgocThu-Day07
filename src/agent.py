from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self._store = store
        self._llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        normalized_question = (question or "").strip()
        if not normalized_question:
            normalized_question = "Please summarize available knowledge."

        retrieved = self._store.search(normalized_question, top_k=top_k)
        context_lines = [item.get("content", "") for item in retrieved if item.get("content")]
        context_block = "\n\n".join(context_lines) if context_lines else "No relevant context found."

        prompt = (
            "You are a helpful knowledge base assistant.\n"
            "Use ONLY the provided context to answer. If information is missing, say so clearly.\n\n"
            f"Question:\n{normalized_question}\n\n"
            f"Context:\n{context_block}\n\n"
            "Answer:"
        )
        return self._llm_fn(prompt)
