import asyncio
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any


# GraphRAG keeps chat and embedding clients in a process-wide singleton.  Those
# clients capture the cache configured for the request that created them.  A
# lock plus an explicit reset prevents a later upload/query from reusing a
# previous index's cache directory.
graphrag_operation_lock = asyncio.Lock()


class OllamaGraphRAGEmbeddings:
    """GraphRAG embedding adapter backed by Ollama's native /api/embed API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        max_retries: int = 4,
    ) -> None:
        self.endpoint = f"{base_url.rstrip('/')}/api/embed"
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        # GraphRAG runs several embedding workflows concurrently. Ollama on
        # Windows is more reliable when those native requests are serialized.
        self._request_lock = threading.Lock()

    def embed_batch(
        self,
        text_list: list[str],
        **_: Any,
    ) -> list[list[float]]:
        if not text_list:
            return []

        payload = json.dumps(
            {"model": self.model, "input": text_list},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )

        last_error: Exception | None = None
        with self._request_lock:
            for attempt in range(self.max_retries + 1):
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        result = json.load(response)
                    embeddings = result.get("embeddings")
                    if not isinstance(embeddings, list) or len(embeddings) != len(text_list):
                        raise RuntimeError("Ollama returned an invalid embedding response")
                    return embeddings
                except (OSError, ValueError, RuntimeError, urllib.error.HTTPError) as exc:
                    last_error = exc
                    if attempt >= self.max_retries:
                        break
                    time.sleep(min(2**attempt, 8))

        raise RuntimeError(
            f"Ollama native embedding failed after {self.max_retries + 1} attempts: "
            f"{last_error}"
        )

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_batch([text], **kwargs)[0]

    async def aembed_batch(
        self,
        text_list: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_batch, text_list, **kwargs)

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        return (await self.aembed_batch([text], **kwargs))[0]


def reset_and_register_graphrag_models(base_url: str, model: str) -> None:
    """Reset request-scoped GraphRAG clients and register native embeddings."""
    from graphrag.language_model.manager import ModelManager

    manager = ModelManager()
    manager.chat_models.clear()
    manager.embedding_models.clear()
    adapter = OllamaGraphRAGEmbeddings(base_url=base_url, model=model)
    for name in ("text_embedding", "local_search_embedding", "default_embedding"):
        manager.embedding_models[name] = adapter


def register_graphrag_ollama_embeddings(base_url: str, model: str) -> None:
    """Backward-compatible alias for resetting and registering GraphRAG models."""
    reset_and_register_graphrag_models(base_url, model)
