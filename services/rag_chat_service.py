import asyncio
import json
import os
import re
from pathlib import Path
from typing import AsyncGenerator, Dict, List

from app.core.config import settings
from app.core.logger import get_logger
from app.services.ollama_graphrag_embeddings import (
    graphrag_operation_lock,
    register_graphrag_ollama_embeddings,
)

logger = get_logger(service="rag_chat")


class RAGChatService:
    """Query a previously built GraphRAG index and return an SSE stream."""

    REQUIRED_TABLES = (
        "entities",
        "communities",
        "community_reports",
        "text_units",
        "relationships",
    )

    def __init__(self):
        self.project_dir = Path(settings.GRAPHRAG_PROJECT_DIR).expanduser().resolve()
        self.data_dir = self.project_dir / settings.GRAPHRAG_DATA_DIR
        self.output_root = self.data_dir / "output"
        self.config_path = self.data_dir / "settings.yaml"
        numba_cache_dir = self.data_dir / "cache" / "_numba"
        numba_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(numba_cache_dir)

    def _resolve_output_dir(self, index_id: str) -> Path:
        if index_id == "default":
            output_dir = self.output_root.resolve()
        else:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", index_id):
                raise ValueError("Invalid GraphRAG index id")
            output_dir = (self.output_root / index_id).resolve()
            if output_dir.parent != self.output_root.resolve():
                raise ValueError("Invalid GraphRAG index path")

        if not output_dir.is_dir():
            raise FileNotFoundError(f"GraphRAG index does not exist: {index_id}")
        return output_dir

    @staticmethod
    def _model_config_overrides() -> Dict[str, str]:
        """Use the same providers already configured for the main application."""
        return {
            "models.default_chat_model.api_base": settings.DEEPSEEK_BASE_URL,
            "models.default_chat_model.api_key": settings.DEEPSEEK_API_KEY,
            "models.default_chat_model.model": settings.DEEPSEEK_MODEL,
            "models.default_chat_model.request_timeout": 60.0,
            "models.default_chat_model.max_retries": 2,
            "models.default_chat_model.concurrent_requests": 4,
            "models.default_embedding_model.api_base": (
                f"{settings.OLLAMA_BASE_URL.rstrip('/')}/v1"
            ),
            "models.default_embedding_model.api_key": "ollama",
            "models.default_embedding_model.model": settings.OLLAMA_EMBEDDING_MODEL,
            "models.default_embedding_model.request_timeout": 120.0,
            "models.default_embedding_model.max_retries": 4,
            "models.default_embedding_model.concurrent_requests": 1,
        }

    @staticmethod
    def _last_user_message(messages: List[Dict[str, str]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user" and message.get("content"):
                return message["content"]
        raise ValueError("No user question was provided")

    async def _query(self, query: str, index_id: str) -> str:
        # GraphRAG imports spaCy/Thinc, which in turn import torch. Importing
        # torch first prevents a Windows DLL initialization-order failure.
        import torch  # noqa: F401
        import graphrag.api as api
        from graphrag.config.load_config import load_config
        from graphrag.storage.file_pipeline_storage import FilePipelineStorage
        from graphrag.utils.storage import load_table_from_storage

        output_dir = self._resolve_output_dir(index_id)
        async with graphrag_operation_lock:
            register_graphrag_ollama_embeddings(
                settings.OLLAMA_BASE_URL,
                settings.OLLAMA_EMBEDDING_MODEL,
            )
            config = load_config(
                self.data_dir,
                self.config_path,
                {
                    **self._model_config_overrides(),
                    "output.base_dir": str(output_dir),
                    "vector_store.default_vector_store.db_uri": str(output_dir / "lancedb"),
                },
            )
            storage = FilePipelineStorage(root_dir=str(output_dir))
            tables = {
                name: await load_table_from_storage(name, storage)
                for name in self.REQUIRED_TABLES
            }
            try:
                covariates = await load_table_from_storage("covariates", storage)
            except ValueError:
                covariates = None

            response, _ = await api.local_search(
                config=config,
                entities=tables["entities"],
                communities=tables["communities"],
                community_reports=tables["community_reports"],
                text_units=tables["text_units"],
                relationships=tables["relationships"],
                covariates=covariates,
                community_level=settings.GRAPHRAG_COMMUNITY_LEVEL,
                response_type=settings.GRAPHRAG_RESPONSE_TYPE,
                query=query,
                callbacks=[],
            )
        if isinstance(response, str):
            return response
        return json.dumps(response, ensure_ascii=False)

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        index_id: str,
    ) -> AsyncGenerator[str, None]:
        try:
            query = self._last_user_message(messages)
            logger.info(f"Querying GraphRAG index {index_id}: {query[:80]}")
            response = await self._query(query, index_id)
            for start in range(0, len(response), 8):
                chunk = response[start : start + 8]
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
        except Exception as exc:
            logger.error(f"GraphRAG query failed: {exc}", exc_info=True)
            message = f"文档问答失败：{exc}"
            yield f"data: {json.dumps(message, ensure_ascii=False)}\n\n"
