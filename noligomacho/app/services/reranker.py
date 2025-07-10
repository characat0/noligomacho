from typing import Any, Self, Optional

from langchain.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
from ollama import Client, AsyncClient
from pydantic import BaseModel, model_validator, PrivateAttr

class ReRankRawResult(BaseModel):
    index: int
    relevance_score: float

class ReRankRawResponse(BaseModel):
    model: str
    results: list[ReRankRawResult]

class ReRankRequest(BaseModel):
    model: str
    query: str
    top_n: int = 1
    documents: list[str]
    return_documents: bool = False


class OllamaCrossEncoder(BaseModel, BaseCrossEncoder):

    model: str
    """Model name to use."""

    base_url: Optional[str] = None
    """Base url the model is hosted under."""

    client_kwargs: Optional[dict] = {}
    """Additional kwargs to pass to the httpx clients. 
    These arguments are passed to both synchronous and async clients.
    Use sync_client_kwargs and async_client_kwargs to pass different arguments
    to synchronous and asynchronous clients.
    """

    async_client_kwargs: Optional[dict] = {}
    """Additional kwargs to merge with client_kwargs before
    passing to the httpx AsyncClient.
    For a full list of the params, see [this link](https://www.python-httpx.org/api/#asyncclient)
    """

    sync_client_kwargs: Optional[dict] = {}
    """Additional kwargs to merge with client_kwargs before
    passing to the httpx Client.
    For a full list of the params, see [this link](https://www.python-httpx.org/api/#client)
    """

    _client: Client = PrivateAttr(default=None)  # type: ignore
    """
    The client to use for making requests.
    """

    _async_client: AsyncClient = PrivateAttr(default=None)  # type: ignore
    """
    The async client to use for making requests.
    """

    @model_validator(mode="after")
    def _set_clients(self) -> Self:
        """Set clients to use for ollama."""
        client_kwargs = self.client_kwargs or {}

        sync_client_kwargs = client_kwargs
        if self.sync_client_kwargs:
            sync_client_kwargs = {**sync_client_kwargs, **self.sync_client_kwargs}

        async_client_kwargs = client_kwargs
        if self.async_client_kwargs:
            async_client_kwargs = {**async_client_kwargs, **self.async_client_kwargs}

        self._client = Client(host=self.base_url, **sync_client_kwargs)
        self._async_client = AsyncClient(host=self.base_url, **async_client_kwargs)
        return self


    def _score_single(self, text_pair: tuple[str, str]) -> float:
        """Score a single text pair using the Ollama model."""
        response = self._client._request(
            ReRankRawResponse,
            'POST',
            "/api/rerank",
            json=ReRankRequest(
                model=self.model,
                query=text_pair[0],
                documents=[text_pair[1]],
                top_n=1,
                return_documents=False
            ).model_dump()
        )
        return response.results[0].relevance_score

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        return [
            self._score_single(text_pair) for text_pair in text_pairs
        ]
