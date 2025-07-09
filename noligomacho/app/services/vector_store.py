from functools import lru_cache
from typing import BinaryIO

import numpy as np
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_elasticsearch import ElasticsearchStore, ElasticsearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from tempfile import SpooledTemporaryFile
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from app.routes.hypothetical_expansion import expansion_chain
from app.services.models import embedding_qwen


@lru_cache(None)
class VectorStoreService:
    def __init__(self):
        self._es_index = "documents-whole"
        self._embeddings = embedding_qwen
        self.vector_store = ElasticsearchStore(
            embedding=self._embeddings,
            index_name="documents",
            es_url="http://localhost:9200",
        )

        self.whole_doc_retriever = ElasticsearchRetriever.from_es_params(
            index_name=self._es_index,
            body_func=self._bm25_vector_query_highlight,
            content_field="text",
            url="http://localhost:9200",
        )
        self.retriever = EnsembleRetriever(
            retrievers=[
                ElasticsearchRetriever.from_es_params(
                    index_name=self._es_index,
                    body_func=self._vector_query,
                    content_field="text",
                    url="http://localhost:9200",
                ),
                ElasticsearchRetriever.from_es_params(
                    index_name=self._es_index,
                    body_func=self._bm25_query,
                    content_field="text",
                    url="http://localhost:9200",
                )
            ],
            weights=[0.5, 0.5],
        )
        self.text_splitter = SentenceTransformersTokenTextSplitter(
            model_name='sentence-transformers/all-mpnet-base-v2',
            tokens_per_chunk=256,  # Limit is 384 for sentence-transformers/all-mpnet-base-v2
            chunk_overlap=32,
        )

    def _vector_query(self, query: str) -> dict:
        vector = expansion_chain.invoke(query).embedding
        return {
            "knn": {
                "field": "vector",
                "query_vector": vector,
                "k": 4,
                "num_candidates": 128,
                "similarity": 0.8,
            }
        }

    def _bm25_query(self, query: str) -> dict:
        """Create a BM25 query for Elasticsearch."""
        return {
            "query": {
                "match": {
                    "text": query,
                }
            }
        }

    def _bm25_vector_query_highlight(self, query: str) -> dict:
        vector = expansion_chain.invoke(query).embedding
        return {
            #  TODO: Get query from Juan
        }

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split a document into smaller chunks."""
        return self.text_splitter.split_documents(documents)

    def add_file(self, file: SpooledTemporaryFile | BinaryIO) -> str:
        """Add a document to the vector store."""
        documents = [
            Document(
                page_content=file.read().decode('utf-8'),
                metadata={"source": file.name}
            )
        ]
        documents = self.text_splitter.split_documents(documents)
        return self.vector_store.add_documents(documents)[0]

    def add_files(self, files: list[tuple[SpooledTemporaryFile | BinaryIO, str]]) -> list[str]:
        """Add multiple documents to the vector store."""
        documents = [
            Document(page_content=file[0].read().decode('utf-8'), metadata={"source": file[1]})
            for file in files
        ]
        documents = self.text_splitter.split_documents(documents)
        return self.vector_store.add_documents(documents)

    def add_whole_files(self, files: list[tuple[SpooledTemporaryFile | BinaryIO, str]]) -> list[str]:
        """Add whole files to the vector store without splitting."""
        documents = [
            Document(page_content=file[0].read().decode('utf-8'), metadata={"source": file[1]})
            for file in files
        ]
        return self.vector_store.add_documents(documents)

    def similarity_search(self, query, k=5):
        """Search for similar documents in the vector store."""
        return self.vector_store.similarity_search(query, k=k)

    def delete(self, document_id):
        """Delete a document from the vector store."""
        self.vector_store.delete(document_id)
