from functools import lru_cache
from typing import BinaryIO

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from tempfile import SpooledTemporaryFile


@lru_cache(None)
class VectorStoreService:
    def __init__(self):
        self.vector_store = InMemoryVectorStore(
            OllamaEmbeddings(model="llama3.2:1b", temperature=0),
        )

    def as_retriever(self):
        """Return the vector store as a retriever."""
        return self.vector_store.as_retriever()

    def add_file(self, file: SpooledTemporaryFile | BinaryIO) -> str:
        """Add a document to the vector store."""
        document = Document(
            page_content=file.read().decode('utf-8'),
            metadata={"source": file.name}
        )
        return self.vector_store.add_documents([document])[0]

    def add_files(self, files: list[SpooledTemporaryFile | BinaryIO]) -> list[str]:
        """Add multiple documents to the vector store."""
        documents = [
            Document(page_content=file.read().decode('utf-8'), metadata={"source": file.name})
            for file in files
        ]
        return self.vector_store.add_documents(documents)

    def similarity_search(self, query, k=5):
        """Search for similar documents in the vector store."""
        return self.vector_store.similarity_search(query, k=k)

    def delete(self, document_id):
        """Delete a document from the vector store."""
        self.vector_store.delete(document_id)
