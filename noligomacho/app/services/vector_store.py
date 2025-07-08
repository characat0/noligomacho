from functools import lru_cache
from typing import BinaryIO

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from tempfile import SpooledTemporaryFile
from langchain_text_splitters import SentenceTransformersTokenTextSplitter


@lru_cache(None)
class VectorStoreService:
    def __init__(self):
        self.vector_store = InMemoryVectorStore(
            OllamaEmbeddings(model="llama3.2:1b", temperature=0),
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 3, 'fetch_k': 50}
        )
        self.text_splitter = SentenceTransformersTokenTextSplitter(
            model_name='sentence-transformers/all-mpnet-base-v2',
            tokens_per_chunk=256,  # Limit is 384 for sentence-transformers/all-mpnet-base-v2
            chunk_overlap=32,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split a document into smaller chunks."""
        # For simplicity, we return the document as is.
        # In a real application, you might want to implement a more sophisticated splitting logic.
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

    def similarity_search(self, query, k=5):
        """Search for similar documents in the vector store."""
        return self.vector_store.similarity_search(query, k=k)

    def delete(self, document_id):
        """Delete a document from the vector store."""
        self.vector_store.delete(document_id)
