from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

embedding_qwen = OllamaEmbeddings(
    model="dengcao/Qwen3-Embedding-0.6B:f16",
)
