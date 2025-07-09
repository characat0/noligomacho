from langchain_huggingface import HuggingFaceEmbeddings

embedding_qwen = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={
        "truncate_dim": 1024,
    },
)
