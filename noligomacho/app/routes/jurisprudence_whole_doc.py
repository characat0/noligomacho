from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document

from app.services.reranker import OllamaCrossEncoder
from app.services.vector_store import VectorStoreService

prompt = PromptTemplate(
    input_variables=["context", "input"],
    template="""
<system>
You are an AI assistant specialized in jurisprudence.
Your task is to predict a likely verdict based on a user's query and relevant legal context_element s (retrieved from other cases).
You must follow these rules:
- Use only the provided legal context_element s to support your prediction.
- Clearly relate the facts in the user's query to the facts, principles, or outcomes in the retrieved cases.
- There should be context_element for you to make a prediction.
- Respond in a scholarly legal tone, offering your verdict prediction and briefly explaining the reasoning using the context.
- Provide citations to the relevant cases in the context to support your prediction.
- Your prediction should go beyond a binary outcome—include likely penalties such as years of imprisonment, fines, or other sentencing details when applicable.
</system>

<question>
{question}
</question>

<context>
{context}
</context>
""")


def augment_single(doc: Document) -> str:
    highlight = doc.metadata.get("highlight", [])
    filter(lambda x: x, highlight)
    text = " [...] ".join(highlight)
    if not text:
        return ""
    tail_length = 1000
    full_text = doc.page_content
    source = doc.metadata.get('_source', {}).get('metadata', {}).get('source', '')
    tail_text = full_text[-tail_length:] if full_text else ""
    return f"<context_element><case>{source[:-4]}</case><text>{text}</text><verdict>{tail_text}</verdict></context_element>".strip()

def augment_with_verdict(docs: list[Document]) -> str:
    augmented_docs = [augment_single(doc) for doc in docs if doc.page_content]
    filter(lambda x: x, augmented_docs)
    return "\n\n".join(augmented_docs)


llm = ChatOllama(
    model="llama3:8b",
    temperature=0.15,
)


reranker = CrossEncoderReranker(
    model=OllamaCrossEncoder(
        model="dengcao/Qwen3-Reranker-0.6B:F16",
        base_url="http://localhost:11435",
    ),
    top_n=2,
)

qa_chain = (
        {
            "context": ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=VectorStoreService().whole_doc_retriever,
            ) | augment_with_verdict,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
)

