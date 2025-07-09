from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document

from app.services.vector_store import VectorStoreService
prompt = PromptTemplate(
    input_variables=["context", "input"],
    template="""
<system>
You are a legal assistant trained in jurisprudence. Use the context below to answer the user's question.
If you don't know, say "I'm not sure based on the available jurisprudence."
If there is no relevant context, guide the user to provide more information or clarify their question.
Do not, under any circumstances, provide legal advice or opinions without proper context.
Refuse to answer questions if there is insufficient context or if the question is outside your expertise.
If the context tags are empty, respond with "I'm not sure based on the available jurisprudence."
</system>

<question>
{question}
</question>

<context>
{context}
</context>

Answer format:
Answer like a legal scholar, referencing any relevant legal principles.
""")


def augment_with_veredict(docs: list[Document]) -> str:
    # TODO: Implement a function that augments the documents with a veredict or summary.
    return None


llm = ChatOllama(
    model="llama3:8b",
    temperature=0,
)


reranker = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(
        model_name="Qwen/Qwen3-Reranker-0.6B",
    ),
    top_n=8,
)

qa_chain = (
        {
            "context": ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=VectorStoreService().whole_doc_retriever,
            ) | augment_with_veredict,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
)

