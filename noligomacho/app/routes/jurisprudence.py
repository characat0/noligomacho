from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
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

def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)

llm = ChatOllama(
    model="llama3:8b",
    temperature=0,
    # other params...
)

qa_chain = (
    {
        "context": VectorStoreService().retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

