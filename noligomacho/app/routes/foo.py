from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0,
    # other params...
)

chain = prompt | llm
