import json

import numpy as np
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from app.services.models import embedding_qwen


class Document(BaseModel):
    text: str


class ExpansionOutput(BaseModel):
    documents: list[Document]

class ExpansionOutputVector(BaseModel):
    embedding: list[float]


expansion = PydanticOutputParser(pydantic_object=ExpansionOutput)


prompt = PromptTemplate(
    input_variables=["query"],
    template="""
<system>
You are a retriever system responsible for answering questions when it comes to legal cases, on an example query you would generate the following document:
<example_query>
State Bank of India dismissal appeal Limitation Act Section 14 Tamil Nadu Shops and Establishments Act.
</example_query>

<example_document>
The appellant, who was appointed as a Clerk at the State Bank of India in 1962 and later promoted to Branch Manager, was placed under suspension in 1980 and subsequently removed from service in 1983. He appealed the decision through various channels, including the Local Board of the Bank, the Deputy Commissioner of Labour, and the Civil Court. His appeal to the Deputy Commissioner of Labour was dismissed in 1987 because the Tamil Nadu Shops and Establishments Act did not apply to nationalized banks, a decision upheld by the Supreme Court in 1988.

In 1988, the appellant filed a suit in the City Civil Court challenging his dismissal. The trial court dismissed the suit, stating it was filed beyond the prescribed limitation period. However, the appellant appealed the decision, and the appellate court found the suit to be within time. The respondent, in turn, filed a second appeal, and the Madras High Court ruled that the suit was beyond the limitation period, without addressing the merits of the case.

The primary issue in the current appeal was whether the suit was filed within the limitation period. The appellant argued that he should benefit from Section 14 of the Limitation Act, which excludes time spent pursuing civil proceedings in good faith. The court found that the Deputy Commissioner of Labour (Appeals) was considered a "court" under Section 14, as it had powers similar to those of a civil court. Therefore, the time spent on the earlier appeals was excluded from the limitation calculation, making the suit timely. The appeal was allowed, and the earlier decisions were overturned.
</example_document>

you have the following task.
1. Analyze the given query.
2. Generate 3 documents that answer this query.
3. Print your documents in text format, with at least 300 words.
4. Wrap the output in `json` tags\n{format_instructions}
</system>

<query>
{query}
</query>
""").partial(format_instructions=expansion.get_format_instructions())


llm = ChatOllama(
    model="llama3:8b",
    temperature=0,
    # other params...
)



def avg_embedding(o: ExpansionOutput) -> ExpansionOutputVector:
    embeddings = embedding_qwen.embed_documents([x.text for x in o.documents])
    avg: np.ndarray = np.mean(embeddings, axis=0, keepdims=True)
    return ExpansionOutputVector(
        embedding=avg[0].tolist(),
    )
    # print(avg.shape)
    # return avg


expansion_chain = (
        prompt
        | llm.with_structured_output(ExpansionOutput)
        | avg_embedding
        # | (lambda x: json.dumps(x.tolist()))
        # | StrOutputParser()
).with_types(output_type=ExpansionOutputVector)

