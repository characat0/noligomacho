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
Delhi University medical postgraduate reservation merit equality constitutional validity
</example_query>

<example_document_1>
Students who had qualified for medical degree course got admission under the All India quota of 15 per cent and migrated to different States to pursue the course of study and are now seeking admission into Postgraduate courses.
Their grievance is that the States or concerned authorities have framed admission rules in such a way that they can neither pursue their studies in the migrated State nor in their home State.
Before we address to the controversy we may briefly survey a few decided cases. In [?CITATION?], the admission rules prescribed by the Delhi University provided that 70% of the seats at the post graduate level in the medical courses shall be reserved for students who had obtained their MBBS degree from the same university and the remaining 30% seats were open to all, including the graduates of Delhi. After considering the decisions rendered till that day, this Court took the view that "university-wise preferential treatment may still be consistent with the rule of equality of opportunity where it is calculated to correct an imbalance or handicap and permit equality in the larger sense. If University-wise classification for post graduate medical education is shown to be relevant and reasonable and the differentia has a nexus the larger goal of equalisation of education opportunities the vice of discrimination may not invalidate the rule." The admission to post graduate medical course are determined on the basis of a common entrance test inasmuch as the students of Delhi University are drawn from all over India and are not confined to the Delhi region.
The rule was held to be not invidious and recognised the desires of the students for institutional continuity in education and recognised as one of the grounds justifying the reservation. The argument of excessive reservation in that case could not be considered on the ground of inadequacy of material on record.
On this basis we think the States/Union Territories/Universities should allow students who had pursued courses outside their home State to participate in the entrance examination held in their home State irrespective of any kind of preference that may have been adopted for selection to P.G.medical course.
Before parting with this case, we make it clear that we are not deciding that vexed question of attaining uniformity in all P.G.courses all over the country except to the extent indicated earlier nor we are in a position to say whether institutional preference based on any study in an institution or requirement of residence or both fully complies with the various directions issued by this Court from time to time. We, therefore, think that it would be appropriate for the concerned States or other authorities to achieve uniformity by adopting institutional and/or residential preference in terms of the decisions referred to by us as otherwise, if challenged, may not stand scrutiny of the Court.
The petitions are allowed to the extent indicated above.
</example_document_1>

<example_document_2>
1980 (2) SCC 768 (28 January 1980)
Constitution of India 1950, Articles 15 and 16- Admission to post-graduate course in medicine-Rule of Delhi University-Reservation of 70 per cent of seats at post graduate level for its own university graduates-Validity of.
The University of Delhi has many post-graduate and diploma courses in the faculty of medicine but all of them put together provide 250 seats. The three medical colleges in Delhi turn out annually 400 medical graduates who get 'house' jobs in the local hospitals and qualify themselves for post-graduate courses. As the graduates from the Delhi University could not be accommodated fully or even in part for the post-graduate courses in medicine and as these graduates were not considered for admission into other universities on account of various regional hurdles such as prescription of domicile, graduation in that very university, registration with the State Medical Council, service in the State Medical service etc., the Delhi University had earmarked some seats at the post-graduate level in medicine for the medical graduates of Delhi University.
The petitioner in his writ petition under Article 32 challenged the rule as violative of Articles 14 and 16 of the Constitution and sought the court's writ to direct the University to admit him to the M.D. Course in Dermatology.
It was contended that the University was sustained by Central Government finances, collected from the whole country and the benefits must likewise belong to all qualified students from everywhere. The University justified the reservation on the ground of exclusivism practised by every other University by forbidding Delhi University graduates from getting admission in their colleges and also on account of the reasonableness of institutional continuity in educational pursuits for students who enter a university for higher studies.
832 Dismissing the writ petition.
Merely because New Delhi is the new Capital of Delhi does not justify a disproportionate treatment of the claim to equality on a national level made by its medical graduates.
The question remains : Is a reservation of 70% excessive ? We have travelled through the record, and I agree with my learned brother that the material is so scanty, fragmentary and unsatisfactory that we are prevented from expressing any definite decision on the point. Although we gave sufficient opportunity to the parties, the requisite material has not been forthcoming. Whether or not a reservation of 70% was called for has not been established conclusively. Indeed, there is hardly anything to show that the authorities applied their mind to a cool dispassionate judgment of the problem facing them. Popular agitation serves at best to arouse and provoke complacent or slumbering authority; the judgment and decision of the authority must be evolved from strictly concrete and unemotional material relevant to the issue before it.
Unfortunately, there is little evidence of that in this case. For that reason, I join my learned brother in the directions proposed by him.
The petitioners have raised other contentions also, principally resting on the allegation that the University of Delhi is a centrally administered institution, but I see no force in those submissions.
Accordingly, subject to the two directions proposed by my learned brother the writ petition is dismissed and the parties shall bear their own costs.
N.V.K. Petition dismissed. 
</example_document_2>

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

