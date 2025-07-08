# Decision Log

Here we maintain an organized log for decisions made during the project.

## 2025-06-07

- Our team will be named `noligomacho` in honor to [this physics paper](https://web3.arxiv.org/abs/1712.02240v1)
which was later renamed.

- We will work on retrieving jurisprudence for legal cases, which consists on
obtaining legal cases for a specific query.

- Our solution consists of a RAG system that:
  - Recieves a query
  - Expands it
  - Retrieves relevant documents
  - Rerank the documents
  - Build a response using information retrieved
