# Decision Log

Here we maintain an organized log for decisions made during the project.

## 2025-06-08

- We will use elastic for document retrieval
- We will do query expansion by:
  - Using the query to generate synthetic documents using llama3:8b quantizied to 4bytes running locally in ollama
  - Compute an average embedding from those documents using Qwen/Qwen3-Embedding-0.6B from hugging face

## 2025-06-07

- Our team will be named `noligomacho` in honor to [this physics paper](https://web3.arxiv.org/abs/1712.02240v1)
which was sadly renamed later.

- We will work on retrieving jurisprudence for legal cases, which consists on
obtaining legal cases for a specific query.

- Our solution consists of a RAG system that:
  - Recieves a query
  - Expands it
  - Retrieves relevant documents
  - Rerank the documents
  - Build a response using information retrieved

### Dataset
- We investigated some datasets designed for legal searches.
  - [CLERK](https://github.com/abehou/CLERC), which is a dataset of 1.8 M U.S. court opinions designed for legal case retrieval and retrieval‑augmented generation, with query–document and passage pairs.
  - [COLIEE](https://coliee.org/resources), which is a benchmark for legal case and statute retrieval, entailment, and question answering, using Canadian and Japanese legal texts. It includes tasks like finding supporting cases and testing legal reasoning, highlighting challenges for retrieval and entailment models.
  - [FIRE2017 IRLeD](https://sites.google.com/view/fire2017irled/track-description) is a legal IR dataset of Indian Supreme Court cases for two tasks: extracting case catchphrases and ranking relevant precedents. It evaluates keyword extraction and citation retrieval using standard IR metrics.
- We eventually decided on FIRE, because it has a reasonable large size for prototyping and high quality legal text. We use a fusion of Task 2's 'Current_Cases' and 'Prior_Cases', neglecting the information on the citations in the texts.
- This resulted in a dataset of 2200 documents with around 7.400 words per document.
