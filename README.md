Overview

This project implements a RAG pipline for Question-Answering task using external unstructured data retrieval.
It combines a retriver and a generator to provide accurate, context-aware answers.

LLMs can hallucinate or lack domain-specific knowledge. This project addresses this by integrating a retriever that fetches relevant chunks from a custom dataset 
and different file formates such as pdf,json,txt,docx,xlsx,csv.

Features
- Vector store for document indexing
- Semantic search for relevant context
- Custome UI and API enpoint
- Document ingestion pipeline

Tech Stack
Frontend - Streamlit
Backend - Python, Flask
LLM - OPENAI GPT 4o
Vector Database - Qdrant
Embeddings - OpenAI
Document Processing - Langchain
Other - HuggingFace, Sentence Transformers
