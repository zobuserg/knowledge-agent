# Knowledge Agent — Sample Document

This document is used to test the ingestion and query pipeline.

## What is an AI Agent?

An AI agent is a system that can perceive its environment, reason about it,
and take actions to achieve a goal. Unlike a simple chatbot that only responds
to prompts, an agent can use tools, search for information, and complete
multi-step tasks autonomously.

## Key components of an agentic RAG system

**Retrieval**: The system searches a vector database to find the most relevant
chunks of text for a given query. Relevance is measured by cosine similarity
between the query embedding and stored document embeddings.

**Augmentation**: The retrieved chunks are passed as context to the language
model along with the user's question.

**Generation**: The language model generates an answer grounded in the
retrieved context, citing specific sources.

## What is ChromaDB?

ChromaDB is an open-source vector database that stores document embeddings
on disk. It supports persistent storage, meaning the data survives restarts.
It is well-suited for local development because it requires no external
services or API keys.

## What are embeddings?

Embeddings are numerical representations of text. Two pieces of text with
similar meaning will have embeddings that are close together in vector space.
This property allows us to search for semantically similar content even if
the exact words don't match.

For example, "machine learning" and "training neural networks" are semantically
related, so their embeddings will be similar even though they share no words.

## LlamaIndex overview

LlamaIndex is a framework for building LLM-powered applications over custom
data. It provides abstractions for:

- Loading documents from various sources (PDF, web, databases)
- Splitting documents into manageable chunks
- Creating and querying vector indexes
- Building agents that can use retrieval as a tool

## Project stack

This project uses:
- LlamaIndex as the agent framework
- ChromaDB as the vector store
- nomic-embed-text (via Ollama) for creating embeddings
- qwen2.5:7b (via Ollama) as the reasoning model
- FastAPI to expose the pipeline as a REST API
