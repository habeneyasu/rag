# The RAG Challenge

Based on the InsureLLM project from Week 5. The knowledge-base contains extended data from the fictional company InsureLLM.

## Quick Start

```bash
# Ingest data
cd implementation && uv run ingest.py

# Run Q&A Chatbot (from project root)
uv run app.py

# Run evaluation (from project root)
uv run evaluator.py
```

## Project Structure

- `implementation/answer.py` - Question answering module with logging (logs to `logs/rag.log`)
- `implementation/ingest.py` - Data ingestion module
- `evaluation/` - Private evaluation code (do not modify)

## Features

- Multi-stage retrieval with reranking
- RAG-Fusion with query variations
- Self-correction and domain knowledge injection
- Optional cross-encoder reranking (`uv add cross-encoder`)

## Configuration

Create a `.env` file with:

```bash
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL=gpt-4.1-nano
MAX_TOKENS=2000
INITIAL_RETRIEVAL_K=12
FINAL_RETRIEVAL_K=12
USE_RERANKING=true
NUM_QUERY_VARIATIONS=4
CHUNKS_PER_QUERY=8
USE_RAG_FUSION=true
USE_CONTEXT_SUMMARIZATION=false
USE_SELF_CORRECTION=true
USE_DOMAIN_KNOWLEDGE=true
DOMAIN_KNOWLEDGE_K=3
```