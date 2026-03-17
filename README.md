# The RAG Challenge

## Introduction

This is based off the InsureLLM project from Week 5 of the course.

The data in knowledge-base is an extended version of the knowledge base from week 5 of the fictional company, InsureLLM.

## How this is organized

In the root directory:

[lab.ipynb](lab.ipynb) is a walk-through notebook  
`cd implementation` and then `uv run ingest.py` to ingest data  
From project root, `uv run app.py` to run the Q&A Chatbot  
From project root, `uv run evaluator.py` to run the evaluation  

In the implementation directory:

[answer.py](implementation/answer.py) is the module that answers a user's question. You can change or rewrite `fetch_context()` and `answer_question()`

[ingest.py](implementation/ingest.py) is the module that loads in the data. You can change any of this!

In the evaluation directory:

Private code that runs the evaluation on test data. Don't change this!

## Your mission

1. Work through the lab to understand the current state and ingest data  
2. Reimplement `ingest.py` and `answer.py` with your ideas  
3. Beat Ed and beat the other teams!  

Good luck!