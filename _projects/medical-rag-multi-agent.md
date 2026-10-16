---
layout: page
title: "Medical RAG Multi-Agent System"
description: "A French-language medical question-answering system combining hybrid retrieval, knowledge graphs, and multiple agents."
importance: 7
category: ai-engineering
---

## Overview

This project explores retrieval-augmented generation (RAG) for medical question answering in French. It combines document retrieval, knowledge representation, and agent-based workflows to help produce answers grounded in retrieved information.

## Architecture

- **Hybrid retrieval:** dense retrieval with LanceDB combined with BM25-based lexical retrieval.
- **Result fusion:** Reciprocal Rank Fusion (RRF) to combine retrieval results.
- **Knowledge graph:** Neo4j to represent relationships between entities.
- **Multi-agent workflow:** specialized agents coordinate parts of the question-answering process.
- **Language models:** integration of available models through Gemini and Groq.
- **Interface:** an interactive Streamlit application.

## Evaluation

The project includes evaluation experiments using metrics such as ROUGE and BLEU, alongside exploration of RAGAS for RAG-specific evaluation. The reported ROUGE-L score in one experiment was approximately 0.715; this result depends on the dataset, configuration, and evaluation procedure.

## Research motivation

This project explores how retrieval quality, evidence grounding, agent coordination, and evaluation influence the reliability of language-model applications in a specialized domain.

## Technologies

Python, RAG, LanceDB, BM25, Neo4j, Streamlit, language models.

## Repository

[View the source code on GitHub](https://github.com/mouhamedamar141/Assistant_Medical)
